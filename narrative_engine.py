from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import time
import warnings
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Mapping

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import google.generativeai as genai  # type: ignore

    _GENAI_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    _GENAI_IMPORT_ERROR = _e

try:
    import scheduleanalytics_secrets as _secrets  # type: ignore
except Exception:  # pragma: no cover
    _secrets = None  # type: ignore[assignment]


# Optional, insecure convenience for early testing only:
# If set, the app can generate narratives without requiring users to enter a key or set up a proxy.
# WARNING: Any key embedded in a desktop app can be extracted by end users. Do not use this for production.
EMBEDDED_GEMINI_API_KEY = ""

LOG_NAME = "schedule_analytics"
_RECENT_LOG_LINES: deque[str] = deque(maxlen=400)
_WINDOWS_TLS_CONFIGURED = False

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MODEL_OPTIONS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
GEMINI_MODEL_ALIASES = {
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3-flash-preview": "gemini-3.5-flash",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
}


def normalize_gemini_model(model: str | None) -> str:
    requested = str(model or "").strip()
    if not requested:
        return DEFAULT_GEMINI_MODEL
    return GEMINI_MODEL_ALIASES.get(requested, requested)


def _configure_windows_tls_trust() -> None:
    """
    On some Windows corporate networks, HTTPS is intercepted (MITM) with a custom root cert.

    urllib can succeed using the OS trust store, while requests/certifi fails with:
    SSLCertVerificationError: unable to get local issuer certificate

    Fix: use the Windows trust store for Python HTTPS (via truststore), with a fallback to certifi-win32.
    """
    global _WINDOWS_TLS_CONFIGURED
    if _WINDOWS_TLS_CONFIGURED:
        return
    _WINDOWS_TLS_CONFIGURED = True

    if platform.system() != "Windows":
        return

    logger = _get_logger()
    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
        logger.info("TLS trust: truststore injected into ssl")
        return
    except Exception as e:
        logger.info("TLS trust: truststore not available (%s)", str(e))

    try:
        import certifi_win32  # type: ignore  # noqa: F401

        logger.info("TLS trust: certifi_win32 loaded")
    except Exception as e:
        logger.info("TLS trust: certifi_win32 not available (%s)", str(e))


def _log_path() -> str:
    system = platform.system()
    if system == "Windows":
        base = os.getenv("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
        return os.path.join(base, "ScheduleAnalytics", "schedule_analytics.log")
    if system == "Darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "ScheduleAnalytics",
            "schedule_analytics.log",
        )
    return os.path.join(os.path.expanduser("~"), ".schedule_analytics", "schedule_analytics.log")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOG_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    class _RecentLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
            try:
                _RECENT_LOG_LINES.append(fmt.format(record))
            except Exception:
                pass

    logger.addHandler(_RecentLogHandler())
    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    except Exception:
        logger.addHandler(logging.NullHandler())
    return logger


def get_recent_logs(*, max_lines: int = 200) -> str:
    """
    Returns recent in-memory log lines (useful when users can't access log files).
    """
    try:
        n = max(1, min(int(max_lines), 400))
    except Exception:
        n = 200
    lines = list(_RECENT_LOG_LINES)[-n:]
    return "\n".join(lines)


def probe_gemini_connectivity(
    *,
    url: str = "https://generativelanguage.googleapis.com",
    timeout_s: int = 5,
) -> dict[str, Any]:
    """
    Lightweight network probe to help diagnose corporate proxies (e.g., Zscaler) without needing logs.

    This does NOT require an API key. It checks:
      - DNS resolution
      - HTTPS reachability and latency
      - Whether the response resembles a proxy block page
    """
    out: dict[str, Any] = {
        "url": url,
        "timeout_s": int(timeout_s),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "genai_transport_env": (os.getenv(GENAI_TRANSPORT_ENV_VAR) or ""),
        "genai_transport_selected": _select_genai_transport() or "default",
    }

    host = url.replace("https://", "").replace("http://", "").split("/", 1)[0]
    out["host"] = host

    # DNS
    t0 = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        ips = sorted({info[4][0] for info in infos if info and info[4]})
        out["dns_ok"] = True
        out["dns_s"] = round(time.perf_counter() - t0, 3)
        out["resolved_ips"] = ips[:10]
    except Exception as e:
        out["dns_ok"] = False
        out["dns_s"] = round(time.perf_counter() - t0, 3)
        out["dns_error"] = str(e)
        return out

    # HTTPS GET
    req = urllib.request.Request(url=url, method="GET", headers={"User-Agent": "ScheduleAnalytics/1.0"})
    t1 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out["http_ok"] = True
            out["http_s"] = round(time.perf_counter() - t1, 3)
            out["status"] = int(getattr(resp, "status", 0) or 0)
            headers = dict(resp.headers.items())
            out["server_header"] = headers.get("Server") or headers.get("server")
            # Read only a small sample to avoid big downloads.
            sample = resp.read(1024).decode("utf-8", errors="replace")
            out["body_sample"] = sample[:300]

            sample_lc = sample.casefold()
            out["likely_proxy_block"] = any(
                s in sample_lc
                for s in [
                    "zscaler",
                    "access denied",
                    "blocked",
                    "forbidden",
                    "your request was blocked",
                    "policy",
                ]
            )
    except urllib.error.HTTPError as e:
        out["http_ok"] = True
        out["http_s"] = round(time.perf_counter() - t1, 3)
        out["status"] = int(e.code)
        try:
            body = e.read(1024).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        out["body_sample"] = body[:300]
        body_lc = body.casefold()
        out["likely_proxy_block"] = any(s in body_lc for s in ["zscaler", "access denied", "blocked", "policy"])
    except Exception as e:
        out["http_ok"] = False
        out["http_s"] = round(time.perf_counter() - t1, 3)
        out["http_error"] = str(e)

    # API-path probe (no API key). Expect 401/403 if reachable.
    api_url = url.rstrip("/") + "/v1beta/models"
    req2 = urllib.request.Request(url=api_url, method="GET", headers={"User-Agent": "ScheduleAnalytics/1.0"})
    t2 = time.perf_counter()
    try:
        with urllib.request.urlopen(req2, timeout=timeout_s) as resp2:
            out["api_ok"] = True
            out["api_s"] = round(time.perf_counter() - t2, 3)
            out["api_status"] = int(getattr(resp2, "status", 0) or 0)
            out["api_body_sample"] = resp2.read(512).decode("utf-8", errors="replace")[:300]
    except urllib.error.HTTPError as e:
        out["api_ok"] = True
        out["api_s"] = round(time.perf_counter() - t2, 3)
        out["api_status"] = int(e.code)
        try:
            out["api_body_sample"] = e.read(512).decode("utf-8", errors="replace")[:300]
        except Exception:
            out["api_body_sample"] = ""
    except Exception as e:
        out["api_ok"] = False
        out["api_s"] = round(time.perf_counter() - t2, 3)
        out["api_error"] = str(e)

    # requests-based probe (mirrors what google-generativeai uses under the hood for REST).
    _configure_windows_tls_trust()
    try:
        import requests  # type: ignore

        t3 = time.perf_counter()
        r = requests.get(api_url, timeout=timeout_s, headers={"User-Agent": "ScheduleAnalytics/1.0"})
        out["requests_ok"] = True
        out["requests_s"] = round(time.perf_counter() - t3, 3)
        out["requests_status"] = int(getattr(r, "status_code", 0) or 0)
        out["requests_body_sample"] = (r.text or "")[:300]
    except Exception as e:
        out["requests_ok"] = False
        out["requests_error"] = str(e)

    return out


SYSTEM_REPORT_GUIDELINES = """You are a Senior Project Planner at EllisDon writing a concise schedule narrative for an owner/GC report.
Your goal is to explain project health, schedule movement, and risk in clear general language using ONLY the provided JSON.

Style and output rules:
- Write in bullet points under the required headings. Keep bullets short, direct, and informative.
- The JSON is organized under report_sections. For each narrative section, primarily use the matching report_sections.section_* input block.
- When a section includes narrative_focus, use narrative_focus first. It is the curated, report-ready summary of the larger raw lists in that section.
- Do NOT write activity IDs in the narrative. Activity IDs in the JSON are internal references only.
- Do not list every activity or every activity name. Mention specific activity names only when they are essential to explain the milestone, a driver, or a material risk.
- Prefer grouped language by location, WBS area, phase, discipline, or work type (e.g., pre-construction, construction, interior, exterior, mechanical, electrical).
- Grouped language must still be concrete. When the JSON includes specific floors, levels, zones, rooms, buildings, areas, disciplines, or endpoints, include the most specific shared descriptor and the practical range/limit (for example, "Levels 8-11", "up to Level 11", "Basement Mechanical Room", or "Tower A exterior"). Do not replace specific supported detail with vague phrases such as "multiple floors", "various areas", "several activities", or "interior work progressed" unless the exact locations/limits are not available.
- When summarizing a group, combine the concrete location/work type with the count or movement from the JSON. Example style: "Interior framing advanced through Level 11 with 6 finishes recorded" rather than "Interior work progressed across multiple floors."
- Use plain construction language. Avoid long paragraphs and avoid repeating the same point in multiple sections.
- Zero-Invention: Use ONLY numbers/facts from the JSON input for dates, variances, float, progress, and causality. Do not fabricate missing reasons or impacts.
- If a required value is missing (null/empty), state "Not available in the provided data."
- Treat all float values as DAYS.
- Critical, near-critical, and eroding-risk status are already calculated in Python. Use the JSON classifications exactly as provided.
- Do not calculate or infer float, slack, criticality, near-criticality, or driving status from date gaps, weekends, holidays, or apparent predecessor/successor gaps. Forecast and actual dates are context only.
- Use precomputed_float_buckets as authoritative whenever it appears. If an activity appears in chronological_target_critical_sequence_0_days, primary_critical_path_0_days, or target_critical_activities_0_days, report it as critical with the provided float_current_days, regardless of its dates.
- Use only near_critical_grouped and eroding_risks to describe near-critical work. Do not move activities between critical and near-critical categories.
- Do not invent causality or planning intent. Do not use words like "optimization", "acceleration", "reforecasting", "revised logic", "mitigation", or "re-sequencing" as a cause unless those exact ideas are explicitly supported in variance_reason or supported_shift_evidence. Forecast date movement shows what changed, not why.
- If variance_reason is missing or null, state that the reason is not determinable from the provided schedule data.
- If date_math_guardrail appears in a section, follow it strictly. Do not mention the guardrail in the report unless it is needed to clarify a calendar-related driver.
- If the JSON indicates "BY OTHERS" or a trade/subcontractor, attribute it. If trade data is not present, state it is not available only when needed.

Required narrative structure:

1) Executive Summary & Milestone Status
   - Briefly state the current update data date and the status of the targeted milestone/activity.
   - Use report_sections.section_1_executive_summary_milestone_status.
   - State the variance against the baseline using milestone_variance.total_variance_days from this section.
   - If the variance changed compared with the previous update, state the movement using milestone_variance.period_variance_days from this section.
   - Keep this section to the bottom line. Do not describe the full critical path here.

2) Strategic Progress & Achievements
   - Use report_sections.section_2_strategic_progress_achievements.
   - Start from narrative_focus.actualized_work.top_progress_groups before using the raw actualized_work.groups.
   - Summarize activities that were actualized between the previous and current update using actualized_work.groups. "Actualized" means actual start and/or actual finish occurred within period_context.window.
   - Group progress by location, phase, WBS area, discipline, or work type. Do not list each activity.
   - Use group-level started_count and finished_count to describe the scale of progress.
   - Use each group's wbs_path, leaf_wbs_path, wbs_hierarchy_root_to_leaf, and representative activity names to preserve exact supported locations/levels/areas. If level/floor/location detail is visible, state the exact range or furthest progressed point instead of saying "multiple floors" or "across areas."
   - Mention meaningful completed/started work and any visible quality/safety/control milestones only if supported by activity names.
   - Also summarize narrative_focus.in_progress_finish_extensions.top_extension_groups and in_progress_finish_extensions.groups: activities that were expected to finish in this update window but remain in progress and have a later current forecast finish.
   - For finish extensions, describe the affected grouped areas, exact available locations/levels, and the scale of movement; do not name every activity.

3) Scope Changes & New Additions
   - Use report_sections.section_3_scope_changes_new_additions.
   - If global_new_scope.count > 0, summarize the new additions even if they are not critical.
   - Use global_new_scope.groups and report the largest or most relevant grouped areas only.
   - Use change_delay_new_scope.groups to identify additions specifically under the Change/Delay WBS area.
   - If Basement, Mechanical Room, Mech, Electrical, or other important discipline/location terms appear, call them out briefly.
   - Use downstream_logic_indicators.has_critical_path_driver and downstream_logic_indicators.has_cross_wbs_connection to distinguish supported critical-path influence from general downstream logic connections.
   - If linkage cannot be supported, use downstream_logic_indicators.impact_statement or state "Downstream influence is not determinable from the provided logic/data."
   - Keep this section short. Do not dump activity lists.

4) Critical Path/Risk
   - Use report_sections.section_4_critical_path_risk.
   - Explain the current target-driving critical path using current_target_critical_path.precomputed_float_buckets and narrative_focus first, especially current_active_critical_driver, active_critical_task_examples, next_not_started_critical_task_examples, upstream_driver_task_examples, downstream_finish_or_milestone_examples, critical_activity_groups, and tied_branch_examples. Use current_target_critical_path.primary_path / precomputed_float_buckets.primary_critical_path_0_days as the source of the logical sequence order (what drives what). Use chronological_target_critical_sequence_0_days only for the complete set of critical activities and date context, never for predecessor order.
   - Treat current_target_critical_path.precomputed_float_buckets.chronological_target_critical_sequence_0_days and target_critical_activities_0_days as the authoritative SET of critical activities (membership and float only). They are sorted by date and are NOT a logical predecessor chain — see the next rule before describing any sequence.
   - For the logical order of the path (what drives what), follow current_target_critical_path.precomputed_float_buckets.primary_critical_path_0_days (equivalently current_target_critical_path.primary_path.activity_chain), in its given upstream-to-target order, together with primary_path.logic_links. Only state that one activity is "followed by", "drives", or "feeds" another when they are adjacent in primary_path or connected in logic_links. Never infer a predecessor/successor link from the date-sorted chronological list, because parallel branches are interleaved there.
   - Use only current_target_critical_path.critical_activities, critical_activity_groups, paths, and their is_critical/on_critical_trace/float_current_days fields to describe critical work. Do not add or remove activities based on date gaps.
   - Use path_method as the path-definition method. The critical path is extracted by treating TASK/TASKPRED as a DAG and tracing backward from the target through predecessors whose P6 total float is less than or equal to the current branch float within tolerance. Keep near-critical discussion for Section 5.
   - If current_target_critical_path.critical_trace_bridges or a logic link trace_bridge is present, explain it as a calendar-based trace bridge only. A bridged predecessor is upstream driver context, not a 0-float critical activity, unless its own is_critical flag is true.
   - If narrative_focus.current_active_critical_driver or current_target_critical_path.critical_status_focus.current_active_critical_driver is present, lead with it as the current active critical driver. Then describe the next not-started critical work. Do not say the critical chain starts with a future/not-started activity when an in-progress critical activity exists.
   - Use current_status exactly as provided. "in_progress" means active critical work; "not_started" means future/next critical work; "completed" means completed upstream context only.
   - Describe critical work from the earliest upstream physical work toward the final finish/milestone by following the logical order in primary_critical_path_0_days (primary_path.activity_chain), not the date-sorted chronological list. When several critical activities run in parallel (different branches in alternate_tied_paths or different critical_activity_groups that converge later), describe them as parallel branches converging toward the target — do not chain them one-after-another.
   - In schedule language, "driving" means upstream predecessor work controlling downstream successor work. Do not state that finalization, commissioning, line painting, paving, off-bridge closeout, or milestone activities drive the path when they appear at the tail end. Say those activities are downstream/tail-end work driven by the upstream critical work.
   - If a single branch of upstream concrete/structural work feeds downstream finish/off-bridge work along the SAME primary_path chain, describe that branch as one continuous sequence into finalization. But when the JSON separates work into different branches (primary_path vs alternate_tied_paths, or distinct critical_activity_groups), keep them as separate parallel branches; do not merge two parallel branches into one linear "X then Y" sequence.
   - If narrative_focus.tied_branch_examples is present, describe those as parallel/tied predecessor branches feeding into the listed shared activity. Prefer each tied branch example's report_phrase when present. Do not say the active critical driver is followed by a tied branch; use "In parallel" or "A tied predecessor branch" language.
   - Lead with the major grouped areas represented by current_target_critical_path.critical_activity_groups. The primary_path is only an example branch and must not be described as the whole critical path when other critical groups are listed.
   - If current_target_critical_path.completed_upstream_summary is present, mention completed upstream work only as completed context when useful. Do not describe completed upstream work as current critical risk.
   - Use logic_links relationship_type and lag_days when they help explain why one activity drives the next.
   - If path_change_from_previous_update.changed is true, use path_change_from_previous_update.path_change_interpretation first. State the previous unique upstream driver sequence, the current unique upstream driver sequence, and the shared downstream sequence when provided.
   - Do not describe path_change_interpretation.previous_unique_upstream_sequence as the whole previous critical path when shared_downstream_sequence is present. Say it flowed into the shared downstream sequence.
   - Do not describe shared_downstream_sequence as the part that shifted. If commissioning, closeout, or other downstream work is in shared_downstream_sequence, describe it as the common downstream continuation after the changed upstream driver portion.
   - Do not say previous upstream work is complete unless path_change_interpretation.previous_unique_upstream_current_status_counts shows it. If statuses are mixed, say the prior upstream driver is no longer the current traced driver rather than saying it is fully complete.
   - Then use path_change_from_previous_update.previous_primary_path and current_primary_path only as backup detail.
   - If current_target_critical_path.primary_path.constraint_drivers is non-empty, mention that the listed activity appears constraint-driven and that the selected critical predecessor may not be the main start driver.
   - Explain only supported observed changes using supported_shift_evidence.possible_shift_causes and supported_shift_evidence.cause_assessment. If supported_shift_evidence.variance_reason is null, say the specific reason for the path shift is not determinable; do not infer intent from observed date changes.
   - Consider relationship changes, lag/type changes, constraint/calendar changes, forecast date movement, duration/status changes, new activities, finish extensions, float erosion, and change/delay links when they appear in supported_shift_evidence.possible_shift_causes.
   - If supported_shift_evidence.upstream_new_activity_links_to_current_critical_path contains chains, use them only as upstream new/change/delay context. State that the new upstream item has logic leading to the current critical chain, but do not call it critical unless its own is_critical and float_current_days fields say it is critical.
   - If no supported cause is available, explicitly state that the path changed but the specific cause is not determinable from the provided schedule data.
   - Use added/removed WBS areas, phases, disciplines, work types, and exact supported locations/levels to explain the shift. Keep the explanation concise but more detailed than a one-line statement.
   - Use change_delay_context only where it shows a supported link from changes to downstream critical-path work.
   - Avoid activity IDs and avoid naming every activity in the path. Name the key driving activities when they clarify the sequence, especially from narrative_focus primary/longest tied path sequences.

5) Risks & Float Erosion (Look-Ahead)
   - Use report_sections.section_5_risks_float_erosion.
   - Do not repeat the critical path discussion from Section 4.
   - Use precomputed_float_buckets.near_critical_bucket_rule and near_critical_grouped as the authoritative near-critical bucket.
   - Use only near_critical_grouped, eroding_risks, and their precomputed float_current_days/float_loss_days fields. Do not infer near-critical exposure from date spacing.
   - Do not classify activities already included in the Section 4 critical trace as near-critical, even if their float is inside the near-critical threshold.
   - Focus on near-critical activities using near_critical_grouped. Near-critical activities are selected by Python from non-summary current-schedule activities using P6 total float greater than the critical threshold and less than or equal to settings.variance_threshold above that threshold, excluding activities already included in the Section 4 critical trace.
   - Summarize near-critical exposure by grouped area, location, phase, discipline, or work type.
   - For each main near-critical area, NAME the key activities (from near_critical_grouped.groups[].representative_items[].task_name) and briefly say what the work is and its float. Do not write only a bare count like "6 activities in this area"; give a concrete picture (e.g. "removals and surface prep — sidewalk concrete removal, expansion joint removal and overhang bracket install — at 1-4 days float"). You may still group, but ground it in the actual activity names. Do not list every activity; pick the most representative.
   - Use near_critical_grouped.driving_links (predecessor links among the near-critical activities, from_task_name -> to_task_name with relationship_type/lag_days) to describe how the near-critical work sequences and drives itself (e.g. "removal drives the repair, which drives forming"). Only state a driving relationship that appears in driving_links; do not infer one from dates or area grouping.
   - Preserve exact supported locations/levels/areas for each near-critical exposure. Avoid generic risk phrases when the grouped data identifies a specific work area.
   - Use eroding_risks to highlight where float is eroding faster than time is passing.
   - Do not discuss in-progress finish extensions here unless they also appear in near_critical_grouped or eroding_risks.
   - Do not repeat change/delay scope discussion here unless near_critical_grouped or eroding_risks directly supports the risk.
   - Call out blockers/constraints such as permits, weather, third-party work, inspections, or procurement only when explicitly indicated in names or summaries.

6) Look-Ahead Window Analysis
   - Use report_sections.section_6_look_ahead_window_analysis.
   - Use upcoming_work.groups to summarize work falling between the current update data date and the user-defined look-ahead horizon.
   - State the window dates and horizon length from look_ahead_window.
   - Group the upcoming work by location, WBS area, phase, discipline, or work type. Do not list every activity.
   - Use group-level forecast_start_count, forecast_finish_count, not_started_count, and in_progress_count to describe the scale of upcoming work.
   - Use exact supported locations/levels/areas from the group fields and representative names. If the data shows upcoming work up to a specific level or within a specific room/zone, state that detail.
   - For each main upcoming group, NAME the key activities (from upcoming_work.groups[].items[].task_name) and briefly describe the work, rather than only giving counts per area. Pick the most representative activities; do not list every one.
   - Use upcoming_work.driving_links (predecessor links among upcoming activities, from_task_name -> to_task_name) to describe how the window's work sequences and drives itself where the data shows it. Only state a driving relationship present in driving_links.
   - Highlight schedule-sensitive upcoming groups using schedule_sensitive_upcoming_work.groups, but do not repeat the full critical path or near-critical risk discussion from Sections 4 and 5.
   - Keep the discussion practical and short: focus on the largest upcoming work groups and the most schedule-sensitive groups.
   - Do not write activity IDs and do not name every activity.

7) Mitigation Strategies
   - Use report_sections.section_7_mitigation_inputs.
   - Section 7 inputs are intentionally compact. Use critical_path_focus, near_critical_focus, and look_ahead_focus to provide 2-3 practical mitigation bullets tailored to the highest-priority areas/trades identified in Sections 4-6.
   - Use current_active_critical_driver, active_critical_task_examples, top_critical_activity_groups, upstream_driver_task_examples, top_near_critical_groups, top_eroding_risks, and top_schedule_sensitive_groups as the main mitigation targets.
   - Recommendations may include crashing, re-sequencing, overlapping/fast-tracking, added supervision, procurement expediting, constraint removal, or trade coordination.
   - Keep recommendations specific to the described drivers, risks, trades, and locations/levels. Do not recommend unrelated actions.
"""

SYSTEM_DELAY_ANALYSIS_GUIDELINES = """You are a Senior Project Planner at EllisDon writing a schedule delay analysis across multiple schedule updates for an owner/GC report.
Your goal is to tell the story of how the target milestone's critical path and completion date evolved across the updates, using ONLY the provided JSON.

Input shape:
- settings: target_activity_id, variance_threshold (days), update_count.
- target: the milestone task_name and baseline_finish_date (the contractual reference).
- update_timeline: one entry per schedule update, in chronological order (oldest to newest). Each has data_date, target_finish_date, variance_vs_baseline_days, and critical_path_summary (active_driver, wbs_groups, driver_sequence with float_current_days in DAYS and status).
- path_change_story: one entry per consecutive transition (previous update -> next update). Each has period_variance_days, path_changed, target_date_moved, path_changed_without_date_movement, shift_classification (evidence tags), drivers_added_to_current_path, drivers_removed_from_previous_path, previous_unique_upstream_status_counts, supported_shift_causes, and date_held_evidence.

Style and output rules:
- Write in clear bullet points under the required headings. Keep bullets short and concrete.
- Do NOT write activity IDs. Use activity names and WBS/area/discipline grouping.
- Treat all float values as DAYS. Criticality is precomputed in Python; use path_changed, status, and float exactly as provided.
- Zero-Invention: use ONLY the numbers, names, dates, and tags in the JSON. Do not invent causes, intent, or impacts.
- Forecast/actual dates are context only. Do not infer criticality, float, or driving status from date gaps.
- If a value is missing (null/empty), state "Not available in the provided data." If a path changed but no supported cause exists in shift_classification or supported_shift_causes, say the specific cause is not determinable from the provided schedule data.

Required structure:

1) Overview & Net Movement
   - State the target milestone name, its baseline_finish_date, and the current (latest update) target_finish_date with the latest variance_vs_baseline_days.
   - State the number of updates analysed and how many transitions had path_changed = true.
   - Give the overall arc in one or two bullets: did the completion date improve, slip, or hold across the series.

2) Update-by-Update Timeline
   - For each update in update_timeline (oldest to newest), start the bullet with data_date, target_finish_date, and variance_vs_baseline_days.
   - Then BRIEFLY DESCRIBE THE WHOLE CRITICAL PATH from the earliest upstream driver through to the target milestone. Use critical_path_summary.driver_sequence (ordered upstream -> target) and critical_path_summary.path_wbs_sequence (the ordered WBS areas/phases the path flows through).
   - Lead with critical_path_summary.active_driver (the current in-progress or next not-started critical activity), then walk the chain downstream through each major WBS area/phase, ending at the target. Describe it as a continuous flow, e.g. "from tender/award through bridge jacking and temporary shoring, into West Pier work and deck placement, then off-bridge sidewalk, paving and final line painting."
   - You MAY group long runs of same-area activities into a phrase (e.g. "deck pour, cure and sidewalk/median forming") rather than naming each one, but you MUST cover the path end-to-end to the target. Do NOT stop at the first upstream driver or only name the active driver.
   - Name the key driving activities and any milestones along the way. Treat float as DAYS from driver_sequence; do not infer criticality from dates.
   - If an update has a critical_path_warning or target_warning, note that the target may not be logic-driven in that update (its path may collapse to just the target) and treat its path cautiously.

3) Critical Path Changes — When and Why
   - For each transition in path_change_story where path_changed = true, write a short paragraph/bullet group that states:
     - WHEN: between which two updates (use from/to data dates).
     - WHAT CHANGED: the new driving work (drivers_added_to_current_path, names + WBS area) and the work that dropped off the path (drivers_removed_from_previous_path).
     - WHY: interpret shift_classification.tags strictly:
        * "previous_path_progressed" -> the prior driving path advanced/completed, so a different chain now governs.
        * "duration_constraint_or_calendar_change_on_current_path" -> a duration, constraint, or calendar change on the new path; cite the specific changed fields from supported_shift_causes.
        * "logic_or_overlap_change_on_current_path" -> a relationship type or lag/overlap change; cite it.
        * "new_activity_became_a_driver" -> a newly added activity is now on the driving path; name it.
        * "cause_not_determinable_from_data" -> state the path changed but the specific cause is not determinable.
     - DATE IMPACT: use period_variance_days. Negative means the completion pulled earlier (improved); positive means it slipped.
     - RESULTING PATH: after explaining the change, briefly state where the new critical path now runs end-to-end, using the later update's critical_path_summary.driver_sequence / path_wbs_sequence (active driver through to the target). Do not stop at the newly added drivers.
   - Do not describe shared downstream work as newly shifted; only the unique upstream driver portion changed.

4) Path Shifted Without Date Movement
   - For any transition where path_changed_without_date_movement = true, explain that the driving path changed but the target completion date did NOT move.
   - Use date_held_evidence to give the supported mechanism: a different chain became governing at the same total length, typically because the previous path progressed (previous_path_progress status counts) while another chain absorbed the slack via the listed task_attribute_changes_on_current_path or relationship_changes_on_current_path (duration, overlap/lag, or calendar).
   - If date_held_evidence has no supported attribute/logic changes, state that the path shifted but the offsetting mechanism is not determinable from the provided data. Do not invent one.
   - If no transition has this flag, omit this section.

5) Risk Outlook & Mitigation
   - First, briefly describe the latest (current) update's critical path end-to-end, from active_driver through each major WBS area to the target, using the latest update's critical_path_summary.driver_sequence / path_wbs_sequence. State its least_float_current_days.
   - Then, based on that path and the most recent transitions, give 2-3 practical, specific mitigation bullets aimed at the current active_driver and the areas/trades that most recently became critical.
   - Recommendations may include re-sequencing, fast-tracking/overlap, procurement expediting, constraint/permit removal, added supervision, or trade coordination. Keep them tied to the named drivers and locations.
"""

DOTENV_FILENAME = ".env"
API_KEY_ENV_VAR = "GEMINI_API_KEY"
GENAI_TRANSPORT_ENV_VAR = "SCHEDULE_ANALYTICS_GENAI_TRANSPORT"


def _select_genai_transport() -> str | None:
    """
    Choose a transport for google-generativeai.

    Why: On some Windows networks, gRPC/HTTP2 can stall while plain REST works.
    Override with SCHEDULE_ANALYTICS_GENAI_TRANSPORT=grpc|rest|grpc_asyncio|auto.
    """
    raw = (os.getenv(GENAI_TRANSPORT_ENV_VAR) or "").strip().casefold()
    if raw in {"", "auto"}:
        return "rest" if platform.system() == "Windows" else None
    if raw in {"rest", "grpc", "grpc_asyncio"}:
        return raw
    return None


def _read_dotenv_key(dotenv_path: str = DOTENV_FILENAME, env_var: str = API_KEY_ENV_VAR) -> str | None:
    """
    Minimal .env reader (no external dependency).
    Supports lines like:
      GEMINI_API_KEY=...
      GEMINI_API_KEY="..."
      GEMINI_API_KEY='...'
    """
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != env_var:
                    continue
                val = v.strip().strip("'").strip('"').strip()
                return val or None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _candidate_dotenv_paths(dotenv_filename: str = DOTENV_FILENAME) -> list[str]:
    paths: list[str] = []

    # 1) Current working directory
    paths.append(os.path.join(os.getcwd(), dotenv_filename))

    # 2) Directory containing this module (dev usage)
    try:
        paths.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), dotenv_filename))
    except Exception:
        pass

    # 3) Directory of the executable (PyInstaller / packaged usage)
    try:
        paths.append(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), dotenv_filename))
    except Exception:
        pass

    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def find_api_key() -> str | None:
    """
    Return API key from environment or local .env (if present), else None.
    """
    def normalize(value: str | None) -> str | None:
        v = (value or "").strip().strip("'").strip('"').strip()
        if not v:
            return None
        if v in {"YOUR_REAL_KEY_HERE", "your_key_here"}:
            return None
        return v

    embedded = normalize(EMBEDDED_GEMINI_API_KEY)
    if embedded:
        return embedded

    # Optional build-time injected module (recommended if you want an EXE with an embedded key
    # without committing the key to git history).
    try:
        embedded_mod = normalize(getattr(_secrets, "EMBEDDED_GEMINI_API_KEY", None) if _secrets else None)
        if embedded_mod:
            return embedded_mod
    except Exception:
        pass

    key = normalize(os.getenv(API_KEY_ENV_VAR))
    if key:
        return key
    for p in _candidate_dotenv_paths():
        k = normalize(_read_dotenv_key(p))
        if k:
            return k
    return None


def _key_diagnostics() -> dict[str, Any]:
    def normalize(value: str | None) -> str | None:
        v = (value or "").strip().strip("'").strip('"').strip()
        if not v:
            return None
        if v in {"YOUR_REAL_KEY_HERE", "your_key_here"}:
            return None
        return v

    diag: dict[str, Any] = {}
    diag["embedded_constant_set"] = bool(normalize(EMBEDDED_GEMINI_API_KEY))
    diag["env_var_set"] = bool(normalize(os.getenv(API_KEY_ENV_VAR)))
    diag["dotenv_candidates"] = _candidate_dotenv_paths()
    diag["dotenv_found"] = []
    for p in diag["dotenv_candidates"]:
        try:
            if os.path.exists(p):
                diag["dotenv_found"].append(p)
        except Exception:
            continue
    diag["secrets_module_imported"] = bool(_secrets is not None)
    try:
        diag["secrets_module_key_set"] = bool(
            normalize(getattr(_secrets, "EMBEDDED_GEMINI_API_KEY", None) if _secrets else None)
        )
    except Exception:
        diag["secrets_module_key_set"] = False
    return diag


def _get_api_key(api_key: str | None = None) -> str:
    key = (api_key or find_api_key() or "").strip()
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Set environment variable GEMINI_API_KEY, or create a local .env file "
            "with GEMINI_API_KEY=..., or pass api_key=... to generate_narrative().\n\n"
            f"Diagnostics: {json.dumps(_key_diagnostics(), indent=2)}"
        )
    return key


def configure_genai_client(api_key: str | None = None) -> None:
    """
    Configure the google-generativeai client (key resolution + Windows TLS trust + transport).

    Shared by the report path (`_run_gemini`) and the Q&A function-calling agent so both honor the
    same key sources and corporate-network workarounds.
    """
    key = _get_api_key(api_key)
    if genai is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'google-generativeai'. Install it with: pip install google-generativeai"
        ) from _GENAI_IMPORT_ERROR
    _configure_windows_tls_trust()
    transport = _select_genai_transport()
    try:
        if transport:
            genai.configure(api_key=key, transport=transport)
        else:
            genai.configure(api_key=key)
    except TypeError:
        genai.configure(api_key=key)


def _run_gemini(system_instruction: str, user_content: str, *, model: str, api_key: str | None = None) -> str:
    """
    Shared Gemini call used by both the narrative and delay-analysis reports.

    Handles key resolution, Windows TLS trust, transport selection, hard timeouts, and error mapping.
    Security note: API keys must NOT be hardcoded. Provide via GEMINI_API_KEY env var or api_key parameter.
    """
    model = normalize_gemini_model(model)
    key = _get_api_key(api_key)

    if genai is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'google-generativeai'. Install it with: pip install google-generativeai"
        ) from _GENAI_IMPORT_ERROR

    logger = _get_logger()
    t0 = time.perf_counter()

    _configure_windows_tls_trust()

    transport = _select_genai_transport()
    try:
        if transport:
            genai.configure(api_key=key, transport=transport)
        else:
            genai.configure(api_key=key)
    except TypeError:
        # Backward-compat for older google-generativeai versions that don't accept transport=.
        genai.configure(api_key=key)
        transport = None

    payload_bytes = len(user_content.encode("utf-8", errors="ignore"))
    logger.info("Gemini start model=%s transport=%s payload_bytes=%s", model, transport or "default", payload_bytes)

    # google-generativeai supports system_instruction on GenerativeModel.
    model_obj = genai.GenerativeModel(model_name=model, system_instruction=system_instruction)
    try:
        t_call = time.perf_counter()
        # Some environments ignore SDK-level timeouts (especially with gRPC/HTTP2).
        # Add a hard timeout so the UI doesn't hang indefinitely.
        request_timeout_s = 120
        hard_timeout_s = 150
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(model_obj.generate_content, user_content, request_options={"timeout": request_timeout_s})
            resp = fut.result(timeout=hard_timeout_s)
        logger.info(
            "Gemini ok model=%s call_s=%.3f total_s=%.3f",
            model,
            time.perf_counter() - t_call,
            time.perf_counter() - t0,
        )
    except FutureTimeoutError as e:
        logger.error(
            "Gemini timeout model=%s transport=%s total_s=%.3f",
            model,
            transport or "default",
            time.perf_counter() - t0,
        )
        raise RuntimeError(
            "Gemini call timed out (hard timeout). On some Windows networks, gRPC/HTTP2 stalls even when HTTPS works. "
            "Try setting environment variable SCHEDULE_ANALYTICS_GENAI_TRANSPORT=rest and retry."
        ) from e
    except Exception as e:
        msg = str(e)
        logger.error(
            "Gemini error model=%s transport=%s payload_bytes=%s total_s=%.3f err=%s",
            model,
            transport or "default",
            payload_bytes,
            time.perf_counter() - t0,
            msg,
        )
        if "429" in msg or "quota" in msg.lower():
            raise RuntimeError(
                "Gemini quota/rate-limit hit (HTTP 429). Try again later, or switch to a lower-tier model "
                "like gemini-3.1-flash-lite, or reduce usage."
            ) from e
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            raise RuntimeError(
                f"Gemini request timed out. Check your internet/firewall and try {DEFAULT_GEMINI_MODEL}. "
                "If you are on a corporate network, the Gemini endpoint may be blocked."
            ) from e
        raise

    text = getattr(resp, "text", None)
    if not text:
        # Fallback for alternate response shapes.
        try:
            text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
        except Exception:
            text = str(resp)
    return str(text).strip()


def generate_narrative(data_digest: Mapping[str, Any], *, api_key: str | None = None, model: str = DEFAULT_GEMINI_MODEL) -> str:
    """Generate a schedule narrative using Gemini."""
    # Keep payload compact to reduce request size / latency.
    user_content = json.dumps(data_digest, default=str)
    return _run_gemini(SYSTEM_REPORT_GUIDELINES, user_content, model=model, api_key=api_key)


SYSTEM_PROJECT_OVERVIEW_GUIDELINES = """You are a Senior Project Planner at EllisDon writing a plain-language overview of a single construction schedule (one XER file) for an owner/GC reader who wants to understand the project at a glance. Use ONLY the provided JSON.

Input shape:
- project: name, data_date, schedule_start, schedule_finish, total_duration_days, activity_count, overall_percent_complete, status_breakdown (completed/in_progress/not_started).
- wbs_scope_top_areas: the largest WBS areas with activity counts (this is the scope/phase breakdown).
- milestones: key milestones with dates, kind (finish vs start_or_other), and status.
- main_paths: the schedule's main backbones (longest driving paths), each with span_days, start/finish dates, percent_complete, is_current_critical_path, starts_constraint_driven, wbs_sequence (the ordered areas the path flows through), and activity_sequence (ordered activity names from upstream to the path end). These are derived from P6's computed dates and logic, not from summing durations.
- data_quality_note: if present, a caution about stale/un-scheduled dates.

Style:
- Write in clear sections with short bullets. Plain construction language.
- Treat any float as DAYS. Do not invent facts, dates, causes, or scope. If a value is missing, say "Not available in the provided data."
- Do NOT write activity IDs.

Required structure:

1) What This Project Is
   - State the project name, the overall schedule window (schedule_start to schedule_finish) and total_duration_days, the activity_count, and overall_percent_complete with the status_breakdown.
   - Describe what kind of project it is and its main scope/phases using wbs_scope_top_areas (e.g. pre-construction, foundations, structure, envelope, interior fit-out, commissioning) and the milestone names. Keep it to what the WBS and milestones actually show.

2) Schedule Status & Key Milestones
   - Summarize how far along the project is (percent complete, what is completed vs in progress vs not started).
   - List the key milestones from milestones (name + date), grouping start vs finish milestones; highlight the major completion milestone(s).

3) Main Paths (Backbones)
   - For each entry in main_paths, describe the path as a continuous flow through its wbs_sequence areas, naming a few key activities from activity_sequence (start, a mid point, and the end). State its span_days, percent_complete, and start/finish dates.
   - Clearly identify the path where is_current_critical_path is true as the project's current driving/critical path.
   - If starts_constraint_driven is true for a path, note that it begins at a constraint-driven activity.
   - Do not merge separate paths into one sequence; describe each backbone on its own.

4) Observations
   - 2-4 neutral, supported observations: which areas carry the most work (from wbs_scope_top_areas), how the backbones relate (e.g. shared early work then splitting by stage/zone), and any data_quality_note. Do not speculate beyond the JSON.
"""


SYSTEM_HANDOVER_BRIEFING_GUIDELINES = """You are a Senior Project Planner at EllisDon. A scheduler is taking over this project and needs a briefing from the schedule. You are given the WHOLE schedule (one XER), so read it and explain the project the way you would brief the incoming scheduler. Use ONLY the provided JSON.

Input shape:
- project_facts: name, data_date, schedule_start, schedule_finish, total_duration_days, activity_count, overall_percent_complete, status_breakdown, wbs_scope_top_areas, milestones.
- current_critical_path: the code-computed current driving/critical chain (ordered activity names with status and finish). Trust this for "what is driving the project."
- all_activities: EVERY activity, each with name, wbs, status, start, finish, total_float_days (DAYS), milestone.
- relationships: predecessor -> successor links as [predecessor_name, relationship_type, successor_name, lag_days].

Grounding rules (important):
- Treat total_float_days as authoritative. An activity at ~0 days total float is critical; near 0 is near-critical. Do NOT infer float, slack, or criticality from dates or gaps.
- For "main drivers / critical path", lead with current_critical_path; you may corroborate with activities at ~0 float and the relationships, but do not invent a different critical path.
- Use real activity names and dates from the data. Do not invent activities, dates, causes, or scope. If something is not in the data, say so. No activity IDs.
- Group by WBS area / phase / discipline; you do not need to list every activity.

Write the briefing with these sections:

1) Project at a Glance
   - What this project is (type, scope, location if evident from the name/WBS), its schedule window (schedule_start to schedule_finish), total duration, and activity count.

2) Where It Stands Now
   - The data date and overall_percent_complete. Summarize what is completed, what is in progress, and what remains (use status_breakdown and the activities). Call out the main areas of recent/active work.

3) Key Dates & Milestones
   - The major start and finish milestones with dates from project_facts.milestones; highlight the contractual/overall completion milestone(s).

4) What Is Driving the Project (Critical & Near-Critical)
   - Describe the current driving/critical path from current_critical_path as a continuous flow through its WBS areas, naming the key activities and the current active driver.
   - Note the main near-critical exposure (activities at low but non-zero total_float_days) by area.

5) Risks, Constraints & Things to Watch
   - Point out anything a new scheduler should know that is supported by the data: constraint-driven starts, long-lead/procurement chains, big remaining scope concentrations (from wbs_scope_top_areas), permits/work-restriction windows, or milestones with little float. Only what the data supports.

6) Where to Dig In
   - 2-3 concrete suggestions of what the incoming scheduler should review first, tied to the drivers and risks above.
"""


def generate_handover_briefing(
    payload: Mapping[str, Any],
    *,
    instruction: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate a full-schedule handover briefing (sends the whole parsed schedule) using Gemini."""
    system_instruction = SYSTEM_HANDOVER_BRIEFING_GUIDELINES
    if instruction and str(instruction).strip():
        system_instruction = (
            SYSTEM_HANDOVER_BRIEFING_GUIDELINES
            + "\n\nAdditional user instruction for this run (follow it unless it conflicts with the no-invention rule):\n"
            + str(instruction).strip()
        )
    user_content = json.dumps(payload, default=str)
    return _run_gemini(system_instruction, user_content, model=model, api_key=api_key)


def generate_project_overview(
    overview_digest: Mapping[str, Any],
    *,
    instruction: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate a single-schedule project overview report using Gemini."""
    system_instruction = SYSTEM_PROJECT_OVERVIEW_GUIDELINES
    if instruction and str(instruction).strip():
        system_instruction = (
            SYSTEM_PROJECT_OVERVIEW_GUIDELINES
            + "\n\nAdditional user instruction for this run (follow it unless it conflicts with the no-invention rule):\n"
            + str(instruction).strip()
        )
    user_content = json.dumps(overview_digest, default=str)
    return _run_gemini(system_instruction, user_content, model=model, api_key=api_key)


def generate_delay_report(
    delay_digest: Mapping[str, Any],
    *,
    instruction: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """
    Generate a multi-update schedule delay-analysis report using Gemini.

    The optional `instruction` is appended to the system guidelines as an extra focus for this run.
    """
    system_instruction = SYSTEM_DELAY_ANALYSIS_GUIDELINES
    if instruction and str(instruction).strip():
        system_instruction = (
            SYSTEM_DELAY_ANALYSIS_GUIDELINES
            + "\n\nAdditional user instruction for this run (follow it unless it conflicts with the Zero-Invention rule):\n"
            + str(instruction).strip()
        )
    user_content = json.dumps(delay_digest, default=str)
    return _run_gemini(system_instruction, user_content, model=model, api_key=api_key)


def generate_narrative_via_proxy(
    data_digest: Mapping[str, Any],
    *,
    proxy_url: str,
    user_token: str,
    model: str,
    timeout_s: int = 120,
) -> str:
    """
    Call a backend proxy that holds the Gemini API key, authenticated with a per-user token.
    """
    model = normalize_gemini_model(model)
    logger = _get_logger()
    t0 = time.perf_counter()
    url = proxy_url.strip().rstrip("/") + "/v1/narrative"
    token = user_token.strip()
    if not token:
        raise RuntimeError("Missing user token for narrative proxy.")

    body = json.dumps({"data_digest": data_digest, "model": model}, default=str).encode("utf-8")
    logger.info("Proxy start model=%s body_bytes=%s url=%s", model, len(body), url)
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        logger.info("Proxy ok model=%s total_s=%.3f", model, time.perf_counter() - t0)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        logger.error("Proxy http_error code=%s total_s=%.3f body=%s", e.code, time.perf_counter() - t0, msg)
        raise RuntimeError(f"Proxy error {e.code}: {msg}") from e
    except urllib.error.URLError as e:
        logger.error("Proxy url_error total_s=%.3f err=%s", time.perf_counter() - t0, str(e))
        raise RuntimeError(f"Could not reach proxy: {e}") from e

    data = json.loads(raw)
    text = data.get("text")
    if not text:
        raise RuntimeError(f"Proxy response missing 'text': {data}")
    return str(text).strip()


def generate_delay_report_via_proxy(
    delay_digest: Mapping[str, Any],
    *,
    proxy_url: str,
    user_token: str,
    model: str,
    instruction: str | None = None,
    timeout_s: int = 120,
) -> str:
    """
    Call the backend proxy to produce a delay-analysis report (proxy holds the Gemini key).
    """
    model = normalize_gemini_model(model)
    logger = _get_logger()
    t0 = time.perf_counter()
    url = proxy_url.strip().rstrip("/") + "/v1/delay"
    token = user_token.strip()
    if not token:
        raise RuntimeError("Missing user token for narrative proxy.")

    payload: dict[str, Any] = {"data_digest": delay_digest, "model": model}
    if instruction and str(instruction).strip():
        payload["instruction"] = str(instruction).strip()
    body = json.dumps(payload, default=str).encode("utf-8")
    logger.info("Proxy(delay) start model=%s body_bytes=%s url=%s", model, len(body), url)
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        logger.info("Proxy(delay) ok model=%s total_s=%.3f", model, time.perf_counter() - t0)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        logger.error("Proxy(delay) http_error code=%s total_s=%.3f body=%s", e.code, time.perf_counter() - t0, msg)
        raise RuntimeError(f"Proxy error {e.code}: {msg}") from e
    except urllib.error.URLError as e:
        logger.error("Proxy(delay) url_error total_s=%.3f err=%s", time.perf_counter() - t0, str(e))
        raise RuntimeError(f"Could not reach proxy: {e}") from e

    data = json.loads(raw)
    text = data.get("text")
    if not text:
        raise RuntimeError(f"Proxy response missing 'text': {data}")
    return str(text).strip()
