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
   - Explain the current target-driving critical path using current_target_critical_path.precomputed_float_buckets and narrative_focus first, especially current_active_critical_driver, active_critical_task_examples, next_not_started_critical_task_examples, chronological_target_critical_sequence_0_days, chronological_critical_task_sequence, upstream_driver_task_examples, downstream_finish_or_milestone_examples, critical_activity_groups, and tied_branch_examples. Use branch_summaries, primary_path_task_sequence, and longest_tied_path_task_sequence only as backup branch detail.
   - Treat current_target_critical_path.precomputed_float_buckets.chronological_target_critical_sequence_0_days and target_critical_activities_0_days as the authoritative critical arrays.
   - Use only current_target_critical_path.critical_activities, critical_activity_groups, paths, and their is_critical/on_critical_trace/float_current_days fields to describe critical work. Do not add or remove activities based on date gaps.
   - Use path_method as the path-definition method. The critical path is extracted by treating TASK/TASKPRED as a DAG and tracing backward from the target through predecessors whose P6 total float is less than or equal to the current branch float within tolerance. Keep near-critical discussion for Section 5.
   - If current_target_critical_path.critical_trace_bridges or a logic link trace_bridge is present, explain it as a calendar-based trace bridge only. A bridged predecessor is upstream driver context, not a 0-float critical activity, unless its own is_critical flag is true.
   - If narrative_focus.current_active_critical_driver or current_target_critical_path.critical_status_focus.current_active_critical_driver is present, lead with it as the current active critical driver. Then describe the next not-started critical work. Do not say the critical chain starts with a future/not-started activity when an in-progress critical activity exists.
   - Use current_status exactly as provided. "in_progress" means active critical work; "not_started" means future/next critical work; "completed" means completed upstream context only.
   - Always describe critical work chronologically from the earliest upstream physical work toward the final finish/milestone. Use the JSON sequence order in chronological_target_critical_sequence_0_days; do not reorder it from apparent date gaps.
   - In schedule language, "driving" means upstream predecessor work controlling downstream successor work. Do not state that finalization, commissioning, line painting, paving, off-bridge closeout, or milestone activities drive the path when they appear at the tail end. Say those activities are downstream/tail-end work driven by the upstream critical work.
   - If upstream concrete/structural work feeds downstream finish/off-bridge work, describe it as one continuous sequence from upstream work into finalization. Do not split one chronological chain into separate "primary" and "tied" narratives unless the JSON explicitly separates them as different branches.
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


def generate_narrative(data_digest: Mapping[str, Any], *, api_key: str | None = None, model: str = "gemini-2.5-flash") -> str:
    """
    Generate a schedule narrative using Gemini.

    Security note: API keys must NOT be hardcoded. Provide via GEMINI_API_KEY env var or api_key parameter.
    """
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

    # Keep payload compact to reduce request size / latency.
    user_content = json.dumps(data_digest, default=str)
    payload_bytes = len(user_content.encode("utf-8", errors="ignore"))
    logger.info("Gemini start model=%s transport=%s payload_bytes=%s", model, transport or "default", payload_bytes)

    # google-generativeai supports system_instruction on GenerativeModel.
    model_obj = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_REPORT_GUIDELINES)
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
                "like gemini-2.5-flash, or reduce usage."
            ) from e
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            raise RuntimeError(
                "Gemini request timed out. Check your internet/firewall and try gemini-2.5-flash. "
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
