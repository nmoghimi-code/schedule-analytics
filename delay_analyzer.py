from __future__ import annotations

"""
Multi-update schedule delay analysis.

This module is intentionally separate from the narrative comparison pipeline, but it
reuses the same critical-path tracing and change-evidence helpers from xer_comparator
so the two tools classify criticality identically.

Flow:
  - A dedicated baseline snapshot provides the contractual reference for target-date variance.
  - A series of update snapshots (>= 1) is analysed in chronological order (auto-sorted by data date).
  - For each update we trace the critical path that drives the target activity and record the
    target finish date + variance vs baseline.
  - For each consecutive transition (X-1 -> X) we diff the driving paths and explain *when* and
    *why* the path changed, including the case where the path shifted but the target date held.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

import xer_comparator as xc


def _iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else ts.isoformat()


def _derive_status(item: Mapping[str, Any]) -> str:
    if item.get("completed"):
        return "completed"
    if item.get("actual_start_date"):
        return "in_progress"
    return "not_started"


def _primary_chain(critical_path: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = critical_path.get("paths", []) or []
    if not paths:
        return []
    first = paths[0] or {}
    return [x for x in (first.get("activity_chain", []) or []) if isinstance(x, dict)]


def _active_driver(chain: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The current physical driver: the in-progress critical activity, else the earliest not-started one."""
    not_started: dict[str, Any] | None = None
    for it in chain:
        status = _derive_status(it)
        if status == "in_progress":
            return {"task_name": it.get("task_name"), "wbs_path": it.get("wbs_path"), "status": status}
        if status == "not_started" and not_started is None:
            not_started = {"task_name": it.get("task_name"), "wbs_path": it.get("wbs_path"), "status": status}
    return not_started


def _wbs_leaf_groups(chain: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in chain:
        path = str(it.get("wbs_path") or it.get("wbs_name") or "").strip()
        leaf = path.split(" / ")[-1].strip() if path else ""
        if leaf and leaf not in seen:
            seen.add(leaf)
            out.append(leaf)
    return out


def _compact_path(critical_path: Mapping[str, Any]) -> dict[str, Any]:
    chain = _primary_chain(critical_path)
    driver_sequence = [
        {
            "task_name": it.get("task_name"),
            "wbs_path": it.get("wbs_path"),
            "float_current_days": it.get("float_current_days"),
            "status": _derive_status(it),
            "forecast_finish_date": it.get("forecast_finish_date"),
        }
        for it in chain
    ]
    return {
        "primary_path_length": len(chain),
        "least_float_current_days": critical_path.get("least_float_current_days"),
        "target_float_current_days": critical_path.get("target_float_current_days"),
        "critical_count": critical_path.get("critical_count"),
        "active_driver": _active_driver(chain),
        "wbs_groups": _wbs_leaf_groups(chain),
        "driver_sequence": driver_sequence,
    }


def _classify_shift(
    change: Mapping[str, Any],
    period_variance_days: int | None,
    new_global: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Evidence-based classification of *why* the driving path changed.

    Tags are derived only from data already computed by xer_comparator; we never invent a cause.
    """
    interp = change.get("path_change_interpretation", {}) or {}
    prev_counts = interp.get("previous_unique_upstream_current_status_counts", {}) or {}
    prev_unique = int(interp.get("previous_unique_upstream_count") or 0)
    prev_done = int(prev_counts.get("completed", 0)) + int(prev_counts.get("in_progress", 0))

    relationship_evidence = change.get("relationship_change_evidence", []) or []
    field_evidence = change.get("task_field_change_evidence", []) or []

    new_ids = {str(x.get("activity_id")) for x in (new_global.get("items", []) or []) if x.get("activity_id")}
    added = change.get("added_to_current_path", []) or []
    new_driver_added = [a for a in added if str(a.get("activity_id")) in new_ids]

    # Did a duration/lag/calendar attribute grow on the new driving path?
    degraded_fields: list[str] = []
    for fe in field_evidence:
        for cf in fe.get("changed_fields", []) or []:
            field = str(cf.get("field", ""))
            if field in {"original_duration", "remaining_duration", "calendar", "constraint_type", "constraint_date"}:
                degraded_fields.append(field)

    tags: list[str] = []
    # Previous path pulled ahead: its unique upstream work is largely completed / in progress.
    if prev_unique > 0 and prev_done >= max(1, prev_unique - 1):
        tags.append("previous_path_progressed")
    if relationship_evidence:
        tags.append("logic_or_overlap_change_on_current_path")
    if degraded_fields:
        tags.append("duration_constraint_or_calendar_change_on_current_path")
    if new_driver_added:
        tags.append("new_activity_became_a_driver")
    if not tags:
        tags.append("cause_not_determinable_from_data")

    return {
        "tags": tags,
        "previous_unique_upstream_count": prev_unique,
        "previous_unique_upstream_status_counts": prev_counts,
        "degraded_fields_on_current_path": sorted(set(degraded_fields)),
        "new_driver_activity_names": [a.get("task_name") for a in new_driver_added],
        "relationship_change_count": len(relationship_evidence),
    }


def analyze_delays(
    baseline: xc.XerSnapshot | None,
    updates: Sequence[xc.XerSnapshot],
    *,
    target_activity_id: str,
    variance_threshold: int,
) -> dict[str, Any]:
    if not updates:
        raise ValueError("At least one update schedule is required for delay analysis.")

    # Auto-order updates chronologically so upload order does not matter.
    ordered = sorted(updates, key=lambda s: (s.data_date is None, s.data_date or pd.Timestamp.min))

    # Warn on duplicate data dates (ambiguous ordering).
    seen_dates: dict[str, str] = {}
    duplicate_data_dates: list[str] = []
    for s in ordered:
        key = _iso(s.data_date) or ""
        if key and key in seen_dates:
            duplicate_data_dates.append(key)
        elif key:
            seen_dates[key] = s.label

    def _finish(snap: xc.XerSnapshot | None) -> dict[str, Any]:
        if snap is None or snap.task is None or snap.task.empty:
            return {"finish_date": None, "task_name": None, "finish_col": None}
        try:
            return xc.milestone_finish(snap.task, target_activity_id)
        except Exception as e:  # target not resolvable in this snapshot
            return {"finish_date": None, "task_name": None, "finish_col": None, "warning": str(e)}

    baseline_ms = _finish(baseline)
    baseline_finish = baseline_ms.get("finish_date")

    per_update: list[dict[str, Any]] = []
    cp_by_index: list[dict[str, Any]] = []
    for snap in ordered:
        try:
            cp = xc.critical_path_to_target(snap, target_activity_id, near_critical_buffer_days=variance_threshold)
        except Exception as e:
            cp = {"paths": [], "warning": str(e)}
        cp_by_index.append(cp)

        ms = _finish(snap)
        finish = ms.get("finish_date")
        per_update.append(
            {
                "label": snap.label,
                "data_date": _iso(snap.data_date),
                "target_task_name": ms.get("task_name"),
                "target_finish_date": finish,
                "target_finish_col": ms.get("finish_col"),
                "variance_vs_baseline_days": xc._variance_days(finish, baseline_finish),
                "critical_path": _compact_path(cp),
                "critical_path_warning": cp.get("warning"),
                "target_warning": cp.get("target_warning"),
                "target_has_driving_logic": (cp.get("target_health", {}) or {}).get("has_driving_logic"),
            }
        )

    transitions: list[dict[str, Any]] = []
    for i in range(1, len(ordered)):
        prev, curr = ordered[i - 1], ordered[i]
        prev_cp, curr_cp = cp_by_index[i - 1], cp_by_index[i]

        try:
            new_global = xc.new_activities_all_wbs(prev, curr)
        except Exception:
            new_global = {}
        try:
            finish_ext = xc.finish_extensions_in_progress(prev, curr)
        except Exception:
            finish_ext = {}
        try:
            trending = xc.near_critical_trending(
                prev, curr, variance_threshold, target_activity_id=target_activity_id, critical_network=curr_cp
            )
        except Exception:
            trending = {}

        change = xc.critical_path_change_summary(
            prev_cp,
            curr_cp,
            last_snapshot=prev,
            current_snapshot=curr,
            new_global=new_global,
            finish_extensions=finish_ext,
            trending=trending,
        )

        prev_finish = per_update[i - 1]["target_finish_date"]
        curr_finish = per_update[i]["target_finish_date"]
        period_var = xc._variance_days(curr_finish, prev_finish)
        date_moved = period_var not in (None, 0)
        path_changed = bool(change.get("changed"))

        classification = _classify_shift(change, period_var, new_global)

        # The hard case: the driving path changed but the target completion date did not move.
        date_held_evidence: dict[str, Any] = {}
        if path_changed and not date_moved:
            date_held_evidence = {
                "explanation": (
                    "A different chain became the governing path to the target at the same total length, so the "
                    "completion date held. Cite the attribute/logic changes below as the supported mechanism."
                ),
                "task_attribute_changes_on_current_path": change.get("task_field_change_evidence", []),
                "relationship_changes_on_current_path": change.get("relationship_change_evidence", []),
                "previous_path_progress": change.get("path_change_interpretation", {}).get(
                    "previous_unique_upstream_current_status_counts"
                ),
            }

        transitions.append(
            {
                "from_label": prev.label,
                "to_label": curr.label,
                "from_data_date": _iso(prev.data_date),
                "to_data_date": _iso(curr.data_date),
                "days_between": (
                    None
                    if prev.data_date is None or curr.data_date is None
                    else int((curr.data_date.normalize() - prev.data_date.normalize()).days)
                ),
                "target_finish_before": prev_finish,
                "target_finish_after": curr_finish,
                "period_variance_days": period_var,
                "target_date_moved": bool(date_moved),
                "path_changed": path_changed,
                "path_changed_without_date_movement": bool(path_changed and not date_moved),
                "shift_classification": classification,
                "date_held_evidence": date_held_evidence,
                "change_detail": change,
                "new_activities": new_global,
            }
        )

    # Roll target-health up across the series: any update lacking driving logic is a hard problem.
    updates_without_logic = [u["label"] for u in per_update if u.get("target_has_driving_logic") is False]
    distinct_target_warnings = sorted({u["target_warning"] for u in per_update if u.get("target_warning")})
    target_validation = {
        "has_driving_logic_all_updates": not updates_without_logic,
        "updates_without_driving_logic": updates_without_logic,
        "warnings": distinct_target_warnings,
    }

    return {
        "settings": {
            "target_activity_id": str(target_activity_id),
            "variance_threshold": int(variance_threshold),
            "update_count": len(ordered),
        },
        "target": {
            "activity_id": str(target_activity_id),
            "task_name": baseline_ms.get("task_name") or (per_update[0]["target_task_name"] if per_update else None),
            "baseline_finish_date": baseline_finish,
            "baseline_label": (baseline.label if baseline is not None else None),
        },
        "target_validation": target_validation,
        "warnings": {"duplicate_data_dates": duplicate_data_dates},
        "updates": per_update,
        "transitions": transitions,
    }


def analyze_delays_from_paths(
    baseline_path: str | Path | None,
    update_paths: Sequence[str | Path],
    *,
    target_activity_id: str,
    variance_threshold: int,
) -> dict[str, Any]:
    baseline = xc.snapshot_from_xer_path("baseline", baseline_path) if baseline_path else None
    updates = [xc.snapshot_from_xer_path(f"update_{i + 1}", p) for i, p in enumerate(update_paths)]
    return analyze_delays(baseline, updates, target_activity_id=target_activity_id, variance_threshold=variance_threshold)


# ---------------------------------------------------------------------------
# AI-ready digest (IDs stripped, large arrays compacted)
# ---------------------------------------------------------------------------

_ID_KEYS = {
    "activity_id",
    "task_id",
    "target_activity_id",
    "from_activity_id",
    "to_activity_id",
    "predecessor_activity_id",
    "successor_activity_id",
}


def _strip_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_ids(v) for k, v in value.items() if k not in _ID_KEYS}
    if isinstance(value, list):
        return [_strip_ids(v) for v in value]
    return value


def _names_wbs(items: Sequence[Mapping[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items[:limit]:
        out.append({"task_name": it.get("task_name"), "wbs_path": it.get("wbs_path") or it.get("wbs_name")})
    return out


def _compact_shift_causes(causes: Sequence[Mapping[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in causes[:limit]:
        out.append(
            {
                "driver_type": c.get("driver_type"),
                "task_name": c.get("task_name"),
                "wbs_path": c.get("wbs_path") or c.get("wbs_name"),
                "changed_fields": c.get("changed_fields"),
                "detail": c.get("detail"),
                "finish_slip_days": c.get("finish_slip_days"),
                "float_loss_days": c.get("float_loss_days"),
            }
        )
    return out


def build_delay_digest(result: Mapping[str, Any], *, instruction: str | None = None) -> dict[str, Any]:
    """Compact, ID-free digest tailored for the delay-analysis LLM prompt."""
    timeline = []
    for u in result.get("updates", []) or []:
        cp = u.get("critical_path", {}) or {}
        timeline.append(
            {
                "label": u.get("label"),
                "data_date": u.get("data_date"),
                "target_finish_date": u.get("target_finish_date"),
                "variance_vs_baseline_days": u.get("variance_vs_baseline_days"),
                "critical_path_warning": u.get("critical_path_warning"),
                "target_warning": u.get("target_warning"),
                "critical_path_summary": {
                    "length": cp.get("primary_path_length"),
                    "least_float_current_days": cp.get("least_float_current_days"),
                    "active_driver": cp.get("active_driver"),
                    # WBS areas in the exact order the path flows through them (upstream -> target).
                    "path_wbs_sequence": cp.get("wbs_groups"),
                    # Full primary path, ordered upstream -> target. Use this to describe the whole chain.
                    "driver_sequence": cp.get("driver_sequence"),
                },
            }
        )

    story = []
    for t in result.get("transitions", []) or []:
        change = t.get("change_detail", {}) or {}
        interp = change.get("path_change_interpretation", {}) or {}
        story.append(
            {
                "from_label": t.get("from_label"),
                "to_label": t.get("to_label"),
                "from_data_date": t.get("from_data_date"),
                "to_data_date": t.get("to_data_date"),
                "days_between": t.get("days_between"),
                "target_finish_before": t.get("target_finish_before"),
                "target_finish_after": t.get("target_finish_after"),
                "period_variance_days": t.get("period_variance_days"),
                "target_date_moved": t.get("target_date_moved"),
                "path_changed": t.get("path_changed"),
                "path_changed_without_date_movement": t.get("path_changed_without_date_movement"),
                "shift_classification": t.get("shift_classification"),
                "drivers_added_to_current_path": _names_wbs(change.get("added_to_current_path", []) or []),
                "drivers_removed_from_previous_path": _names_wbs(change.get("removed_from_previous_path", []) or []),
                "previous_unique_upstream_status_counts": interp.get(
                    "previous_unique_upstream_current_status_counts"
                ),
                "shared_downstream_count": interp.get("shared_downstream_count"),
                "supported_shift_causes": _compact_shift_causes(change.get("possible_shift_causes", []) or []),
                "date_held_evidence": _strip_ids(t.get("date_held_evidence", {}) or {}),
                "new_activity_count": (t.get("new_activities", {}) or {}).get("count"),
            }
        )

    digest = {
        "report_type": "schedule_delay_analysis",
        "settings": result.get("settings"),
        "target": result.get("target"),
        "target_validation": result.get("target_validation"),
        "warnings": result.get("warnings"),
        "update_timeline": timeline,
        "path_change_story": story,
    }
    if instruction and str(instruction).strip():
        digest["user_instruction"] = str(instruction).strip()
    return _strip_ids(digest)
