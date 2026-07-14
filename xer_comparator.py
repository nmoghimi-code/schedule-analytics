from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Mapping

import pandas as pd

import xer_parser as xp


@dataclass(frozen=True)
class XerSnapshot:
    label: str
    project: pd.DataFrame
    task: pd.DataFrame
    taskpred: pd.DataFrame
    wbs: pd.DataFrame
    calendar: pd.DataFrame
    data_date: pd.Timestamp | None
    data_date_col: str | None
    # Kept on the existing snapshot type so downstream analysis can apply
    # source-specific safeguards without changing the established XER path.
    source_format: str = "xer"


# Shared with the parser module; kept as a single implementation there.
_pick_col = xp._pick_col


def _extract_project_data_date(project_df: pd.DataFrame) -> tuple[pd.Timestamp | None, str | None]:
    if project_df is None or project_df.empty:
        return None, None

    # Prefer P6 recalculation date if present; fall back to data_date.
    preferred = ["last_recalc_date", "lastrecalcdate", "data_date", "datadate"]
    col = _pick_col(project_df, preferred)
    if not col:
        recalc_like = [c for c in project_df.columns if "recalc" in c.lower() and "date" in c.lower()]
        if recalc_like:
            col = recalc_like[0]
        else:
            data_like = [c for c in project_df.columns if "data" in c.lower() and "date" in c.lower()]
            col = data_like[0] if data_like else None
    if not col:
        return None, None

    raw = project_df.iloc[0].get(col)
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None, col
    return pd.Timestamp(ts), col


def snapshot_from_tables(label: str, tables: Mapping[str, pd.DataFrame]) -> XerSnapshot:
    def get(name: str) -> pd.DataFrame:
        for k, v in tables.items():
            if k.strip().upper() == name:
                return v
        return pd.DataFrame()

    project = get("PROJECT")
    task = get("TASK")
    taskpred = get("TASKPRED")
    wbs = get("WBS")
    projwbs = get("PROJWBS")
    calendar = get("CALENDAR")
    if (wbs is None or wbs.empty) and (projwbs is not None and not projwbs.empty):
        wbs = projwbs

    # Enrich TASK with WBS and milestone fields used by the comparator.
    task = xp.merge_wbs_names(task, wbs)
    task = xp.add_is_milestone(task)
    task = xp.add_full_wbs_path(task, wbs)

    data_date, data_date_col = _extract_project_data_date(project)
    source_format = "xer"
    for frame in (project, task):
        if frame is None or frame.empty or "source_format" not in frame.columns:
            continue
        values = frame["source_format"].dropna().astype(str).str.strip()
        if not values.empty and values.iloc[0]:
            source_format = values.iloc[0]
            break
    return XerSnapshot(
        label=label,
        project=project,
        task=task,
        taskpred=taskpred,
        wbs=wbs,
        calendar=calendar,
        data_date=data_date,
        data_date_col=data_date_col,
        source_format=source_format,
    )


def snapshot_from_xer_path(label: str, xer_path: str | Path) -> XerSnapshot:
    tables = xp.read_xer_tables(xer_path, ["PROJECT", "TASK", "TASKPRED", "WBS", "PROJWBS", "CALENDAR"])
    return snapshot_from_tables(label, tables)


def snapshot_from_schedule_path(label: str, schedule_path: str | Path) -> XerSnapshot:
    """Load either a Primavera XER or Microsoft Project XML schedule."""
    path = Path(schedule_path)
    suffix = path.suffix.casefold()
    if suffix == ".xer":
        return snapshot_from_xer_path(label, path)
    if suffix == ".xml":
        import schedule_xml_parser as sxp

        return snapshot_from_tables(label, sxp.read_mspdi_tables(path))
    raise ValueError(
        f"Unsupported schedule format '{path.suffix or '(none)'}'. Select a Primavera .xer or "
        "Microsoft Project .xml file."
    )


def _is_mspdi(snapshot: XerSnapshot | None) -> bool:
    return snapshot is not None and snapshot.source_format == "mspdi_xml"


def _float_basis_text(snapshot: XerSnapshot, *, short: bool = False) -> str:
    if _is_mspdi(snapshot):
        if short:
            return (
                "Microsoft Project TotalSlack converted from tenths of a minute to days using the "
                "exported MinutesPerDay value; do not infer slack from date gaps."
            )
        return (
            "Microsoft Project TotalSlack converted from tenths of a minute to days using the exported "
            "MinutesPerDay value; dates are context only."
        )
    if short:
        return "P6 total float converted to days in Python using the activity calendar; do not infer float from date gaps."
    return "P6 total float converted to days in Python using the activity calendar; dates are context only."


def _resolve_target_row(task_df: pd.DataFrame, target_activity_id: str) -> pd.Series:
    if task_df is None or task_df.empty:
        raise ValueError("TASK is empty; cannot resolve target activity.")

    target = str(target_activity_id).strip()
    if not target:
        raise ValueError("target_activity_id is empty.")

    task_id_col = _pick_col(task_df, ["task_id"])
    activity_id_col = _pick_col(task_df, ["task_code", "activity_id"])
    name_col = _pick_col(task_df, ["task_name", "task_title", "activity_name"])

    if activity_id_col:
        match = task_df[task_df[activity_id_col].astype(str) == target]
        if len(match) == 1:
            return match.iloc[0]
        if len(match) > 1:
            raise ValueError(f"Multiple rows match {activity_id_col}='{target}'.")

    if task_id_col:
        match = task_df[task_df[task_id_col].astype(str) == target]
        if len(match) == 1:
            return match.iloc[0]

    if name_col:
        normalized = task_df[name_col].astype(str).str.strip()
        match = task_df[normalized.str.casefold() == target.casefold()]
        if len(match) == 1:
            return match.iloc[0]
        if len(match) > 1:
            raise ValueError(f"Multiple rows match {name_col}='{target}'.")

    raise ValueError(f"Could not resolve target_activity_id='{target}'.")


def _detect_finish_col(task_df: pd.DataFrame) -> str | None:
    candidates = [
        "act_end_date",
        "actual_finish_date",
        "end_date",
        "finish_date",
        "target_end_date",
        "early_end_date",
        "late_end_date",
    ]
    col = _pick_col(task_df, candidates)
    if col:
        return col
    date_like = [c for c in task_df.columns if "end" in c.lower() and "date" in c.lower()]
    if date_like:
        return date_like[0]
    return None


def _detect_actual_cols(task_df: pd.DataFrame) -> tuple[str | None, str | None]:
    act_start = _pick_col(task_df, ["act_start_date", "actual_start_date"])
    act_finish = _pick_col(task_df, ["act_end_date", "actual_finish_date"])
    return act_start, act_finish


def _select_finish_date_with_priority(task_df: pd.DataFrame, row: pd.Series) -> tuple[pd.Timestamp | None, str | None]:
    """
    Smart finish date selection for milestone/target activities.

    Priority (as requested):
      1) act_end_date (Actual Finish)
      2) target_finish_date (or target_end_date)
      3) early_end_date

    If none exist/parse, falls back to any detected finish-like column.
    """
    priority_groups: list[list[str]] = [
        ["act_end_date", "actual_finish_date"],
        ["target_finish_date", "target_end_date"],
        ["early_end_date", "early_finish_date"],
    ]

    for group in priority_groups:
        col = _pick_col(task_df, group)
        if not col:
            continue
        raw = row.get(col)
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts), col

    fallback_col = _detect_finish_col(task_df)
    if fallback_col:
        raw = row.get(fallback_col)
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts), fallback_col
        return None, fallback_col

    return None, None


def _calendar_hours_per_day_by_id(calendar_df: pd.DataFrame | None) -> dict[str, float]:
    if calendar_df is None or calendar_df.empty:
        return {}
    clndr_id_col = _pick_col(calendar_df, ["clndr_id", "calendar_id"])
    day_hr_col = _pick_col(calendar_df, ["day_hr_cnt", "hours_per_day"])
    if not clndr_id_col or not day_hr_col:
        return {}

    out: dict[str, float] = {}
    for _, row in calendar_df.iterrows():
        clndr_id = _clean_optional(row.get(clndr_id_col))
        if not clndr_id:
            continue
        val = pd.to_numeric(row.get(day_hr_col), errors="coerce")
        if pd.isna(val) or float(val) <= 0:
            continue
        out[str(clndr_id)] = float(val)
    return out


def _calendar_name_by_id(calendar_df: pd.DataFrame | None) -> dict[str, str]:
    if calendar_df is None or calendar_df.empty:
        return {}
    clndr_id_col = _pick_col(calendar_df, ["clndr_id", "calendar_id"])
    name_col = _pick_col(calendar_df, ["clndr_name", "calendar_name"])
    if not clndr_id_col or not name_col:
        return {}
    out: dict[str, str] = {}
    for _, row in calendar_df.iterrows():
        clndr_id = _clean_optional(row.get(clndr_id_col))
        name = _clean_optional(row.get(name_col))
        if clndr_id and name:
            out[str(clndr_id)] = name
    return out


def _task_float_series_to_days(
    task_df: pd.DataFrame,
    float_col: str,
    values: pd.Series,
    calendar_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Convert P6 float hours to days using each activity's assigned calendar when possible.

    Critical/near-critical classification must come from P6 float fields, not date gaps.
    This helper only changes the reporting unit from hours to calendar-specific days.
    """
    if "hr" not in str(float_col).lower():
        return values

    calendar_hours = _calendar_hours_per_day_by_id(calendar_df)
    clndr_id_col = _pick_col(task_df, ["clndr_id", "calendar_id"])
    if not clndr_id_col or not calendar_hours:
        return values / xp.P6_HOURS_PER_DAY

    hours_per_day = task_df[clndr_id_col].astype(str).map(calendar_hours)
    hours_per_day = pd.to_numeric(hours_per_day, errors="coerce").fillna(xp.P6_HOURS_PER_DAY)
    hours_per_day = hours_per_day.where(hours_per_day > 0, xp.P6_HOURS_PER_DAY)
    hours_per_day = hours_per_day.reindex(values.index).fillna(xp.P6_HOURS_PER_DAY)
    return values / hours_per_day


def _task_calendar_context(task_df: pd.DataFrame, calendar_df: pd.DataFrame | None, row: pd.Series) -> dict[str, Any]:
    clndr_id_col = _pick_col(task_df, ["clndr_id", "calendar_id"])
    clndr_id = _clean_optional(row.get(clndr_id_col)) if clndr_id_col else None
    hours_by_id = _calendar_hours_per_day_by_id(calendar_df)
    names_by_id = _calendar_name_by_id(calendar_df)
    hours = hours_by_id.get(str(clndr_id)) if clndr_id else None
    return {
        "calendar_id": clndr_id,
        "calendar_name": names_by_id.get(str(clndr_id)) if clndr_id else None,
        "calendar_hours_per_day": hours if hours is not None else xp.P6_HOURS_PER_DAY,
        "calendar_hours_source": "CALENDAR.day_hr_cnt" if hours is not None else "default_8h_fallback",
    }


def data_date_sync(baseline: XerSnapshot, last: XerSnapshot, current: XerSnapshot) -> dict[str, Any]:
    def iso(ts: pd.Timestamp | None) -> str | None:
        return None if ts is None else ts.isoformat()

    last_to_current_days: int | None = None
    if last.data_date is not None and current.data_date is not None:
        last_to_current_days = int((current.data_date.normalize() - last.data_date.normalize()).days)

    return {
        "baseline_data_date": iso(baseline.data_date),
        "baseline_data_date_col": baseline.data_date_col,
        "last_data_date": iso(last.data_date),
        "last_data_date_col": last.data_date_col,
        "current_data_date": iso(current.data_date),
        "current_data_date_col": current.data_date_col,
        "days_last_to_current": last_to_current_days,
    }


def milestone_finish(task_df: pd.DataFrame, target_activity_id: str) -> dict[str, Any]:
    row = _resolve_target_row(task_df, target_activity_id)
    finish, finish_col = _select_finish_date_with_priority(task_df, row)

    activity_id_col = _pick_col(task_df, ["task_code", "activity_id"])
    task_id_col = _pick_col(task_df, ["task_id"])
    name_col = _pick_col(task_df, ["task_name", "task_title", "activity_name"])

    return {
        "task_id": (None if not task_id_col else str(row.get(task_id_col))),
        "activity_id": (None if not activity_id_col else str(row.get(activity_id_col))),
        "task_name": (None if not name_col else str(row.get(name_col))),
        "finish_col": finish_col,
        "finish_date": (None if finish is None else finish.isoformat()),
    }


def _variance_days(a_iso: str | None, b_iso: str | None) -> int | None:
    if not a_iso or not b_iso:
        return None
    a = pd.to_datetime(a_iso, errors="coerce")
    b = pd.to_datetime(b_iso, errors="coerce")
    if pd.isna(a) or pd.isna(b):
        return None
    return int((pd.Timestamp(a).normalize() - pd.Timestamp(b).normalize()).days)


def near_critical_trending(
    last: XerSnapshot,
    current: XerSnapshot,
    variance_threshold: int,
    *,
    target_activity_id: str | None = None,
    critical_network: Mapping[str, Any] | None = None,
    exclude_activity_ids: set[str] | None = None,
) -> dict[str, Any]:
    if last.data_date is None or current.data_date is None:
        days_between = None
    else:
        days_between = int((current.data_date.normalize() - last.data_date.normalize()).days)

    if critical_network is None:
        if not target_activity_id:
            return {
                "days_between": days_between,
                "least_float_current_days": None,
                "cutoff_current_days": None,
                "excluded_activity_count": int(len(exclude_activity_ids or set())),
                "near_critical_count": 0,
                "near_critical": [],
                "eroding_risk_count": 0,
                "eroding_risks": [],
                "warning": "Near-critical classification requires target_activity_id or a critical_network object.",
            }
        critical_network = critical_path_to_target(
            current,
            target_activity_id,
            near_critical_buffer_days=variance_threshold,
        )

    try:
        float_col = xp.detect_total_float_column(current.task)
    except Exception:
        float_col = None
    if not float_col:
        return {
            "days_between": days_between,
            "least_float_current_days": None,
            "cutoff_current_days": None,
            "excluded_activity_count": int(len(exclude_activity_ids or set())),
            "near_critical_count": 0,
            "near_critical": [],
            "eroding_risk_count": 0,
            "eroding_risks": [],
            "warning": "Current TASK missing total float column.",
        }

    activity_id_col_curr = _pick_col(current.task, ["task_code", "activity_id"])
    activity_id_col_last = _pick_col(last.task, ["task_code", "activity_id"])
    if not activity_id_col_curr or not activity_id_col_last:
        raise ValueError("TASK missing activity id columns in one of the snapshots.")

    last_float_col = xp.detect_total_float_column(last.task)

    curr = current.task[[activity_id_col_curr, float_col]].copy()
    curr = curr.rename(columns={activity_id_col_curr: "activity_id", float_col: "float_current_raw_hours"})
    curr["float_current_raw_hours"] = pd.to_numeric(curr["float_current_raw_hours"], errors="coerce")
    curr["float_current_days"] = _task_float_series_to_days(
        current.task,
        str(float_col),
        pd.to_numeric(current.task[float_col], errors="coerce"),
        current.calendar,
    )

    prev = last.task[[activity_id_col_last, last_float_col]].copy()
    prev = prev.rename(columns={activity_id_col_last: "activity_id", last_float_col: "float_last_raw_hours"})
    prev["float_last_raw_hours"] = pd.to_numeric(prev["float_last_raw_hours"], errors="coerce")
    prev["float_last_days"] = _task_float_series_to_days(
        last.task,
        str(last_float_col),
        pd.to_numeric(last.task[last_float_col], errors="coerce"),
        last.calendar,
    )

    exclude_activity_ids = {str(x) for x in (exclude_activity_ids or set()) if str(x).strip()}
    wanted = {str(x) for x in (critical_network.get("near_critical_activity_ids", []) or []) if str(x).strip()}
    if exclude_activity_ids:
        wanted = wanted - exclude_activity_ids
    curr = curr[curr["activity_id"].astype(str).isin(wanted)].copy()

    merged = curr.merge(prev, on="activity_id", how="left")
    float_current_days = pd.to_numeric(merged["float_current_days"], errors="coerce")
    float_last_days = pd.to_numeric(merged["float_last_days"], errors="coerce")
    float_change_days = float_current_days - float_last_days

    # Add story-friendly fields from CURRENT snapshot.
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    clndr_id_col = _pick_col(current.task, ["clndr_id", "calendar_id"])
    if name_col or wbs_name_col or wbs_path_col or clndr_id_col:
        cols = [activity_id_col_curr]
        if name_col:
            cols.append(name_col)
        if wbs_name_col:
            cols.append(wbs_name_col)
        if wbs_path_col:
            cols.append(wbs_path_col)
        if clndr_id_col:
            cols.append(clndr_id_col)
        details = current.task[cols].copy()
        details = details.rename(columns={activity_id_col_curr: "activity_id"})
        if name_col:
            details = details.rename(columns={name_col: "task_name"})
        if wbs_name_col:
            details = details.rename(columns={wbs_name_col: "wbs_name"})
        if wbs_path_col:
            details = details.rename(columns={wbs_path_col: "wbs_path"})
        if clndr_id_col:
            details = details.rename(columns={clndr_id_col: "calendar_id"})
        details = details.drop_duplicates(subset=["activity_id"])
        merged = merged.merge(details, on="activity_id", how="left")

    calendar_names = _calendar_name_by_id(current.calendar)
    calendar_hours = _calendar_hours_per_day_by_id(current.calendar)

    loss_days = float_last_days - float_current_days

    erosion_assessment_warning: str | None = None
    if days_between is None:
        days_passed = None
        eroding_mask = pd.Series([False] * len(merged), index=merged.index)
    elif _is_mspdi(current) and days_between <= 0:
        # Keep file-to-file slack changes visible, but do not call them period
        # erosion when Microsoft Project exported no advancing StatusDate.
        days_passed = float(max(0, days_between))
        eroding_mask = pd.Series([False] * len(merged), index=merged.index)
        erosion_assessment_warning = (
            "Microsoft Project StatusDate did not advance between the previous and current XML files. "
            "Slack changes are file-version differences; period float erosion cannot be assessed."
        )
    else:
        days_passed = float(max(0, days_between))
        eroding_mask = loss_days > days_passed

    rows = []
    for idx, r in merged.iterrows():
        rows.append(
            {
                "activity_id": str(r["activity_id"]),
                "task_name": (None if "task_name" not in merged.columns else (None if pd.isna(r.get("task_name")) else str(r.get("task_name")))),
                "wbs_name": (None if "wbs_name" not in merged.columns else (None if pd.isna(r.get("wbs_name")) else str(r.get("wbs_name")))),
                "wbs_path": (None if "wbs_path" not in merged.columns else (None if pd.isna(r.get("wbs_path")) else str(r.get("wbs_path")))),
                "float_current_days": (None if pd.isna(float_current_days.loc[idx]) else float(float_current_days.loc[idx])),
                "float_last_days": (None if pd.isna(float_last_days.loc[idx]) else float(float_last_days.loc[idx])),
                "float_current_raw_hours": (None if pd.isna(r.get("float_current_raw_hours")) else float(r.get("float_current_raw_hours"))),
                "float_last_raw_hours": (None if pd.isna(r.get("float_last_raw_hours")) else float(r.get("float_last_raw_hours"))),
                "float_change_days": (None if pd.isna(float_change_days.loc[idx]) else float(float_change_days.loc[idx])),
                "float_loss_days": (None if pd.isna(loss_days.loc[idx]) else float(loss_days.loc[idx])),
                "calendar_id": (None if "calendar_id" not in merged.columns or pd.isna(r.get("calendar_id")) else str(r.get("calendar_id"))),
                "calendar_name": (
                    None
                    if "calendar_id" not in merged.columns or pd.isna(r.get("calendar_id"))
                    else calendar_names.get(str(r.get("calendar_id")))
                ),
                "calendar_hours_per_day": (
                    None
                    if "calendar_id" not in merged.columns or pd.isna(r.get("calendar_id"))
                    else calendar_hours.get(str(r.get("calendar_id")), xp.P6_HOURS_PER_DAY)
                ),
                "float_basis": _float_basis_text(current),
                "days_passed": days_passed,
                "eroding_risk": bool(eroding_mask.loc[idx]),
            }
        )

    eroding = [x for x in rows if x["eroding_risk"]]

    # Driving relationships among the near-critical activities themselves, so the narrative can describe
    # how near-critical work sequences (e.g. "removal drives repair drives forming") instead of bare counts.
    nc_name_by_id = {str(r["activity_id"]): r.get("task_name") for r in rows}
    nc_ids = set(nc_name_by_id.keys())
    driving_links: list[dict[str, Any]] = []
    if nc_ids:
        for (pred_aid, succ_aid), rec in _relationship_records_by_pair(current).items():
            if pred_aid in nc_ids and succ_aid in nc_ids:
                driving_links.append(
                    {
                        "from_task_name": nc_name_by_id.get(pred_aid),
                        "to_task_name": nc_name_by_id.get(succ_aid),
                        "relationship_type": rec.get("relationship_type"),
                        "lag_days": rec.get("lag_days"),
                    }
                )

    result = {
        "days_between": days_between,
        "least_float_current_days": critical_network.get("least_float_current_days"),
        "cutoff_current_days": critical_network.get("absolute_near_critical_threshold_days"),
        "near_critical_buffer_days": critical_network.get("near_critical_buffer_days"),
        "excluded_activity_count": int(len(exclude_activity_ids)),
        "near_critical_count": int(len(rows)),
        "near_critical": rows,
        "driving_links": driving_links,
        "method": critical_network.get("method"),
        "eroding_risk_count": int(len(eroding)),
        "eroding_risks": eroding,
    }
    if erosion_assessment_warning:
        result["erosion_assessment_warning"] = erosion_assessment_warning
    return result


def wbs_monitor_change_and_delay(last: XerSnapshot, current: XerSnapshot, *, term: str = "change") -> dict[str, Any]:
    # Requirement:
    # - Wildcard/partial match: typing "change" should match "Changes", "Changing", "Change Orders", etc.
    # - Highest hierarchy: identify the *mother* WBS node(s) that match, then compare activities across the
    #   entire section (mother + all descendants).
    term_norm = str(term or "").strip()

    def section_activity_ids(snapshot: XerSnapshot) -> tuple[set[str], dict[str, Any]]:
        activity_id_col = _pick_col(snapshot.task, ["task_code", "activity_id"])
        task_wbs_id_col = _pick_col(snapshot.task, ["wbs_id"])
        if not activity_id_col or not task_wbs_id_col or not term_norm:
            return set(), {"mother_wbs": [], "included_wbs_count": 0, "_included_wbs_ids": set()}

        wbs_df = snapshot.wbs
        wbs_id_col = _pick_col(wbs_df, ["wbs_id"]) if wbs_df is not None and not wbs_df.empty else None
        wbs_name_col = _pick_col(wbs_df, ["wbs_name", "wbs_short_name", "wbs_code"]) if wbs_id_col else None
        parent_wbs_id_col = _pick_col(wbs_df, ["parent_wbs_id", "par_wbs_id"]) if wbs_id_col else None

        # If we can build hierarchy, do it (preferred).
        if wbs_id_col and wbs_name_col:
            if not parent_wbs_id_col:
                for c in wbs_df.columns:
                    lc = c.lower()
                    if "parent" in lc and "wbs" in lc and lc.endswith("_id"):
                        parent_wbs_id_col = c
                        break

            name_by_id: dict[str, str] = {}
            parent_by_id: dict[str, str | None] = {}
            for _, r in wbs_df.iterrows():
                wid = str(r.get(wbs_id_col, "")).strip()
                if not wid or wid.lower() == "nan":
                    continue
                name_by_id[wid] = str(r.get(wbs_name_col, "")).strip()
                if parent_wbs_id_col:
                    pid_raw = str(r.get(parent_wbs_id_col, "")).strip()
                    parent_by_id[wid] = None if (not pid_raw or pid_raw.lower() == "nan") else pid_raw
                else:
                    parent_by_id[wid] = None

            children_by_parent: dict[str | None, set[str]] = defaultdict(set)
            for wid, pid in parent_by_id.items():
                children_by_parent[pid].add(wid)

            path_cache: dict[str, str] = {}

            def wbs_path(wid: str) -> str:
                if wid in path_cache:
                    return path_cache[wid]
                parts: list[str] = []
                seen: set[str] = set()
                cur: str | None = wid
                while cur and cur not in seen:
                    seen.add(cur)
                    nm = name_by_id.get(cur, "")
                    if nm:
                        parts.append(nm)
                    cur = parent_by_id.get(cur)
                parts.reverse()
                out = " / ".join(parts)
                path_cache[wid] = out
                return out

            # Wildcard match: simple partial contains on the full WBS path (case-insensitive).
            needle = term_norm.casefold()
            matched = {wid for wid in name_by_id.keys() if needle in (wbs_path(wid) or "").casefold()}
            if not matched:
                return set(), {"mother_wbs": [], "included_wbs_count": 0, "_included_wbs_ids": set()}

            # Highest-hierarchy selection: remove any matched node that has a matched ancestor.
            mothers: set[str] = set()
            for wid in matched:
                cur = parent_by_id.get(wid)
                seen: set[str] = set()
                while cur and cur not in seen:
                    seen.add(cur)
                    if cur in matched:
                        break
                    cur = parent_by_id.get(cur)
                else:
                    mothers.add(wid)

            included_wbs: set[str] = set()
            q: list[str] = sorted(mothers)
            while q:
                cur = q.pop()
                if cur in included_wbs:
                    continue
                included_wbs.add(cur)
                for child in children_by_parent.get(cur, set()):
                    if child not in included_wbs:
                        q.append(child)

            task_wbs_ids = snapshot.task[task_wbs_id_col].astype(str).str.strip()
            mask = task_wbs_ids.isin(included_wbs)
            ids = set(snapshot.task.loc[mask, activity_id_col].astype(str).tolist())

            mother_list = []
            for wid in sorted(mothers):
                mother_list.append({"wbs_id": wid, "wbs_name": name_by_id.get(wid), "wbs_path": wbs_path(wid)})

            return ids, {"mother_wbs": mother_list, "included_wbs_count": int(len(included_wbs)), "_included_wbs_ids": included_wbs}

        # Fallback: no WBS hierarchy available. Use partial match on TASK.wbs_name only.
        if "wbs_name" not in snapshot.task.columns:
            return set(), {"mother_wbs": [], "included_wbs_count": 0, "_included_wbs_ids": set()}
        needle = term_norm.casefold()
        hay = snapshot.task["wbs_name"].astype(str).str.casefold()
        mask = hay.str.contains(needle, na=False)
        ids = set(snapshot.task.loc[mask, activity_id_col].astype(str).tolist())
        return ids, {"mother_wbs": [{"wbs_id": None, "wbs_name": None, "wbs_path": None}], "included_wbs_count": 0, "_included_wbs_ids": set()}

    current_ids, current_meta = section_activity_ids(current)
    last_ids, last_meta = section_activity_ids(last)
    new_ids = sorted(current_ids - last_ids)

    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    clndr_id_col = _pick_col(current.task, ["clndr_id", "calendar_id"])

    details: dict[str, dict[str, Any]] = {}
    if activity_id_col:
        cols = [activity_id_col]
        if name_col:
            cols.append(name_col)
        if wbs_name_col:
            cols.append(wbs_name_col)
        if wbs_path_col:
            cols.append(wbs_path_col)
        if clndr_id_col:
            cols.append(clndr_id_col)
        task_id_col = _pick_col(current.task, ["task_id"])
        wbs_id_col = _pick_col(current.task, ["wbs_id"])
        float_col = None
        try:
            float_col = xp.detect_total_float_column(current.task)
        except Exception:
            float_col = None
        if task_id_col:
            cols.append(task_id_col)
        if wbs_id_col:
            cols.append(wbs_id_col)
        if float_col:
            cols.append(float_col)
        tmp = current.task[cols].copy()
        tmp = tmp.rename(columns={activity_id_col: "activity_id"})
        if name_col:
            tmp = tmp.rename(columns={name_col: "task_name"})
        if wbs_name_col:
            tmp = tmp.rename(columns={wbs_name_col: "wbs_name"})
        if wbs_path_col:
            tmp = tmp.rename(columns={wbs_path_col: "wbs_path"})
        if clndr_id_col:
            tmp = tmp.rename(columns={clndr_id_col: "calendar_id"})
        if task_id_col:
            tmp = tmp.rename(columns={task_id_col: "task_id"})
        if wbs_id_col:
            tmp = tmp.rename(columns={wbs_id_col: "wbs_id"})
        if float_col:
            tmp = tmp.rename(columns={float_col: "total_float_raw"})
        tmp = tmp.drop_duplicates(subset=["activity_id"])
        calendar_names = _calendar_name_by_id(current.calendar)
        calendar_hours = _calendar_hours_per_day_by_id(current.calendar)
        for _, r in tmp.iterrows():
            tf_raw = None
            if "total_float_raw" in tmp.columns:
                tf_raw = pd.to_numeric(r.get("total_float_raw"), errors="coerce")
            tf_days = None
            if tf_raw is not None and not pd.isna(tf_raw):
                tf_days_series = _task_float_series_to_days(tmp, str(float_col), pd.Series([tf_raw], index=[r.name]), current.calendar)
                tf_days = float(tf_days_series.iloc[0])
            details[str(r["activity_id"])] = {
                "activity_id": str(r["activity_id"]),
                "task_name": (None if "task_name" not in tmp.columns else str(r.get("task_name"))),
                "wbs_name": (None if "wbs_name" not in tmp.columns else str(r.get("wbs_name"))),
                "wbs_path": (None if "wbs_path" not in tmp.columns else str(r.get("wbs_path"))),
                "task_id": (None if "task_id" not in tmp.columns else str(r.get("task_id"))),
                "wbs_id": (None if "wbs_id" not in tmp.columns else str(r.get("wbs_id"))),
                "calendar_id": (None if "calendar_id" not in tmp.columns else str(r.get("calendar_id"))),
                "calendar_name": (
                    None
                    if "calendar_id" not in tmp.columns or pd.isna(r.get("calendar_id"))
                    else calendar_names.get(str(r.get("calendar_id")))
                ),
                "calendar_hours_per_day": (
                    None
                    if "calendar_id" not in tmp.columns or pd.isna(r.get("calendar_id"))
                    else calendar_hours.get(str(r.get("calendar_id")), xp.P6_HOURS_PER_DAY)
                ),
                "total_float_raw_hours": None if tf_raw is None or pd.isna(tf_raw) else float(tf_raw),
                "total_float_days": tf_days,
            }

    def _path_impact_tracker() -> dict[str, Any]:
        """
        Build a successor-path tracker for new activities in the matched WBS section.
        """
        if not new_ids:
            return {"new_paths": [], "driving_delay_items": [], "cross_wbs_alerts": []}

        if current.taskpred is None or current.taskpred.empty:
            return {"new_paths": [], "driving_delay_items": [], "cross_wbs_alerts": [], "warning": "TASKPRED is empty."}

        task_id_col_curr = _pick_col(current.task, ["task_id"])
        activity_id_col_curr = _pick_col(current.task, ["task_code", "activity_id"])
        if not task_id_col_curr or not activity_id_col_curr:
            return {"new_paths": [], "driving_delay_items": [], "cross_wbs_alerts": [], "warning": "TASK missing task_id/activity_id."}

        # Determine least float in CURRENT (project-wide, in DAYS) to label driving delay items.
        least_float_current_days: float | None
        float_col = None
        try:
            float_col = xp.detect_total_float_column(current.task)
        except Exception:
            float_col = None
        if float_col:
            numeric = pd.to_numeric(current.task[float_col], errors="coerce")
            numeric_days = _task_float_series_to_days(current.task, float_col, numeric, current.calendar)
            least_float_current_days = None if numeric_days.notna().sum() == 0 else float(numeric_days.min(skipna=True))
        else:
            least_float_current_days = None

        # Build internal id mappings.
        id_map = current.task[[task_id_col_curr, activity_id_col_curr]].copy()
        id_map[task_id_col_curr] = id_map[task_id_col_curr].astype(str).str.strip()
        id_map[activity_id_col_curr] = id_map[activity_id_col_curr].astype(str).str.strip()
        task_id_by_activity: dict[str, str] = dict(
            zip(id_map[activity_id_col_curr].tolist(), id_map[task_id_col_curr].tolist(), strict=False)
        )
        activity_by_task_id: dict[str, str] = dict(
            zip(id_map[task_id_col_curr].tolist(), id_map[activity_id_col_curr].tolist(), strict=False)
        )

        # Successor adjacency from TASKPRED (task_id is successor; pred_task_id is predecessor).
        succ_col = _pick_col(current.taskpred, ["task_id"])
        pred_col = _pick_col(current.taskpred, ["pred_task_id"])
        if not succ_col or not pred_col:
            return {"new_paths": [], "driving_delay_items": [], "cross_wbs_alerts": [], "warning": "TASKPRED missing task_id/pred_task_id."}

        succ_by_pred: dict[str, set[str]] = defaultdict(set)
        dfp = current.taskpred[[succ_col, pred_col]].copy()
        dfp[succ_col] = dfp[succ_col].astype(str).str.strip()
        dfp[pred_col] = dfp[pred_col].astype(str).str.strip()
        for _, r in dfp.iterrows():
            succ = str(r[succ_col]).strip()
            pred = str(r[pred_col]).strip()
            if not succ or not pred or succ.lower() == "nan" or pred.lower() == "nan":
                continue
            succ_by_pred[pred].add(succ)

        # Newness across the *whole schedule* (not just change WBS section).
        last_activity_id_col = _pick_col(last.task, ["task_code", "activity_id"])
        last_all_ids = set(last.task[last_activity_id_col].astype(str).str.strip().tolist()) if last_activity_id_col else set()
        current_all_ids = set(current.task[activity_id_col_curr].astype(str).str.strip().tolist())
        global_new_ids = current_all_ids - last_all_ids

        included_change_wbs_ids: set[str] = set(current_meta.get("_included_wbs_ids", set()) or set())

        def top_wbs(path: str | None) -> str | None:
            if not path:
                return None
            s = str(path).strip()
            if not s:
                return None
            return s.split(" / ", 1)[0].strip() or None

        driving_delay_items: dict[str, dict[str, Any]] = {}
        cross_wbs_alerts: list[dict[str, Any]] = []
        new_paths: list[dict[str, Any]] = []

        eps = 1e-9

        def is_driving(activity_id: str) -> bool:
            if least_float_current_days is None:
                return False
            info = details.get(activity_id, {})
            tf = info.get("total_float_days")
            if tf is None:
                return False
            return abs(float(tf) - float(least_float_current_days)) <= eps

        for root_aid in new_ids:
            root = details.get(root_aid, {"activity_id": root_aid, "task_name": None, "wbs_name": None, "wbs_path": None, "task_id": None, "wbs_id": None})
            root_tid = root.get("task_id") or task_id_by_activity.get(root_aid)
            if not root_tid:
                new_paths.append({"root_activity_id": root_aid, "warning": "Missing task_id; cannot traverse successors.", "edges": []})
                continue

            q: list[str] = [root_aid]
            expanded: set[str] = set()
            edges: list[dict[str, Any]] = []
            touched: set[str] = {root_aid}

            if is_driving(root_aid):
                driving_delay_items[root_aid] = {
                    "activity_id": root_aid,
                    "task_name": root.get("task_name"),
                    "wbs_path": root.get("wbs_path"),
                    "total_float_days": root.get("total_float_days"),
                    "total_float_raw_hours": root.get("total_float_raw_hours"),
                    "calendar_id": root.get("calendar_id"),
                    "calendar_name": root.get("calendar_name"),
                    "calendar_hours_per_day": root.get("calendar_hours_per_day"),
                }

            while q:
                from_aid = q.pop(0)
                if from_aid in expanded:
                    continue
                expanded.add(from_aid)

                from_tid = (details.get(from_aid, {}).get("task_id")) or task_id_by_activity.get(from_aid)
                if not from_tid:
                    continue

                for succ_tid in sorted(succ_by_pred.get(str(from_tid), set())):
                    succ_aid = activity_by_task_id.get(str(succ_tid))
                    if not succ_aid:
                        continue
                    touched.add(succ_aid)

                    in_last = succ_aid in last_all_ids
                    is_new_global = succ_aid in global_new_ids
                    status = "existing" if in_last else ("new" if is_new_global else "unknown")

                    succ_info = details.get(succ_aid, {"activity_id": succ_aid, "task_name": None, "wbs_name": None, "wbs_path": None, "task_id": str(succ_tid), "wbs_id": None})

                    cross_wbs = False
                    cross_wbs_name = None
                    if included_change_wbs_ids:
                        succ_wbs_id = succ_info.get("wbs_id")
                        if succ_wbs_id and str(succ_wbs_id).strip() not in included_change_wbs_ids:
                            cross_wbs = True
                            cross_wbs_name = top_wbs(succ_info.get("wbs_path")) or succ_info.get("wbs_name")

                    edge = {
                        "from_activity_id": from_aid,
                        "to_activity_id": succ_aid,
                        "to_status_vs_last": status,
                        "to_task_name": succ_info.get("task_name"),
                        "to_wbs_name": succ_info.get("wbs_name"),
                        "to_wbs_path": succ_info.get("wbs_path"),
                        "cross_wbs": cross_wbs,
                        "cross_wbs_top": cross_wbs_name,
                        "driving_delay_item": is_driving(succ_aid),
                    }
                    edges.append(edge)

                    if edge["driving_delay_item"]:
                        driving_delay_items[succ_aid] = {
                            "activity_id": succ_aid,
                            "task_name": succ_info.get("task_name"),
                            "wbs_path": succ_info.get("wbs_path"),
                            "total_float_days": succ_info.get("total_float_days"),
                            "total_float_raw_hours": succ_info.get("total_float_raw_hours"),
                            "calendar_id": succ_info.get("calendar_id"),
                            "calendar_name": succ_info.get("calendar_name"),
                            "calendar_hours_per_day": succ_info.get("calendar_hours_per_day"),
                        }

                    if cross_wbs and cross_wbs_name:
                        cross_wbs_alerts.append(
                            {
                                "from_activity_id": from_aid,
                                "to_activity_id": succ_aid,
                                "to_task_name": succ_info.get("task_name"),
                                "to_wbs_top": cross_wbs_name,
                            }
                        )

                    # Trace forward only when successor is also new (globally new vs last).
                    if is_new_global and succ_aid not in expanded:
                        q.append(succ_aid)

            new_paths.append(
                {
                    "root_activity_id": root_aid,
                    "root_task_name": root.get("task_name"),
                    "root_wbs_path": root.get("wbs_path"),
                    "edges": edges,
                    "touched_activity_ids": sorted(touched),
                }
            )

        return {
            "least_float_current_days": least_float_current_days,
            "new_paths": new_paths,
            "driving_delay_item_count": int(len(driving_delay_items)),
            "driving_delay_items": sorted(driving_delay_items.values(), key=lambda x: (x.get("activity_id") or "")),
            "cross_wbs_alert_count": int(len(cross_wbs_alerts)),
            "cross_wbs_alerts": cross_wbs_alerts,
        }

    path_impact = _path_impact_tracker()

    critical_path_successors = critical_path_successor_summaries(
        last,
        current,
        new_activity_ids=new_ids,
        least_float_current_days=path_impact.get("least_float_current_days"),
        included_change_wbs_ids=set(current_meta.get("_included_wbs_ids", set()) or set()),
    )

    return {
        "term": term_norm,
        "current_count": int(len(current_ids)),
        "last_count": int(len(last_ids)),
        "current_mother_wbs": current_meta.get("mother_wbs", []),
        "last_mother_wbs": last_meta.get("mother_wbs", []),
        "current_included_wbs_count": current_meta.get("included_wbs_count"),
        "last_included_wbs_count": last_meta.get("included_wbs_count"),
        "new_activity_ids": new_ids,
        "new_activities": [details.get(aid, {"activity_id": aid, "task_name": None, "wbs_name": None}) for aid in new_ids],
        "path_impact_tracker": path_impact,
        "critical_path_successors": critical_path_successors,
    }


def new_activities_all_wbs(last: XerSnapshot, current: XerSnapshot) -> dict[str, Any]:
    if current.task is None or current.task.empty:
        return {"count": 0, "items": [], "warning": "Current TASK is empty."}

    activity_id_col_curr = _pick_col(current.task, ["task_code", "activity_id"])
    if not activity_id_col_curr:
        return {"count": 0, "items": [], "warning": "Current TASK missing activity id column."}

    activity_id_col_last = _pick_col(last.task, ["task_code", "activity_id"]) if last.task is not None else None
    last_ids = (
        set(last.task[activity_id_col_last].astype(str).str.strip().tolist())
        if activity_id_col_last and last.task is not None and not last.task.empty
        else set()
    )
    current_ids = set(current.task[activity_id_col_curr].astype(str).str.strip().tolist())
    new_ids = current_ids - last_ids
    if not new_ids:
        return {"count": 0, "items": []}

    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])

    cols = [activity_id_col_curr]
    if name_col:
        cols.append(name_col)
    if wbs_name_col:
        cols.append(wbs_name_col)
    if wbs_path_col:
        cols.append(wbs_path_col)

    df = current.task[cols].copy()
    df = df.rename(columns={activity_id_col_curr: "activity_id"})
    if name_col:
        df = df.rename(columns={name_col: "task_name"})
    if wbs_name_col:
        df = df.rename(columns={wbs_name_col: "wbs_name"})
    if wbs_path_col:
        df = df.rename(columns={wbs_path_col: "wbs_path"})

    df["activity_id"] = df["activity_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["activity_id"])
    df = df[df["activity_id"].isin(new_ids)].copy()

    _clean = _clean_optional

    items: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        path = None if "wbs_path" not in df.columns else _clean(r.get("wbs_path"))
        if not path and "wbs_name" in df.columns:
            path = _clean(r.get("wbs_name"))

        parts: list[str] = []
        if path:
            parts = [p.strip() for p in str(path).split(" / ") if p.strip()]
        items.append(
            {
                "activity_id": str(r.get("activity_id")),
                "task_name": (None if "task_name" not in df.columns else _clean(r.get("task_name"))),
                "wbs_name": (None if "wbs_name" not in df.columns else _clean(r.get("wbs_name"))),
                "wbs_path": path,
                "wbs_hierarchy_root_to_leaf": parts,
                "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
            }
        )

    groups_by_leaf: dict[str, dict[str, Any]] = {}
    for it in items:
        path = it.get("wbs_path")
        parts = it.get("wbs_hierarchy_root_to_leaf") or []
        leaf_name = parts[-1] if parts else it.get("wbs_name")
        leaf_path = path if path else leaf_name
        key = str(leaf_path or "Unknown").strip() or "Unknown"
        if key not in groups_by_leaf:
            groups_by_leaf[key] = {
                "leaf_wbs_name": leaf_name,
                "leaf_wbs_path": path,
                "wbs_hierarchy_root_to_leaf": parts,
                "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                "items": [],
            }
        groups_by_leaf[key]["items"].append(
            {
                "activity_id": it.get("activity_id"),
                "task_name": it.get("task_name"),
            }
        )

    groups = sorted(
        groups_by_leaf.values(),
        key=lambda g: (g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or ""),
    )

    return {
        "count": int(len(items)),
        "group_count": int(len(groups)),
        "groups": groups,
        "items": items,
    }


def critical_path_to_target(
    current: XerSnapshot,
    target_activity_id: str,
    *,
    critical_threshold_days: float | None = None,
    near_critical_buffer_days: float | None = None,
) -> dict[str, Any]:
    if current.task is None or current.task.empty:
        return {"paths": [], "warning": "Current TASK is empty."}
    if current.taskpred is None or current.taskpred.empty:
        return {"paths": [], "warning": "Current TASKPRED is empty."}

    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    task_id_col = _pick_col(current.task, ["task_id"])
    if not activity_id_col or not task_id_col:
        return {"paths": [], "warning": "Current TASK missing activity_id/task_id."}

    try:
        float_col = xp.detect_total_float_column(current.task)
    except Exception:
        float_col = None
    if not float_col:
        return {"paths": [], "warning": "Current TASK missing total float column."}

    numeric = pd.to_numeric(current.task[float_col], errors="coerce")
    numeric_days = _task_float_series_to_days(current.task, float_col, numeric, current.calendar)
    if numeric_days.notna().sum() == 0:
        return {"paths": [], "warning": "Total float values are not available."}

    project_least_float_current_days = float(numeric_days.min(skipna=True))

    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    clndr_id_col = _pick_col(current.task, ["clndr_id", "calendar_id"])
    task_type_col = _pick_col(current.task, ["task_type", "task_type_code", "activity_type"])
    act_start_col, actual_finish_col = _detect_actual_cols(current.task)
    status_col = _pick_col(current.task, ["status_code", "task_status"])
    constraint_type_col = _pick_col(current.task, ["cstr_type", "constraint_type", "primary_constraint_type"])
    constraint_date_col = _pick_col(current.task, ["cstr_date", "constraint_date", "primary_constraint_date"])
    constraint_type2_col = _pick_col(current.task, ["cstr_type2", "secondary_constraint_type"])
    constraint_date2_col = _pick_col(current.task, ["cstr_date2", "secondary_constraint_date"])
    start_context_cols = [
        c
        for c in [
            _pick_col(current.task, ["restart_date"]),
            _pick_col(current.task, ["early_start_date", "early_start"]),
            _pick_col(current.task, ["target_start_date", "target_start"]),
            _pick_col(current.task, ["start_date"]),
            act_start_col,
        ]
        if c
    ]
    finish_context_cols = [
        c
        for c in [
            _pick_col(current.task, ["reend_date"]),
            _pick_col(current.task, ["early_end_date", "early_finish_date"]),
            _pick_col(current.task, ["target_end_date", "target_finish_date"]),
            _pick_col(current.task, ["end_date", "finish_date"]),
            actual_finish_col,
        ]
        if c
    ]

    cols = [task_id_col, activity_id_col, float_col]
    for c in [
        name_col,
        wbs_name_col,
        wbs_path_col,
        clndr_id_col,
        task_type_col,
        act_start_col,
        actual_finish_col,
        status_col,
        constraint_type_col,
        constraint_date_col,
        constraint_type2_col,
        constraint_date2_col,
        *start_context_cols,
        *finish_context_cols,
    ]:
        if c:
            cols.append(c)
    cols = list(dict.fromkeys(cols))

    tmp = current.task[cols].copy()
    tmp[task_id_col] = tmp[task_id_col].astype(str).str.strip()
    tmp[activity_id_col] = tmp[activity_id_col].astype(str).str.strip()
    tmp = tmp.drop_duplicates(subset=[activity_id_col])

    activity_by_task_id = dict(zip(tmp[task_id_col].tolist(), tmp[activity_id_col].tolist(), strict=False))
    row_by_activity = {
        str(row.get(activity_id_col)).strip(): row
        for _, row in tmp.iterrows()
        if str(row.get(activity_id_col, "")).strip()
    }

    name_by_activity = {
        str(aid): str(nm)
        for aid, nm in zip(tmp[activity_id_col].tolist(), tmp[name_col].astype(str).tolist(), strict=False)
    } if name_col else {}
    wbs_name_by_activity = {
        str(aid): str(wnm)
        for aid, wnm in zip(tmp[activity_id_col].tolist(), tmp[wbs_name_col].astype(str).tolist(), strict=False)
    } if wbs_name_col else {}
    wbs_path_by_activity = {
        str(aid): str(wpath)
        for aid, wpath in zip(tmp[activity_id_col].tolist(), tmp[wbs_path_col].astype(str).tolist(), strict=False)
    } if wbs_path_col else {}

    float_days_by_activity: dict[str, float] = {}
    vals_days = _task_float_series_to_days(tmp, float_col, pd.to_numeric(tmp[float_col], errors="coerce"), current.calendar)
    for aid, v in zip(tmp[activity_id_col].tolist(), vals_days.tolist(), strict=False):
        if pd.isna(v):
            continue
        float_days_by_activity[str(aid)] = float(v)
    float_raw_by_activity: dict[str, float] = {}
    for aid, v in zip(tmp[activity_id_col].tolist(), pd.to_numeric(tmp[float_col], errors="coerce").tolist(), strict=False):
        if pd.isna(v):
            continue
        float_raw_by_activity[str(aid)] = float(v)

    completed_activity_ids: set[str] = set()
    excluded_summary_activity_ids: set[str] = set()
    for _, row in tmp.iterrows():
        aid = str(row.get(activity_id_col, "")).strip()
        if not aid or aid.lower() == "nan":
            continue
        status = str(row.get(status_col, "")).casefold() if status_col else ""
        actual_finish = _parse_date(row.get(actual_finish_col)) if actual_finish_col else None
        if actual_finish is not None or "complete" in status:
            completed_activity_ids.add(aid)

        task_type = str(row.get(task_type_col, "")).strip().casefold() if task_type_col else ""
        if any(token in task_type for token in ["loe", "level of effort", "wbs", "summary"]):
            excluded_summary_activity_ids.add(aid)

    try:
        target_row = _resolve_target_row(current.task, target_activity_id)
    except Exception as e:
        return {"paths": [], "warning": f"Could not resolve target_activity_id: {e}"}

    target_aid = str(target_row.get(activity_id_col, "")).strip()
    if not target_aid:
        target_tid = str(target_row.get(task_id_col, "")).strip()
        target_aid = activity_by_task_id.get(target_tid, "")
    if not target_aid:
        return {"paths": [], "warning": "Could not resolve target activity id."}
    excluded_summary_activity_ids.discard(target_aid)

    succ_col = _pick_col(current.taskpred, ["task_id"])
    pred_col = _pick_col(current.taskpred, ["pred_task_id"])
    if not succ_col or not pred_col:
        return {"paths": [], "warning": "Current TASKPRED missing task_id/pred_task_id."}

    pred_by_succ: dict[str, set[str]] = defaultdict(set)
    succ_by_pred: dict[str, set[str]] = defaultdict(set)
    dfp = current.taskpred[[succ_col, pred_col]].copy()
    dfp[succ_col] = dfp[succ_col].astype(str).str.strip()
    dfp[pred_col] = dfp[pred_col].astype(str).str.strip()
    for _, r in dfp.iterrows():
        succ_aid = activity_by_task_id.get(str(r[succ_col]).strip())
        pred_aid = activity_by_task_id.get(str(r[pred_col]).strip())
        if not succ_aid or not pred_aid:
            continue
        pred_by_succ[str(succ_aid)].add(str(pred_aid))
        succ_by_pred[str(pred_aid)].add(str(succ_aid))

    target_float_days = float_days_by_activity.get(target_aid)
    if critical_threshold_days is None:
        governing_critical_float_days = target_float_days if target_float_days is not None else project_least_float_current_days
    else:
        governing_critical_float_days = float(critical_threshold_days)

    try:
        near_buffer_days = max(0.0, float(near_critical_buffer_days or 0.0))
    except Exception:
        near_buffer_days = 0.0
    absolute_near_critical_threshold_days = governing_critical_float_days + near_buffer_days

    # Keep tolerance small so true 1-day float activities do not get reported as critical.
    # This only absorbs minor float rounding noise after converting XER hours to days.
    float_tolerance_days = 0.01
    max_depth = 100
    max_trace_paths = 1000
    path_generation_truncated = False

    upstream_activity_ids: set[str] = {target_aid}
    q: deque[str] = deque([target_aid])
    while q:
        cur = q.popleft()
        for pred_id in sorted(pred_by_succ.get(cur, set())):
            if pred_id in upstream_activity_ids or pred_id in excluded_summary_activity_ids:
                continue
            upstream_activity_ids.add(pred_id)
            q.append(pred_id)

    # Target health: warn when the chosen target cannot drive a meaningful critical/delay analysis.
    # The most common trap is a contractual-date milestone with no predecessor logic (e.g. "Substantial
    # Completion Date" set only by a constraint). It traces back to itself and gets labelled critical
    # despite carrying large float, which makes the critical-path and delay story meaningless.
    target_direct_pred_count = len(pred_by_succ.get(target_aid, set()))
    target_upstream_count = len(upstream_activity_ids) - 1  # exclude the target itself
    target_is_off_critical = (
        target_float_days is not None
        and target_float_days > project_least_float_current_days + max(near_buffer_days, float_tolerance_days)
    )
    target_warnings: list[str] = []
    if target_direct_pred_count == 0:
        target_warnings.append(
            "Target activity has no predecessor logic, so it has no driving path. The critical-path/delay "
            "analysis will collapse to the target alone. Pick a logic-driven completion milestone instead."
        )
    elif target_upstream_count == 0:
        target_warnings.append(
            "Target activity has predecessors but none resolve to traceable upstream activities "
            "(they may be summary/LOE rows or missing from TASK)."
        )
    if target_is_off_critical:
        target_warnings.append(
            f"Target total float is {target_float_days:.1f} days vs the project minimum of "
            f"{project_least_float_current_days:.1f} days, so the target is not on (or near) the project critical "
            "path. Verify this is the intended driving milestone."
        )
    target_health = {
        "target_direct_predecessor_count": int(target_direct_pred_count),
        "target_upstream_activity_count": int(target_upstream_count),
        "target_float_current_days": target_float_days,
        "project_least_float_current_days": project_least_float_current_days,
        "target_is_off_critical_path": bool(target_is_off_critical),
        "has_driving_logic": bool(target_direct_pred_count > 0 and target_upstream_count > 0),
        "warnings": target_warnings,
    }

    critical_ids = {
        aid
        for aid in upstream_activity_ids
        if float_days_by_activity.get(aid) is not None
        and float_days_by_activity[aid] <= governing_critical_float_days + float_tolerance_days
    }
    # Scope near-critical to the target's own upstream network (everything that feeds the target),
    # mirroring the critical-path scope. This keeps near-critical relevant to the target: when the
    # target is a mid-project milestone (e.g. a Stage 1 completion), unrelated parallel work (e.g.
    # Stage 2 abutments) is excluded rather than reported as near-critical risk to this target.
    near_critical_candidate_ids = upstream_activity_ids - excluded_summary_activity_ids
    near_critical_ids_set = {
        aid
        for aid in near_critical_candidate_ids
        if aid not in critical_ids
        and float_days_by_activity.get(aid) is not None
        and float_days_by_activity[aid] > governing_critical_float_days + float_tolerance_days
        and float_days_by_activity[aid] <= absolute_near_critical_threshold_days + float_tolerance_days
    }
    completed_upstream_ids = {
        aid
        for aid in upstream_activity_ids
        if aid in completed_activity_ids
    }

    critical_path_ids: list[list[str]] = []
    seen_paths: set[tuple[str, ...]] = set()
    trace_bridge_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

    def qualifies_as_critical(pred_float: float, branch_float: float) -> bool:
        return pred_float <= branch_float + float_tolerance_days

    def _first_date_for_trace(row: pd.Series | None, cols_to_try: list[str]) -> pd.Timestamp | None:
        if row is None:
            return None
        for col in cols_to_try:
            ts = _parse_date(row.get(col))
            if ts is not None:
                return ts
        return None

    def _constraint_appears_to_drive(aid: str) -> bool:
        row = row_by_activity.get(aid)
        if row is None:
            return False
        start_date = _first_date_for_trace(row, start_context_cols)
        finish_date = _first_date_for_trace(row, finish_context_cols)

        constraints: list[tuple[Any, pd.Timestamp | None]] = []
        if constraint_type_col or constraint_date_col:
            constraints.append(
                (
                    row.get(constraint_type_col) if constraint_type_col else None,
                    _parse_date(row.get(constraint_date_col)) if constraint_date_col else None,
                )
            )
        if constraint_type2_col or constraint_date2_col:
            constraints.append(
                (
                    row.get(constraint_type2_col) if constraint_type2_col else None,
                    _parse_date(row.get(constraint_date2_col)) if constraint_date2_col else None,
                )
            )

        for raw_type, constraint_date in constraints:
            if constraint_date is None:
                continue
            ctype = str(raw_type or "").strip().casefold()
            start_type = any(token in ctype for token in ["start", "mso", "snet", "snlt"])
            finish_type = any(token in ctype for token in ["finish", "fin", "meo", "fnlt", "fnet"])
            aligns_with_start = (
                start_date is not None
                and abs((constraint_date.normalize() - start_date.normalize()).days) <= 1
            )
            aligns_with_finish = (
                finish_date is not None
                and abs((constraint_date.normalize() - finish_date.normalize()).days) <= 1
            )
            if (start_type and aligns_with_start) or (finish_type and aligns_with_finish):
                return True
            if not start_type and not finish_type and (aligns_with_start or aligns_with_finish):
                return True
        return False

    def _calendar_signature(aid: str) -> tuple[str | None, str | None, float | None]:
        row = row_by_activity.get(aid)
        if row is None:
            return None, None, None
        ctx = _task_calendar_context(tmp, current.calendar, row)
        return (
            _clean_optional(ctx.get("calendar_id")),
            _clean_optional(ctx.get("calendar_name")),
            ctx.get("calendar_hours_per_day"),
        )

    def _calendar_context_differs(pred_aid: str, succ_aid: str) -> bool:
        pred_id, pred_name, pred_hours = _calendar_signature(pred_aid)
        succ_id, succ_name, succ_hours = _calendar_signature(succ_aid)
        if pred_id and succ_id and pred_id != succ_id:
            return True
        if pred_name and succ_name and pred_name != succ_name:
            return True
        try:
            if pred_hours is not None and succ_hours is not None and abs(float(pred_hours) - float(succ_hours)) > 0.01:
                return True
        except Exception:
            pass
        return False

    def _calendar_bridge_candidates(cur_aid: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if _constraint_appears_to_drive(cur_aid):
            return []
        return [
            (pred_id, pred_float)
            for pred_id, pred_float in candidates
            if _calendar_context_differs(pred_id, cur_aid)
        ]

    def trace_path(cur_aid: str, path: list[str], current_branch_float: float) -> None:
        nonlocal path_generation_truncated
        if len(critical_path_ids) >= max_trace_paths:
            path_generation_truncated = True
            return
        if len(path) >= max_depth:
            path_generation_truncated = True
            return

        critical_preds: list[tuple[str, float]] = []
        all_preds: list[tuple[str, float]] = []
        for pred_id in sorted(pred_by_succ.get(cur_aid, set())):
            if pred_id in path or pred_id in excluded_summary_activity_ids:
                continue
            pred_float = float_days_by_activity.get(pred_id)
            if pred_float is None:
                continue
            all_preds.append((pred_id, pred_float))

            if qualifies_as_critical(pred_float, current_branch_float):
                critical_preds.append((pred_id, pred_float))

        # Bridge float gaps only when there is calendar evidence. If a constraint appears
        # to drive the current activity, the trace should stop at the constraint-driven item.
        if not critical_preds and all_preds:
            min_float = min(p[1] for p in all_preds)
            min_float_candidates = [
                (p_id, p_float)
                for p_id, p_float in all_preds
                if p_float <= min_float + float_tolerance_days
            ]
            calendar_candidates = _calendar_bridge_candidates(cur_aid, min_float_candidates)
            if calendar_candidates:
                for p_id, p_float in calendar_candidates:
                    critical_preds.append((p_id, p_float))
                    pred_sig = _calendar_signature(p_id)
                    succ_sig = _calendar_signature(cur_aid)
                    trace_bridge_by_pair[(p_id, cur_aid)] = {
                        "bridge_type": "calendar_float_gap_bridge",
                        "reason": (
                            "No predecessor matched the current branch float, so the minimum-float predecessor was traced "
                            "only because the predecessor and successor use different calendar context."
                        ),
                        "predecessor_float_days": p_float,
                        "successor_branch_float_days": current_branch_float,
                        "predecessor_calendar_id": pred_sig[0],
                        "predecessor_calendar_name": pred_sig[1],
                        "predecessor_calendar_hours_per_day": pred_sig[2],
                        "successor_calendar_id": succ_sig[0],
                        "successor_calendar_name": succ_sig[1],
                        "successor_calendar_hours_per_day": succ_sig[2],
                    }

        if not critical_preds:
            key = tuple(path)
            if key not in seen_paths:
                seen_paths.add(key)
                critical_path_ids.append(path)
            return

        for pred_id, pred_float in critical_preds:
            trace_path(pred_id, [pred_id] + path, pred_float)

    trace_path(target_aid, [target_aid], governing_critical_float_days)

    critical_trace_activity_ids = {
        aid
        for path in critical_path_ids
        for aid in path
    }
    near_critical_ids_set = near_critical_ids_set - critical_trace_activity_ids

    completed_critical_ids = critical_ids & completed_activity_ids
    relation_by_pair = _relationship_records_by_pair(current)

    def _first_date_from_cols(row: pd.Series, cols_to_try: list[str]) -> tuple[pd.Timestamp | None, str | None]:
        for col in cols_to_try:
            ts = _parse_date(row.get(col))
            if ts is not None:
                return ts, col
        return None, None

    def _constraint_kind(raw_type: Any, constraint_date: pd.Timestamp | None, start_date: pd.Timestamp | None) -> str:
        ctype = str(raw_type or "").strip().casefold()
        if "start" in ctype or "mso" in ctype or "snet" in ctype or "snlt" in ctype:
            return "start"
        if "finish" in ctype or "fin" in ctype or "meo" in ctype or "fnlt" in ctype or "fnet" in ctype:
            return "finish"
        if constraint_date is not None and start_date is not None:
            if abs((constraint_date.normalize() - start_date.normalize()).days) <= 1:
                return "start"
        return "unknown"

    def _constraint_driver_for(aid: str, predecessor_aid: str | None) -> dict[str, Any] | None:
        row = row_by_activity.get(aid)
        if row is None:
            return None

        start_date, start_col = _first_date_from_cols(row, start_context_cols)
        constraints: list[tuple[Any, pd.Timestamp | None, str | None]] = []
        if constraint_type_col or constraint_date_col:
            constraints.append(
                (
                    row.get(constraint_type_col) if constraint_type_col else None,
                    _parse_date(row.get(constraint_date_col)) if constraint_date_col else None,
                    "primary",
                )
            )
        if constraint_type2_col or constraint_date2_col:
            constraints.append(
                (
                    row.get(constraint_type2_col) if constraint_type2_col else None,
                    _parse_date(row.get(constraint_date2_col)) if constraint_date2_col else None,
                    "secondary",
                )
            )

        best: dict[str, Any] | None = None
        for raw_type, constraint_date, source in constraints:
            if constraint_date is None:
                continue
            kind = _constraint_kind(raw_type, constraint_date, start_date)
            if kind != "start":
                continue

            predecessor_finish = None
            predecessor_finish_col = None
            predecessor_completed = None
            predecessor_task_name = None
            if predecessor_aid:
                pred_row = row_by_activity.get(predecessor_aid)
                predecessor_task_name = name_by_activity.get(predecessor_aid)
                predecessor_completed = predecessor_aid in completed_activity_ids
                if pred_row is not None:
                    predecessor_finish, predecessor_finish_col = _first_date_from_cols(pred_row, finish_context_cols)

            aligns_with_start = (
                start_date is not None
                and abs((constraint_date.normalize() - start_date.normalize()).days) <= 1
            )
            predecessor_not_later_than_constraint = (
                predecessor_finish is None
                or predecessor_finish.normalize() <= constraint_date.normalize()
            )
            possible_driver = bool(aligns_with_start and predecessor_not_later_than_constraint)
            if not possible_driver:
                continue

            best = {
                "activity_task_name": name_by_activity.get(aid),
                "constraint_source": source,
                "constraint_type": _clean_optional(raw_type),
                "constraint_date": constraint_date.isoformat(),
                "forecast_start_date": None if start_date is None else start_date.isoformat(),
                "forecast_start_col": start_col,
                "predecessor_task_name": predecessor_task_name,
                "predecessor_forecast_finish_date": None if predecessor_finish is None else predecessor_finish.isoformat(),
                "predecessor_forecast_finish_col": predecessor_finish_col,
                "predecessor_completed": predecessor_completed,
                "assessment": (
                    "Start appears constraint-driven: the constraint date aligns with the activity forecast/restart start, "
                    "and the selected critical predecessor is complete or does not finish later than the constraint date."
                ),
            }
            break
        return best

    def item_for(aid: str, predecessor_aid: str | None = None) -> dict[str, Any]:
        row = row_by_activity.get(aid)
        forecast_start = forecast_start_col = forecast_finish = forecast_finish_col = None
        actual_start = actual_finish = None
        if row is not None:
            forecast_start, forecast_start_col = _first_date_from_cols(row, start_context_cols)
            forecast_finish, forecast_finish_col = _first_date_from_cols(row, finish_context_cols)
            actual_start = _parse_date(row.get(act_start_col)) if act_start_col else None
            actual_finish = _parse_date(row.get(actual_finish_col)) if actual_finish_col else None
        item = {
            "activity_id": aid,
            "task_name": name_by_activity.get(aid),
            "wbs_path": wbs_path_by_activity.get(aid),
            "wbs_name": wbs_name_by_activity.get(aid),
            "is_critical": aid in critical_ids,
            "is_near_critical": aid in near_critical_ids_set,
            "on_critical_trace": aid in critical_trace_activity_ids,
            "float_current_days": float_days_by_activity.get(aid),
            "float_current_raw_hours": float_raw_by_activity.get(aid),
            "float_basis": _float_basis_text(current, short=True),
            "completed": aid in completed_activity_ids,
            "forecast_start_date": None if forecast_start is None else forecast_start.isoformat(),
            "forecast_finish_date": None if forecast_finish is None else forecast_finish.isoformat(),
            "forecast_start_col": forecast_start_col,
            "forecast_finish_col": forecast_finish_col,
            "actual_start_date": None if actual_start is None else actual_start.isoformat(),
            "actual_finish_date": None if actual_finish is None else actual_finish.isoformat(),
        }
        if row is not None:
            item.update(_task_calendar_context(tmp, current.calendar, row))
        constraint_driver = _constraint_driver_for(aid, predecessor_aid)
        if constraint_driver:
            item["constraint_driver"] = constraint_driver
        return item

    def link_for(pred_aid: str, succ_aid: str) -> dict[str, Any]:
        rel = relation_by_pair.get((pred_aid, succ_aid), {}) or {}
        out = {
            "from_activity_id": pred_aid,
            "to_activity_id": succ_aid,
            "from_task_name": name_by_activity.get(pred_aid),
            "to_task_name": name_by_activity.get(succ_aid),
            "relationship_type": rel.get("relationship_type"),
            "lag_days": rel.get("lag_days"),
        }
        bridge = trace_bridge_by_pair.get((pred_aid, succ_aid))
        if bridge:
            out["trace_bridge"] = bridge
        return out

    def path_object(p: list[str], branch_type: str) -> dict[str, Any]:
        chain = [item_for(aid, p[i - 1] if i > 0 else None) for i, aid in enumerate(p)]
        constraint_drivers = [x["constraint_driver"] for x in chain if x.get("constraint_driver")]
        logic_links = [link_for(p[i], p[i + 1]) for i in range(len(p) - 1)]
        floats = [x.get("float_current_days") for x in chain if x.get("float_current_days") is not None]
        return {
            "branch_type": branch_type,
            "length": len(p),
            "branch_min_float_days": min(floats) if floats else None,
            "branch_max_float_days": max(floats) if floats else None,
            "activity_chain": chain,
            "logic_links": logic_links,
            "constraint_driver_count": int(len(constraint_drivers)),
            "constraint_drivers": constraint_drivers,
        }

    out_paths = [path_object(p, "critical") for p in sorted(critical_path_ids, key=lambda x: (-len(x), x))]
    links = []
    for p in critical_path_ids:
        for i in range(len(p) - 1):
            links.append(link_for(p[i], p[i + 1]))

    # De-duplicate links
    seen_links: set[tuple[str, str]] = set()
    dedup_links = []
    for l in links:
        key = (str(l.get("from_activity_id")), str(l.get("to_activity_id")))
        if key in seen_links:
            continue
        seen_links.add(key)
        dedup_links.append(l)

    method_description = (
        "Treats TASK/TASKPRED as a DAG and traces backward from the target. Critical branches continue through "
        "predecessors whose total float is less than or equal to the current branch float within tolerance. If no "
        "predecessor matches the current branch float, the trace bridges only to the minimum-float predecessor(s) "
        "when a predecessor/successor calendar-context difference is detected and the current activity is not "
        "constraint-driven; this calendar bridge is independent of the near-critical threshold. Near-critical work "
        "is not path-traced; it is classified across non-summary current-schedule activities by "
        + ("Microsoft Project TotalSlack" if _is_mspdi(current) else "P6 total float")
        + " threshold and excludes activities already included in the critical trace."
    )
    method_float_basis = (
        "Microsoft Project TotalSlack values are converted from tenths of a minute to days using the exported "
        "MinutesPerDay value."
        if _is_mspdi(current)
        else "P6 total float values are read from TASK and converted from hours to days in Python using each activity's assigned CALENDAR.day_hr_cnt where available."
    )

    return {
        "target_activity_id": target_aid,
        "target_task_name": name_by_activity.get(str(target_aid)),
        "target_wbs_path": wbs_path_by_activity.get(str(target_aid)),
        "target_health": target_health,
        "target_warning": (" ".join(target_health["warnings"]) or None),
        "target_float_current_days": target_float_days,
        "least_float_current_days": governing_critical_float_days,
        "project_least_float_current_days": project_least_float_current_days,
        "critical_threshold_days": governing_critical_float_days,
        "near_critical_buffer_days": near_buffer_days,
        "absolute_near_critical_threshold_days": absolute_near_critical_threshold_days,
        "critical_count": int(len(critical_ids)),
        "completed_critical_count": int(len(completed_critical_ids)),
        "critical_activity_ids": sorted(critical_ids),
        "critical_trace_activity_ids": sorted(critical_trace_activity_ids),
        "critical_trace_bridge_count": int(len(trace_bridge_by_pair)),
        "critical_trace_bridges": [
            {
                "from_activity_id": pred_aid,
                "to_activity_id": succ_aid,
                "from_task_name": name_by_activity.get(pred_aid),
                "to_task_name": name_by_activity.get(succ_aid),
                **bridge,
            }
            for (pred_aid, succ_aid), bridge in sorted(trace_bridge_by_pair.items())
        ],
        "critical_activities": [item_for(aid) for aid in sorted(critical_ids)],
        "near_critical_count": int(len(near_critical_ids_set)),
        "near_critical_activity_ids": sorted(near_critical_ids_set),
        "near_critical_activities": [item_for(aid) for aid in sorted(near_critical_ids_set)],
        "completed_upstream_count": int(len(completed_upstream_ids)),
        "completed_upstream_activities": [item_for(aid) for aid in sorted(completed_upstream_ids)],
        "path_count": int(len(out_paths)),
        "paths": out_paths,
        "links": dedup_links,
        "method": {
            "name": "dag_two_tier_float_trace",
            "description": method_description,
            "float_col": float_col,
            "float_basis": method_float_basis,
            "date_math_guardrail": "Forecast/actual dates are context only; do not infer float, slack, or near-critical status from date gaps, weekends, or apparent gaps between linked activities.",
            "float_tolerance_days": float_tolerance_days,
            "critical_threshold_days": governing_critical_float_days,
            "near_critical_buffer_days": near_buffer_days,
            "absolute_near_critical_threshold_days": absolute_near_critical_threshold_days,
            "loe_and_wbs_summary_filtered": True,
            "excluded_summary_activity_count": int(len(excluded_summary_activity_ids)),
            "completed_activities_included": True,
            "max_trace_paths_returned": max_trace_paths,
            "max_depth": max_depth,
            "path_generation_truncated": path_generation_truncated,
        },
    }


def critical_path_successor_summaries(
    last: XerSnapshot,
    current: XerSnapshot,
    *,
    new_activity_ids: list[str],
    least_float_current_days: float | None = None,
    included_change_wbs_ids: set[str] | None = None,
    max_depth: int = 50,
) -> dict[str, Any]:
    """
    Critical Path Successor:
      1) For each new activity (typically from the "Changes" WBS section), find immediate successors (TASKPRED).
      2) If a successor's total_float matches least_float_current, label it a Critical Path Driver.
      3) If that successor is also new, keep tracing successors until reaching an existing activity (present in Last).
      4) Output narrative-ready strings, e.g.:
         'CO#04 (New) -> Impacts -> Actuator Procurement (New) -> Drives -> Mechanical Re-installation (Critical Path).'
    """
    included_change_wbs_ids = included_change_wbs_ids or set()

    if not new_activity_ids:
        return {"count": 0, "items": []}
    if current.task is None or current.task.empty:
        return {"count": 0, "items": [], "warning": "Current TASK is empty."}
    if current.taskpred is None or current.taskpred.empty:
        return {"count": 0, "items": [], "warning": "Current TASKPRED is empty."}

    task_id_col = _pick_col(current.task, ["task_id"])
    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_id_col = _pick_col(current.task, ["wbs_id"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    clndr_id_col = _pick_col(current.task, ["clndr_id", "calendar_id"])
    if not task_id_col or not activity_id_col:
        return {"count": 0, "items": [], "warning": "Current TASK missing task_id/task_code."}

    float_col = None
    try:
        float_col = xp.detect_total_float_column(current.task)
    except Exception:
        float_col = None

    # Determine least float (in DAYS) if not provided.
    if least_float_current_days is None and float_col:
        numeric = pd.to_numeric(current.task[float_col], errors="coerce")
        numeric_days = _task_float_series_to_days(current.task, float_col, numeric, current.calendar)
        if numeric_days.notna().sum() > 0:
            least_float_current_days = float(numeric_days.min(skipna=True))

    eps = 1e-9

    def is_critical(activity_id: str) -> bool:
        if least_float_current_days is None or not float_col:
            return False
        tf = float_days_by_activity.get(activity_id)
        if tf is None:
            return False
        return abs(float(tf) - float(least_float_current_days)) <= eps

    # Lookup tables from CURRENT.
    cols = [task_id_col, activity_id_col]
    if name_col:
        cols.append(name_col)
    if wbs_id_col:
        cols.append(wbs_id_col)
    if wbs_name_col:
        cols.append(wbs_name_col)
    if wbs_path_col:
        cols.append(wbs_path_col)
    if clndr_id_col:
        cols.append(clndr_id_col)
    if float_col:
        cols.append(float_col)

    tmp = current.task[cols].copy()
    tmp[task_id_col] = tmp[task_id_col].astype(str).str.strip()
    tmp[activity_id_col] = tmp[activity_id_col].astype(str).str.strip()
    tmp = tmp.drop_duplicates(subset=[activity_id_col])

    task_id_by_activity = dict(zip(tmp[activity_id_col].tolist(), tmp[task_id_col].tolist(), strict=False))
    activity_by_task_id = dict(zip(tmp[task_id_col].tolist(), tmp[activity_id_col].tolist(), strict=False))

    name_by_activity: dict[str, str] = {}
    if name_col:
        for aid, nm in zip(tmp[activity_id_col].tolist(), tmp[name_col].astype(str).tolist(), strict=False):
            name_by_activity[str(aid)] = str(nm)

    wbs_id_by_activity: dict[str, str] = {}
    if wbs_id_col:
        for aid, wid in zip(tmp[activity_id_col].tolist(), tmp[wbs_id_col].astype(str).tolist(), strict=False):
            wbs_id_by_activity[str(aid)] = str(wid).strip()

    wbs_name_by_activity: dict[str, str] = {}
    if wbs_name_col:
        for aid, wnm in zip(tmp[activity_id_col].tolist(), tmp[wbs_name_col].astype(str).tolist(), strict=False):
            wbs_name_by_activity[str(aid)] = str(wnm)

    wbs_path_by_activity: dict[str, str] = {}
    if wbs_path_col:
        for aid, wpath in zip(tmp[activity_id_col].tolist(), tmp[wbs_path_col].astype(str).tolist(), strict=False):
            wbs_path_by_activity[str(aid)] = str(wpath)

    float_days_by_activity: dict[str, float] = {}
    if float_col:
        vals_raw = pd.to_numeric(tmp[float_col], errors="coerce")
        vals_days = _task_float_series_to_days(tmp, float_col, vals_raw, current.calendar)
        for aid, v in zip(tmp[activity_id_col].tolist(), vals_days.tolist(), strict=False):
            if pd.isna(v):
                continue
            float_days_by_activity[str(aid)] = float(v)

    # Successor adjacency (by internal task_id) from CURRENT TASKPRED.
    succ_col = _pick_col(current.taskpred, ["task_id"])
    pred_col = _pick_col(current.taskpred, ["pred_task_id"])
    if not succ_col or not pred_col:
        return {"count": 0, "items": [], "warning": "Current TASKPRED missing task_id/pred_task_id."}

    succ_by_pred: dict[str, set[str]] = defaultdict(set)
    dfp = current.taskpred[[succ_col, pred_col]].copy()
    dfp[succ_col] = dfp[succ_col].astype(str).str.strip()
    dfp[pred_col] = dfp[pred_col].astype(str).str.strip()
    for _, r in dfp.iterrows():
        succ_tid = str(r[succ_col]).strip()
        pred_tid = str(r[pred_col]).strip()
        if not succ_tid or not pred_tid or succ_tid.lower() == "nan" or pred_tid.lower() == "nan":
            continue
        succ_by_pred[pred_tid].add(succ_tid)

    # Newness relative to LAST (global across schedule).
    last_aid_col = _pick_col(last.task, ["task_code", "activity_id"])
    last_ids = set(last.task[last_aid_col].astype(str).str.strip().tolist()) if last_aid_col and last.task is not None and not last.task.empty else set()
    current_ids = set(current.task[activity_id_col].astype(str).str.strip().tolist())
    global_new_ids = current_ids - last_ids

    def status(aid: str) -> str:
        if aid in last_ids:
            return "Existing"
        if aid in global_new_ids:
            return "New"
        return "Unknown"

    def label(aid: str) -> str:
        nm = name_by_activity.get(aid) or "Unnamed activity"
        st = status(aid)
        crit = is_critical(aid)
        if crit:
            return f"{nm} ({st}, Critical Path)"
        return f"{nm} ({st})"

    def arrow(to_aid: str) -> str:
        return " -> Drives -> " if is_critical(to_aid) else " -> Impacts -> "

    def build_summary(path: list[str]) -> str:
        if not path:
            return ""
        s = label(path[0])
        for nxt in path[1:]:
            s += arrow(nxt) + label(nxt)
        return s + "."

    items: list[dict[str, Any]] = []

    for root_aid in new_activity_ids:
        root_aid = str(root_aid).strip()
        if not root_aid:
            continue
        root_tid = task_id_by_activity.get(root_aid)
        if not root_tid:
            items.append(
                {
                    "root_activity_id": root_aid,
                    "summary": "Unnamed new activity -> Impacts -> (No successors: missing task_id).",
                    "path_activity_ids": [root_aid],
                }
            )
            continue

        # BFS through successors, expanding only while node is globally new.
        # We prefer reaching an EXISTING endpoint, especially if it's Critical Path.
        start_path = [root_aid]
        q: deque[list[str]] = deque([start_path])
        visited: set[str] = {root_aid}
        endpoints: list[list[str]] = []

        while q:
            path = q.popleft()
            if len(path) > max_depth:
                endpoints.append(path)
                continue
            cur_aid = path[-1]

            # Stop expanding when we hit an existing activity (not new vs last).
            if cur_aid != root_aid and status(cur_aid) == "Existing":
                endpoints.append(path)
                continue

            # Only continue tracing successors when the current node is also new (per requirement).
            if cur_aid != root_aid and status(cur_aid) != "New":
                endpoints.append(path)
                continue

            cur_tid = task_id_by_activity.get(cur_aid)
            if not cur_tid:
                endpoints.append(path)
                continue

            succ_tids = sorted(succ_by_pred.get(cur_tid, set()))
            if not succ_tids:
                endpoints.append(path)
                continue

            for succ_tid in succ_tids:
                succ_aid = activity_by_task_id.get(succ_tid)
                if not succ_aid:
                    continue
                succ_aid = str(succ_aid).strip()
                if not succ_aid:
                    continue
                new_path = path + [succ_aid]
                endpoints.append(new_path) if status(succ_aid) == "Existing" else None
                # still allow enqueuing if new (or unknown) and not revisiting
                if succ_aid not in visited and status(succ_aid) == "New":
                    visited.add(succ_aid)
                    q.append(new_path)
                elif succ_aid not in visited and status(succ_aid) != "Existing":
                    # Unknown: treat as endpoint; don't keep expanding.
                    endpoints.append(new_path)

        if not endpoints:
            endpoints = [start_path]

        def score(p: list[str]) -> tuple[int, int, int, int]:
            end = p[-1]
            end_existing = 0 if (end != root_aid and status(end) == "Existing") else 1
            end_critical = 0 if is_critical(end) else 1
            any_critical = 0 if any(is_critical(x) for x in p) else 1
            return (end_existing, end_critical, any_critical, len(p))

        best = sorted(endpoints, key=score)[0]

        cross_wbs = False
        cross_wbs_top = None
        if included_change_wbs_ids and wbs_id_col:
            for aid in best[1:]:
                wid = wbs_id_by_activity.get(aid)
                if wid and wid not in included_change_wbs_ids:
                    cross_wbs = True
                    wpath = wbs_path_by_activity.get(aid) or ""
                    cross_wbs_top = (wpath.split(" / ", 1)[0].strip() if " / " in wpath else (wpath.strip() or wbs_name_by_activity.get(aid)))
                    break

        items.append(
            {
                "root_activity_id": root_aid,
                "summary": build_summary(best),
                "path_activity_ids": best,
                "has_critical_path_driver": any(is_critical(x) for x in best[1:]),
                "cross_wbs": cross_wbs,
                "cross_wbs_top": cross_wbs_top,
            }
        )

    return {
        "least_float_current_days": least_float_current_days,
        "count": int(len(items)),
        "items": items,
    }


def work_accomplished(last: XerSnapshot, current: XerSnapshot) -> dict[str, Any]:
    if last.data_date is None or current.data_date is None:
        return {"window": None, "count": 0, "activities": []}

    start = last.data_date.normalize()
    end = current.data_date.normalize()
    if end < start:
        start, end = end, start
    msp_period_unavailable = _is_mspdi(last) and _is_mspdi(current) and end <= start
    msp_period_warning = (
        "Microsoft Project StatusDate did not advance between the previous and current XML files; "
        "period progress cannot be assessed."
        if msp_period_unavailable
        else None
    )

    act_start_col, act_finish_col = _detect_actual_cols(current.task)
    if not act_start_col and not act_finish_col:
        result = {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": 0, "activities": []}
        if msp_period_warning:
            result.update({"period_assessment_available": False, "warning": msp_period_warning})
        return result

    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])

    df = current.task.copy()
    if act_start_col:
        df["_act_start"] = pd.to_datetime(df[act_start_col], errors="coerce")
    else:
        df["_act_start"] = pd.NaT
    if act_finish_col:
        df["_act_finish"] = pd.to_datetime(df[act_finish_col], errors="coerce")
    else:
        df["_act_finish"] = pd.NaT

    in_start = df["_act_start"].notna() & (df["_act_start"].dt.normalize().between(start, end, inclusive="both"))
    in_finish = df["_act_finish"].notna() & (df["_act_finish"].dt.normalize().between(start, end, inclusive="both"))
    df = df[in_start | in_finish].copy()

    activities: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        activities.append(
            {
                "activity_id": (None if not activity_id_col else str(r.get(activity_id_col))),
                "task_name": (None if not name_col else str(r.get(name_col))),
                "wbs_name": (None if not wbs_name_col else str(r.get(wbs_name_col))),
                "wbs_path": (None if not wbs_path_col else str(r.get(wbs_path_col))),
                "actual_start_date": (None if pd.isna(r["_act_start"]) else pd.Timestamp(r["_act_start"]).isoformat()),
                "actual_finish_date": (None if pd.isna(r["_act_finish"]) else pd.Timestamp(r["_act_finish"]).isoformat()),
            }
        )

    # de-dupe by activity_id if present
    if activity_id_col:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for a in activities:
            aid = a.get("activity_id")
            if aid is None:
                deduped.append(a)
                continue
            if aid in seen:
                continue
            seen.add(aid)
            deduped.append(a)
        activities = deduped

    result = {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": int(len(activities)), "activities": activities}
    if msp_period_warning:
        result.update({"period_assessment_available": False, "warning": msp_period_warning})
    return result


def microsoft_project_progress_changes(last: XerSnapshot, current: XerSnapshot) -> dict[str, Any]:
    """Compare MSPDI progress fields without altering the established XER progress logic."""
    if not (_is_mspdi(last) and _is_mspdi(current)):
        return {}

    days_between = (
        None
        if last.data_date is None or current.data_date is None
        else int((current.data_date.normalize() - last.data_date.normalize()).days)
    )
    assessment_available = days_between is not None and days_between > 0
    warning = None
    if not assessment_available:
        warning = (
            "Microsoft Project StatusDate must advance between XML updates to attribute progress to a reporting "
            "period. The values below are file-version differences only."
        )

    aid_last = _pick_col(last.task, ["task_code", "activity_id"])
    aid_current = _pick_col(current.task, ["task_code", "activity_id"])
    if not aid_last or not aid_current:
        return {
            "period_assessment_available": assessment_available,
            "days_between_status_dates": days_between,
            "warning": warning or "Microsoft Project task identifiers are unavailable.",
            "count": 0,
            "items": [],
        }

    previous = {
        str(row.get(aid_last, "")).strip(): row
        for _, row in last.task.iterrows()
        if str(row.get(aid_last, "")).strip()
    }
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    task_type_col = _pick_col(current.task, ["task_type", "task_type_code", "activity_type"])

    def numeric(row: pd.Series, col: str) -> float | None:
        if col not in row.index:
            return None
        value = pd.to_numeric(row.get(col), errors="coerce")
        return None if pd.isna(value) else float(value)

    items: list[dict[str, Any]] = []
    for _, row in current.task.iterrows():
        aid = str(row.get(aid_current, "")).strip()
        before = previous.get(aid)
        if before is None:
            continue
        if task_type_col:
            task_type = str(row.get(task_type_col, "")).strip().casefold()
            if any(token in task_type for token in ["wbs", "summary", "loe", "level of effort"]):
                continue

        pct_before = numeric(before, "msp_percent_complete")
        pct_current = numeric(row, "msp_percent_complete")
        physical_before = numeric(before, "msp_physical_percent_complete")
        physical_current = numeric(row, "msp_physical_percent_complete")
        remaining_before = numeric(before, "remain_drtn_hr_cnt")
        remaining_current = numeric(row, "remain_drtn_hr_cnt")

        pct_delta = None if pct_before is None or pct_current is None else pct_current - pct_before
        physical_delta = (
            None if physical_before is None or physical_current is None else physical_current - physical_before
        )
        remaining_delta = (
            None if remaining_before is None or remaining_current is None else remaining_current - remaining_before
        )
        if not any(
            delta is not None and abs(delta) > 1e-9
            for delta in (pct_delta, physical_delta, remaining_delta)
        ):
            continue

        items.append(
            {
                "task_name": _clean_optional(row.get(name_col)) if name_col else None,
                "wbs_name": _clean_optional(row.get(wbs_name_col)) if wbs_name_col else None,
                "wbs_path": _clean_optional(row.get(wbs_path_col)) if wbs_path_col else None,
                "percent_complete_previous": pct_before,
                "percent_complete_current": pct_current,
                "percent_complete_change": pct_delta,
                "physical_percent_complete_previous": physical_before,
                "physical_percent_complete_current": physical_current,
                "physical_percent_complete_change": physical_delta,
                "remaining_duration_previous_hours": remaining_before,
                "remaining_duration_current_hours": remaining_current,
                "remaining_duration_change_hours": remaining_delta,
            }
        )

    items.sort(
        key=lambda item: (
            -float(item.get("percent_complete_change") or item.get("physical_percent_complete_change") or 0.0),
            str(item.get("wbs_path") or ""),
            str(item.get("task_name") or ""),
        )
    )
    return {
        "period_assessment_available": assessment_available,
        "days_between_status_dates": days_between,
        "warning": warning,
        "count": len(items),
        "items": items,
    }


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return None
    return s


def _parse_date(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _select_forecast_start_date(task_df: pd.DataFrame, row: pd.Series) -> tuple[pd.Timestamp | None, str | None]:
    """
    Select a planned/forecast start date for look-ahead checks.
    """
    priority_groups: list[list[str]] = [
        ["early_start_date", "early_start"],
        ["target_start_date", "target_start"],
        ["start_date"],
        ["late_start_date", "late_start"],
        ["act_start_date", "actual_start_date"],
    ]

    for group in priority_groups:
        col = _pick_col(task_df, group)
        if not col:
            continue
        ts = _parse_date(row.get(col))
        if ts is not None:
            return ts, col
    return None, None


def _select_forecast_finish_date(task_df: pd.DataFrame, row: pd.Series) -> tuple[pd.Timestamp | None, str | None]:
    """
    Select a non-actual finish date for look-back slippage checks.

    P6 exports vary by configuration. Early finish is usually the best forecast for in-progress work,
    while target finish is a useful fallback for not-started work.
    """
    priority_groups: list[list[str]] = [
        ["early_end_date", "early_finish_date"],
        ["target_finish_date", "target_end_date"],
        ["end_date", "finish_date"],
        ["late_end_date", "late_finish_date"],
    ]

    for group in priority_groups:
        col = _pick_col(task_df, group)
        if not col:
            continue
        ts = _parse_date(row.get(col))
        if ts is not None:
            return ts, col
    return None, None


def _group_items_by_wbs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups_by_leaf: dict[str, dict[str, Any]] = {}
    for it in items:
        path = _clean_optional(it.get("wbs_path")) or _clean_optional(it.get("wbs_name"))
        parts = [p.strip() for p in str(path).split(" / ") if p.strip()] if path else []
        leaf_name = parts[-1] if parts else _clean_optional(it.get("wbs_name"))
        key = str(path or leaf_name or "Unknown").strip() or "Unknown"
        if key not in groups_by_leaf:
            groups_by_leaf[key] = {
                "leaf_wbs_name": leaf_name,
                "leaf_wbs_path": path,
                "wbs_hierarchy_root_to_leaf": parts,
                "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                "items": [],
            }
        groups_by_leaf[key]["items"].append(it)

    return sorted(groups_by_leaf.values(), key=lambda g: (g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or ""))


def _task_lookup_by_activity(snapshot: XerSnapshot) -> dict[str, pd.Series]:
    if snapshot.task is None or snapshot.task.empty:
        return {}
    activity_id_col = _pick_col(snapshot.task, ["task_code", "activity_id"])
    if not activity_id_col:
        return {}
    out: dict[str, pd.Series] = {}
    for _, row in snapshot.task.iterrows():
        aid = str(row.get(activity_id_col, "")).strip()
        if aid and aid.lower() != "nan" and aid not in out:
            out[aid] = row
    return out


def _activity_task_id_maps(snapshot: XerSnapshot) -> tuple[dict[str, str], dict[str, str]]:
    if snapshot.task is None or snapshot.task.empty:
        return {}, {}
    activity_id_col = _pick_col(snapshot.task, ["task_code", "activity_id"])
    task_id_col = _pick_col(snapshot.task, ["task_id"])
    if not activity_id_col or not task_id_col:
        return {}, {}
    activity_by_task_id: dict[str, str] = {}
    task_id_by_activity: dict[str, str] = {}
    for _, row in snapshot.task.iterrows():
        aid = str(row.get(activity_id_col, "")).strip()
        tid = str(row.get(task_id_col, "")).strip()
        if not aid or aid.lower() == "nan" or not tid or tid.lower() == "nan":
            continue
        task_id_by_activity[aid] = tid
        activity_by_task_id[tid] = aid
    return task_id_by_activity, activity_by_task_id


def _selected_task_fields(snapshot: XerSnapshot, activity_id: str) -> dict[str, Any]:
    row = _task_lookup_by_activity(snapshot).get(str(activity_id))
    if row is None:
        return {}

    def col(names: list[str]) -> str | None:
        return _pick_col(snapshot.task, names)

    field_groups = {
        "forecast_start": ["early_start_date", "early_start", "target_start_date", "target_start", "start_date"],
        "forecast_finish": ["early_end_date", "early_finish_date", "target_finish_date", "target_end_date", "end_date", "finish_date"],
        "constraint_type": ["cstr_type", "constraint_type", "primary_constraint_type"],
        "constraint_date": ["cstr_date", "constraint_date", "primary_constraint_date"],
        "secondary_constraint_type": ["cstr_type2", "secondary_constraint_type"],
        "secondary_constraint_date": ["cstr_date2", "secondary_constraint_date"],
        "calendar": ["clndr_id", "calendar_id", "calendar_name"],
        "remaining_duration": ["remain_drtn_hr_cnt", "remaining_duration", "remain_drtn"],
        "original_duration": ["target_drtn_hr_cnt", "orig_drtn_hr_cnt", "duration"],
        "physical_percent_complete": ["phys_complete_pct", "physical_percent_complete"],
        "status": ["status_code", "task_status"],
    }

    out: dict[str, Any] = {}
    for key, names in field_groups.items():
        c = col(names)
        if not c:
            continue
        value = row.get(c)
        cleaned = _clean_optional(value)
        if cleaned is not None:
            out[key] = cleaned
            out[f"{key}_col"] = c
    return out


def _task_field_changes(last: XerSnapshot, current: XerSnapshot, activity_id: str) -> list[dict[str, Any]]:
    before = _selected_task_fields(last, activity_id)
    after = _selected_task_fields(current, activity_id)
    keys = sorted({k for k in before.keys() | after.keys() if not k.endswith("_col")})
    changes: list[dict[str, Any]] = []
    for key in keys:
        before_val = before.get(key)
        after_val = after.get(key)
        if before_val == after_val:
            continue
        change: dict[str, Any] = {
            "field": key,
            "previous_value": before_val,
            "current_value": after_val,
        }

        before_date = _parse_date(before_val)
        after_date = _parse_date(after_val)
        if before_date is not None and after_date is not None:
            change["delta_days"] = int((after_date.normalize() - before_date.normalize()).days)

        changes.append(change)
    return changes


def _relationship_records_by_pair(snapshot: XerSnapshot) -> dict[tuple[str, str], dict[str, Any]]:
    if snapshot.taskpred is None or snapshot.taskpred.empty:
        return {}
    _, activity_by_task_id = _activity_task_id_maps(snapshot)
    succ_col = _pick_col(snapshot.taskpred, ["task_id"])
    pred_col = _pick_col(snapshot.taskpred, ["pred_task_id"])
    if not succ_col or not pred_col:
        return {}

    type_col = _pick_col(snapshot.taskpred, ["pred_type", "link_type", "relationship_type"])
    lag_col = _pick_col(snapshot.taskpred, ["lag_hr_cnt", "lag_drtn_hr_cnt", "lag", "lag_days"])
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in snapshot.taskpred.iterrows():
        succ_tid = str(row.get(succ_col, "")).strip()
        pred_tid = str(row.get(pred_col, "")).strip()
        pred_aid = activity_by_task_id.get(pred_tid)
        succ_aid = activity_by_task_id.get(succ_tid)
        if not pred_aid or not succ_aid:
            continue
        rec: dict[str, Any] = {
            "predecessor_activity_id": pred_aid,
            "successor_activity_id": succ_aid,
        }
        if type_col:
            rec["relationship_type"] = _clean_optional(row.get(type_col))
        if lag_col:
            raw_lag = pd.to_numeric(row.get(lag_col), errors="coerce")
            lag_days = None if pd.isna(raw_lag) else float(xp.float_series_to_days(lag_col, pd.Series([raw_lag])).iloc[0])
            rec["lag_days"] = lag_days
        out[(pred_aid, succ_aid)] = rec
    return out


def _relationship_change_evidence(
    last: XerSnapshot,
    current: XerSnapshot,
    *,
    current_path_ids: list[str],
    previous_path_ids: list[str],
) -> list[dict[str, Any]]:
    last_rels = _relationship_records_by_pair(last)
    curr_rels = _relationship_records_by_pair(current)
    current_pairs = list(zip(current_path_ids, current_path_ids[1:], strict=False))
    previous_pairs = list(zip(previous_path_ids, previous_path_ids[1:], strict=False))

    current_lookup = _task_lookup_by_activity(current)
    last_lookup = _task_lookup_by_activity(last)

    def name(snapshot_rows: dict[str, pd.Series], aid: str) -> str | None:
        row = snapshot_rows.get(aid)
        if row is None:
            return None
        name_col = _pick_col(current.task if snapshot_rows is current_lookup else last.task, ["task_name", "task_title", "activity_name"])
        return _clean_optional(row.get(name_col)) if name_col else None

    out: list[dict[str, Any]] = []
    for pair in current_pairs:
        cur = curr_rels.get(pair)
        prev = last_rels.get(pair)
        if cur and not prev:
            out.append(
                {
                    "driver_type": "relationship_added_on_current_path",
                    "predecessor_task_name": name(current_lookup, pair[0]),
                    "successor_task_name": name(current_lookup, pair[1]),
                    "relationship_type": cur.get("relationship_type"),
                    "lag_days": cur.get("lag_days"),
                    "detail": "A predecessor relationship on the current primary critical path was not present in the previous update.",
                }
            )
        elif cur and prev:
            changed_fields: dict[str, Any] = {}
            for field in ["relationship_type", "lag_days"]:
                if cur.get(field) != prev.get(field):
                    changed_fields[field] = {"previous": prev.get(field), "current": cur.get(field)}
            if changed_fields:
                out.append(
                    {
                        "driver_type": "relationship_changed_on_current_path",
                        "predecessor_task_name": name(current_lookup, pair[0]),
                        "successor_task_name": name(current_lookup, pair[1]),
                        "changed_fields": changed_fields,
                        "detail": "A relationship type or lag value changed on the current primary critical path.",
                    }
                )

    current_pair_set = set(current_pairs)
    for pair in previous_pairs:
        if pair in current_pair_set:
            continue
        prev = last_rels.get(pair)
        if prev and pair not in curr_rels:
            out.append(
                {
                    "driver_type": "relationship_removed_from_previous_path",
                    "predecessor_task_name": name(last_lookup, pair[0]),
                    "successor_task_name": name(last_lookup, pair[1]),
                    "relationship_type": prev.get("relationship_type"),
                    "lag_days": prev.get("lag_days"),
                    "detail": "A relationship that was part of the previous primary critical path is no longer present.",
                }
            )
    return out


def look_ahead_window_analysis(
    current: XerSnapshot,
    *,
    horizon_days: int | None,
    near_critical_activity_ids: set[str] | None = None,
    critical_activity_ids: set[str] | None = None,
    critical_path_activity_ids: set[str] | None = None,
) -> dict[str, Any]:
    """
    Find current-schedule activities with forecast start/finish dates inside the user-defined look-ahead window.
    """
    if current.data_date is None:
        return {"window": None, "count": 0, "groups": [], "items": [], "warning": "Current data date is not available."}
    if current.task is None or current.task.empty:
        return {"window": None, "count": 0, "groups": [], "items": [], "warning": "Current TASK is empty."}

    try:
        horizon = max(0, int(horizon_days if horizon_days is not None else 0))
    except Exception:
        horizon = 0

    start = current.data_date.normalize()
    end = start + pd.Timedelta(days=horizon)

    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    if not activity_id_col:
        return {"window": {"start": start.isoformat(), "end": end.isoformat(), "horizon_days": horizon}, "count": 0, "groups": [], "items": [], "warning": "Current TASK missing activity id column."}

    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    task_type_col = _pick_col(current.task, ["task_type", "task_type_code", "activity_type"])
    act_start_col, act_finish_col = _detect_actual_cols(current.task)

    float_col = None
    float_days: pd.Series | None = None
    try:
        float_col = xp.detect_total_float_column(current.task)
        float_days = _task_float_series_to_days(
            current.task,
            float_col,
            pd.to_numeric(current.task[float_col], errors="coerce"),
            current.calendar,
        )
    except Exception:
        float_col = None
        float_days = None

    near_critical_activity_ids = near_critical_activity_ids or set()
    critical_activity_ids = critical_activity_ids or set()
    critical_path_activity_ids = critical_path_activity_ids or set()

    items: list[dict[str, Any]] = []
    for idx, row in current.task.iterrows():
        aid = str(row.get(activity_id_col, "")).strip()
        if not aid or aid.lower() == "nan":
            continue
        if _is_mspdi(current) and task_type_col:
            task_type = str(row.get(task_type_col, "")).strip().casefold()
            if any(token in task_type for token in ["wbs", "summary", "loe", "level of effort"]):
                continue

        actual_finish = _parse_date(row.get(act_finish_col)) if act_finish_col else None
        if actual_finish is not None and actual_finish.normalize() < start:
            continue

        forecast_start, forecast_start_col = _select_forecast_start_date(current.task, row)
        forecast_finish, forecast_finish_col = _select_forecast_finish_date(current.task, row)

        dates_in_window: list[str] = []
        if forecast_start is not None and start <= forecast_start.normalize() <= end:
            dates_in_window.append("start")
        if forecast_finish is not None and start <= forecast_finish.normalize() <= end:
            dates_in_window.append("finish")
        if not dates_in_window:
            continue

        status = "not_started"
        actual_start = _parse_date(row.get(act_start_col)) if act_start_col else None
        if actual_finish is not None:
            status = "finished"
        elif actual_start is not None:
            status = "in_progress"

        tf = None
        if float_days is not None:
            val = float_days.loc[idx]
            tf = None if pd.isna(val) else float(val)

        items.append(
            {
                "activity_id": aid,
                "task_name": _clean_optional(row.get(name_col)) if name_col else None,
                "wbs_name": _clean_optional(row.get(wbs_name_col)) if wbs_name_col else None,
                "wbs_path": _clean_optional(row.get(wbs_path_col)) if wbs_path_col else None,
                "forecast_start_date": None if forecast_start is None else forecast_start.isoformat(),
                "forecast_start_col": forecast_start_col,
                "forecast_finish_date": None if forecast_finish is None else forecast_finish.isoformat(),
                "forecast_finish_col": forecast_finish_col,
                "dates_in_window": dates_in_window,
                "current_status": status,
                "total_float_days": tf,
                "near_critical_current": aid in near_critical_activity_ids,
                "critical_current": aid in critical_activity_ids,
                "on_current_critical_path": aid in critical_path_activity_ids,
            }
        )

    items.sort(
        key=lambda x: (
            not bool(x.get("on_current_critical_path")),
            not bool(x.get("critical_current")),
            not bool(x.get("near_critical_current")),
            x.get("forecast_start_date") or x.get("forecast_finish_date") or "",
            x.get("wbs_path") or "",
        )
    )
    groups = _group_items_by_wbs(items)
    risk_items = [x for x in items if x.get("critical_current") or x.get("near_critical_current") or x.get("on_current_critical_path")]
    risk_groups = _group_items_by_wbs(risk_items)

    # Driving relationships among the upcoming activities, so the look-ahead can describe how the
    # window's work sequences rather than only listing counts per area.
    name_by_id = {str(x.get("activity_id")): x.get("task_name") for x in items}
    upcoming_ids = set(name_by_id.keys())
    driving_links: list[dict[str, Any]] = []
    if upcoming_ids:
        for (pred_aid, succ_aid), rec in _relationship_records_by_pair(current).items():
            if pred_aid in upcoming_ids and succ_aid in upcoming_ids:
                driving_links.append(
                    {
                        "from_task_name": name_by_id.get(pred_aid),
                        "to_task_name": name_by_id.get(succ_aid),
                        "relationship_type": rec.get("relationship_type"),
                        "lag_days": rec.get("lag_days"),
                    }
                )

    return {
        "window": {"start": start.isoformat(), "end": end.isoformat(), "horizon_days": horizon},
        "count": int(len(items)),
        "group_count": int(len(groups)),
        "critical_or_near_critical_count": int(len(risk_items)),
        "critical_or_near_critical_group_count": int(len(risk_groups)),
        "groups": groups,
        "critical_or_near_critical_groups": risk_groups,
        "items": items,
        "driving_links": driving_links,
    }


def finish_extensions_in_progress(
    last: XerSnapshot,
    current: XerSnapshot,
) -> dict[str, Any]:
    """
    Find activities that were expected to finish during the last-current update window,
    remain in progress in the current update, and now forecast a later finish.
    """
    if last.data_date is None or current.data_date is None:
        return {"window": None, "count": 0, "groups": [], "items": []}
    if last.task is None or last.task.empty or current.task is None or current.task.empty:
        return {"window": None, "count": 0, "groups": [], "items": [], "warning": "TASK data is empty."}

    start = last.data_date.normalize()
    end = current.data_date.normalize()
    if end < start:
        start, end = end, start

    last_aid_col = _pick_col(last.task, ["task_code", "activity_id"])
    curr_aid_col = _pick_col(current.task, ["task_code", "activity_id"])
    if not last_aid_col or not curr_aid_col:
        return {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": 0, "groups": [], "items": [], "warning": "Missing activity id column."}

    _, curr_act_finish_col = _detect_actual_cols(current.task)
    last_act_start_col, last_act_finish_col = _detect_actual_cols(last.task)
    if not last_act_start_col:
        return {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": 0, "groups": [], "items": [], "warning": "Last TASK missing actual start column."}
    if not last_act_finish_col:
        return {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": 0, "groups": [], "items": [], "warning": "Last TASK missing actual finish column."}
    if not curr_act_finish_col:
        return {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": 0, "groups": [], "items": [], "warning": "Current TASK missing actual finish column."}

    current_df = current.task.copy()
    last_df = last.task.copy()
    current_df["_activity_id_key"] = current_df[curr_aid_col].astype(str).str.strip()
    last_df["_activity_id_key"] = last_df[last_aid_col].astype(str).str.strip()
    current_by_aid = current_df.drop_duplicates(subset=["_activity_id_key"]).set_index("_activity_id_key")

    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])

    items: list[dict[str, Any]] = []
    for _, last_row in last_df.iterrows():
        aid = str(last_row.get("_activity_id_key", "")).strip()
        if not aid or aid.lower() == "nan" or aid not in current_by_aid.index:
            continue

        last_actual_start = _parse_date(last_row.get(last_act_start_col)) if last_act_start_col else None
        last_actual_finish = _parse_date(last_row.get(last_act_finish_col)) if last_act_finish_col else None
        if last_actual_start is None:
            continue
        if last_actual_finish is not None:
            continue

        last_forecast_finish, last_finish_col = _select_forecast_finish_date(last.task, last_row)
        if last_forecast_finish is None:
            continue
        last_finish_day = last_forecast_finish.normalize()
        if not (start <= last_finish_day <= end):
            continue

        curr_row = current_by_aid.loc[aid]
        curr_actual_finish = _parse_date(curr_row.get(curr_act_finish_col)) if curr_act_finish_col else None
        if curr_actual_finish is not None:
            continue

        curr_forecast_finish, curr_finish_col = _select_forecast_finish_date(current.task, curr_row)
        if curr_forecast_finish is None:
            continue
        if curr_forecast_finish.normalize() <= last_forecast_finish.normalize():
            continue

        slip_days = int((curr_forecast_finish.normalize() - last_forecast_finish.normalize()).days)
        items.append(
            {
                "activity_id": aid,
                "task_name": _clean_optional(curr_row.get(name_col)) if name_col else None,
                "wbs_name": _clean_optional(curr_row.get(wbs_name_col)) if wbs_name_col else None,
                "wbs_path": _clean_optional(curr_row.get(wbs_path_col)) if wbs_path_col else None,
                "last_forecast_finish_date": last_forecast_finish.isoformat(),
                "last_finish_col": last_finish_col,
                "current_forecast_finish_date": curr_forecast_finish.isoformat(),
                "current_finish_col": curr_finish_col,
                "finish_slip_days": slip_days,
                "previous_status": "in_progress",
                "current_status": "in_progress",
            }
        )

    items.sort(key=lambda x: (-(x.get("finish_slip_days") or 0), x.get("wbs_path") or ""))
    groups = _group_items_by_wbs(items)
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "count": int(len(items)),
        "group_count": int(len(groups)),
        "groups": groups,
        "items": items,
    }


def critical_path_change_summary(
    last_path: Mapping[str, Any],
    current_path: Mapping[str, Any],
    *,
    last_snapshot: XerSnapshot | None = None,
    current_snapshot: XerSnapshot | None = None,
    new_global: Mapping[str, Any] | None = None,
    finish_extensions: Mapping[str, Any] | None = None,
    trending: Mapping[str, Any] | None = None,
    wbs_monitor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def primary(path_obj: Mapping[str, Any]) -> list[dict[str, Any]]:
        paths = path_obj.get("paths", []) or []
        if not paths:
            return []
        first = paths[0] or {}
        chain = first.get("activity_chain", []) or []
        return [x for x in chain if isinstance(x, dict)]

    last_chain = primary(last_path)
    curr_chain = primary(current_path)
    last_ids = [str(x.get("activity_id")) for x in last_chain if x.get("activity_id")]
    curr_ids = [str(x.get("activity_id")) for x in curr_chain if x.get("activity_id")]

    added_ids = [x for x in curr_ids if x not in set(last_ids)]
    removed_ids = [x for x in last_ids if x not in set(curr_ids)]

    def summarize(ids: list[str], chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {str(x.get("activity_id")): x for x in chain if x.get("activity_id")}
        out: list[dict[str, Any]] = []
        for aid in ids:
            it = by_id.get(aid, {})
            out.append(
                {
                    "activity_id": aid,
                    "task_name": it.get("task_name"),
                    "wbs_name": it.get("wbs_name"),
                    "wbs_path": it.get("wbs_path"),
                }
            )
        return out

    def _current_status_lookup(snapshot: XerSnapshot | None) -> dict[str, dict[str, Any]]:
        if snapshot is None or snapshot.task is None or snapshot.task.empty:
            return {}
        activity_col = _pick_col(snapshot.task, ["task_code", "activity_id"])
        if not activity_col:
            return {}
        name_col = _pick_col(snapshot.task, ["task_name", "task_title", "activity_name"])
        status_col = _pick_col(snapshot.task, ["status_code", "task_status"])
        actual_start_col, actual_finish_col = _detect_actual_cols(snapshot.task)
        remaining_col = _pick_col(snapshot.task, ["remain_drtn_hr_cnt", "remaining_duration"])
        out: dict[str, dict[str, Any]] = {}
        for _, row in snapshot.task.iterrows():
            aid = str(row.get(activity_col, "")).strip()
            if not aid or aid.lower() == "nan":
                continue
            actual_start = _parse_date(row.get(actual_start_col)) if actual_start_col else None
            actual_finish = _parse_date(row.get(actual_finish_col)) if actual_finish_col else None
            raw_status = _clean_optional(row.get(status_col)) if status_col else None
            status_text = str(raw_status or "").casefold()
            remaining = pd.to_numeric(row.get(remaining_col), errors="coerce") if remaining_col else None
            completed = bool(actual_finish is not None or "complete" in status_text)
            in_progress = bool(not completed and (actual_start is not None or "active" in status_text))
            current_status = "completed" if completed else ("in_progress" if in_progress else "not_started")
            out[aid] = {
                "task_name": _clean_optional(row.get(name_col)) if name_col else None,
                "current_status": current_status,
                "current_status_raw": raw_status,
                "actual_start_date": None if actual_start is None else actual_start.isoformat(),
                "actual_finish_date": None if actual_finish is None else actual_finish.isoformat(),
                "remaining_duration_raw": None if remaining_col is None or pd.isna(remaining) else float(remaining),
            }
        return out

    current_status_by_id = _current_status_lookup(current_snapshot)

    def summarize_with_current_status(ids: list[str], chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = summarize(ids, chain)
        for item in out:
            aid = str(item.get("activity_id") or "")
            status = current_status_by_id.get(aid)
            if status:
                item.update(
                    {
                        "current_status": status.get("current_status"),
                        "actual_start_date": status.get("actual_start_date"),
                        "actual_finish_date": status.get("actual_finish_date"),
                        "remaining_duration_raw": status.get("remaining_duration_raw"),
                    }
                )
        return out

    def _common_downstream_suffix(left_ids: list[str], right_ids: list[str]) -> list[str]:
        common: list[str] = []
        li = len(left_ids) - 1
        ri = len(right_ids) - 1
        while li >= 0 and ri >= 0 and left_ids[li] == right_ids[ri]:
            common.append(left_ids[li])
            li -= 1
            ri -= 1
        return list(reversed(common))

    shared_downstream_ids = _common_downstream_suffix(last_ids, curr_ids)
    previous_unique_upstream_ids = last_ids[: max(0, len(last_ids) - len(shared_downstream_ids))]
    current_unique_upstream_ids = curr_ids[: max(0, len(curr_ids) - len(shared_downstream_ids))]

    def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"completed": 0, "in_progress": 0, "not_started": 0, "unknown": 0}
        for item in items:
            status = _clean_optional(item.get("current_status")) or "unknown"
            counts[status if status in counts else "unknown"] += 1
        return counts

    previous_unique_upstream_items = summarize_with_current_status(previous_unique_upstream_ids, last_chain)
    current_unique_upstream_items = summarize(current_unique_upstream_ids, curr_chain)
    shared_downstream_items = summarize(shared_downstream_ids, curr_chain)
    path_change_interpretation = {
        "purpose": (
            "Use this compact comparison to explain the path shift. Describe unique upstream driver changes first, "
            "then acknowledge the shared downstream sequence so the previous path is not shortened incorrectly."
        ),
        "changed_portion_rule": (
            "The path change is the difference between previous_unique_upstream_sequence and current_unique_upstream_sequence. "
            "Do not describe shared_downstream_sequence as newly shifted work; it is the downstream continuation common to both traced paths."
        ),
        "previous_unique_upstream_count": len(previous_unique_upstream_items),
        "previous_unique_upstream_sequence": previous_unique_upstream_items,
        "previous_unique_upstream_current_status_counts": _status_counts(previous_unique_upstream_items),
        "current_unique_upstream_count": len(current_unique_upstream_items),
        "current_unique_upstream_sequence": current_unique_upstream_items,
        "shared_downstream_count": len(shared_downstream_items),
        "shared_downstream_sequence": shared_downstream_items,
        "shared_downstream_role": (
            "Common downstream continuation after the upstream driver portion; mention it as the path continuation, "
            "not as the portion that shifted."
            if shared_downstream_items
            else None
        ),
        "narrative_rule": (
            "Do not describe only the previous unique upstream activities as the whole previous critical path. "
            "State that they flowed into the shared downstream sequence when shared_downstream_sequence is present. "
            "Do not say all previous upstream work is complete unless the current status counts support that."
        ),
    }

    last_wbs = {str(x.get("wbs_path") or x.get("wbs_name") or "").strip() for x in last_chain}
    curr_wbs = {str(x.get("wbs_path") or x.get("wbs_name") or "").strip() for x in curr_chain}
    last_wbs.discard("")
    curr_wbs.discard("")

    def chain_summary(chain: list[dict[str, Any]]) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for it in chain:
            items.append(
                {
                    "activity_id": it.get("activity_id"),
                    "task_name": it.get("task_name"),
                    "wbs_name": it.get("wbs_name"),
                    "wbs_path": it.get("wbs_path"),
                    "float_current_days": it.get("float_current_days"),
                }
            )
        groups = _group_items_by_wbs(items)
        return {
            "length": len(items),
            "wbs_paths": sorted({str(x.get("wbs_path") or x.get("wbs_name") or "").strip() for x in items if x.get("wbs_path") or x.get("wbs_name")}),
            "groups": groups,
            "items": items,
        }

    new_global = new_global or {}
    finish_extensions = finish_extensions or {}
    trending = trending or {}
    wbs_monitor = wbs_monitor or {}

    new_ids = {str(x.get("activity_id")) for x in (new_global.get("items", []) or []) if x.get("activity_id")}
    finish_extension_by_id = {
        str(x.get("activity_id")): x for x in (finish_extensions.get("items", []) or []) if x.get("activity_id")
    }
    eroding_by_id = {str(x.get("activity_id")): x for x in (trending.get("eroding_risks", []) or []) if x.get("activity_id")}
    change_new_ids = {str(x) for x in (wbs_monitor.get("new_activity_ids", []) or [])}

    pit = wbs_monitor.get("path_impact_tracker", {}) or {}
    change_driving_ids = {str(x.get("activity_id")) for x in (pit.get("driving_delay_items", []) or []) if x.get("activity_id")}
    change_alert_ids = {
        str(x.get("to_activity_id"))
        for x in (pit.get("cross_wbs_alerts", []) or [])
        if x.get("to_activity_id")
    }

    curr_by_id = {str(x.get("activity_id")): x for x in curr_chain if x.get("activity_id")}
    drivers: list[dict[str, Any]] = []

    def add_driver(kind: str, aid: str, source: Mapping[str, Any], detail: str) -> None:
        item = curr_by_id.get(aid, {})
        drivers.append(
            {
                "driver_type": kind,
                "activity_id": aid,
                "task_name": source.get("task_name") or item.get("task_name"),
                "wbs_name": source.get("wbs_name") or item.get("wbs_name"),
                "wbs_path": source.get("wbs_path") or item.get("wbs_path"),
                "detail": detail,
                "finish_slip_days": source.get("finish_slip_days"),
                "float_loss_days": source.get("float_loss_days"),
                "float_current_days": source.get("float_current_days") or item.get("float_current_days"),
            }
        )

    for aid in added_ids:
        if aid in finish_extension_by_id:
            src = finish_extension_by_id[aid]
            add_driver(
                "finish_extension_on_current_path",
                aid,
                src,
                "An in-progress activity now forecasts later than it did in the previous update and is on the current critical path.",
            )
        if aid in eroding_by_id:
            src = eroding_by_id[aid]
            add_driver(
                "float_erosion_on_current_path",
                aid,
                src,
                "Float eroded faster than time passed and the activity is now part of the current critical path.",
            )
        if aid in new_ids:
            add_driver(
                "new_activity_on_current_path",
                aid,
                curr_by_id.get(aid, {}),
                "A newly added activity is now part of the current critical path.",
            )
        if aid in change_new_ids or aid in change_driving_ids or aid in change_alert_ids:
            add_driver(
                "change_related_current_path_driver",
                aid,
                curr_by_id.get(aid, {}),
                "Change/delay logic indicates this current-path activity is linked to change-related work or downstream cross-WBS impact.",
            )

    relationship_evidence: list[dict[str, Any]] = []
    task_field_evidence: list[dict[str, Any]] = []
    if last_snapshot is not None and current_snapshot is not None:
        relationship_evidence = _relationship_change_evidence(
            last_snapshot,
            current_snapshot,
            current_path_ids=curr_ids,
            previous_path_ids=last_ids,
        )

        current_path_candidate_ids = list(dict.fromkeys(added_ids + [aid for aid in curr_ids if aid in set(last_ids)]))
        for aid in current_path_candidate_ids:
            changes = _task_field_changes(last_snapshot, current_snapshot, aid)
            relevant_changes = [
                ch
                for ch in changes
                if ch.get("field")
                in {
                    "forecast_start",
                    "forecast_finish",
                    "constraint_type",
                    "constraint_date",
                    "secondary_constraint_type",
                    "secondary_constraint_date",
                    "calendar",
                    "remaining_duration",
                    "original_duration",
                    "status",
                }
            ]
            if not relevant_changes:
                continue
            item = curr_by_id.get(aid, {})
            task_field_evidence.append(
                {
                    "driver_type": "task_attribute_changed_on_current_path",
                    "activity_id": aid,
                    "task_name": item.get("task_name"),
                    "wbs_name": item.get("wbs_name"),
                    "wbs_path": item.get("wbs_path"),
                    "changed_fields": relevant_changes,
                    "detail": "One or more schedule attributes changed on an activity in the current primary critical path.",
                }
            )

    # De-duplicate driver records while preserving evidence order.
    seen_driver_keys: set[tuple[str, str]] = set()
    deduped_drivers: list[dict[str, Any]] = []
    for d in drivers:
        key = (str(d.get("driver_type")), str(d.get("activity_id")))
        if key in seen_driver_keys:
            continue
        seen_driver_keys.add(key)
        deduped_drivers.append(d)

    possible_shift_causes = deduped_drivers + relationship_evidence + task_field_evidence
    changed = bool(added_ids or removed_ids)
    return {
        "changed": changed,
        "comparison_basis": "primary critical path to target, last update vs current update",
        "last_path_length": len(last_ids),
        "current_path_length": len(curr_ids),
        "path_change_interpretation": path_change_interpretation,
        "previous_primary_path": chain_summary(last_chain),
        "current_primary_path": chain_summary(curr_chain),
        "added_to_current_path_count": len(added_ids),
        "removed_from_previous_path_count": len(removed_ids),
        "added_to_current_path": summarize(added_ids, curr_chain),
        "removed_from_previous_path": summarize(removed_ids, last_chain),
        "current_wbs_added": sorted(curr_wbs - last_wbs),
        "previous_wbs_removed": sorted(last_wbs - curr_wbs),
        "likely_shift_drivers": deduped_drivers,
        "relationship_change_evidence": relationship_evidence,
        "task_field_change_evidence": task_field_evidence,
        "possible_shift_causes": possible_shift_causes,
        "cause_assessment": (
            "Supported shift drivers are listed in possible_shift_causes."
            if possible_shift_causes
            else (
                "The current and previous primary critical paths differ, but the specific cause is not determinable from the provided schedule fields."
                if changed
                else "No critical path shift detected."
            )
        ),
        "warning": (last_path.get("warning") or current_path.get("warning")),
    }


def upstream_new_activity_links_to_critical_path(
    current: XerSnapshot,
    new_global: Mapping[str, Any],
    critical_activity_ids: set[str],
    *,
    max_depth: int = 8,
    max_chains: int = 20,
) -> dict[str, Any]:
    """
    Trace newly added activities forward through existing logic until they reach a current critical activity.

    This is context evidence only. A new activity in these chains is not critical unless its own P6 float says so.
    """
    if current.task is None or current.task.empty:
        return {"count": 0, "chains": [], "warning": "Current TASK is empty."}
    if current.taskpred is None or current.taskpred.empty:
        return {"count": 0, "chains": [], "warning": "Current TASKPRED is empty."}

    root_item_by_id = {
        str(item.get("activity_id")).strip(): item
        for item in (new_global.get("items", []) or [])
        if isinstance(item, Mapping) and _clean_optional(item.get("activity_id"))
    }
    root_ids = list(root_item_by_id.keys())
    root_ids = list(dict.fromkeys(root_ids))
    if not root_ids or not critical_activity_ids:
        return {"count": 0, "chains": []}

    priority_terms = ["leak", "delay", "rfi", "change", "constraint", "conflict"]

    def priority_text(value: Mapping[str, Any] | None) -> str:
        if not isinstance(value, Mapping):
            return ""
        return " ".join(
            _clean_optional(value.get(key)) or ""
            for key in ["task_name", "wbs_name", "wbs_path"]
        ).casefold()

    root_ids = sorted(
        root_ids,
        key=lambda aid: (
            0 if any(term in priority_text(root_item_by_id.get(aid)) for term in priority_terms) else 1,
            priority_text(root_item_by_id.get(aid)),
            aid,
        ),
    )

    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    status_col = _pick_col(current.task, ["status_code", "task_status"])
    act_start_col, act_finish_col = _detect_actual_cols(current.task)
    if not activity_id_col:
        return {"count": 0, "chains": [], "warning": "Current TASK missing activity id column."}

    relation_by_pair = _relationship_records_by_pair(current)
    succ_by_pred: dict[str, set[str]] = defaultdict(set)
    for pred_aid, succ_aid in relation_by_pair:
        succ_by_pred[str(pred_aid)].add(str(succ_aid))

    task_df = current.task.copy()
    task_df["_activity_id_key"] = task_df[activity_id_col].astype(str).str.strip()
    task_by_aid = task_df.drop_duplicates(subset=["_activity_id_key"]).set_index("_activity_id_key")

    float_by_activity: dict[str, float] = {}
    try:
        float_col = xp.detect_total_float_column(current.task)
        numeric = pd.to_numeric(current.task[float_col], errors="coerce")
        numeric_days = _task_float_series_to_days(current.task, float_col, numeric, current.calendar)
        for aid, value in zip(current.task[activity_id_col].astype(str).str.strip(), numeric_days, strict=False):
            if not pd.isna(value):
                float_by_activity[str(aid)] = float(value)
    except Exception:
        float_by_activity = {}

    def status_for(row: pd.Series | None) -> str:
        if row is None:
            return "unknown"
        status = str(row.get(status_col, "")).casefold() if status_col else ""
        actual_start = _parse_date(row.get(act_start_col)) if act_start_col else None
        actual_finish = _parse_date(row.get(act_finish_col)) if act_finish_col else None
        if actual_finish is not None or "complete" in status:
            return "completed"
        if actual_start is not None:
            return "in_progress"
        return "not_started"

    def item_fact(aid: str) -> dict[str, Any]:
        row = task_by_aid.loc[aid] if aid in task_by_aid.index else None
        return {
            "activity_id": aid,
            "task_name": None if row is None or not name_col else _clean_optional(row.get(name_col)),
            "wbs_name": None if row is None or not wbs_name_col else _clean_optional(row.get(wbs_name_col)),
            "wbs_path": None if row is None or not wbs_path_col else _clean_optional(row.get(wbs_path_col)),
            "is_critical": aid in critical_activity_ids,
            "is_near_critical": False,
            "float_current_days": float_by_activity.get(aid),
            "completed": status_for(row) == "completed",
            "current_status": status_for(row),
        }

    def link_fact(pred_aid: str, succ_aid: str) -> dict[str, Any]:
        rel = relation_by_pair.get((pred_aid, succ_aid), {}) or {}
        pred = item_fact(pred_aid)
        succ = item_fact(succ_aid)
        return {
            "from_activity_id": pred_aid,
            "to_activity_id": succ_aid,
            "from_task_name": pred.get("task_name"),
            "to_task_name": succ.get("task_name"),
            "relationship_type": rel.get("relationship_type"),
            "lag_days": rel.get("lag_days"),
        }

    chains: list[dict[str, Any]] = []
    truncated = False
    for root_aid in root_ids:
        if len(chains) >= max_chains:
            truncated = True
            break

        q: deque[list[str]] = deque([[root_aid]])
        seen_at_depth: set[tuple[str, int]] = {(root_aid, 1)}
        found_for_root = False
        while q and not found_for_root:
            path = q.popleft()
            cur = path[-1]
            if len(path) >= max_depth:
                truncated = True
                continue

            for succ_aid in sorted(succ_by_pred.get(cur, set())):
                if succ_aid in path:
                    continue
                next_path = [*path, succ_aid]
                if succ_aid in critical_activity_ids and succ_aid != root_aid:
                    chain_items = [item_fact(aid) for aid in next_path]
                    chains.append(
                        {
                            "root_activity_id": root_aid,
                            "root_task_name": chain_items[0].get("task_name") if chain_items else None,
                            "root_float_current_days": chain_items[0].get("float_current_days") if chain_items else None,
                            "root_current_status": chain_items[0].get("current_status") if chain_items else None,
                            "reached_critical_activity_id": succ_aid,
                            "reached_critical_task_name": chain_items[-1].get("task_name") if chain_items else None,
                            "reached_critical_float_current_days": chain_items[-1].get("float_current_days") if chain_items else None,
                            "chain_length": len(next_path),
                            "activity_chain": chain_items,
                            "logic_links": [link_fact(next_path[i], next_path[i + 1]) for i in range(len(next_path) - 1)],
                            "interpretation": (
                                "A newly added activity has successor logic reaching a current critical activity. "
                                "This is upstream context; classify each activity by its own float_current_days."
                            ),
                        }
                    )
                    found_for_root = True
                    break

                key = (succ_aid, len(next_path))
                if key in seen_at_depth:
                    continue
                seen_at_depth.add(key)
                q.append(next_path)

            if len(chains) >= max_chains:
                truncated = True
                break

    return {
        "count": int(len(chains)),
        "truncated": truncated,
        "max_depth": int(max_depth),
        "chains": chains,
    }


def compare_three_way(
    baseline: XerSnapshot,
    last: XerSnapshot,
    current: XerSnapshot,
    *,
    variance_threshold: int,
    target_activity_id: str,
    change_term: str = "change",
    look_ahead_horizon_days: int | None = None,
) -> dict[str, Any]:
    period = data_date_sync(baseline, last, current)

    ms_base = milestone_finish(baseline.task, target_activity_id)
    ms_last = milestone_finish(last.task, target_activity_id)
    ms_curr = milestone_finish(current.task, target_activity_id)

    total_var = _variance_days(ms_curr["finish_date"], ms_base["finish_date"])
    period_var = _variance_days(ms_curr["finish_date"], ms_last["finish_date"])

    critical_path_last = critical_path_to_target(
        last,
        target_activity_id,
        near_critical_buffer_days=variance_threshold,
    )
    critical_path = critical_path_to_target(
        current,
        target_activity_id,
        near_critical_buffer_days=variance_threshold,
    )
    critical_float_ids = {str(x) for x in (critical_path.get("critical_activity_ids", []) or []) if str(x).strip()}
    critical_trace_ids = {str(x) for x in (critical_path.get("critical_trace_activity_ids", []) or []) if str(x).strip()}
    if not critical_trace_ids:
        critical_trace_ids = {
            str(x.get("activity_id"))
            for p in critical_path.get("paths", []) or []
            for x in (p.get("activity_chain", []) or [])
            if x.get("activity_id")
        }
    critical_path_ids = critical_trace_ids or critical_float_ids
    trending_all = near_critical_trending(
        last,
        current,
        variance_threshold,
        target_activity_id=target_activity_id,
        critical_network=critical_path,
    )
    trending = near_critical_trending(
        last,
        current,
        variance_threshold,
        target_activity_id=target_activity_id,
        critical_network=critical_path,
        exclude_activity_ids=critical_path_ids,
    )
    wbs = wbs_monitor_change_and_delay(last, current, term=change_term)
    new_global = new_activities_all_wbs(last, current)
    accomplished = work_accomplished(last, current)
    msp_progress = microsoft_project_progress_changes(last, current) if _is_mspdi(current) else None
    upstream_new_critical_links = upstream_new_activity_links_to_critical_path(
        current,
        new_global,
        critical_path_ids,
    )

    near_ids = {str(x) for x in trending.get("activity_ids", []) or []}
    near_ids.update(str(x.get("activity_id")) for x in trending.get("near_critical", []) or [] if x.get("activity_id"))
    critical_ids = set(critical_float_ids)
    finish_extensions = finish_extensions_in_progress(last, current)
    critical_path_change = critical_path_change_summary(
        critical_path_last,
        critical_path,
        last_snapshot=last,
        current_snapshot=current,
        new_global=new_global,
        finish_extensions=finish_extensions,
        trending=trending_all,
        wbs_monitor=wbs,
    )
    look_ahead = look_ahead_window_analysis(
        current,
        horizon_days=look_ahead_horizon_days,
        near_critical_activity_ids=near_ids,
        critical_activity_ids=critical_ids,
        critical_path_activity_ids=critical_path_ids,
    )

    result = {
        "settings": {
            "variance_threshold": int(variance_threshold),
            "look_ahead_horizon_days": (None if look_ahead_horizon_days is None else int(look_ahead_horizon_days)),
            "change_term": str(change_term),
        },
        "update_period": period,
        "milestone": {
            "target_activity_id": str(target_activity_id),
            "baseline": ms_base,
            "last": ms_last,
            "current": ms_curr,
            "total_variance_days": total_var,
            "period_variance_days": period_var,
        },
        "near_critical_trending": trending,
        "wbs_monitor": wbs,
        "new_activities_global": new_global,
        "work_accomplished": accomplished,
        "finish_extensions_in_progress": finish_extensions,
        "look_ahead_window_analysis": look_ahead,
        "critical_path_change": critical_path_change,
        "upstream_new_activity_links_to_critical_path": upstream_new_critical_links,
        "previous_critical_path_to_target": critical_path_last,
        "critical_path_to_target": critical_path,
    }
    if _is_mspdi(current):
        result["source_format"] = "mspdi_xml"
        result["microsoft_project_progress_changes"] = msp_progress or {}
    return result


def get_ai_ready_digest(compare_result: Mapping[str, Any]) -> dict[str, Any]:
    """
    Produce a narrative-friendly digest:
      - removes technical database IDs (e.g., task_id)
      - removes activity IDs from the AI-facing payload where possible
      - keeps task names, WBS groups, and the key variances/trends
    """
    update_period = compare_result.get("update_period", {}) or {}
    milestone = compare_result.get("milestone", {}) or {}
    trending = compare_result.get("near_critical_trending", {}) or {}
    wbs = compare_result.get("wbs_monitor", {}) or {}
    new_global = compare_result.get("new_activities_global", {}) or {}
    accomplished = compare_result.get("work_accomplished", {}) or {}
    finish_extensions = compare_result.get("finish_extensions_in_progress", {}) or {}
    look_ahead = compare_result.get("look_ahead_window_analysis", {}) or {}
    critical_path_change = compare_result.get("critical_path_change", {}) or {}
    critical_path = compare_result.get("critical_path_to_target", {}) or {}
    upstream_new_critical_links = compare_result.get("upstream_new_activity_links_to_critical_path", {}) or {}
    settings = compare_result.get("settings", {}) or {}
    is_mspdi = compare_result.get("source_format") == "mspdi_xml"
    msp_progress = compare_result.get("microsoft_project_progress_changes", {}) or {}

    def _without_activity_ids(value: Any) -> Any:
        id_keys = {"activity_id", "target_activity_id", "root_activity_id", "from_activity_id", "to_activity_id", "path_activity_ids"}
        if isinstance(value, dict):
            return {k: _without_activity_ids(v) for k, v in value.items() if k not in id_keys}
        if isinstance(value, list):
            return [_without_activity_ids(v) for v in value]
        return value

    def ms_part(label: str) -> dict[str, Any]:
        part = milestone.get(label, {}) or {}
        return {
            "task_name": part.get("task_name"),
            "finish_date": part.get("finish_date"),
        }

    eroding = []
    for r in (trending.get("eroding_risks", []) or []):
        eroding.append(
            {
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
                "wbs_path": r.get("wbs_path"),
                "float_last_days": r.get("float_last_days"),
                "float_current_days": r.get("float_current_days"),
                "float_change_days": r.get("float_change_days"),
                "float_loss_days": r.get("float_loss_days"),
                "days_passed": r.get("days_passed"),
                "eroding_risk": r.get("eroding_risk"),
            }
        )

    new_change = []
    for r in (wbs.get("new_activities", []) or []):
        new_change.append(
            {
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
            }
        )

    cps = wbs.get("critical_path_successors", {}) or {}
    cps_items = []
    for r in (cps.get("items", []) or []):
        cps_items.append(
            {
                "summary": r.get("summary"),
                "cross_wbs_top": r.get("cross_wbs_top"),
                "has_critical_path_driver": r.get("has_critical_path_driver"),
            }
        )

    pit = wbs.get("path_impact_tracker", {}) or {}
    cross_wbs_alerts = []
    for r in (pit.get("cross_wbs_alerts", []) or []):
        cross_wbs_alerts.append(
            {
                "to_task_name": r.get("to_task_name"),
                "to_wbs_top": r.get("to_wbs_top"),
            }
        )

    def _group_near_critical(src: Mapping[str, Any]) -> dict[str, Any]:
        items = src.get("near_critical", []) or []
        if not items:
            return {
                "count": 0,
                "group_count": 0,
                "excluded_activity_count": src.get("excluded_activity_count"),
                "least_float_current_days": src.get("least_float_current_days"),
                "cutoff_current_days": src.get("cutoff_current_days"),
                "groups": [],
            }

        _clean = _clean_optional

        grouped: dict[str, dict[str, Any]] = {}
        for r in items:
            path = _clean(r.get("wbs_path")) or _clean(r.get("wbs_name"))
            parts = [p.strip() for p in str(path).split(" / ") if p.strip()] if path else []
            leaf_name = parts[-1] if parts else _clean(r.get("wbs_name"))
            key = str(path or leaf_name or "Unknown").strip() or "Unknown"

            if key not in grouped:
                grouped[key] = {
                    "leaf_wbs_name": leaf_name,
                    "leaf_wbs_path": path,
                    "wbs_hierarchy_root_to_leaf": parts,
                    "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                    "items": [],
                }

            grouped[key]["items"].append(
                {
                    "task_name": r.get("task_name"),
                    "float_last_days": r.get("float_last_days"),
                    "float_current_days": r.get("float_current_days"),
                    "float_change_days": r.get("float_change_days"),
                    "float_loss_days": r.get("float_loss_days"),
                    "days_passed": r.get("days_passed"),
                    "eroding_risk": r.get("eroding_risk"),
                }
            )

        compact_groups: list[dict[str, Any]] = []
        for group in grouped.values():
            group_items = group.get("items") or []
            current_floats = [
                float(item["float_current_days"])
                for item in group_items
                if item.get("float_current_days") is not None
            ]
            float_losses = [
                float(item["float_loss_days"])
                for item in group_items
                if item.get("float_loss_days") is not None
            ]
            representative_items = sorted(
                group_items,
                key=lambda item: (
                    item.get("float_current_days") is None,
                    item.get("float_current_days") if item.get("float_current_days") is not None else float("inf"),
                    -(item.get("float_loss_days") or 0),
                    item.get("task_name") or "",
                ),
            )[:8]
            compact_groups.append(
                {
                    "leaf_wbs_name": group.get("leaf_wbs_name"),
                    "leaf_wbs_path": group.get("leaf_wbs_path"),
                    "wbs_hierarchy_root_to_leaf": group.get("wbs_hierarchy_root_to_leaf"),
                    "wbs_hierarchy_leaf_to_root": group.get("wbs_hierarchy_leaf_to_root"),
                    "near_critical_count": int(len(group_items)),
                    "eroding_risk_count": int(sum(1 for item in group_items if item.get("eroding_risk"))),
                    "min_float_current_days": min(current_floats) if current_floats else None,
                    "max_float_current_days": max(current_floats) if current_floats else None,
                    "max_float_loss_days": max(float_losses) if float_losses else None,
                    "representative_items": representative_items,
                }
            )

        groups = sorted(
            compact_groups,
            key=lambda g: (
                -(g.get("near_critical_count") or 0),
                g.get("min_float_current_days") is None,
                g.get("min_float_current_days") if g.get("min_float_current_days") is not None else float("inf"),
                g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or "",
            ),
        )
        return {
            "count": int(len(items)),
            "group_count": int(len(groups)),
            "excluded_activity_count": src.get("excluded_activity_count"),
            "least_float_current_days": src.get("least_float_current_days"),
            "cutoff_current_days": src.get("cutoff_current_days"),
            "groups": groups,
            # Predecessor links among near-critical activities; use to describe how near-critical work sequences.
            "driving_links": (src.get("driving_links") or [])[:40],
        }

    new_global_items = []
    for r in (new_global.get("items", []) or []):
        new_global_items.append(
            {
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
                "wbs_path": r.get("wbs_path"),
                "wbs_hierarchy_leaf_to_root": r.get("wbs_hierarchy_leaf_to_root"),
                "wbs_hierarchy_root_to_leaf": r.get("wbs_hierarchy_root_to_leaf"),
            }
        )

    new_global_groups = []
    for g in (new_global.get("groups", []) or []):
        new_global_groups.append(
            {
                "leaf_wbs_name": g.get("leaf_wbs_name"),
                "leaf_wbs_path": g.get("leaf_wbs_path"),
                "wbs_hierarchy_leaf_to_root": g.get("wbs_hierarchy_leaf_to_root"),
                "wbs_hierarchy_root_to_leaf": g.get("wbs_hierarchy_root_to_leaf"),
                "items": g.get("items"),
            }
        )

    def _clean_digest_value(val: Any) -> str | None:
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none", "nat"}:
            return None
        return s

    MAX_AI_GROUPS = 25
    MAX_AI_GROUP_ITEMS = 5
    MAX_AI_TOP_LEVEL_ITEMS = 80
    MAX_AI_BRANCH_SUMMARIES = 12
    MAX_AI_ALTERNATE_PATH_EXAMPLES = 8
    MAX_AI_LOGIC_LINKS = 60
    MAX_AI_CRITICAL_ACTIVITIES = 160
    MAX_AI_PRIMARY_PATH_ACTIVITIES = 120
    MAX_AI_SHIFT_EVIDENCE_ITEMS = 25

    def _limit_list(items: Any, max_items: int) -> tuple[list[Any], int, int]:
        if not isinstance(items, list):
            return [], 0, 0
        shown = items[: max(0, max_items)]
        total = len(items)
        return shown, total, max(0, total - len(shown))

    def _cap_group_details(group: Any, *, max_items: int = MAX_AI_GROUP_ITEMS) -> dict[str, Any]:
        if not isinstance(group, Mapping):
            return {}
        out = dict(group)
        for key in ["items", "representative_items"]:
            value = out.get(key)
            if not isinstance(value, list):
                continue
            shown, total, omitted = _limit_list(value, max_items)
            out[key] = shown
            out[f"{key}_shown_count"] = len(shown)
            out[f"{key}_omitted_count"] = omitted
            if key == "items":
                out.setdefault("item_count", total)
        return out

    def _cap_grouped_summary(
        summary: Mapping[str, Any],
        *,
        max_groups: int = MAX_AI_GROUPS,
        max_group_items: int = MAX_AI_GROUP_ITEMS,
        max_items: int = MAX_AI_TOP_LEVEL_ITEMS,
        max_driving_links: int = MAX_AI_LOGIC_LINKS,
    ) -> dict[str, Any]:
        out = dict(summary or {})
        groups = out.get("groups")
        if isinstance(groups, list):
            shown_groups, total_groups, omitted_groups = _limit_list(groups, max_groups)
            out["groups"] = [
                group
                for group in (_cap_group_details(g, max_items=max_group_items) for g in shown_groups)
                if group
            ]
            out.setdefault("group_count", total_groups)
            out["groups_shown_count"] = len(out["groups"])
            out["groups_omitted_count"] = omitted_groups

        items = out.get("items")
        if isinstance(items, list):
            shown_items, total_items, omitted_items = _limit_list(items, max_items)
            out["items"] = shown_items
            out.setdefault("count", total_items)
            out["items_shown_count"] = len(shown_items)
            out["items_omitted_count"] = omitted_items

        links = out.get("driving_links")
        if isinstance(links, list):
            shown_links, total_links, omitted_links = _limit_list(links, max_driving_links)
            out["driving_links"] = shown_links
            out["driving_links_count"] = total_links
            out["driving_links_shown_count"] = len(shown_links)
            out["driving_links_omitted_count"] = omitted_links
        return out

    def _cap_named_lists(
        summary: Mapping[str, Any],
        keys: list[str],
        *,
        max_items: int = MAX_AI_SHIFT_EVIDENCE_ITEMS,
    ) -> dict[str, Any]:
        out = dict(summary or {})
        for key in keys:
            value = out.get(key)
            if not isinstance(value, list):
                continue
            shown, total, omitted = _limit_list(value, max_items)
            out[key] = shown
            out[f"{key}_count"] = total
            out[f"{key}_shown_count"] = len(shown)
            out[f"{key}_omitted_count"] = omitted
        return out

    def _wbs_parts(path: Any, fallback_name: Any = None) -> tuple[str | None, str | None, list[str]]:
        clean_path = _clean_digest_value(path)
        clean_name = _clean_digest_value(fallback_name)
        parts = [p.strip() for p in str(clean_path).split(" / ") if p.strip()] if clean_path else []
        leaf_name = parts[-1] if parts else clean_name
        return clean_path or clean_name, leaf_name, parts

    def _summarize_actualized_work(src: Mapping[str, Any]) -> dict[str, Any]:
        activities = src.get("activities", []) or []
        grouped: dict[str, dict[str, Any]] = {}
        cleaned_items: list[dict[str, Any]] = []
        started_count = 0
        finished_count = 0
        started_and_finished_count = 0
        window = src.get("window") or {}
        window_start = _parse_date(window.get("start")) if isinstance(window, Mapping) else None
        window_end = _parse_date(window.get("end")) if isinstance(window, Mapping) else None

        def _date_in_actualized_window(value: Any) -> bool:
            ts = _parse_date(value)
            if ts is None:
                return False
            if window_start is None or window_end is None:
                return True
            return bool(window_start.normalize() <= ts.normalize() <= window_end.normalize())

        for r in activities:
            actual_start = _clean_digest_value(r.get("actual_start_date"))
            actual_finish = _clean_digest_value(r.get("actual_finish_date"))
            started = _date_in_actualized_window(actual_start)
            finished = _date_in_actualized_window(actual_finish)
            if started:
                started_count += 1
            if finished:
                finished_count += 1
            if started and finished:
                started_and_finished_count += 1

            wbs_path, leaf_name, parts = _wbs_parts(r.get("wbs_path"), r.get("wbs_name"))
            key = wbs_path or leaf_name or "Unknown"
            if key not in grouped:
                grouped[key] = {
                    "leaf_wbs_name": leaf_name,
                    "leaf_wbs_path": wbs_path,
                    "wbs_hierarchy_root_to_leaf": parts,
                    "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                    "started_count": 0,
                    "finished_count": 0,
                    "started_and_finished_count": 0,
                    "item_count": 0,
                    "items": [],
                }

            item = {
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
                "wbs_path": wbs_path,
                "actual_start_date": actual_start,
                "actual_finish_date": actual_finish,
                "progress_events": [
                    event
                    for event, happened in [
                        ("actual_start", started),
                        ("actual_finish", finished),
                    ]
                    if happened
                ],
            }
            cleaned_items.append(item)
            grouped[key]["items"].append(item)
            grouped[key]["item_count"] += 1
            if started:
                grouped[key]["started_count"] += 1
            if finished:
                grouped[key]["finished_count"] += 1
            if started and finished:
                grouped[key]["started_and_finished_count"] += 1

        groups = sorted(
            grouped.values(),
            key=lambda g: (-(g.get("item_count") or 0), g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or ""),
        )
        return {
            "window": src.get("window"),
            "count": int(len(cleaned_items)),
            "started_count": int(started_count),
            "finished_count": int(finished_count),
            "started_and_finished_count": int(started_and_finished_count),
            "group_count": int(len(groups)),
            "groups": groups,
            "items": cleaned_items,
        }

    def _summarize_finish_extensions(src: Mapping[str, Any]) -> dict[str, Any]:
        items = src.get("items", []) or []
        grouped: dict[str, dict[str, Any]] = {}
        cleaned_items: list[dict[str, Any]] = []

        for r in items:
            slip_days = r.get("finish_slip_days")
            try:
                slip_days_int = int(slip_days)
            except Exception:
                slip_days_int = None

            wbs_path, leaf_name, parts = _wbs_parts(r.get("wbs_path"), r.get("wbs_name"))
            key = wbs_path or leaf_name or "Unknown"
            if key not in grouped:
                grouped[key] = {
                    "leaf_wbs_name": leaf_name,
                    "leaf_wbs_path": wbs_path,
                    "wbs_hierarchy_root_to_leaf": parts,
                    "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                    "item_count": 0,
                    "max_finish_slip_days": None,
                    "items": [],
                }

            item = {
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
                "wbs_path": wbs_path,
                "last_forecast_finish_date": r.get("last_forecast_finish_date"),
                "current_forecast_finish_date": r.get("current_forecast_finish_date"),
                "finish_slip_days": slip_days_int,
                "previous_status": r.get("previous_status"),
                "current_status": r.get("current_status"),
            }
            cleaned_items.append(item)
            grouped[key]["items"].append(item)
            grouped[key]["item_count"] += 1
            if slip_days_int is not None:
                current_max = grouped[key]["max_finish_slip_days"]
                grouped[key]["max_finish_slip_days"] = slip_days_int if current_max is None else max(current_max, slip_days_int)

        groups = sorted(
            grouped.values(),
            key=lambda g: (
                -(g.get("max_finish_slip_days") or 0),
                -(g.get("item_count") or 0),
                g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or "",
            ),
        )
        cleaned_items = sorted(
            cleaned_items,
            key=lambda x: (-(x.get("finish_slip_days") or 0), x.get("wbs_path") or x.get("wbs_name") or ""),
        )
        return {
            "window": src.get("window"),
            "count": int(len(cleaned_items)),
            "group_count": int(len(groups)),
            "groups": groups,
            "items": cleaned_items,
        }

    def _compact_wbs_context(group: Mapping[str, Any], *, max_parts: int = 4) -> str | None:
        parts = group.get("wbs_hierarchy_root_to_leaf") or []
        if not isinstance(parts, list) or not parts:
            return _clean_digest_value(group.get("leaf_wbs_path")) or _clean_digest_value(group.get("leaf_wbs_name"))
        # Drop the project root so the context starts at a useful schedule area.
        useful = [_clean_digest_value(p) for p in parts[1:] if _clean_digest_value(p)]
        if not useful:
            useful = [_clean_digest_value(p) for p in parts if _clean_digest_value(p)]
        return " / ".join(useful[-max_parts:]) if useful else None

    def _item_names(items: Any, *, limit: int = 5) -> list[str]:
        names: list[str] = []
        if not isinstance(items, list):
            return names
        for item in items:
            if not isinstance(item, Mapping):
                continue
            name = _clean_digest_value(item.get("task_name"))
            if not name or name in names:
                continue
            names.append(name)
            if len(names) >= limit:
                break
        return names

    def _constraint_driver_fact(driver: Any) -> dict[str, Any]:
        if not isinstance(driver, Mapping):
            return {}
        return {
            "activity_task_name": driver.get("activity_task_name"),
            "constraint_source": driver.get("constraint_source"),
            "constraint_type": driver.get("constraint_type"),
            "predecessor_task_name": driver.get("predecessor_task_name"),
            "predecessor_completed": driver.get("predecessor_completed"),
            "assessment": driver.get("assessment"),
        }

    def _activity_current_status(item: Mapping[str, Any]) -> str:
        provided_status = _clean_digest_value(item.get("current_status"))
        if provided_status in {"completed", "in_progress", "not_started", "unknown"}:
            return provided_status
        if bool(item.get("completed")):
            return "completed"
        if _clean_digest_value(item.get("actual_finish_date")):
            return "completed"
        if _clean_digest_value(item.get("actual_start_date")):
            return "in_progress"
        return "not_started"

    def _activity_float_fact(item: Any) -> dict[str, Any]:
        if not isinstance(item, Mapping):
            return {}
        fact = {
            "task_name": item.get("task_name"),
            "wbs_name": item.get("wbs_name"),
            "wbs_path": item.get("wbs_path"),
            "is_critical": bool(item.get("is_critical")),
            "is_near_critical": bool(item.get("is_near_critical")),
            "on_critical_trace": bool(item.get("on_critical_trace")),
            "float_current_days": item.get("float_current_days"),
            "completed": bool(item.get("completed")),
            "current_status": _activity_current_status(item),
            "calendar_name": item.get("calendar_name"),
            "float_classification_basis": (
                "precomputed_by_python_from_microsoft_project_totalslack_and_minutes_per_day"
                if is_mspdi
                else "precomputed_by_python_from_p6_total_float_and_activity_calendar"
            ),
        }
        constraint_driver = _constraint_driver_fact(item.get("constraint_driver"))
        if constraint_driver:
            fact["constraint_driver"] = constraint_driver
        return fact

    def _logic_link_fact(link: Any) -> dict[str, Any]:
        if not isinstance(link, Mapping):
            return {}
        out = {
            "from_task_name": link.get("from_task_name"),
            "to_task_name": link.get("to_task_name"),
            "relationship_type": link.get("relationship_type"),
            "lag_days": link.get("lag_days"),
        }
        bridge = link.get("trace_bridge")
        if isinstance(bridge, Mapping):
            out["trace_bridge"] = {
                "bridge_type": bridge.get("bridge_type"),
                "reason": bridge.get("reason"),
                "predecessor_float_days": bridge.get("predecessor_float_days"),
                "successor_branch_float_days": bridge.get("successor_branch_float_days"),
                "predecessor_calendar_name": bridge.get("predecessor_calendar_name"),
                "predecessor_calendar_hours_per_day": bridge.get("predecessor_calendar_hours_per_day"),
                "successor_calendar_name": bridge.get("successor_calendar_name"),
                "successor_calendar_hours_per_day": bridge.get("successor_calendar_hours_per_day"),
            }
        return out

    def _trace_bridge_fact(bridge: Any) -> dict[str, Any]:
        if not isinstance(bridge, Mapping):
            return {}
        return {
            "from_task_name": bridge.get("from_task_name"),
            "to_task_name": bridge.get("to_task_name"),
            "bridge_type": bridge.get("bridge_type"),
            "reason": bridge.get("reason"),
            "predecessor_float_days": bridge.get("predecessor_float_days"),
            "successor_branch_float_days": bridge.get("successor_branch_float_days"),
            "predecessor_calendar_name": bridge.get("predecessor_calendar_name"),
            "predecessor_calendar_hours_per_day": bridge.get("predecessor_calendar_hours_per_day"),
            "successor_calendar_name": bridge.get("successor_calendar_name"),
            "successor_calendar_hours_per_day": bridge.get("successor_calendar_hours_per_day"),
        }

    def _path_fact(path: Any) -> dict[str, Any] | None:
        if not isinstance(path, Mapping):
            return None
        chain = [_activity_float_fact(item) for item in (path.get("activity_chain") or [])]
        chain = [item for item in chain if item]
        constraint_drivers = [_constraint_driver_fact(item) for item in (path.get("constraint_drivers") or [])]
        constraint_drivers = [item for item in constraint_drivers if item]
        logic_links = [_logic_link_fact(link) for link in (path.get("logic_links") or [])]
        logic_links = [item for item in logic_links if item]
        return {
            "branch_type": path.get("branch_type"),
            "length": path.get("length"),
            "branch_min_float_days": path.get("branch_min_float_days"),
            "branch_max_float_days": path.get("branch_max_float_days"),
            "activity_chain": chain,
            "logic_links": logic_links,
            "constraint_driver_count": len(constraint_drivers),
            "constraint_drivers": constraint_drivers,
            "date_math_rule": "Do not infer float or driving status from forecast/actual date gaps.",
        }

    def _activity_sequence_sort_key(item: Mapping[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
        status = _activity_current_status(item)
        actual_start = _parse_date(item.get("actual_start_date"))
        forecast_start = _parse_date(item.get("forecast_start_date"))
        start = (
            actual_start
            if status == "in_progress" and actual_start is not None
            else (
                forecast_start
                or actual_start
                or _parse_date(item.get("forecast_finish_date"))
                or _parse_date(item.get("actual_finish_date"))
                or pd.Timestamp.max
            )
        )
        finish = (
            _parse_date(item.get("forecast_finish_date"))
            or _parse_date(item.get("actual_finish_date"))
            or _parse_date(item.get("forecast_start_date"))
            or _parse_date(item.get("actual_start_date"))
            or pd.Timestamp.max
        )
        return (
            start,
            finish,
            _clean_digest_value(item.get("wbs_path")) or "",
            _clean_digest_value(item.get("task_name")) or "",
        )

    def _chronological_activity_facts(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        sorted_items = sorted(
            [item for item in items if isinstance(item, Mapping)],
            key=_activity_sequence_sort_key,
        )
        facts: list[dict[str, Any]] = []
        for pos, item in enumerate(sorted_items, start=1):
            fact = _activity_float_fact(item)
            if not fact:
                continue
            fact["chronological_order"] = pos
            facts.append(fact)
        return facts

    def _critical_status_focus(facts: list[dict[str, Any]]) -> dict[str, Any]:
        in_progress = [fact for fact in facts if fact.get("current_status") == "in_progress"]
        not_started = [fact for fact in facts if fact.get("current_status") == "not_started"]
        completed = [fact for fact in facts if fact.get("current_status") == "completed"]
        first_active = in_progress[0] if in_progress else None
        first_open = first_active or (not_started[0] if not_started else None)
        return {
            "status_counts": {
                "in_progress": int(len(in_progress)),
                "not_started": int(len(not_started)),
                "completed": int(len(completed)),
            },
            "current_active_critical_driver": first_active,
            "first_open_critical_activity": first_open,
            "active_critical_activity_examples": in_progress[:8],
            "next_not_started_critical_activity_examples": not_started[:8],
            "status_language_rule": (
                "If current_active_critical_driver is present, describe it as the current active critical driver before future/not-started critical work. "
                "Do not say the current critical chain starts with a future activity when an in-progress critical activity exists."
            ),
        }

    def _is_downstream_finish_or_milestone(fact: Mapping[str, Any]) -> bool:
        text = " ".join(
            _clean_digest_value(fact.get(key)) or ""
            for key in ["task_name", "wbs_name", "wbs_path"]
        ).casefold()
        return any(
            term in text
            for term in [
                "completion",
                "milestone",
                "line painting",
                "road closure",
                "approach slab",
                "final",
                "closeout",
                "commission",
                "turnover",
            ]
        )

    def _critical_sequence_role_examples(facts: list[dict[str, Any]]) -> dict[str, Any]:
        active_facts = [fact for fact in facts if not fact.get("completed")]
        non_tail = [fact for fact in active_facts if not _is_downstream_finish_or_milestone(fact)]
        tail = [fact for fact in active_facts if _is_downstream_finish_or_milestone(fact)]
        if not tail and active_facts:
            tail = active_facts[-6:]
        return {
            "upstream_driver_examples": non_tail[:8],
            "downstream_finish_or_milestone_examples": tail[-8:],
            "driver_language_rule": (
                "Describe the critical sequence chronologically. Upstream predecessor work controls downstream successors; "
                "tail-end finish/off-bridge/milestone work is driven by the upstream critical work and must not be called the driver."
            ),
        }

    def _compact_focus_group(group: Any, *, max_examples: int = 5) -> dict[str, Any]:
        if not isinstance(group, Mapping):
            return {}
        out: dict[str, Any] = {
            "area": group.get("area") or _compact_wbs_context(group),
            "leaf_wbs_name": group.get("leaf_wbs_name"),
            "leaf_wbs_path": group.get("leaf_wbs_path"),
        }
        for key in [
            "activity_count",
            "item_count",
            "started_count",
            "finished_count",
            "forecast_start_count",
            "forecast_finish_count",
            "not_started_count",
            "in_progress_count",
            "critical_path_count",
            "critical_count",
            "near_critical_count",
            "eroding_risk_count",
            "completed_count",
            "min_float_days",
            "max_float_days",
            "min_float_current_days",
            "max_float_current_days",
            "max_float_loss_days",
            "max_finish_slip_days",
        ]:
            if group.get(key) is not None:
                out[key] = group.get(key)

        representative_names = group.get("representative_task_names")
        if isinstance(representative_names, list) and representative_names:
            out["representative_task_names"] = [
                name for name in (_clean_digest_value(n) for n in representative_names) if name
            ][:max_examples]
        else:
            out["representative_task_names"] = _item_names(
                group.get("representative_items") or group.get("items") or [],
                limit=max_examples,
            )

        return {k: v for k, v in out.items() if v not in (None, [], {})}

    def _compact_focus_path(path: Any, *, max_tasks: int = 8) -> dict[str, Any]:
        if not isinstance(path, Mapping):
            return {}
        chain = path.get("activity_chain") or []
        out = {
            "branch_type": path.get("branch_type"),
            "length": path.get("length"),
            "branch_min_float_days": path.get("branch_min_float_days"),
            "branch_max_float_days": path.get("branch_max_float_days"),
            "task_sequence": _item_names(chain, limit=max_tasks),
            "wbs_sequence": _path_wbs_sequence(path)[:5],
            "constraint_driver_count": path.get("constraint_driver_count"),
            "constraint_drivers": (path.get("constraint_drivers") or [])[:3],
        }
        return {k: v for k, v in out.items() if v not in (None, [], {})}

    def _compact_eroding_risk(item: Any) -> dict[str, Any]:
        if not isinstance(item, Mapping):
            return {}
        out = {
            "task_name": item.get("task_name"),
            "wbs_name": item.get("wbs_name"),
            "wbs_path": item.get("wbs_path"),
            "float_current_days": item.get("float_current_days"),
            "float_loss_days": item.get("float_loss_days"),
            "days_passed": item.get("days_passed"),
            "eroding_risk": item.get("eroding_risk"),
        }
        return {k: v for k, v in out.items() if v not in (None, [], {})}

    def _compact_shift_cause(cause: Any) -> dict[str, Any]:
        if not isinstance(cause, Mapping):
            return {}
        changed_fields = []
        for field_change in cause.get("changed_fields") or []:
            if not isinstance(field_change, Mapping):
                continue
            changed_fields.append(
                {
                    "field": field_change.get("field"),
                    "delta_days": field_change.get("delta_days"),
                }
            )
            if len(changed_fields) >= 4:
                break
        out = {
            "driver_type": cause.get("driver_type"),
            "task_name": cause.get("task_name"),
            "wbs_name": cause.get("wbs_name"),
            "changed_fields": changed_fields,
            "detail": cause.get("detail"),
        }
        return {k: v for k, v in out.items() if v not in (None, [], {})}

    def _compact_change_delay_context(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "has_critical_path_driver": summary.get("has_critical_path_driver"),
            "has_cross_wbs_connection": summary.get("has_cross_wbs_connection"),
            "critical_path_driver_count": summary.get("critical_path_driver_count"),
            "cross_wbs_alert_count": summary.get("cross_wbs_alert_count"),
            "impact_statement": summary.get("impact_statement"),
        }

    def _compact_path_change_path(path: Any, *, max_items: int = 24, max_groups: int = 8) -> dict[str, Any] | None:
        if not isinstance(path, Mapping):
            return None
        raw_items = path.get("items") or []
        shown_items, total_items, omitted_items = _limit_list(raw_items, max_items)
        groups = [
            group
            for group in (_compact_focus_group(g) for g in (path.get("groups") or [])[:max_groups])
            if group
        ]
        return {
            "length": path.get("length"),
            "wbs_paths": (path.get("wbs_paths") or [])[:12],
            "groups": groups,
            "groups_shown_count": len(groups),
            "groups_omitted_count": max(0, len(path.get("groups") or []) - len(groups)),
            "task_sequence": _item_names(raw_items, limit=max_items),
            "items": [
                fact
                for fact in (_activity_float_fact(item) for item in shown_items)
                if fact
            ],
            "items_shown_count": len(shown_items),
            "items_omitted_count": omitted_items,
            "item_count": total_items,
        }

    def _compact_path_change_interpretation(value: Any, *, max_sequence_items: int = 24) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        out = dict(value)
        for key in [
            "previous_unique_upstream_sequence",
            "current_unique_upstream_sequence",
            "shared_downstream_sequence",
        ]:
            sequence = out.get(key)
            if not isinstance(sequence, list):
                continue
            shown, total, omitted = _limit_list(sequence, max_sequence_items)
            out[key] = _item_names(shown, limit=max_sequence_items)
            out[f"{key}_count"] = total
            out[f"{key}_shown_count"] = len(out[key])
            out[f"{key}_omitted_count"] = omitted
        return out

    def _compact_alternate_path_examples(paths: list[Any], *, max_examples: int = MAX_AI_ALTERNATE_PATH_EXAMPLES) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        seen_keys: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for path in paths:
            compact = _compact_focus_path(_path_fact(path), max_tasks=12)
            if not compact:
                continue
            key = (
                tuple(compact.get("task_sequence") or []),
                tuple(compact.get("wbs_sequence") or []),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            examples.append(compact)
            if len(examples) >= max_examples:
                break
        return examples

    def _summarize_completed_upstream(items: list[dict[str, Any]], *, max_groups: int = 8, max_examples: int = 5) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            wbs_path, leaf_name, parts = _wbs_parts(item.get("wbs_path"), item.get("wbs_name"))
            key = wbs_path or leaf_name or "Unknown"
            if key not in grouped:
                grouped[key] = {
                    "area": None,
                    "leaf_wbs_name": leaf_name,
                    "leaf_wbs_path": wbs_path,
                    "wbs_hierarchy_root_to_leaf": parts,
                    "activity_count": 0,
                    "representative_task_names": [],
                }
                grouped[key]["area"] = _compact_wbs_context(grouped[key])
            group = grouped[key]
            group["activity_count"] += 1
            name = _clean_digest_value(item.get("task_name"))
            if name and name not in group["representative_task_names"] and len(group["representative_task_names"]) < max_examples:
                group["representative_task_names"].append(name)

        groups = sorted(
            grouped.values(),
            key=lambda g: (-(g.get("activity_count") or 0), g.get("area") or g.get("leaf_wbs_name") or ""),
        )
        shown_groups = groups[:max_groups]
        return {
            "count": int(len(items)),
            "group_count": int(len(groups)),
            "shown_group_count": int(len(shown_groups)),
            "additional_group_count": max(0, int(len(groups) - len(shown_groups))),
            "groups": shown_groups,
            "usage_rule": "Completed upstream activities are completed context only. Do not describe them as current critical risk.",
        }

    def _compact_upstream_new_critical_links(summary: Mapping[str, Any], *, max_chains: int = 10) -> dict[str, Any]:
        priority_terms = ["leak", "delay", "rfi", "change", "constraint", "conflict"]

        def chain_priority(chain: Any) -> tuple[int, float, int, str]:
            if not isinstance(chain, Mapping):
                return (9, 999999.0, 999999, "")
            names = " ".join(
                _clean_digest_value(item.get("task_name")) or ""
                for item in (chain.get("activity_chain") or [])
                if isinstance(item, Mapping)
            ).casefold()
            root_float = chain.get("root_float_current_days")
            try:
                root_float_value = float(root_float)
            except Exception:
                root_float_value = 999999.0
            return (
                0 if any(term in names for term in priority_terms) else 1,
                root_float_value,
                int(chain.get("chain_length") or 999999),
                _clean_digest_value(chain.get("root_task_name")) or "",
            )

        chains = []
        sorted_chains = sorted(summary.get("chains") or [], key=chain_priority)
        for chain in sorted_chains[:max_chains]:
            if not isinstance(chain, Mapping):
                continue
            activity_chain = [
                fact
                for fact in (_activity_float_fact(item) for item in (chain.get("activity_chain") or []))
                if fact
            ]
            logic_links = [
                fact
                for fact in (_logic_link_fact(link) for link in (chain.get("logic_links") or []))
                if fact
            ]
            chains.append(
                {
                    "root_task_name": chain.get("root_task_name"),
                    "root_float_current_days": chain.get("root_float_current_days"),
                    "root_current_status": chain.get("root_current_status"),
                    "reached_critical_task_name": chain.get("reached_critical_task_name"),
                    "reached_critical_float_current_days": chain.get("reached_critical_float_current_days"),
                    "chain_length": chain.get("chain_length"),
                    "task_sequence": _item_names(activity_chain, limit=12),
                    "status_sequence": [
                        item.get("current_status")
                        for item in activity_chain
                        if isinstance(item, Mapping)
                    ][:12],
                    "float_sequence_days": [
                        item.get("float_current_days")
                        for item in activity_chain
                        if isinstance(item, Mapping)
                    ][:12],
                    "activity_chain": activity_chain[:12],
                    "logic_links": logic_links[:12],
                    "interpretation": chain.get("interpretation"),
                }
            )
        return {
            "count": summary.get("count"),
            "included_chain_count": int(len(chains)),
            "truncated": summary.get("truncated"),
            "max_depth": summary.get("max_depth"),
            "chains": chains,
            "usage_rule": (
                "Use these chains only as upstream new/change/delay context. "
                "Do not call the root activity critical unless its own is_critical and float_current_days fields say so."
            ),
        }

    def _progress_focus(summary: Mapping[str, Any], *, max_groups: int = 6) -> dict[str, Any]:
        highlights = []
        for group in (summary.get("groups") or [])[:max_groups]:
            items = group.get("items") or []
            started_and_finished = []
            started_only = []
            finished_only = []
            for item in items:
                events = set(item.get("progress_events") or [])
                name = _clean_digest_value(item.get("task_name"))
                if not name:
                    continue
                if {"actual_start", "actual_finish"}.issubset(events):
                    started_and_finished.append(name)
                elif "actual_start" in events:
                    started_only.append(name)
                elif "actual_finish" in events:
                    finished_only.append(name)

            highlights.append(
                {
                    "area": _compact_wbs_context(group),
                    "leaf_wbs_name": group.get("leaf_wbs_name"),
                    "started_count": group.get("started_count"),
                    "finished_count": group.get("finished_count"),
                    "started_and_finished_count": group.get("started_and_finished_count"),
                    "started_and_finished_examples": started_and_finished[:5],
                    "started_not_finished_examples": started_only[:5],
                    "finished_only_examples": finished_only[:5],
                    "representative_task_names": _item_names(items, limit=6),
                }
            )
        return {
            "total_started_count": summary.get("started_count"),
            "total_finished_count": summary.get("finished_count"),
            "total_started_and_finished_count": summary.get("started_and_finished_count"),
            "top_progress_groups": highlights,
        }

    def _finish_extension_focus(summary: Mapping[str, Any], *, max_groups: int = 6) -> dict[str, Any]:
        highlights = []
        for group in (summary.get("groups") or [])[:max_groups]:
            examples = []
            for item in (group.get("items") or [])[:5]:
                examples.append(
                    {
                        "task_name": item.get("task_name"),
                        "finish_slip_days": item.get("finish_slip_days"),
                        "last_forecast_finish_date": item.get("last_forecast_finish_date"),
                        "current_forecast_finish_date": item.get("current_forecast_finish_date"),
                    }
                )
            highlights.append(
                {
                    "area": _compact_wbs_context(group),
                    "leaf_wbs_name": group.get("leaf_wbs_name"),
                    "item_count": group.get("item_count"),
                    "max_finish_slip_days": group.get("max_finish_slip_days"),
                    "examples": examples,
                }
            )
        return {
            "count": summary.get("count"),
            "top_extension_groups": highlights,
        }

    def _path_task_sequence(path: Mapping[str, Any] | None) -> list[str]:
        if not isinstance(path, Mapping):
            return []
        return _item_names(path.get("activity_chain"), limit=20)

    def _path_wbs_sequence(path: Mapping[str, Any] | None) -> list[str]:
        if not isinstance(path, Mapping):
            return []
        sequence: list[str] = []
        for item in path.get("activity_chain") or []:
            if not isinstance(item, Mapping):
                continue
            parts = [p.strip() for p in str(item.get("wbs_path") or "").split(" / ") if p.strip()]
            useful = [p for p in parts[1:] if p]
            context = " / ".join(useful[-3:]) if useful else _clean_digest_value(item.get("wbs_name"))
            if context and context not in sequence:
                sequence.append(context)
        return sequence

    def _critical_path_narrative_focus(
        paths: list[dict[str, Any]],
        *,
        target_task_name: Any,
        least_float_current_days: Any,
    ) -> dict[str, Any]:
        branch_summaries = []
        grouped_areas: dict[str, dict[str, Any]] = {}
        longest_path = max(paths, key=lambda p: len(p.get("activity_chain") or []), default=None)
        primary_chain = (paths[0].get("activity_chain") or []) if paths else []
        primary_ids = {
            str(item.get("activity_id"))
            for item in primary_chain
            if isinstance(item, Mapping) and item.get("activity_id")
        }
        tied_branch_examples: list[dict[str, Any]] = []
        seen_tied_branch_keys: set[tuple[str, str]] = set()

        for idx, path in enumerate(paths, start=1):
            chain = path.get("activity_chain") or []
            floats = [item.get("float_current_days") for item in chain if isinstance(item, Mapping) and item.get("float_current_days") is not None]
            if len(branch_summaries) < MAX_AI_BRANCH_SUMMARIES:
                branch_summaries.append(
                    {
                        "branch_number": idx,
                        "length": path.get("length"),
                        "task_sequence": _path_task_sequence(path),
                        "wbs_sequence": _path_wbs_sequence(path),
                        "min_float_days": min(floats) if floats else None,
                        "max_float_days": max(floats) if floats else None,
                        "constraint_drivers": (path.get("constraint_drivers") or [])[:3],
                    }
                )

            if idx > 1 and primary_ids:
                first_unique_idx = None
                first_unique_item = None
                for item_idx, item in enumerate(chain):
                    if not isinstance(item, Mapping):
                        continue
                    aid = str(item.get("activity_id") or "")
                    if aid and aid not in primary_ids:
                        first_unique_idx = item_idx
                        first_unique_item = item
                        break

                feed_item = None
                feed_idx = None
                if first_unique_idx is not None:
                    for item_idx, item in enumerate(chain[first_unique_idx + 1:], start=first_unique_idx + 1):
                        if not isinstance(item, Mapping):
                            continue
                        aid = str(item.get("activity_id") or "")
                        if aid and aid in primary_ids:
                            feed_item = item
                            feed_idx = item_idx
                            break

                if first_unique_item is not None and feed_item is not None and feed_idx is not None:
                    key = (
                        str(first_unique_item.get("activity_id") or first_unique_item.get("task_name")),
                        str(feed_item.get("activity_id") or feed_item.get("task_name")),
                    )
                    if key not in seen_tied_branch_keys:
                        seen_tied_branch_keys.add(key)
                        tied_branch_examples.append(
                            {
                                "branch_role": "parallel_or_tied_predecessor_branch",
                                "branch_start_task_name": first_unique_item.get("task_name"),
                                "branch_start_wbs_path": first_unique_item.get("wbs_path"),
                                "branch_start_float_days": first_unique_item.get("float_current_days"),
                                "feeds_into_task_name": feed_item.get("task_name"),
                                "feeds_into_wbs_path": feed_item.get("wbs_path"),
                                "report_phrase": (
                                    f"In parallel, {first_unique_item.get('task_name')} also feeds into "
                                    f"{feed_item.get('task_name')}."
                                ),
                                "wording_rule": (
                                    "Use 'In parallel' or 'A tied predecessor branch' language. Do not use 'followed by' "
                                    "for this branch because it is not sequentially after the primary active driver."
                                ),
                                "sequence_to_shared_path": _item_names(
                                    chain[: (feed_idx + 1)],
                                    limit=8,
                                ),
                            }
                        )

            for item in chain:
                if not isinstance(item, Mapping):
                    continue
                path_key, leaf_name, parts = _wbs_parts(item.get("wbs_path"), item.get("wbs_name"))
                key = path_key or leaf_name or "Unknown"
                if key not in grouped_areas:
                    grouped_areas[key] = {
                        "area": None,
                        "leaf_wbs_name": leaf_name,
                        "leaf_wbs_path": path_key,
                        "wbs_hierarchy_root_to_leaf": parts,
                        "activity_count": 0,
                        "representative_task_names": [],
                        "min_float_days": None,
                        "max_float_days": None,
                        "_seen_activity_keys": set(),
                    }
                    grouped_areas[key]["area"] = _compact_wbs_context(grouped_areas[key])

                group = grouped_areas[key]
                activity_key = (
                    _clean_digest_value(item.get("activity_id"))
                    or _clean_digest_value(item.get("task_name"))
                    or f"path_{idx}_item"
                )
                if activity_key in group["_seen_activity_keys"]:
                    continue
                group["_seen_activity_keys"].add(activity_key)
                group["activity_count"] += 1
                task_name = _clean_digest_value(item.get("task_name"))
                if task_name and task_name not in group["representative_task_names"]:
                    group["representative_task_names"].append(task_name)
                flt = item.get("float_current_days")
                if flt is not None:
                    group["min_float_days"] = flt if group["min_float_days"] is None else min(group["min_float_days"], flt)
                    group["max_float_days"] = flt if group["max_float_days"] is None else max(group["max_float_days"], flt)

        area_highlights = sorted(
            grouped_areas.values(),
            key=lambda g: (-(g.get("activity_count") or 0), g.get("area") or ""),
        )
        for group in area_highlights:
            group["representative_task_names"] = group["representative_task_names"][:6]
            group.pop("_seen_activity_keys", None)

        return {
            "target_task_name": target_task_name,
            "least_float_current_days": least_float_current_days,
            "path_count": len(paths),
            "branch_summary_count": len(paths),
            "branch_summary_shown_count": len(branch_summaries),
            "branch_summary_omitted_count": max(0, len(paths) - len(branch_summaries)),
            "primary_path_task_sequence": _path_task_sequence(paths[0] if paths else None),
            "primary_path_wbs_sequence": _path_wbs_sequence(paths[0] if paths else None),
            "longest_tied_path_task_sequence": _path_task_sequence(longest_path),
            "longest_tied_path_wbs_sequence": _path_wbs_sequence(longest_path),
            "branch_summaries": branch_summaries,
            "path_area_highlights": area_highlights,
            "tied_branch_examples": tied_branch_examples[:8],
            "tied_branch_language_rule": (
                "Describe tied branch examples as parallel/predecessor branches feeding into the named shared activity. "
                "Do not write that the primary active driver is followed by the tied branch unless the sequence_to_shared_path shows that order."
            ),
        }

    def _group_critical_path_activity_items(paths: list[dict[str, Any]], extra_items: list[dict[str, Any]]) -> dict[str, Any]:
        unique: dict[str, dict[str, Any]] = {}
        fallback_index = 0
        for path in paths:
            for item in path.get("activity_chain") or []:
                if not isinstance(item, Mapping):
                    continue
                key = _clean_digest_value(item.get("activity_id"))
                if not key:
                    fallback_index += 1
                    key = f"item_{fallback_index}"
                unique.setdefault(key, dict(item))
        for item in extra_items:
            if not isinstance(item, Mapping):
                continue
            key = _clean_digest_value(item.get("activity_id"))
            if not key:
                fallback_index += 1
                key = f"extra_{fallback_index}"
            unique.setdefault(key, dict(item))

        grouped: dict[str, dict[str, Any]] = {}
        for item in unique.values():
            wbs_path, leaf_name, parts = _wbs_parts(item.get("wbs_path"), item.get("wbs_name"))
            key = wbs_path or leaf_name or "Unknown"
            if key not in grouped:
                grouped[key] = {
                    "area": None,
                    "leaf_wbs_name": leaf_name,
                    "leaf_wbs_path": wbs_path,
                    "wbs_hierarchy_root_to_leaf": parts,
                    "activity_count": 0,
                    "completed_count": 0,
                    "in_progress_count": 0,
                    "not_started_count": 0,
                    "min_float_days": None,
                    "max_float_days": None,
                    "items": [],
                }
                grouped[key]["area"] = _compact_wbs_context(grouped[key])

            group = grouped[key]
            group["activity_count"] += 1
            item_status = _activity_current_status(item)
            if item_status == "completed":
                group["completed_count"] += 1
            elif item_status == "in_progress":
                group["in_progress_count"] += 1
            elif item_status == "not_started":
                group["not_started_count"] += 1
            flt = item.get("float_current_days")
            if flt is not None:
                group["min_float_days"] = flt if group["min_float_days"] is None else min(group["min_float_days"], flt)
                group["max_float_days"] = flt if group["max_float_days"] is None else max(group["max_float_days"], flt)
            group["items"].append(
                {
                    "task_name": item.get("task_name"),
                    "float_current_days": item.get("float_current_days"),
                    "completed": item.get("completed"),
                    "current_status": item_status,
                }
            )

        groups = sorted(
            grouped.values(),
            key=lambda g: (-(g.get("activity_count") or 0), g.get("area") or g.get("leaf_wbs_name") or ""),
        )
        return {
            "count": int(len(unique)),
            "group_count": int(len(groups)),
            "groups": groups,
        }

    def _group_simple_scope_items(items: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        cleaned_items: list[dict[str, Any]] = []
        for r in items:
            wbs_path, leaf_name, parts = _wbs_parts(r.get("wbs_path"), r.get("wbs_name"))
            key = wbs_path or leaf_name or "Unknown"
            if key not in grouped:
                grouped[key] = {
                    "leaf_wbs_name": leaf_name,
                    "leaf_wbs_path": wbs_path,
                    "wbs_hierarchy_root_to_leaf": parts,
                    "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                    "item_count": 0,
                    "items": [],
                }
            item = {
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
                "wbs_path": wbs_path,
            }
            cleaned_items.append(item)
            grouped[key]["items"].append(item)
            grouped[key]["item_count"] += 1

        groups = sorted(
            grouped.values(),
            key=lambda g: (-(g.get("item_count") or 0), g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or ""),
        )
        return {
            "count": int(len(cleaned_items)),
            "group_count": int(len(groups)),
            "groups": groups,
            "items": cleaned_items,
        }

    def _summarize_downstream_logic(cps: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> dict[str, Any]:
        critical_summaries = [r for r in cps if bool(r.get("has_critical_path_driver"))]
        cross_wbs_summaries = [r for r in cps if _clean_digest_value(r.get("cross_wbs_top"))]
        return {
            "has_any_downstream_logic": bool(cps or alerts),
            "has_critical_path_driver": bool(critical_summaries),
            "has_cross_wbs_connection": bool(cross_wbs_summaries or alerts),
            "critical_path_driver_count": int(len(critical_summaries)),
            "cross_wbs_alert_count": int(len(alerts)),
            "critical_path_successor_summaries": cps,
            "critical_path_driver_summaries": critical_summaries,
            "cross_wbs_summaries": cross_wbs_summaries,
            "cross_wbs_alerts": alerts,
            "impact_statement": (
                "Downstream critical-path influence is supported by traced successor logic."
                if critical_summaries
                else (
                    "Downstream logic connections exist, but downstream critical-path influence is not indicated."
                    if (cps or alerts)
                    else "Downstream influence is not determinable from the provided logic/data."
                )
            ),
        }

    def _summarize_look_ahead(src: Mapping[str, Any]) -> dict[str, Any]:
        items = src.get("items", []) or []
        grouped: dict[str, dict[str, Any]] = {}
        sensitive_grouped: dict[str, dict[str, Any]] = {}
        cleaned_items: list[dict[str, Any]] = []

        def ensure_group(store: dict[str, dict[str, Any]], r: Mapping[str, Any]) -> dict[str, Any]:
            wbs_path, leaf_name, parts = _wbs_parts(r.get("wbs_path"), r.get("wbs_name"))
            key = wbs_path or leaf_name or "Unknown"
            if key not in store:
                store[key] = {
                    "leaf_wbs_name": leaf_name,
                    "leaf_wbs_path": wbs_path,
                    "wbs_hierarchy_root_to_leaf": parts,
                    "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                    "item_count": 0,
                    "forecast_start_count": 0,
                    "forecast_finish_count": 0,
                    "not_started_count": 0,
                    "in_progress_count": 0,
                    "critical_path_count": 0,
                    "critical_count": 0,
                    "near_critical_count": 0,
                    "items": [],
                }
            return store[key]

        for r in items:
            dates_in_window = r.get("dates_in_window", []) or []
            current_status = r.get("current_status")
            item = {
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
                "wbs_path": r.get("wbs_path"),
                "forecast_start_date": r.get("forecast_start_date"),
                "forecast_finish_date": r.get("forecast_finish_date"),
                "dates_in_window": dates_in_window,
                "current_status": current_status,
                "total_float_days": r.get("total_float_days"),
                "on_current_critical_path": bool(r.get("on_current_critical_path")),
                "critical_current": bool(r.get("critical_current")),
                "near_critical_current": bool(r.get("near_critical_current")),
            }
            cleaned_items.append(item)

            for store in [grouped]:
                group = ensure_group(store, r)
                group["item_count"] += 1
                if "start" in dates_in_window:
                    group["forecast_start_count"] += 1
                if "finish" in dates_in_window:
                    group["forecast_finish_count"] += 1
                if current_status == "not_started":
                    group["not_started_count"] += 1
                if current_status == "in_progress":
                    group["in_progress_count"] += 1
                if item["on_current_critical_path"]:
                    group["critical_path_count"] += 1
                if item["critical_current"]:
                    group["critical_count"] += 1
                if item["near_critical_current"]:
                    group["near_critical_count"] += 1
                group["items"].append(item)

            if item["on_current_critical_path"] or item["critical_current"] or item["near_critical_current"]:
                group = ensure_group(sensitive_grouped, r)
                group["item_count"] += 1
                if "start" in dates_in_window:
                    group["forecast_start_count"] += 1
                if "finish" in dates_in_window:
                    group["forecast_finish_count"] += 1
                if current_status == "not_started":
                    group["not_started_count"] += 1
                if current_status == "in_progress":
                    group["in_progress_count"] += 1
                if item["on_current_critical_path"]:
                    group["critical_path_count"] += 1
                if item["critical_current"]:
                    group["critical_count"] += 1
                if item["near_critical_current"]:
                    group["near_critical_count"] += 1
                group["items"].append(item)

        def sort_groups(groups_dict: dict[str, dict[str, Any]], *, sensitive: bool = False) -> list[dict[str, Any]]:
            if sensitive:
                return sorted(
                    groups_dict.values(),
                    key=lambda g: (
                        -(g.get("critical_path_count") or 0),
                        -(g.get("critical_count") or 0),
                        -(g.get("near_critical_count") or 0),
                        -(g.get("item_count") or 0),
                        g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or "",
                    ),
                )
            return sorted(
                groups_dict.values(),
                key=lambda g: (-(g.get("item_count") or 0), g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or ""),
            )

        groups = sort_groups(grouped)
        sensitive_groups = sort_groups(sensitive_grouped, sensitive=True)
        return {
            "window": src.get("window"),
            "upcoming_work": {
                "count": int(len(cleaned_items)),
                "group_count": int(len(groups)),
                "groups": groups,
                # Predecessor links among upcoming activities; use to describe how the window's work sequences.
                "driving_links": (src.get("driving_links") or [])[:40],
            },
            "schedule_sensitive_upcoming_work": {
                "count": int(sum(g.get("item_count", 0) for g in sensitive_groups)),
                "group_count": int(len(sensitive_groups)),
                "groups": sensitive_groups,
            },
            "warning": src.get("warning"),
        }

    settings_summary = {
        "variance_threshold": settings.get("variance_threshold"),
        "look_ahead_horizon_days": settings.get("look_ahead_horizon_days"),
        "change_term": settings.get("change_term"),
    }
    if is_mspdi:
        settings_summary["schedule_source"] = "microsoft_project_xml"
        date_math_guardrail_summary = {
            "rule": "Critical and near-critical status is precomputed by Python from Microsoft Project TotalSlack, not inferred from activity date gaps.",
            "float_conversion": "Python converts Microsoft Project TotalSlack from tenths of a minute to days using the exported MinutesPerDay value before the JSON is sent to the LLM.",
            "llm_instruction": "Do not calculate slack, float, or driving status from forecast/actual dates, weekends, holidays, or apparent gaps between predecessor and successor dates.",
        }
    else:
        date_math_guardrail_summary = {
            "rule": "Critical and near-critical status is precomputed by Python from P6 total float, not inferred from activity date gaps.",
            "float_conversion": "Python converts P6 total float to days using each activity's assigned calendar before the JSON is sent to the LLM.",
            "llm_instruction": "Do not calculate slack, float, or driving status from forecast/actual dates, weekends, holidays, or apparent gaps between predecessor and successor dates.",
        }
    update_period_summary = {
        "baseline_data_date": update_period.get("baseline_data_date"),
        "last_data_date": update_period.get("last_data_date"),
        "current_data_date": update_period.get("current_data_date"),
        "days_last_to_current": update_period.get("days_last_to_current"),
    }
    milestone_variance_summary = {
        "baseline": ms_part("baseline"),
        "last": ms_part("last"),
        "current": ms_part("current"),
        "total_variance_days": milestone.get("total_variance_days"),
        "period_variance_days": milestone.get("period_variance_days"),
    }
    eroding_risks_summary = {
        "least_float_current_days": trending.get("least_float_current_days"),
        "cutoff_current_days": trending.get("cutoff_current_days"),
        "excluded_activity_count": trending.get("excluded_activity_count"),
        "days_between": trending.get("days_between"),
        "count": trending.get("eroding_risk_count"),
        "items": eroding,
    }
    if is_mspdi and trending.get("erosion_assessment_warning"):
        eroding_risks_summary["erosion_assessment_warning"] = trending.get("erosion_assessment_warning")
    near_critical_grouped_summary = _group_near_critical(trending)
    new_activities_global_summary = {
        "count": new_global.get("count"),
        "group_count": new_global.get("group_count"),
        "warning": new_global.get("warning"),
        "groups": new_global_groups,
        "items": new_global_items,
    }
    actualized_work_summary = _summarize_actualized_work(accomplished)
    if is_mspdi and accomplished.get("period_assessment_available") is False:
        actualized_work_summary["period_assessment_available"] = False
        actualized_work_summary["warning"] = accomplished.get("warning")
    msp_progress_summary: dict[str, Any] = {}
    if is_mspdi:
        msp_items = msp_progress.get("items", []) or []
        msp_progress_summary = {
            "period_assessment_available": msp_progress.get("period_assessment_available"),
            "days_between_status_dates": msp_progress.get("days_between_status_dates"),
            "warning": msp_progress.get("warning"),
            "count": msp_progress.get("count"),
            "items": msp_items[:30],
            "items_shown_count": min(len(msp_items), 30),
            "items_omitted_count": max(0, len(msp_items) - 30),
        }
    in_progress_finish_extensions_summary = _summarize_finish_extensions(finish_extensions)
    global_new_scope_summary = {
        "count": new_activities_global_summary.get("count"),
        "group_count": new_activities_global_summary.get("group_count"),
        "warning": new_activities_global_summary.get("warning"),
        "groups": new_activities_global_summary.get("groups"),
        "items": new_activities_global_summary.get("items"),
    }
    change_delay_new_scope_summary = _group_simple_scope_items(new_change)
    downstream_logic_indicators_summary = _summarize_downstream_logic(cps_items, cross_wbs_alerts)
    look_ahead_summary = _summarize_look_ahead(look_ahead)
    actualized_work_payload_summary = _cap_grouped_summary(actualized_work_summary)
    in_progress_finish_extensions_payload_summary = _cap_grouped_summary(in_progress_finish_extensions_summary)
    global_new_scope_payload_summary = _cap_grouped_summary(global_new_scope_summary)
    change_delay_new_scope_payload_summary = _cap_grouped_summary(change_delay_new_scope_summary)
    downstream_logic_indicators_payload_summary = _cap_named_lists(
        downstream_logic_indicators_summary,
        [
            "critical_path_successor_summaries",
            "critical_path_driver_summaries",
            "cross_wbs_summaries",
            "cross_wbs_alerts",
        ],
    )
    eroding_risks_payload_summary = _cap_grouped_summary(eroding_risks_summary, max_groups=0)
    near_critical_grouped_payload_summary = _cap_grouped_summary(near_critical_grouped_summary)
    look_ahead_payload_summary = dict(look_ahead_summary)
    look_ahead_payload_summary["upcoming_work"] = _cap_grouped_summary(
        (look_ahead_summary.get("upcoming_work") or {}),
        max_groups=MAX_AI_GROUPS,
        max_group_items=MAX_AI_GROUP_ITEMS,
    )
    look_ahead_payload_summary["schedule_sensitive_upcoming_work"] = _cap_grouped_summary(
        (look_ahead_summary.get("schedule_sensitive_upcoming_work") or {}),
        max_groups=MAX_AI_GROUPS,
        max_group_items=MAX_AI_GROUP_ITEMS,
    )
    critical_paths = critical_path.get("paths", []) or []
    current_primary_path = critical_paths[0] if critical_paths else None
    alternate_tied_paths = critical_paths[1:]
    current_primary_path_fact = _path_fact(current_primary_path)
    alternate_tied_path_examples = _compact_alternate_path_examples(alternate_tied_paths)
    alternate_tied_paths_summary = {
        "total_count": int(len(alternate_tied_paths)),
        "shown_example_count": int(len(alternate_tied_path_examples)),
        "omitted_count": max(0, int(len(alternate_tied_paths) - len(alternate_tied_path_examples))),
        "examples": alternate_tied_path_examples,
        "usage_rule": (
            "These are compact representative alternate/tied branch examples only. Use path_count for the total "
            "branch count and critical_activity_groups for the overall critical scope."
        ),
    }
    critical_activity_facts = _chronological_activity_facts(critical_path.get("critical_activities", []) or [])
    critical_activity_facts_payload, critical_activity_facts_total, critical_activity_facts_omitted = _limit_list(
        critical_activity_facts,
        MAX_AI_CRITICAL_ACTIVITIES,
    )
    completed_upstream_activity_facts = [
        fact
        for fact in (_activity_float_fact(item) for item in (critical_path.get("completed_upstream_activities", []) or []))
        if fact
    ]
    completed_upstream_summary = _summarize_completed_upstream(completed_upstream_activity_facts)
    primary_critical_chain_facts = (current_primary_path_fact or {}).get("activity_chain", []) if current_primary_path_fact else []
    primary_critical_chain_facts_payload, primary_critical_chain_total, primary_critical_chain_omitted = _limit_list(
        primary_critical_chain_facts,
        MAX_AI_PRIMARY_PATH_ACTIVITIES,
    )
    critical_sequence_role_summary = _critical_sequence_role_examples(critical_activity_facts)
    critical_status_focus_summary = _critical_status_focus(critical_activity_facts)
    chronological_critical_sequence_summary = {
        "definition": "The COMPLETE SET of target critical activities, listed by date only. This is NOT a logical predecessor chain.",
        "sequence_order": "earliest_forecast_or_restart_start_to_latest_finish_or_milestone",
        "sequence_is_complete_target_critical_set": True,
        "is_logical_predecessor_order": False,
        "narration_rule": (
            "Use this only as the set of critical activities and for date context. Do NOT narrate it as one linear "
            "chain and do NOT say one activity is 'followed by' or 'drives' the next based on this order, because "
            "parallel branches are interleaved by date. For logical predecessor order use primary_critical_path_0_days "
            "and the primary_path.logic_links; describe tied/parallel branches as parallel, not sequential."
        ),
        "activity_count": critical_activity_facts_total,
        "activity_shown_count": len(critical_activity_facts_payload),
        "activity_omitted_count": critical_activity_facts_omitted,
        "activity_chain": critical_activity_facts_payload,
        "current_status_focus": critical_status_focus_summary,
        **critical_sequence_role_summary,
    }
    precomputed_float_buckets_summary = {
        "classification_source": (
            "python_precomputed_from_microsoft_project_totalslack_and_minutes_per_day"
            if is_mspdi
            else "python_precomputed_from_p6_total_float_and_activity_calendar"
        ),
        "date_math_rule": "Never infer float, slack, criticality, or driving status from dates or apparent gaps.",
        "sequence_language_rule": critical_sequence_role_summary.get("driver_language_rule"),
        "critical_threshold_days": critical_path.get("least_float_current_days"),
        "near_critical_upper_threshold_days": critical_path.get("absolute_near_critical_threshold_days"),
        "chronological_target_critical_sequence_0_days": chronological_critical_sequence_summary,
        "current_status_focus": critical_status_focus_summary,
        "primary_critical_path_0_days": primary_critical_chain_facts_payload,
        "primary_critical_path_activity_count": primary_critical_chain_total,
        "primary_critical_path_omitted_count": primary_critical_chain_omitted,
        "target_critical_activities_0_days": critical_activity_facts_payload,
        "target_critical_activity_count": critical_activity_facts_total,
        "target_critical_activity_omitted_count": critical_activity_facts_omitted,
    }
    near_critical_float_buckets_summary = {
        "classification_source": precomputed_float_buckets_summary["classification_source"],
        "date_math_rule": precomputed_float_buckets_summary["date_math_rule"],
        "critical_threshold_days": precomputed_float_buckets_summary["critical_threshold_days"],
        "near_critical_upper_threshold_days": precomputed_float_buckets_summary["near_critical_upper_threshold_days"],
        "near_critical_activity_count": near_critical_grouped_summary.get("count"),
        "near_critical_group_count": near_critical_grouped_summary.get("group_count"),
        "excluded_critical_trace_activity_count": near_critical_grouped_summary.get("excluded_activity_count"),
        "near_critical_bucket_rule": (
            (
                "Near-critical activities are selected by Microsoft Project TotalSlack only: greater than the critical "
                "threshold and less than or equal to the near-critical threshold, excluding activities already included "
                "in the Section 4 critical trace."
            )
            if is_mspdi
            else (
                "Near-critical activities are selected by P6 total float only: greater than the critical threshold and "
                "less than or equal to the near-critical threshold, excluding activities already included in the Section 4 critical trace."
            )
        ),
    }
    actualized_work_focus_summary = _progress_focus(actualized_work_summary)
    in_progress_finish_extensions_focus_summary = _finish_extension_focus(in_progress_finish_extensions_summary)
    critical_path_narrative_focus_summary = _critical_path_narrative_focus(
        critical_paths,
        target_task_name=critical_path.get("target_task_name"),
        least_float_current_days=critical_path.get("least_float_current_days"),
    )
    critical_path_narrative_focus_summary["chronological_critical_task_sequence"] = _item_names(
        critical_activity_facts,
        limit=30,
    )
    critical_path_narrative_focus_summary["critical_status_counts"] = critical_status_focus_summary.get("status_counts")
    critical_path_narrative_focus_summary["current_active_critical_driver"] = critical_status_focus_summary.get("current_active_critical_driver")
    critical_path_narrative_focus_summary["first_open_critical_activity"] = critical_status_focus_summary.get("first_open_critical_activity")
    critical_path_narrative_focus_summary["active_critical_task_examples"] = _item_names(
        critical_status_focus_summary.get("active_critical_activity_examples"),
        limit=8,
    )
    critical_path_narrative_focus_summary["next_not_started_critical_task_examples"] = _item_names(
        critical_status_focus_summary.get("next_not_started_critical_activity_examples"),
        limit=8,
    )
    critical_path_narrative_focus_summary["upstream_driver_task_examples"] = _item_names(
        critical_sequence_role_summary.get("upstream_driver_examples"),
        limit=8,
    )
    critical_path_narrative_focus_summary["downstream_finish_or_milestone_examples"] = _item_names(
        critical_sequence_role_summary.get("downstream_finish_or_milestone_examples"),
        limit=8,
    )
    critical_path_narrative_focus_summary["sequence_language_rule"] = critical_sequence_role_summary.get("driver_language_rule")
    critical_path_narrative_focus_summary["status_language_rule"] = critical_status_focus_summary.get("status_language_rule")
    critical_path_narrative_focus_summary["branch_interpretation_rule"] = (
        "Branch summaries are examples of target-traced branches and may represent terminal segments. "
        "Use the chronological critical task sequence to explain what drives the path."
    )
    critical_activity_groups_summary = _group_critical_path_activity_items(
        critical_paths,
        critical_path.get("critical_activities", []) or [],
    )
    critical_activity_groups_payload_summary = _cap_grouped_summary(
        critical_activity_groups_summary,
        max_groups=30,
        max_group_items=MAX_AI_GROUP_ITEMS,
        max_items=0,
    )
    critical_path_narrative_focus_summary["critical_activity_groups"] = critical_activity_groups_payload_summary.get("groups")
    critical_path_narrative_focus_summary["critical_activity_group_count"] = critical_activity_groups_payload_summary.get("group_count")
    path_change_summary = {
        "changed": critical_path_change.get("changed"),
        "comparison_basis": critical_path_change.get("comparison_basis"),
        "path_change_interpretation": _compact_path_change_interpretation(critical_path_change.get("path_change_interpretation")),
        "previous_primary_path": _compact_path_change_path(critical_path_change.get("previous_primary_path")),
        "current_primary_path": _compact_path_change_path(critical_path_change.get("current_primary_path")),
        "added_to_current_path_count": critical_path_change.get("added_to_current_path_count"),
        "removed_from_previous_path_count": critical_path_change.get("removed_from_previous_path_count"),
        "added_to_current_path": critical_path_change.get("added_to_current_path"),
        "removed_from_previous_path": critical_path_change.get("removed_from_previous_path"),
        "current_wbs_added": critical_path_change.get("current_wbs_added"),
        "previous_wbs_removed": critical_path_change.get("previous_wbs_removed"),
        "warning": critical_path_change.get("warning"),
    }
    path_change_payload_summary = _cap_named_lists(
        path_change_summary,
        [
            "added_to_current_path",
            "removed_from_previous_path",
            "current_wbs_added",
            "previous_wbs_removed",
        ],
    )
    upstream_new_critical_context_summary = _compact_upstream_new_critical_links(upstream_new_critical_links)
    supported_shift_evidence_summary = {
        "variance_reason": None,
        "causality_limit": (
            "Schedule data can show observed field, logic, float, and path changes, but it does not prove the planning intent or root cause. "
            "Do not describe observed forecast movement as optimization, acceleration, reforecasting, or revised logic unless those words are explicitly present in the evidence."
        ),
        "cause_assessment": critical_path_change.get("cause_assessment"),
        "possible_shift_causes": critical_path_change.get("possible_shift_causes"),
        "likely_shift_drivers": critical_path_change.get("likely_shift_drivers"),
        "relationship_change_evidence": critical_path_change.get("relationship_change_evidence"),
        "task_field_change_evidence": critical_path_change.get("task_field_change_evidence"),
        "upstream_new_activity_links_to_current_critical_path": upstream_new_critical_context_summary,
    }
    supported_shift_evidence_payload_summary = _cap_named_lists(
        supported_shift_evidence_summary,
        [
            "possible_shift_causes",
            "likely_shift_drivers",
            "relationship_change_evidence",
            "task_field_change_evidence",
        ],
    )
    change_delay_context_summary = {
        "has_critical_path_driver": downstream_logic_indicators_summary.get("has_critical_path_driver"),
        "has_cross_wbs_connection": downstream_logic_indicators_summary.get("has_cross_wbs_connection"),
        "critical_path_driver_count": downstream_logic_indicators_summary.get("critical_path_driver_count"),
        "cross_wbs_alert_count": downstream_logic_indicators_summary.get("cross_wbs_alert_count"),
        "critical_path_successor_summaries": downstream_logic_indicators_summary.get("critical_path_successor_summaries"),
        "cross_wbs_alerts": downstream_logic_indicators_summary.get("cross_wbs_alerts"),
        "impact_statement": downstream_logic_indicators_summary.get("impact_statement"),
    }
    critical_path_focus_summary = {
        "target_task_name": critical_path.get("target_task_name"),
        "target_wbs_path": critical_path.get("target_wbs_path"),
        "least_float_current_days": critical_path.get("least_float_current_days"),
        "float_classification_source": precomputed_float_buckets_summary.get("classification_source"),
        "critical_activity_count": critical_path.get("critical_count"),
        "critical_group_count": critical_activity_groups_summary.get("group_count"),
        "sequence_language_rule": critical_path_narrative_focus_summary.get("sequence_language_rule"),
        "status_language_rule": critical_path_narrative_focus_summary.get("status_language_rule"),
        "critical_status_counts": critical_path_narrative_focus_summary.get("critical_status_counts"),
        "current_active_critical_driver": critical_path_narrative_focus_summary.get("current_active_critical_driver"),
        "chronological_critical_task_sequence": critical_path_narrative_focus_summary.get("chronological_critical_task_sequence"),
        "active_critical_task_examples": critical_path_narrative_focus_summary.get("active_critical_task_examples"),
        "next_not_started_critical_task_examples": critical_path_narrative_focus_summary.get("next_not_started_critical_task_examples"),
        "upstream_driver_task_examples": critical_path_narrative_focus_summary.get("upstream_driver_task_examples"),
        "downstream_finish_or_milestone_examples": critical_path_narrative_focus_summary.get("downstream_finish_or_milestone_examples"),
        "primary_branch_example": _compact_focus_path(current_primary_path_fact),
        "top_critical_activity_groups": [
            group
            for group in (_compact_focus_group(g) for g in (critical_activity_groups_summary.get("groups") or [])[:5])
            if group
        ],
        "path_changed_from_previous_update": path_change_payload_summary.get("changed"),
        "current_wbs_added": path_change_payload_summary.get("current_wbs_added"),
        "previous_wbs_removed": path_change_payload_summary.get("previous_wbs_removed"),
        "cause_assessment": supported_shift_evidence_payload_summary.get("cause_assessment"),
        "top_shift_causes": [
            cause
            for cause in (
                _compact_shift_cause(c) for c in (supported_shift_evidence_payload_summary.get("possible_shift_causes") or [])[:5]
            )
            if cause
        ],
        "constraint_drivers": (current_primary_path_fact or {}).get("constraint_drivers", [])[:3] if current_primary_path_fact else [],
        "change_delay_context": _compact_change_delay_context(change_delay_context_summary),
        "upstream_new_activity_links_to_current_critical_path": upstream_new_critical_context_summary,
    }
    near_critical_focus_summary = {
        "variance_threshold": settings_summary.get("variance_threshold"),
        "near_critical_count": near_critical_grouped_summary.get("count"),
        "near_critical_group_count": near_critical_grouped_summary.get("group_count"),
        "least_float_current_days": near_critical_grouped_summary.get("least_float_current_days"),
        "cutoff_current_days": near_critical_grouped_summary.get("cutoff_current_days"),
        "float_classification_source": near_critical_float_buckets_summary.get("classification_source"),
        "top_near_critical_groups": [
            group
            for group in (_compact_focus_group(g) for g in (near_critical_grouped_summary.get("groups") or [])[:5])
            if group
        ],
        "eroding_risk_count": eroding_risks_summary.get("count"),
        "top_eroding_risks": [
            risk
            for risk in (_compact_eroding_risk(item) for item in (eroding_risks_summary.get("items") or [])[:5])
            if risk
        ],
    }
    look_ahead_focus_summary = {
        "window": look_ahead_summary.get("window"),
        "top_upcoming_groups": [
            group
            for group in (
                _compact_focus_group(g) for g in (((look_ahead_summary.get("upcoming_work") or {}).get("groups") or [])[:5])
            )
            if group
        ],
        "top_schedule_sensitive_groups": [
            group
            for group in (
                _compact_focus_group(g)
                for g in (((look_ahead_summary.get("schedule_sensitive_upcoming_work") or {}).get("groups") or [])[:5])
            )
            if group
        ],
    }

    digest = {
        "report_sections": {
            "section_1_executive_summary_milestone_status": {
                "update_period": update_period_summary,
                "milestone_variance": {
                    **milestone_variance_summary,
                    "variance_reason": None,
                    "variance_reason_rule": "If variance_reason is null, do not infer a cause from dates or path movement.",
                },
            },
            "section_2_strategic_progress_achievements": {
                "period_context": {
                    "window": actualized_work_summary.get("window") or in_progress_finish_extensions_summary.get("window"),
                    "purpose": "Actualized work and in-progress finish extensions during the update period.",
                },
                "narrative_focus": {
                    "actualized_work": actualized_work_focus_summary,
                    "in_progress_finish_extensions": in_progress_finish_extensions_focus_summary,
                },
                "actualized_work": actualized_work_payload_summary,
                "in_progress_finish_extensions": in_progress_finish_extensions_payload_summary,
            },
            "section_3_scope_changes_new_additions": {
                "global_new_scope": global_new_scope_payload_summary,
                "change_delay_new_scope": change_delay_new_scope_payload_summary,
                "downstream_logic_indicators": downstream_logic_indicators_payload_summary,
            },
            "section_4_critical_path_risk": {
                "date_math_guardrail": date_math_guardrail_summary,
                "path_method": critical_path.get("method"),
                "narrative_focus": critical_path_narrative_focus_summary,
                "current_target_critical_path": {
                    "target_task_name": critical_path.get("target_task_name"),
                    "target_wbs_path": critical_path.get("target_wbs_path"),
                    "least_float_current_days": critical_path.get("least_float_current_days"),
                    "project_least_float_current_days": critical_path.get("project_least_float_current_days"),
                    "target_float_current_days": critical_path.get("target_float_current_days"),
                    "absolute_near_critical_threshold_days": critical_path.get("absolute_near_critical_threshold_days"),
                    "critical_count": critical_path.get("critical_count"),
                    "completed_critical_count": critical_path.get("completed_critical_count"),
                    "critical_trace_activity_count": len(critical_path.get("critical_trace_activity_ids") or []),
                    "critical_trace_bridge_count": critical_path.get("critical_trace_bridge_count"),
                    "critical_trace_bridges": [
                        bridge
                        for bridge in (
                            _trace_bridge_fact(item) for item in (critical_path.get("critical_trace_bridges") or [])
                        )
                        if bridge
                    ],
                    "precomputed_float_buckets": precomputed_float_buckets_summary,
                    "critical_status_focus": critical_status_focus_summary,
                    "critical_activities": critical_activity_facts_payload,
                    "critical_activity_payload_summary": {
                        "total_count": critical_activity_facts_total,
                        "shown_count": len(critical_activity_facts_payload),
                        "omitted_count": critical_activity_facts_omitted,
                    },
                    "completed_upstream_count": critical_path.get("completed_upstream_count"),
                    "completed_upstream_summary": completed_upstream_summary,
                    "path_count": critical_path.get("path_count"),
                    "critical_activity_groups": critical_activity_groups_payload_summary,
                    "primary_path": current_primary_path_fact,
                    "alternate_tied_paths": alternate_tied_path_examples,
                    "alternate_tied_paths_summary": alternate_tied_paths_summary,
                    "links": [_logic_link_fact(link) for link in (critical_path.get("links") or [])[:MAX_AI_LOGIC_LINKS]],
                    "links_summary": {
                        "total_count": len(critical_path.get("links") or []),
                        "shown_count": min(len(critical_path.get("links") or []), MAX_AI_LOGIC_LINKS),
                        "omitted_count": max(0, len(critical_path.get("links") or []) - MAX_AI_LOGIC_LINKS),
                    },
                },
                "path_change_from_previous_update": path_change_payload_summary,
                "supported_shift_evidence": supported_shift_evidence_payload_summary,
                "change_delay_context": _cap_named_lists(
                    change_delay_context_summary,
                    ["critical_path_successor_summaries", "cross_wbs_alerts"],
                ),
            },
            "section_5_risks_float_erosion": {
                "date_math_guardrail": date_math_guardrail_summary,
                "settings": settings_summary,
                "precomputed_float_buckets": near_critical_float_buckets_summary,
                "near_critical_grouped": near_critical_grouped_payload_summary,
                "eroding_risks": eroding_risks_payload_summary,
            },
            "section_6_look_ahead_window_analysis": {
                "settings": settings_summary,
                "look_ahead_window": look_ahead_payload_summary.get("window"),
                "upcoming_work": look_ahead_payload_summary.get("upcoming_work"),
                "schedule_sensitive_upcoming_work": look_ahead_payload_summary.get("schedule_sensitive_upcoming_work"),
                "warning": look_ahead_payload_summary.get("warning"),
            },
            "section_7_mitigation_inputs": {
                "critical_path_focus": critical_path_focus_summary,
                "near_critical_focus": near_critical_focus_summary,
                "look_ahead_focus": look_ahead_focus_summary,
            },
        },
    }
    if is_mspdi:
        digest["schedule_source"] = "mspdi_xml"
        progress_section = digest["report_sections"]["section_2_strategic_progress_achievements"]
        progress_section["period_context"]["period_assessment_available"] = msp_progress_summary.get(
            "period_assessment_available"
        )
        progress_section["period_context"]["warning"] = msp_progress_summary.get("warning") or accomplished.get(
            "warning"
        )
        progress_section["microsoft_project_progress_changes"] = msp_progress_summary
    return _without_activity_ids(digest)


def _main() -> int:
    p = argparse.ArgumentParser(description="Three-way schedule comparison (Baseline vs Last vs Current).")
    p.add_argument("baseline_xer", help="Path to baseline .XER or Microsoft Project .XML")
    p.add_argument("last_xer", help="Path to last update .XER or Microsoft Project .XML")
    p.add_argument("current_xer", help="Path to current .XER or Microsoft Project .XML")
    p.add_argument("--target-activity-id", required=True, help="Milestone/target activity (task_code/name/task_id)")
    p.add_argument("--variance-threshold", required=True, type=int, help="Near-critical threshold above least float")
    p.add_argument("--change-term", default="change", help="Term used to identify the Change/Delay WBS section")
    args = p.parse_args()

    baseline = snapshot_from_schedule_path("baseline", args.baseline_xer)
    last = snapshot_from_schedule_path("last", args.last_xer)
    current = snapshot_from_schedule_path("current", args.current_xer)

    out = compare_three_way(
        baseline,
        last,
        current,
        variance_threshold=args.variance_threshold,
        target_activity_id=args.target_activity_id,
        change_term=args.change_term,
    )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
