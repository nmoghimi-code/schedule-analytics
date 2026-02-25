from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from collections import defaultdict, deque
from typing import Any, Literal, Mapping

import pandas as pd

import xer_parser as xp


@dataclass(frozen=True)
class XerSnapshot:
    label: str
    project: pd.DataFrame
    task: pd.DataFrame
    taskpred: pd.DataFrame
    wbs: pd.DataFrame
    data_date: pd.Timestamp | None
    data_date_col: str | None


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


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
    if (wbs is None or wbs.empty) and (projwbs is not None and not projwbs.empty):
        wbs = projwbs

    # Apply the same enrichment as parse_xer does.
    task = xp.merge_wbs_names(task, wbs)
    task = xp.add_is_milestone(task)
    task = xp.add_full_wbs_path(task, wbs)

    data_date, data_date_col = _extract_project_data_date(project)
    return XerSnapshot(
        label=label,
        project=project,
        task=task,
        taskpred=taskpred,
        wbs=wbs,
        data_date=data_date,
        data_date_col=data_date_col,
    )


def snapshot_from_xer_path(label: str, xer_path: str | Path) -> XerSnapshot:
    tables = xp.read_xer_tables(xer_path, ["PROJECT", "TASK", "TASKPRED", "WBS", "PROJWBS"])
    return snapshot_from_tables(label, tables)


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


def _float_units_to_days(float_col: str, values: pd.Series) -> pd.Series:
    return xp.float_series_to_days(float_col, values)


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


def near_critical_ids(
    task_df: pd.DataFrame,
    variance_threshold: int,
    *,
    scope: Literal["project"] = "project",
) -> dict[str, Any]:
    if task_df is None or task_df.empty:
        return {"least_float": None, "cutoff": None, "float_col": None, "activity_ids": []}

    float_col = xp.detect_total_float_column(task_df)
    numeric = pd.to_numeric(task_df[float_col], errors="coerce")
    numeric_days = _float_units_to_days(float_col, numeric)
    lf_days = float(numeric_days.min(skipna=True))
    cutoff_days = lf_days + float(variance_threshold)

    activity_id_col = _pick_col(task_df, ["task_code", "activity_id"])
    if not activity_id_col:
        raise ValueError("TASK is missing an activity id column (expected 'task_code' or 'activity_id').")

    df = task_df.copy()
    df["_float_days"] = numeric_days
    df = df[df["_float_days"].le(cutoff_days)].copy()

    ids = df[activity_id_col].astype(str).tolist()
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)

    return {
        "float_col": float_col,
        "least_float_days": lf_days,
        "cutoff_days": cutoff_days,
        "activity_ids": out,
    }


def near_critical_trending(
    last: XerSnapshot,
    current: XerSnapshot,
    variance_threshold: int,
) -> dict[str, Any]:
    if last.data_date is None or current.data_date is None:
        days_between = None
    else:
        days_between = int((current.data_date.normalize() - last.data_date.normalize()).days)

    current_near = near_critical_ids(current.task, variance_threshold)
    float_col = current_near["float_col"]
    if not float_col:
        return {
            "days_between": days_between,
            "least_float_current_days": None,
            "cutoff_current_days": None,
            "near_critical_count": 0,
            "eroding_risks": [],
        }

    activity_id_col_curr = _pick_col(current.task, ["task_code", "activity_id"])
    activity_id_col_last = _pick_col(last.task, ["task_code", "activity_id"])
    if not activity_id_col_curr or not activity_id_col_last:
        raise ValueError("TASK missing activity id columns in one of the snapshots.")

    last_float_col = xp.detect_total_float_column(last.task)

    curr = current.task[[activity_id_col_curr, float_col]].copy()
    curr = curr.rename(columns={activity_id_col_curr: "activity_id", float_col: "float_current"})
    curr["float_current"] = pd.to_numeric(curr["float_current"], errors="coerce")

    prev = last.task[[activity_id_col_last, last_float_col]].copy()
    prev = prev.rename(columns={activity_id_col_last: "activity_id", last_float_col: "float_last"})
    prev["float_last"] = pd.to_numeric(prev["float_last"], errors="coerce")

    wanted = set(current_near["activity_ids"])
    curr = curr[curr["activity_id"].astype(str).isin(wanted)].copy()

    merged = curr.merge(prev, on="activity_id", how="left")
    # Convert to days for all downstream logic/output.
    float_current_days = _float_units_to_days(str(float_col), merged["float_current"])
    float_last_days = _float_units_to_days(str(last_float_col), merged["float_last"])
    float_change_days = float_current_days - float_last_days

    # Add story-friendly fields from CURRENT snapshot.
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])
    if name_col or wbs_name_col or wbs_path_col:
        cols = [activity_id_col_curr]
        if name_col:
            cols.append(name_col)
        if wbs_name_col:
            cols.append(wbs_name_col)
        if wbs_path_col:
            cols.append(wbs_path_col)
        details = current.task[cols].copy()
        details = details.rename(columns={activity_id_col_curr: "activity_id"})
        if name_col:
            details = details.rename(columns={name_col: "task_name"})
        if wbs_name_col:
            details = details.rename(columns={wbs_name_col: "wbs_name"})
        if wbs_path_col:
            details = details.rename(columns={wbs_path_col: "wbs_path"})
        details = details.drop_duplicates(subset=["activity_id"])
        merged = merged.merge(details, on="activity_id", how="left")

    loss_days = float_last_days - float_current_days

    if days_between is None:
        days_passed = None
        eroding_mask = pd.Series([False] * len(merged), index=merged.index)
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
                "float_change_days": (None if pd.isna(float_change_days.loc[idx]) else float(float_change_days.loc[idx])),
                "float_loss_days": (None if pd.isna(loss_days.loc[idx]) else float(loss_days.loc[idx])),
                "days_passed": days_passed,
                "eroding_risk": bool(eroding_mask.loc[idx]),
            }
        )

    eroding = [x for x in rows if x["eroding_risk"]]

    return {
        "days_between": days_between,
        "least_float_current_days": current_near["least_float_days"],
        "cutoff_current_days": current_near["cutoff_days"],
        "near_critical_count": int(len(rows)),
        "near_critical": rows,
        "eroding_risk_count": int(len(eroding)),
        "eroding_risks": eroding,
    }


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

    details: dict[str, dict[str, Any]] = {}
    if activity_id_col:
        cols = [activity_id_col]
        if name_col:
            cols.append(name_col)
        if wbs_name_col:
            cols.append(wbs_name_col)
        wbs_path_col = _pick_col(current.task, ["wbs_path"])
        if wbs_path_col:
            cols.append(wbs_path_col)
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
        if task_id_col:
            tmp = tmp.rename(columns={task_id_col: "task_id"})
        if wbs_id_col:
            tmp = tmp.rename(columns={wbs_id_col: "wbs_id"})
        if float_col:
            tmp = tmp.rename(columns={float_col: "total_float_raw"})
        tmp = tmp.drop_duplicates(subset=["activity_id"])
        for _, r in tmp.iterrows():
            tf_raw = None
            if "total_float_raw" in tmp.columns:
                tf_raw = pd.to_numeric(r.get("total_float_raw"), errors="coerce")
            tf_days = None if tf_raw is None or pd.isna(tf_raw) else float(xp.float_series_to_days(str(float_col), pd.Series([tf_raw])).iloc[0])
            details[str(r["activity_id"])] = {
                "activity_id": str(r["activity_id"]),
                "task_name": (None if "task_name" not in tmp.columns else str(r.get("task_name"))),
                "wbs_name": (None if "wbs_name" not in tmp.columns else str(r.get("wbs_name"))),
                "wbs_path": (None if "wbs_path" not in tmp.columns else str(r.get("wbs_path"))),
                "task_id": (None if "task_id" not in tmp.columns else str(r.get("task_id"))),
                "wbs_id": (None if "wbs_id" not in tmp.columns else str(r.get("wbs_id"))),
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
            numeric_days = xp.float_series_to_days(float_col, numeric)
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
                driving_delay_items[root_aid] = {"activity_id": root_aid, "task_name": root.get("task_name"), "wbs_path": root.get("wbs_path"), "total_float": root.get("total_float")}

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
                            "total_float": succ_info.get("total_float"),
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

    def _clean(val: Any) -> str | None:
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        return s

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


def critical_activities_all_wbs(current: XerSnapshot) -> dict[str, Any]:
    if current.task is None or current.task.empty:
        return {"count": 0, "items": [], "warning": "Current TASK is empty."}

    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    if not activity_id_col:
        return {"count": 0, "items": [], "warning": "Current TASK missing activity id column."}

    try:
        float_col = xp.detect_total_float_column(current.task)
    except Exception:
        float_col = None
    if not float_col:
        return {"count": 0, "items": [], "warning": "Current TASK missing total float column."}

    numeric = pd.to_numeric(current.task[float_col], errors="coerce")
    numeric_days = xp.float_series_to_days(float_col, numeric)
    if numeric_days.notna().sum() == 0:
        return {"count": 0, "items": [], "warning": "Total float values are not available."}

    least_float_current_days = float(numeric_days.min(skipna=True))
    eps = 1e-9
    mask = numeric_days.notna() & (numeric_days - least_float_current_days).abs().le(eps)

    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])

    cols = [activity_id_col, float_col]
    if name_col:
        cols.append(name_col)
    if wbs_name_col:
        cols.append(wbs_name_col)
    if wbs_path_col:
        cols.append(wbs_path_col)

    df = current.task.loc[mask, cols].copy()
    df = df.rename(columns={activity_id_col: "activity_id", float_col: "float_current_raw"})
    if name_col:
        df = df.rename(columns={name_col: "task_name"})
    if wbs_name_col:
        df = df.rename(columns={wbs_name_col: "wbs_name"})
    if wbs_path_col:
        df = df.rename(columns={wbs_path_col: "wbs_path"})

    df["activity_id"] = df["activity_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["activity_id"])

    def _clean(val: Any) -> str | None:
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        return s

    items: list[dict[str, Any]] = []
    for idx, r in df.iterrows():
        path = None if "wbs_path" not in df.columns else _clean(r.get("wbs_path"))
        if not path and "wbs_name" in df.columns:
            path = _clean(r.get("wbs_name"))

        parts: list[str] = []
        if path:
            parts = [p.strip() for p in str(path).split(" / ") if p.strip()]

        float_days = numeric_days.loc[idx]
        items.append(
            {
                "activity_id": str(r.get("activity_id")),
                "task_name": (None if "task_name" not in df.columns else _clean(r.get("task_name"))),
                "wbs_name": (None if "wbs_name" not in df.columns else _clean(r.get("wbs_name"))),
                "wbs_path": path,
                "wbs_hierarchy_root_to_leaf": parts,
                "wbs_hierarchy_leaf_to_root": list(reversed(parts)),
                "float_current_days": (None if pd.isna(float_days) else float(float_days)),
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
                "float_current_days": it.get("float_current_days"),
            }
        )

    groups = sorted(
        groups_by_leaf.values(),
        key=lambda g: (g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or ""),
    )

    return {
        "least_float_current_days": least_float_current_days,
        "count": int(len(items)),
        "group_count": int(len(groups)),
        "groups": groups,
        "items": items,
    }


def critical_path_to_target(current: XerSnapshot, target_activity_id: str) -> dict[str, Any]:
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
    numeric_days = xp.float_series_to_days(float_col, numeric)
    if numeric_days.notna().sum() == 0:
        return {"paths": [], "warning": "Total float values are not available."}

    least_float_current_days = float(numeric_days.min(skipna=True))
    eps = 1e-9

    # Map ids and names.
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])
    wbs_path_col = _pick_col(current.task, ["wbs_path"])

    cols = [task_id_col, activity_id_col, float_col]
    if name_col:
        cols.append(name_col)
    if wbs_name_col:
        cols.append(wbs_name_col)
    if wbs_path_col:
        cols.append(wbs_path_col)

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

    wbs_name_by_activity: dict[str, str] = {}
    if wbs_name_col:
        for aid, wnm in zip(tmp[activity_id_col].tolist(), tmp[wbs_name_col].astype(str).tolist(), strict=False):
            wbs_name_by_activity[str(aid)] = str(wnm)

    wbs_path_by_activity: dict[str, str] = {}
    if wbs_path_col:
        for aid, wpath in zip(tmp[activity_id_col].tolist(), tmp[wbs_path_col].astype(str).tolist(), strict=False):
            wbs_path_by_activity[str(aid)] = str(wpath)

    float_days_by_activity: dict[str, float] = {}
    vals_raw = pd.to_numeric(tmp[float_col], errors="coerce")
    vals_days = xp.float_series_to_days(float_col, vals_raw)
    for aid, v in zip(tmp[activity_id_col].tolist(), vals_days.tolist(), strict=False):
        if pd.isna(v):
            continue
        float_days_by_activity[str(aid)] = float(v)

    critical_ids = {aid for aid, v in float_days_by_activity.items() if abs(v - least_float_current_days) <= eps}

    # Resolve target activity.
    try:
        target_row = _resolve_target_row(current.task, target_activity_id)
    except Exception as e:
        return {"paths": [], "warning": f"Could not resolve target_activity_id: {e}"}

    target_aid = None if not activity_id_col else str(target_row.get(activity_id_col))
    target_tid = None if not task_id_col else str(target_row.get(task_id_col))
    if not target_aid and target_tid:
        target_aid = activity_by_task_id.get(target_tid)

    if not target_aid:
        return {"paths": [], "warning": "Could not resolve target activity id."}

    # Build predecessor mapping: succ_aid -> list[pred_aid] (critical only).
    succ_col = _pick_col(current.taskpred, ["task_id"])
    pred_col = _pick_col(current.taskpred, ["pred_task_id"])
    if not succ_col or not pred_col:
        return {"paths": [], "warning": "Current TASKPRED missing task_id/pred_task_id."}

    pred_by_succ: dict[str, set[str]] = defaultdict(set)
    dfp = current.taskpred[[succ_col, pred_col]].copy()
    dfp[succ_col] = dfp[succ_col].astype(str).str.strip()
    dfp[pred_col] = dfp[pred_col].astype(str).str.strip()
    for _, r in dfp.iterrows():
        succ_tid = str(r[succ_col]).strip()
        pred_tid = str(r[pred_col]).strip()
        if not succ_tid or not pred_tid or succ_tid.lower() == "nan" or pred_tid.lower() == "nan":
            continue
        succ_aid = activity_by_task_id.get(succ_tid)
        pred_aid = activity_by_task_id.get(pred_tid)
        if not succ_aid or not pred_aid:
            continue
        if pred_aid in critical_ids:
            pred_by_succ[str(succ_aid)].add(str(pred_aid))

    max_depth = 50
    max_paths = 5
    paths: list[list[str]] = []

    def dfs(cur_aid: str, path: list[str], visited: set[str]) -> None:
        if len(path) > max_depth:
            paths.append(path)
            return
        preds = sorted(pred_by_succ.get(cur_aid, set()))
        if not preds:
            paths.append(path)
            return
        for p in preds:
            if p in visited:
                continue
            dfs(p, [p] + path, visited | {p})

    dfs(str(target_aid), [str(target_aid)], {str(target_aid)})

    # Prefer longer chains (more critical activities).
    paths = sorted(paths, key=len, reverse=True)[:max_paths]

    def item_for(aid: str) -> dict[str, Any]:
        return {
            "activity_id": aid,
            "task_name": name_by_activity.get(aid),
            "wbs_path": wbs_path_by_activity.get(aid),
            "wbs_name": wbs_name_by_activity.get(aid),
            "is_critical": aid in critical_ids,
            "float_current_days": float_days_by_activity.get(aid),
        }

    out_paths = []
    links = []
    for p in paths:
        chain = [item_for(aid) for aid in p]
        out_paths.append({"length": len(p), "activity_chain": chain})
        for i in range(len(p) - 1):
            links.append({"from_activity_id": p[i], "to_activity_id": p[i + 1]})

    # De-duplicate links
    seen_links: set[tuple[str, str]] = set()
    dedup_links = []
    for l in links:
        key = (str(l.get("from_activity_id")), str(l.get("to_activity_id")))
        if key in seen_links:
            continue
        seen_links.add(key)
        dedup_links.append(l)

    return {
        "target_activity_id": target_aid,
        "target_task_name": name_by_activity.get(str(target_aid)),
        "target_wbs_path": wbs_path_by_activity.get(str(target_aid)),
        "least_float_current_days": least_float_current_days,
        "critical_count": int(len(critical_ids)),
        "path_count": int(len(out_paths)),
        "paths": out_paths,
        "links": dedup_links,
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
        numeric_days = xp.float_series_to_days(float_col, numeric)
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
        vals_days = xp.float_series_to_days(float_col, vals_raw)
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
        nm = name_by_activity.get(aid) or aid
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
                    "summary": f"{root_aid} (New) -> Impacts -> (No successors: missing task_id).",
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

    act_start_col, act_finish_col = _detect_actual_cols(current.task)
    if not act_start_col and not act_finish_col:
        return {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": 0, "activities": []}

    activity_id_col = _pick_col(current.task, ["task_code", "activity_id"])
    name_col = _pick_col(current.task, ["task_name", "task_title", "activity_name"])
    wbs_name_col = _pick_col(current.task, ["wbs_name"])

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

    return {"window": {"start": start.isoformat(), "end": end.isoformat()}, "count": int(len(activities)), "activities": activities}


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

    trending = near_critical_trending(last, current, variance_threshold)
    wbs = wbs_monitor_change_and_delay(last, current, term=change_term)
    critical_global = critical_activities_all_wbs(current)
    new_global = new_activities_all_wbs(last, current)
    accomplished = work_accomplished(last, current)
    critical_path = critical_path_to_target(current, target_activity_id)

    return {
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
        "critical_activities_global": critical_global,
        "new_activities_global": new_global,
        "work_accomplished": accomplished,
        "critical_path_to_target": critical_path,
    }


def get_ai_ready_digest(compare_result: Mapping[str, Any]) -> dict[str, Any]:
    """
    Produce a narrative-friendly digest:
      - removes technical database IDs (e.g., task_id)
      - keeps activity_id, task_name, and the key variances/trends
    """
    update_period = compare_result.get("update_period", {}) or {}
    milestone = compare_result.get("milestone", {}) or {}
    trending = compare_result.get("near_critical_trending", {}) or {}
    wbs = compare_result.get("wbs_monitor", {}) or {}
    critical_global = compare_result.get("critical_activities_global", {}) or {}
    new_global = compare_result.get("new_activities_global", {}) or {}
    accomplished = compare_result.get("work_accomplished", {}) or {}
    critical_path = compare_result.get("critical_path_to_target", {}) or {}
    settings = compare_result.get("settings", {}) or {}

    def ms_part(label: str) -> dict[str, Any]:
        part = milestone.get(label, {}) or {}
        return {
            "activity_id": part.get("activity_id"),
            "task_name": part.get("task_name"),
            "finish_date": part.get("finish_date"),
        }

    eroding = []
    for r in (trending.get("eroding_risks", []) or []):
        eroding.append(
            {
                "activity_id": r.get("activity_id"),
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
                "activity_id": r.get("activity_id"),
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
            }
        )

    cps = wbs.get("critical_path_successors", {}) or {}
    cps_items = []
    for r in (cps.get("items", []) or []):
        cps_items.append(
            {
                "root_activity_id": r.get("root_activity_id"),
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
                "from_activity_id": r.get("from_activity_id"),
                "to_activity_id": r.get("to_activity_id"),
                "to_task_name": r.get("to_task_name"),
                "to_wbs_top": r.get("to_wbs_top"),
            }
        )

    def _group_near_critical(src: Mapping[str, Any]) -> dict[str, Any]:
        items = src.get("near_critical", []) or []
        if not items:
            return {"count": 0, "group_count": 0, "groups": []}

        def _clean(val: Any) -> str | None:
            if val is None:
                return None
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass
            s = str(val).strip()
            if not s or s.lower() in {"nan", "none"}:
                return None
            return s

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
                    "activity_id": r.get("activity_id"),
                    "task_name": r.get("task_name"),
                    "float_last_days": r.get("float_last_days"),
                    "float_current_days": r.get("float_current_days"),
                    "float_change_days": r.get("float_change_days"),
                    "float_loss_days": r.get("float_loss_days"),
                    "days_passed": r.get("days_passed"),
                    "eroding_risk": r.get("eroding_risk"),
                }
            )

        groups = sorted(
            grouped.values(),
            key=lambda g: (g.get("leaf_wbs_path") or g.get("leaf_wbs_name") or ""),
        )
        return {"count": int(len(items)), "group_count": int(len(groups)), "groups": groups}

    new_global_items = []
    for r in (new_global.get("items", []) or []):
        new_global_items.append(
            {
                "activity_id": r.get("activity_id"),
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

    critical_items = []
    for r in (critical_global.get("items", []) or []):
        critical_items.append(
            {
                "activity_id": r.get("activity_id"),
                "task_name": r.get("task_name"),
                "wbs_name": r.get("wbs_name"),
                "wbs_path": r.get("wbs_path"),
                "wbs_hierarchy_leaf_to_root": r.get("wbs_hierarchy_leaf_to_root"),
                "wbs_hierarchy_root_to_leaf": r.get("wbs_hierarchy_root_to_leaf"),
                "float_current_days": r.get("float_current_days"),
            }
        )

    critical_groups = []
    for g in (critical_global.get("groups", []) or []):
        critical_groups.append(
            {
                "leaf_wbs_name": g.get("leaf_wbs_name"),
                "leaf_wbs_path": g.get("leaf_wbs_path"),
                "wbs_hierarchy_leaf_to_root": g.get("wbs_hierarchy_leaf_to_root"),
                "wbs_hierarchy_root_to_leaf": g.get("wbs_hierarchy_root_to_leaf"),
                "items": g.get("items"),
            }
        )

    return {
        "settings": {
            "variance_threshold": settings.get("variance_threshold"),
            "look_ahead_horizon_days": settings.get("look_ahead_horizon_days"),
            "change_term": settings.get("change_term"),
        },
        "update_period": {
            "baseline_data_date": update_period.get("baseline_data_date"),
            "last_data_date": update_period.get("last_data_date"),
            "current_data_date": update_period.get("current_data_date"),
            "days_last_to_current": update_period.get("days_last_to_current"),
        },
        "milestone_variance": {
            "target_activity_id": milestone.get("target_activity_id"),
            "baseline": ms_part("baseline"),
            "last": ms_part("last"),
            "current": ms_part("current"),
            "total_variance_days": milestone.get("total_variance_days"),
            "period_variance_days": milestone.get("period_variance_days"),
        },
        "eroding_risks": {
            "least_float_current_days": trending.get("least_float_current_days"),
            "cutoff_current_days": trending.get("cutoff_current_days"),
            "days_between": trending.get("days_between"),
            "count": trending.get("eroding_risk_count"),
            "items": eroding,
        },
        "near_critical_grouped": _group_near_critical(trending),
        "change_delay_wbs_new_activities": {
            "count": (len(new_change)),
            "items": new_change,
        },
        "change_impact": {
            "critical_path_successor_summaries": cps_items,
            "cross_wbs_alerts": cross_wbs_alerts,
        },
        "critical_activities_global": {
            "least_float_current_days": critical_global.get("least_float_current_days"),
            "count": critical_global.get("count"),
            "group_count": critical_global.get("group_count"),
            "warning": critical_global.get("warning"),
            "groups": critical_groups,
            "items": critical_items,
        },
        "new_activities_global": {
            "count": new_global.get("count"),
            "group_count": new_global.get("group_count"),
            "warning": new_global.get("warning"),
            "groups": new_global_groups,
            "items": new_global_items,
        },
        "work_accomplished": accomplished,
        "critical_path_to_target": critical_path,
    }


def _main() -> int:
    p = argparse.ArgumentParser(description="Three-way Primavera P6 XER comparison (Baseline vs Last vs Current).")
    p.add_argument("baseline_xer", help="Path to baseline .XER")
    p.add_argument("last_xer", help="Path to last update .XER")
    p.add_argument("current_xer", help="Path to current .XER")
    p.add_argument("--target-activity-id", required=True, help="Milestone/target activity (task_code/name/task_id)")
    p.add_argument("--variance-threshold", required=True, type=int, help="Near-critical threshold above least float")
    p.add_argument("--change-term", default="change", help="Term used to identify the Change/Delay WBS section")
    args = p.parse_args()

    baseline = snapshot_from_xer_path("baseline", args.baseline_xer)
    last = snapshot_from_xer_path("last", args.last_xer)
    current = snapshot_from_xer_path("current", args.current_xer)

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
