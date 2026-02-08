from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd


P6_HOURS_PER_DAY = 8.0


@dataclass(frozen=True)
class XerTables:
    task: pd.DataFrame
    taskpred: pd.DataFrame
    project: pd.DataFrame
    wbs: pd.DataFrame


def _finalize_table(
    tables: dict[str, pd.DataFrame],
    table_name: str | None,
    columns: list[str] | None,
    rows: list[list[str]],
    wanted: set[str],
) -> None:
    if not table_name or table_name not in wanted:
        return
    if not columns:
        tables[table_name] = pd.DataFrame()
        return
    normalized_rows: list[list[str]] = []
    width = len(columns)
    for row in rows:
        if len(row) < width:
            normalized_rows.append(row + [""] * (width - len(row)))
        elif len(row) > width:
            normalized_rows.append(row[:width])
        else:
            normalized_rows.append(row)
    tables[table_name] = pd.DataFrame.from_records(normalized_rows, columns=columns)


def read_xer_tables(xer_path: str | Path, table_names: Iterable[str]) -> dict[str, pd.DataFrame]:
    """
    Read selected tables from a Primavera P6 .XER file.

    XER is a tab-delimited text file where each table section typically contains:
      - %T <TABLE_NAME>
      - %F <col1> <col2> ...
      - %R <val1> <val2> ... (repeated)
    """
    wanted = {name.strip().upper() for name in table_names}
    tables: dict[str, pd.DataFrame] = {}

    current_table: str | None = None
    current_columns: list[str] | None = None
    current_rows: list[list[str]] = []

    xer_path = Path(xer_path)
    with xer_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            rec_type = parts[0].strip()

            if rec_type == "%T" and len(parts) >= 2:
                _finalize_table(tables, current_table, current_columns, current_rows, wanted)
                current_table = parts[1].strip().upper()
                current_columns = None
                current_rows = []
                continue

            if current_table not in wanted:
                continue

            if rec_type == "%F":
                current_columns = [c.strip() for c in parts[1:]]
                continue

            if rec_type == "%R":
                current_rows.append(parts[1:])
                continue

            if rec_type == "%E":
                break

    _finalize_table(tables, current_table, current_columns, current_rows, wanted)
    for name in wanted:
        tables.setdefault(name, pd.DataFrame())
    return tables


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _try_parse_p6_datetime(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Common P6 exports: '01-JAN-24', '01-JAN-2024', sometimes with time.
    formats = [
        "%d-%b-%y",
        "%d-%b-%Y",
        "%d-%b-%y %H:%M",
        "%d-%b-%Y %H:%M",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        ts = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts)

    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(ts):
        return pd.Timestamp(ts)
    return None


def identify_data_date(
    project_df: pd.DataFrame,
    task_df: pd.DataFrame | None = None,
    project_hint: str | None = None,
) -> str | None:
    """
    Identify the schedule Data Date from the PROJECT table.

    Returns an ISO-8601 string (date or datetime) when possible, else None.
    """
    if project_df is None or project_df.empty:
        return None

    def find_col(possible: list[str]) -> str | None:
        cols_lower = {c.lower(): c for c in project_df.columns}
        for cand in possible:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    data_date_col = find_col(["data_date", "datadate", "last_recalc_date", "lastrecalcdate"])
    if data_date_col is None:
        for c in project_df.columns:
            if "data" in c.lower() and "date" in c.lower():
                data_date_col = c
                break
    if data_date_col is None:
        return None

    project_row: pd.Series
    if len(project_df) == 1:
        project_row = project_df.iloc[0]
    else:
        if project_hint:
            hint = project_hint.strip()
            for col in ["proj_short_name", "proj_name", "proj_id", "project_id"]:
                if col in project_df.columns:
                    match = project_df[project_df[col].astype(str) == hint]
                    if len(match) == 1:
                        project_row = match.iloc[0]
                        break
            else:
                project_row = project_df.iloc[0]
        elif task_df is not None and not task_df.empty and "proj_id" in task_df.columns and "proj_id" in project_df.columns:
            dominant_proj_id = task_df["proj_id"].astype(str).mode(dropna=True)
            if len(dominant_proj_id) > 0:
                dominant_proj_id = dominant_proj_id.iloc[0]
                match = project_df[project_df["proj_id"].astype(str) == str(dominant_proj_id)]
                project_row = match.iloc[0] if len(match) > 0 else project_df.iloc[0]
            else:
                project_row = project_df.iloc[0]
        else:
            project_row = project_df.iloc[0]

    ts = _try_parse_p6_datetime(project_row.get(data_date_col))
    return ts.isoformat() if ts is not None else str(project_row.get(data_date_col)).strip() or None


def merge_wbs_names(task_df: pd.DataFrame, wbs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge WBS names into TASK using wbs_id so each task has a 'wbs_name' column (when possible).
    """
    if task_df is None or task_df.empty or wbs_df is None or wbs_df.empty:
        return task_df

    task_wbs_id_col = _pick_col(task_df, ["wbs_id"])
    wbs_id_col = _pick_col(wbs_df, ["wbs_id"])
    if not task_wbs_id_col or not wbs_id_col:
        return task_df

    wbs_name_col = _pick_col(wbs_df, ["wbs_name", "wbs_short_name", "wbs_code"])
    if not wbs_name_col:
        for c in wbs_df.columns:
            lc = c.lower()
            if "wbs" in lc and "name" in lc:
                wbs_name_col = c
                break
    if not wbs_name_col:
        return task_df

    # Force string dtype + strip whitespace on join keys (explicit, per requirement).
    task_df = task_df.copy()
    wbs_df = wbs_df.copy()
    task_df[task_wbs_id_col] = task_df[task_wbs_id_col].astype(str).str.strip()
    wbs_df[wbs_id_col] = wbs_df[wbs_id_col].astype(str).str.strip()

    def _norm_key(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        # common mismatch when something was treated as numeric elsewhere: "100.0" vs "100"
        s = s.str.replace(r"\.0$", "", regex=True)
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
        return s

    mapping = wbs_df[[wbs_id_col, wbs_name_col]].copy()
    mapping["_wbs_id_key"] = _norm_key(mapping[wbs_id_col])
    mapping["wbs_name"] = mapping[wbs_name_col].astype(str).str.strip()
    mapping = mapping.dropna(subset=["_wbs_id_key"]).drop_duplicates(subset=["_wbs_id_key"])[["_wbs_id_key", "wbs_name"]]

    task_with_key = task_df.copy()
    task_with_key["_wbs_id_key"] = _norm_key(task_with_key[task_wbs_id_col])

    if "wbs_name" in task_df.columns:
        mapping2 = mapping.rename(columns={"wbs_name": "_wbs_name_ref"})
        merged = task_with_key.merge(mapping2, how="left", on="_wbs_id_key")
        merged.drop(columns=["_wbs_id_key"], inplace=True, errors="ignore")
        orig = merged["wbs_name"]
        non_empty = orig.notna() & (orig.astype(str).str.strip() != "")
        merged["wbs_name"] = orig.where(non_empty, merged["_wbs_name_ref"])
        merged["wbs_name"] = merged["wbs_name"].fillna(merged["_wbs_name_ref"])
        merged.drop(columns=["_wbs_name_ref"], inplace=True, errors="ignore")
        return merged

    merged = task_with_key.merge(mapping, how="left", on="_wbs_id_key")
    merged.drop(columns=["_wbs_id_key"], inplace=True, errors="ignore")
    return merged


def add_full_wbs_path(
    task_df: pd.DataFrame,
    wbs_df: pd.DataFrame,
    *,
    path_col: str = "wbs_path",
    sep: str = " / ",
) -> pd.DataFrame:
    """
    Add a 'full WBS path' column to TASK using WBS hierarchy (parent_wbs_id) when available.

    This helps downstream logic (e.g., searching for "change") operate on a stable string even when
    only ancestor WBS names contain the term.
    """
    if task_df is None or task_df.empty:
        return task_df
    if wbs_df is None or wbs_df.empty:
        df = task_df.copy()
        if path_col not in df.columns:
            df[path_col] = df.get("wbs_name", pd.Series([""] * len(df))).astype(str)
        return df

    task_wbs_id_col = _pick_col(task_df, ["wbs_id"])
    wbs_id_col = _pick_col(wbs_df, ["wbs_id"])
    if not task_wbs_id_col or not wbs_id_col:
        df = task_df.copy()
        if path_col not in df.columns:
            df[path_col] = df.get("wbs_name", pd.Series([""] * len(df))).astype(str)
        return df

    wbs_name_col = _pick_col(wbs_df, ["wbs_name", "wbs_short_name", "wbs_code"])
    if not wbs_name_col:
        df = task_df.copy()
        if path_col not in df.columns:
            df[path_col] = df.get("wbs_name", pd.Series([""] * len(df))).astype(str)
        return df

    parent_wbs_id_col = _pick_col(wbs_df, ["parent_wbs_id", "par_wbs_id"])
    if not parent_wbs_id_col:
        for c in wbs_df.columns:
            lc = c.lower()
            if "parent" in lc and "wbs" in lc and lc.endswith("_id"):
                parent_wbs_id_col = c
                break

    def norm_id(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
        return s

    w = wbs_df.copy()
    w["_wid"] = norm_id(w[wbs_id_col])
    w["_wname"] = w[wbs_name_col].astype(str).str.strip()
    if parent_wbs_id_col:
        w["_pid"] = norm_id(w[parent_wbs_id_col])
    else:
        w["_pid"] = pd.NA
    w = w.dropna(subset=["_wid"]).drop_duplicates(subset=["_wid"])

    name_by_id = dict(zip(w["_wid"].astype(str), w["_wname"].astype(str), strict=False))
    parent_by_id = dict(zip(w["_wid"].astype(str), w["_pid"].astype(str), strict=False))
    for k, v in list(parent_by_id.items()):
        if not v or v.lower() == "nan":
            parent_by_id[k] = ""

    path_cache: dict[str, str] = {}

    def build_path(wid: str) -> str:
        wid = str(wid).strip()
        if wid in path_cache:
            return path_cache[wid]
        parts: list[str] = []
        seen: set[str] = set()
        cur: str = wid
        while cur and cur not in seen:
            seen.add(cur)
            nm = name_by_id.get(cur, "")
            if nm:
                parts.append(nm)
            cur = parent_by_id.get(cur, "") or ""
        parts.reverse()
        out = sep.join(parts)
        path_cache[wid] = out
        return out

    df = task_df.copy()
    df["_wid"] = norm_id(df[task_wbs_id_col])
    df[path_col] = df["_wid"].astype(str).map(build_path)
    df.drop(columns=["_wid"], inplace=True, errors="ignore")

    # If no hierarchy path was found, fall back to mapped leaf name.
    if "wbs_name" in df.columns:
        df[path_col] = df[path_col].where(df[path_col].astype(str).str.strip() != "", df["wbs_name"].astype(str))
    return df


def add_is_milestone(task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean 'is_milestone' to TASK where task_type is TT_FinMile or TT_StartMile.
    """
    if task_df is None or task_df.empty:
        return task_df
    task_type_col = _pick_col(task_df, ["task_type"])
    if not task_type_col:
        task_df = task_df.copy()
        task_df["is_milestone"] = False
        return task_df

    df = task_df.copy()
    df["is_milestone"] = df[task_type_col].astype(str).isin(["TT_FinMile", "TT_StartMile"])
    return df


def activity_ids_in_wbs(
    task_df: pd.DataFrame,
    wbs_df: pd.DataFrame,
    wbs_term: str,
    *,
    match: Literal["contains", "exact"] = "contains",
    highest_hierarchy: bool = True,
    include_children: bool = True,
    case_sensitive: bool = False,
) -> list[str]:
    """
    Return a list of Activity IDs inside a WBS section identified by a term.

    This is designed to avoid a common bug: when both a top-level WBS and its child WBS nodes
    contain the same term (e.g., "Change"), we select only the *highest* matching WBS node(s),
    then include all descendant WBS nodes to capture the whole section.
    """
    if task_df is None or task_df.empty:
        return []
    if wbs_df is None or wbs_df.empty:
        return []

    task_wbs_id_col = _pick_col(task_df, ["wbs_id"])
    if not task_wbs_id_col:
        raise ValueError("TASK is missing 'wbs_id'; cannot filter by WBS section.")

    activity_id_col = _pick_col(task_df, ["task_code", "activity_id"])
    if not activity_id_col:
        raise ValueError("TASK is missing an activity id column (expected 'task_code' or 'activity_id').")

    wbs_id_col = _pick_col(wbs_df, ["wbs_id"])
    if not wbs_id_col:
        raise ValueError("WBS is missing 'wbs_id'; cannot build hierarchy.")

    parent_wbs_id_col = _pick_col(wbs_df, ["parent_wbs_id", "par_wbs_id"])
    if not parent_wbs_id_col:
        for c in wbs_df.columns:
            lc = c.lower()
            if "parent" in lc and "wbs" in lc and lc.endswith("_id"):
                parent_wbs_id_col = c
                break

    wbs_name_col = _pick_col(wbs_df, ["wbs_name", "wbs_short_name", "wbs_code"])
    if not wbs_name_col:
        raise ValueError("WBS is missing a name column (expected 'wbs_name' or similar).")

    needle = str(wbs_term).strip()
    if not needle:
        return []

    names = wbs_df[wbs_name_col].astype(str).str.strip()
    if not case_sensitive:
        needle_cmp = needle.casefold()
        names_cmp = names.str.casefold()
    else:
        needle_cmp = needle
        names_cmp = names

    if match == "exact":
        matched = wbs_df[names_cmp == needle_cmp]
    else:
        matched = wbs_df[names_cmp.str.contains(needle_cmp, na=False)]

    matched_ids: set[str] = set(matched[wbs_id_col].astype(str).str.strip().tolist())
    matched_ids.discard("")
    matched_ids.discard("nan")
    if not matched_ids:
        return []

    parent_by_id: dict[str, str | None] = {}
    if parent_wbs_id_col:
        for _, row in wbs_df[[wbs_id_col, parent_wbs_id_col]].iterrows():
            wid = str(row[wbs_id_col]).strip()
            pid = str(row[parent_wbs_id_col]).strip()
            if not wid or wid.lower() == "nan":
                continue
            if not pid or pid.lower() == "nan":
                parent_by_id[wid] = None
            else:
                parent_by_id[wid] = pid
    else:
        for wid in wbs_df[wbs_id_col].astype(str).str.strip().tolist():
            if wid and wid.lower() != "nan":
                parent_by_id[wid] = None

    highest_ids: set[str]
    if highest_hierarchy and parent_by_id:
        highest_ids = set()
        for wid in matched_ids:
            current = parent_by_id.get(wid)
            while current:
                if current in matched_ids:
                    break
                current = parent_by_id.get(current)
            else:
                highest_ids.add(wid)
    else:
        highest_ids = matched_ids

    wbs_ids_to_include: set[str]
    if include_children:
        children_by_parent: dict[str | None, set[str]] = defaultdict(set)
        for wid, pid in parent_by_id.items():
            children_by_parent[pid].add(wid)

        wbs_ids_to_include = set()
        q: deque[str] = deque(sorted(highest_ids))
        while q:
            cur = q.popleft()
            if cur in wbs_ids_to_include:
                continue
            wbs_ids_to_include.add(cur)
            for child in children_by_parent.get(cur, set()):
                if child not in wbs_ids_to_include:
                    q.append(child)
    else:
        wbs_ids_to_include = highest_ids

    task_wbs_ids = task_df[task_wbs_id_col].astype(str).str.strip()
    mask = task_wbs_ids.isin(wbs_ids_to_include)

    ids = task_df.loc[mask, activity_id_col].astype(str).tolist()
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def activity_ids_in_change_wbs(task_df: pd.DataFrame, wbs_df: pd.DataFrame, *, term: str = "change") -> list[str]:
    """
    Convenience wrapper to identify the highest-level WBS section containing 'change' and return all Activity IDs.
    """
    return activity_ids_in_wbs(
        task_df,
        wbs_df,
        term,
        match="contains",
        highest_hierarchy=True,
        include_children=True,
        case_sensitive=False,
    )


def detect_total_float_column(task_df: pd.DataFrame) -> str:
    if task_df is None or task_df.empty:
        raise ValueError("TASK table is empty; cannot detect a float column.")

    candidates = [
        "total_float_hr_cnt",
        "total_float",
        "total_float_cnt",
        "total_float_day_cnt",
        "total_float_days_cnt",
        "float_total",
    ]
    cols_lower = {c.lower(): c for c in task_df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    float_like = [c for c in task_df.columns if "float" in c.lower()]
    if not float_like:
        raise ValueError(
            "Could not find a total float column in TASK. Expected something like 'total_float_hr_cnt'."
        )

    # Pick the float-like column that looks most numeric.
    best_col = float_like[0]
    best_numeric_ratio = -1.0
    for c in float_like:
        numeric = pd.to_numeric(task_df[c], errors="coerce")
        ratio = float(numeric.notna().mean()) if len(task_df) else 0.0
        if ratio > best_numeric_ratio:
            best_col = c
            best_numeric_ratio = ratio
    return best_col


def float_series_to_days(float_col: str, values: pd.Series, *, hours_per_day: float = P6_HOURS_PER_DAY) -> pd.Series:
    """
    Convert a P6 total-float series to days when the source column is in hours.
    """
    col_lc = str(float_col).lower()
    if "hr" in col_lc:
        return values / float(hours_per_day)
    return values


def least_float(task_df: pd.DataFrame, float_col: str | None = None) -> float:
    """
    Return the minimum (least) float value in the TASK table (in DAYS), including negatives.
    """
    if task_df is None or task_df.empty:
        raise ValueError("TASK table is empty; cannot compute least float.")
    float_col = float_col or detect_total_float_column(task_df)
    values = pd.to_numeric(task_df[float_col], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"Column '{float_col}' could not be converted to numeric.")
    values_days = float_series_to_days(float_col, values)
    return float(values_days.min(skipna=True))


def _resolve_target_task_id(task_df: pd.DataFrame, target_activity_id: str) -> str:
    if task_df is None or task_df.empty:
        raise ValueError("TASK table is empty; cannot resolve target activity.")

    target = str(target_activity_id).strip()
    if not target:
        raise ValueError("target_activity_id is empty.")

    cols_lower = {c.lower(): c for c in task_df.columns}
    internal_id_col = cols_lower.get("task_id")
    activity_id_col = cols_lower.get("task_code") or cols_lower.get("activity_id")
    name_col = cols_lower.get("task_name") or cols_lower.get("task_title") or cols_lower.get("activity_name")

    if activity_id_col and internal_id_col:
        match = task_df[task_df[activity_id_col].astype(str) == target]
        if len(match) == 1:
            return str(match.iloc[0][internal_id_col])
        if len(match) > 1:
            raise ValueError(f"Multiple TASK rows match {activity_id_col}='{target}'.")

    if internal_id_col:
        match = task_df[task_df[internal_id_col].astype(str) == target]
        if len(match) == 1:
            return str(match.iloc[0][internal_id_col])

    if name_col and internal_id_col:
        normalized = task_df[name_col].astype(str).str.strip()
        match = task_df[normalized.str.casefold() == target.casefold()]
        if len(match) == 1:
            return str(match.iloc[0][internal_id_col])
        if len(match) > 1:
            raise ValueError(f"Multiple TASK rows match {name_col}='{target}'.")

    if internal_id_col:
        raise ValueError(
            f"Could not find target '{target}' in TASK using "
            f"{activity_id_col or 'task_code'} (Activity ID), {internal_id_col} (internal id)"
            f"{f', or {name_col} (Activity Name)' if name_col else ''}."
        )
    raise ValueError("TASK is missing 'task_id'; cannot traverse dependencies reliably.")


def upstream_task_ids(
    taskpred_df: pd.DataFrame, target_task_id: str, max_nodes: int = 500_000
) -> set[str]:
    """
    Return the set of upstream (predecessor) task_id values that can reach target_task_id.
    """
    if taskpred_df is None or taskpred_df.empty:
        return {str(target_task_id)}

    cols_lower = {c.lower(): c for c in taskpred_df.columns}
    succ_col = cols_lower.get("task_id")
    pred_col = cols_lower.get("pred_task_id")
    if not succ_col or not pred_col:
        # Some exports can use different naming; best-effort fallback.
        possible_succ = [c for c in taskpred_df.columns if c.lower().endswith("task_id") and "pred" not in c.lower()]
        possible_pred = [c for c in taskpred_df.columns if "pred" in c.lower() and c.lower().endswith("task_id")]
        succ_col = succ_col or (possible_succ[0] if possible_succ else None)
        pred_col = pred_col or (possible_pred[0] if possible_pred else None)
    if not succ_col or not pred_col:
        return {str(target_task_id)}

    preds_by_succ: dict[str, set[str]] = defaultdict(set)
    for _, row in taskpred_df[[succ_col, pred_col]].iterrows():
        succ = str(row[succ_col]).strip()
        pred = str(row[pred_col]).strip()
        if succ and pred and succ.lower() != "nan" and pred.lower() != "nan":
            preds_by_succ[succ].add(pred)

    visited: set[str] = set()
    q: deque[str] = deque([str(target_task_id)])
    while q:
        current = q.popleft()
        if current in visited:
            continue
        visited.add(current)
        if len(visited) > max_nodes:
            raise RuntimeError(f"Upstream traversal exceeded {max_nodes} nodes; possible cycle or malformed data.")
        for pred in preds_by_succ.get(current, set()):
            if pred not in visited:
                q.append(pred)
    return visited


def flag_near_critical(
    task_df: pd.DataFrame,
    taskpred_df: pd.DataFrame,
    target_activity_id: str,
    variance_threshold: float,
    *,
    float_col: str | None = None,
    scope: Literal["network", "project"] = "network",
) -> dict[str, Any]:
    """
    Flag near-critical activities based on:
      - target_activity_id: an Activity ID (task_code) or internal task_id
      - variance_threshold: value added above the least float
      - scope:
          - 'network': compute least float within target's upstream network
          - 'project': compute least float across all tasks
    """
    float_col = float_col or detect_total_float_column(task_df)
    target_task_id = _resolve_target_task_id(task_df, target_activity_id)
    upstream_ids = upstream_task_ids(taskpred_df, target_task_id)

    id_col = next((c for c in task_df.columns if c.lower() == "task_id"), None)
    if id_col is None:
        raise ValueError("TASK table missing 'task_id'.")

    df = task_df.copy()
    df["_total_float_numeric"] = pd.to_numeric(df[float_col], errors="coerce")
    df["_total_float_days"] = float_series_to_days(float_col, df["_total_float_numeric"])

    if scope == "network":
        scope_df = df[df[id_col].astype(str).isin(upstream_ids)]
    else:
        scope_df = df

    if scope_df.empty:
        raise ValueError("No TASK rows found for the selected scope; cannot compute least float / near-critical set.")
    if scope_df["_total_float_days"].notna().sum() == 0:
        raise ValueError(f"Float column '{float_col}' has no numeric values in the selected scope.")

    lf_days = float(scope_df["_total_float_days"].min(skipna=True))
    near_cutoff_days = lf_days + float(variance_threshold)

    near_df = scope_df[scope_df["_total_float_days"].le(near_cutoff_days)].copy()
    near_df.sort_values(by="_total_float_days", ascending=True, inplace=True)

    def pick(col_names: list[str]) -> str | None:
        cols_lower = {c.lower(): c for c in df.columns}
        for name in col_names:
            if name.lower() in cols_lower:
                return cols_lower[name.lower()]
        return None

    activity_id_col = pick(["task_code", "activity_id"])
    name_col = pick(["task_name", "task_title", "activity_name"])
    wbs_name_col = pick(["wbs_name"])

    keep_cols = [id_col]
    if activity_id_col:
        keep_cols.append(activity_id_col)
    if name_col:
        keep_cols.append(name_col)
    if wbs_name_col:
        keep_cols.append(wbs_name_col)
    if float_col not in keep_cols:
        keep_cols.append(float_col)

    activities = []
    for _, r in near_df[keep_cols + ["_total_float_days"]].iterrows():
        activities.append(
            {
                "task_id": str(r[id_col]),
                "activity_id": (str(r[activity_id_col]) if activity_id_col else None),
                "task_name": (str(r[name_col]) if name_col else None),
                "wbs_name": (str(r[wbs_name_col]) if wbs_name_col else None),
                "float_col": float_col,
                "total_float_days": (None if pd.isna(r["_total_float_days"]) else float(r["_total_float_days"])),
                "near_critical": True,
            }
        )

    return {
        "target_activity_id": str(target_activity_id),
        "target_task_id": str(target_task_id),
        "scope": scope,
        "float_column": float_col,
        "least_float_days": lf_days,
        "variance_threshold": float(variance_threshold),
        "near_critical_cutoff_days": near_cutoff_days,
        "upstream_task_count": int(len(upstream_ids)),
        "near_critical_activities": activities,
    }


def parse_xer(
    xer_path: str | Path,
    *,
    target_activity_id: str,
    variance_threshold: float,
    project_hint: str | None = None,
    scope: Literal["network", "project"] = "network",
    include_tables: Literal["none", "records"] = "none",
) -> dict[str, Any]:
    tables = read_xer_tables(xer_path, ["TASK", "TASKPRED", "PROJECT", "WBS", "PROJWBS"])
    task = tables["TASK"]
    taskpred = tables["TASKPRED"]
    project = tables["PROJECT"]
    wbs = tables["WBS"]
    projwbs = tables["PROJWBS"]
    if (wbs is None or wbs.empty) and (projwbs is not None and not projwbs.empty):
        wbs = projwbs

    task = merge_wbs_names(task, wbs)
    task = add_is_milestone(task)
    task = add_full_wbs_path(task, wbs)

    data_date = identify_data_date(project, task_df=task, project_hint=project_hint)
    float_col = detect_total_float_column(task) if not task.empty else None
    global_least_days = least_float(task, float_col=float_col) if not task.empty else None

    near = flag_near_critical(
        task,
        taskpred,
        target_activity_id=target_activity_id,
        variance_threshold=variance_threshold,
        float_col=float_col,
        scope=scope,
    )

    result: dict[str, Any] = {
        "xer_path": str(Path(xer_path)),
        "tables_read": {
            "PROJECT_rows": int(len(project)),
            "TASK_rows": int(len(task)),
            "TASKPRED_rows": int(len(taskpred)),
            "WBS_rows": int(len(tables["WBS"])),
            "PROJWBS_rows": int(len(tables["PROJWBS"])),
            "WBS_used": ("WBS" if (tables["WBS"] is not None and not tables["WBS"].empty) else ("PROJWBS" if (tables["PROJWBS"] is not None and not tables["PROJWBS"].empty) else None)),
        },
        "data_date": data_date,
        "global_least_float_days": global_least_days,
        "near_critical": near,
    }

    if include_tables == "records":
        result["tables"] = {
            "PROJECT": project.to_dict(orient="records"),
            "TASK": task.to_dict(orient="records"),
            "TASKPRED": taskpred.to_dict(orient="records"),
            "WBS": tables["WBS"].to_dict(orient="records"),
            "PROJWBS": tables["PROJWBS"].to_dict(orient="records"),
        }

    return result


def _main() -> int:
    p = argparse.ArgumentParser(description="Parse a Primavera P6 .XER file and flag near-critical activities.")
    p.add_argument("xer_path", help="Path to the .XER file")
    p.add_argument("--target-activity-id", required=True, help="Activity ID (task_code) or internal task_id")

    def _float_or_launchjson_hint(value: str) -> float:
        try:
            return float(value)
        except ValueError as e:
            if "${input:" in value:
                raise argparse.ArgumentTypeError(
                    "Got an unresolved VS Code launch.json input like '${input:...}'. "
                    "Select the 'Run XER Parser' debug configuration (it should prompt), "
                    "or use 'Run XER Parser (No Prompts)' and edit args directly."
                ) from e
            raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from e

    p.add_argument(
        "--variance-threshold",
        required=True,
        type=_float_or_launchjson_hint,
        help="Range above least float (in DAYS) to flag near-critical activities",
    )
    p.add_argument(
        "--scope",
        choices=["network", "project"],
        default="network",
        help="Use least float from target's upstream network ('network') or whole project ('project')",
    )
    p.add_argument(
        "--project-hint",
        default=None,
        help="Optional project identifier to select the correct PROJECT row when multiple exist",
    )
    p.add_argument(
        "--include-tables",
        choices=["none", "records"],
        default="none",
        help="Include raw tables in output (can be large)",
    )
    args = p.parse_args()

    if isinstance(args.xer_path, str) and "${input:" in args.xer_path:
        raise SystemExit(
            "Got an unresolved VS Code launch.json input for xer_path like '${input:...}'. "
            "Select the 'Run XER Parser' debug configuration (it should prompt), "
            "or use 'Run XER Parser (No Prompts)' and edit args directly."
        )

    result = parse_xer(
        args.xer_path,
        target_activity_id=args.target_activity_id,
        variance_threshold=args.variance_threshold,
        project_hint=args.project_hint,
        scope=args.scope,
        include_tables=args.include_tables,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
