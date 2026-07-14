from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


P6_HOURS_PER_DAY = 8.0


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
        # pandas 3 keeps missing values as floating NaN in its string dtype,
        # including after astype(str). Normalize before using string methods.
        if pd.isna(v) or not str(v).strip() or str(v).strip().casefold() in {"nan", "none", "<na>"}:
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
