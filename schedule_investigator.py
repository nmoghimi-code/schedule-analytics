from __future__ import annotations

"""
Single-XER schedule investigation.

Produces a "what is this project" overview plus the schedule's main longest paths
(backbones), for a one-file investigation tab. Reuses xer_comparator helpers so the
parsing, WBS paths, calendars, and critical-path logic stay consistent with the other tools.

Correctness note on "longest path":
    We do NOT sum activity durations. With SS/FF/SF relationships, lags, mixed calendars,
    and constraints, durations do not add. P6 already ran the full CPM forward/backward pass
    and stored the answer as computed dates and total float. We therefore trace the *driving
    (binding) predecessor* using those dates + the relationship type + lag, and measure path
    length as the real calendar span. The result is cross-checked against the float-based
    critical path for the incomplete tail.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

import xer_comparator as xc
import xer_parser as xp


def _iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else ts.isoformat()


# Priority order: actual dates first (so completed/in-progress work uses real dates),
# then forecast/remaining, then planned/target.
_START_COLS = ["act_start_date", "restart_date", "early_start_date", "target_start_date", "start_date"]
_FINISH_COLS = ["act_end_date", "reend_date", "early_end_date", "target_end_date", "end_date", "finish_date"]


def _first_date(row: pd.Series, cols: Sequence[str]) -> pd.Timestamp | None:
    for col in cols:
        if col in row.index:
            ts = xc._parse_date(row.get(col))
            if ts is not None:
                return ts
    return None


def _derive_status(row: pd.Series, status_col: str | None, act_start: pd.Timestamp | None, act_finish: pd.Timestamp | None) -> str:
    status = str(row.get(status_col, "")).casefold() if status_col else ""
    if act_finish is not None or "complete" in status:
        return "completed"
    if act_start is not None or "active" in status:
        return "in_progress"
    return "not_started"


def _activity_records(snapshot: xc.XerSnapshot) -> dict[str, dict[str, Any]]:
    """aid -> {name, wbs_path, status, start, finish, task_type, is_milestone, constraint_*}."""
    t = snapshot.task
    if t is None or t.empty:
        return {}
    aid_col = xp._pick_col(t, ["task_code", "activity_id"])
    name_col = xp._pick_col(t, ["task_name", "task_title", "activity_name"])
    wbs_path_col = xp._pick_col(t, ["wbs_path"])
    wbs_name_col = xp._pick_col(t, ["wbs_name"])
    status_col = xp._pick_col(t, ["status_code", "task_status"])
    type_col = xp._pick_col(t, ["task_type", "task_type_code", "activity_type"])
    act_start_col, act_finish_col = xc._detect_actual_cols(t)
    cstr_type_col = xp._pick_col(t, ["cstr_type", "constraint_type", "primary_constraint_type"])
    cstr_date_col = xp._pick_col(t, ["cstr_date", "constraint_date", "primary_constraint_date"])
    if not aid_col:
        return {}

    # Total float (days) per activity, for marking the current critical backbone.
    float_by_aid: dict[str, float] = {}
    try:
        float_col = xp.detect_total_float_column(t)
        fvals = xc._task_float_series_to_days(t, float_col, pd.to_numeric(t[float_col], errors="coerce"), snapshot.calendar)
        for aid_v, fv in zip(t[aid_col].astype(str).str.strip().tolist(), fvals.tolist(), strict=False):
            if not pd.isna(fv):
                float_by_aid[aid_v] = float(fv)
    except Exception:
        pass

    out: dict[str, dict[str, Any]] = {}
    for _, row in t.iterrows():
        aid = str(row.get(aid_col, "")).strip()
        if not aid or aid.lower() == "nan" or aid in out:
            continue
        task_type = str(row.get(type_col, "")).strip().casefold() if type_col else ""
        # Exclude level-of-effort / summary rows from the path network.
        if any(tok in task_type for tok in ["loe", "level of effort", "wbs", "summary"]):
            continue
        act_start = xc._parse_date(row.get(act_start_col)) if act_start_col else None
        act_finish = xc._parse_date(row.get(act_finish_col)) if act_finish_col else None
        out[aid] = {
            "activity_id": aid,
            "task_name": (None if not name_col else xc._clean_optional(row.get(name_col))),
            "wbs_path": (None if not wbs_path_col else xc._clean_optional(row.get(wbs_path_col))),
            "wbs_name": (None if not wbs_name_col else xc._clean_optional(row.get(wbs_name_col))),
            "status": _derive_status(row, status_col, act_start, act_finish),
            "start": _first_date(row, _START_COLS),
            "finish": _first_date(row, _FINISH_COLS),
            "float_days": float_by_aid.get(aid),
            "is_milestone": "mile" in task_type,
            "is_finish_milestone": "finmile" in task_type,
            "constraint_type": (None if not cstr_type_col else xc._clean_optional(row.get(cstr_type_col))),
            "constraint_date": _iso(xc._parse_date(row.get(cstr_date_col))) if cstr_date_col else None,
        }
    return out


def _preds_by_succ(snapshot: xc.XerSnapshot) -> dict[str, list[dict[str, Any]]]:
    """succ_aid -> [{pred_aid, relationship_type, lag_days}]."""
    rels = xc._relationship_records_by_pair(snapshot)
    out: dict[str, list[dict[str, Any]]] = {}
    for (pred_aid, succ_aid), rec in rels.items():
        out.setdefault(succ_aid, []).append(
            {
                "pred_aid": pred_aid,
                "relationship_type": rec.get("relationship_type"),
                "lag_days": rec.get("lag_days") or 0.0,
            }
        )
    return out


def _impose_start_date(pred: Mapping[str, Any], rel_type: str | None, lag_days: float) -> pd.Timestamp | None:
    """The successor-start date this predecessor imposes via its relationship + lag."""
    ps, pf = pred.get("start"), pred.get("finish")
    rel = str(rel_type or "").upper()
    base = pf if rel in ("PR_FS", "PR_FF", "") else ps  # FS/FF key off pred finish; SS/SF off pred start
    if base is None:
        base = pf if pf is not None else ps
    if base is None:
        return None
    try:
        return base + pd.Timedelta(days=float(lag_days or 0.0))
    except Exception:
        return base


def _driving_predecessor(
    succ_aid: str,
    records: Mapping[str, dict[str, Any]],
    preds_by_succ: Mapping[str, list[dict[str, Any]]],
) -> tuple[str | None, bool]:
    """
    Return (driving_pred_aid, constraint_driven_start).

    The driving predecessor is the one imposing the LATEST start on the successor (P6 sets early
    start to the max over predecessors). If that latest imposed date is materially earlier than the
    successor's own start, the start is constraint/calendar-driven (no binding predecessor).
    """
    succ = records.get(succ_aid)
    if succ is None:
        return None, False
    candidates = preds_by_succ.get(succ_aid, [])
    best_aid: str | None = None
    best_date: pd.Timestamp | None = None
    for link in candidates:
        pred = records.get(link["pred_aid"])
        if pred is None:
            continue
        imposed = _impose_start_date(pred, link.get("relationship_type"), link.get("lag_days") or 0.0)
        if imposed is None:
            continue
        if best_date is None or imposed > best_date:
            best_date = imposed
            best_aid = link["pred_aid"]
    if best_aid is None:
        return None, False
    succ_start = succ.get("start")
    constraint_driven = False
    if succ_start is not None and best_date is not None:
        # Successor starts materially later than any predecessor forces -> constraint/calendar driven.
        if (succ_start.normalize() - best_date.normalize()).days > 2:
            constraint_driven = bool(succ.get("constraint_type")) or True
    return best_aid, constraint_driven


def _trace_driving_path(end_aid: str, records, preds_by_succ) -> list[str]:
    # Follow the driving (latest-imposing) predecessor back to a source. A constraint-driven start is
    # recorded as a label on the first activity (see _backbone_object); it is NOT a stop condition,
    # because ordinary scheduling gaps would otherwise truncate real backbones.
    path = [end_aid]
    visited = {end_aid}
    cur = end_aid
    while True:
        dp, _constraint_driven = _driving_predecessor(cur, records, preds_by_succ)
        if dp is None or dp in visited:
            break
        path.append(dp)
        visited.add(dp)
        cur = dp
    path.reverse()
    return path


def _path_span_days(path: Sequence[str], records: Mapping[str, dict[str, Any]]) -> float:
    starts = [records[a]["start"] for a in path if records.get(a) and records[a].get("start") is not None]
    finishes = [records[a]["finish"] for a in path if records.get(a) and records[a].get("finish") is not None]
    if not starts or not finishes:
        return 0.0
    return float((max(finishes).normalize() - min(starts).normalize()).days)


def longest_backbones(
    snapshot: xc.XerSnapshot,
    *,
    k: int = 5,
    max_overlap: float = 0.5,
) -> dict[str, Any]:
    records = _activity_records(snapshot)
    if not records:
        return {"backbones": [], "warning": "No usable activities found."}
    preds_by_succ = _preds_by_succ(snapshot)

    # Candidate end-points: activities with no successors (sinks) plus finish milestones.
    has_successor: set[str] = set()
    for succ_aid, links in preds_by_succ.items():
        for link in links:
            has_successor.add(link["pred_aid"])
    candidates = [
        aid
        for aid, rec in records.items()
        if aid not in has_successor or rec.get("is_finish_milestone")
    ]

    # A real backbone is a multi-activity chain; a 2-3 node path is usually a permit/restriction
    # window or an isolated milestone, not a project backbone.
    min_length = 4
    traced: list[tuple[float, list[str]]] = []
    for end_aid in candidates:
        path = _trace_driving_path(end_aid, records, preds_by_succ)
        if len(path) < min_length:
            continue
        traced.append((_path_span_days(path, records), path))
    traced.sort(key=lambda x: (-x[0], -len(x[1])))

    # Greedy distinct extraction with overlap suppression.
    kept: list[tuple[float, list[str]]] = []
    for span, path in traced:
        pset = set(path)
        if any(len(pset & set(kp)) / max(1, len(pset)) >= max_overlap for _, kp in kept):
            continue
        kept.append((span, path))
        if len(kept) >= k:
            break

    backbones = [_backbone_object(span, path, records, preds_by_succ) for span, path in kept]
    # Mark the current driving (critical) backbone: the one whose incomplete activities are
    # predominantly at/near zero total float (P6's critical path runs through it).
    best_idx, best_score = -1, 0.0
    for i, b in enumerate(backbones):
        score = b.get("_critical_fraction", 0.0)
        if score > best_score:
            best_idx, best_score = i, score
    if best_idx >= 0 and best_score >= 0.5:
        backbones[best_idx]["is_current_critical_path"] = True
    for b in backbones:
        b.pop("_critical_fraction", None)

    return {
        "backbone_count": len(backbones),
        "candidate_endpoint_count": len(candidates),
        "backbones": backbones,
    }


def _backbone_object(span: float, path: Sequence[str], records, preds_by_succ) -> dict[str, Any]:
    activities = []
    for aid in path:
        r = records[aid]
        activities.append(
            {
                "task_name": r.get("task_name"),
                "wbs_path": r.get("wbs_path"),
                "status": r.get("status"),
                "start_date": _iso(r.get("start")),
                "finish_date": _iso(r.get("finish")),
                "is_milestone": r.get("is_milestone"),
            }
        )
    _, constraint_driven = _driving_predecessor(path[0], records, preds_by_succ)
    statuses = [records[a]["status"] for a in path]
    completed = sum(1 for s in statuses if s == "completed")
    wbs_seq: list[str] = []
    for aid in path:
        leaf = str(records[aid].get("wbs_path") or records[aid].get("wbs_name") or "").split(" / ")[-1].strip()
        if leaf and (not wbs_seq or wbs_seq[-1] != leaf):
            wbs_seq.append(leaf)

    # Fraction of the incomplete activities that are at/near zero total float (current critical driver).
    incomplete = [a for a in path if records[a]["status"] != "completed"]
    near_zero = [a for a in incomplete if records[a].get("float_days") is not None and records[a]["float_days"] <= 1.0]
    critical_fraction = (len(near_zero) / len(incomplete)) if incomplete else 0.0

    return {
        "length": len(path),
        "span_days": span,
        "start_date": activities[0]["start_date"],
        "finish_date": activities[-1]["finish_date"],
        "percent_complete": round(100.0 * completed / max(1, len(path)), 1),
        "starts_constraint_driven": bool(constraint_driven),
        "wbs_sequence": wbs_seq,
        "activity_chain": activities,
        "is_current_critical_path": False,
        "_critical_fraction": critical_fraction,
    }


# ---------------------------------------------------------------------------
# Project overview facts
# ---------------------------------------------------------------------------

def project_overview(snapshot: xc.XerSnapshot) -> dict[str, Any]:
    t = snapshot.task
    if t is None or t.empty:
        return {"warning": "TASK table is empty."}
    records = _activity_records(snapshot)

    starts = [r["start"] for r in records.values() if r.get("start") is not None]
    finishes = [r["finish"] for r in records.values() if r.get("finish") is not None]
    proj_start = min(starts) if starts else None
    proj_finish = max(finishes) if finishes else None

    status_counts = {"completed": 0, "in_progress": 0, "not_started": 0}
    for r in records.values():
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    pct_col = xp._pick_col(t, ["phys_complete_pct", "physical_percent_complete"])
    if pct_col:
        pct = pd.to_numeric(t[pct_col], errors="coerce")
        overall_pct = round(float(pct.mean(skipna=True)), 1) if pct.notna().any() else None
    else:
        overall_pct = round(100.0 * status_counts["completed"] / max(1, len(records)), 1)

    # WBS scope: top levels with activity counts.
    wbs_level_counts: dict[str, int] = {}
    for r in records.values():
        path = str(r.get("wbs_path") or "").strip()
        if not path:
            continue
        parts = [p.strip() for p in path.split(" / ") if p.strip()]
        # levels 2-3 (skip the project root at level 1)
        for depth in (2, 3):
            if len(parts) >= depth:
                key = " / ".join(parts[1:depth])
                wbs_level_counts[key] = wbs_level_counts.get(key, 0) + 1
    top_wbs = sorted(wbs_level_counts.items(), key=lambda kv: -kv[1])[:18]

    milestones = []
    for aid, r in records.items():
        if r.get("is_milestone"):
            milestones.append(
                {
                    "task_name": r.get("task_name"),
                    "wbs_path": r.get("wbs_path"),
                    "date": _iso(r.get("finish") or r.get("start")),
                    "kind": "finish" if r.get("is_finish_milestone") else "start_or_other",
                    "status": r.get("status"),
                }
            )
    milestones.sort(key=lambda m: (m["date"] or ""))

    proj_name = None
    if snapshot.wbs is not None and not snapshot.wbs.empty:
        root_name_col = xp._pick_col(snapshot.wbs, ["wbs_name", "wbs_short_name"])
        if root_name_col and len(snapshot.wbs):
            proj_name = xc._clean_optional(snapshot.wbs.iloc[0].get(root_name_col))

    return {
        "project_name": proj_name,
        "data_date": _iso(snapshot.data_date),
        "schedule_start": _iso(proj_start),
        "schedule_finish": _iso(proj_finish),
        "total_duration_days": (None if not (proj_start and proj_finish) else int((proj_finish.normalize() - proj_start.normalize()).days)),
        "activity_count": len(records),
        "status_breakdown": status_counts,
        "overall_percent_complete": overall_pct,
        "milestone_count": len(milestones),
        "wbs_scope_top_areas": [{"area": k, "activity_count": v} for k, v in top_wbs],
        "milestones": milestones[:40],
    }


# ---------------------------------------------------------------------------
# AI-ready digest
# ---------------------------------------------------------------------------

def build_investigation_digest(
    overview: Mapping[str, Any],
    backbones: Mapping[str, Any],
    *,
    instruction: str | None = None,
) -> dict[str, Any]:
    digest = {
        "report_type": "single_schedule_project_overview",
        "project": {
            "name": overview.get("project_name"),
            "data_date": overview.get("data_date"),
            "schedule_start": overview.get("schedule_start"),
            "schedule_finish": overview.get("schedule_finish"),
            "total_duration_days": overview.get("total_duration_days"),
            "activity_count": overview.get("activity_count"),
            "overall_percent_complete": overview.get("overall_percent_complete"),
            "status_breakdown": overview.get("status_breakdown"),
        },
        "wbs_scope_top_areas": overview.get("wbs_scope_top_areas"),
        "milestones": overview.get("milestones"),
        "main_paths": [
            {
                "span_days": b.get("span_days"),
                "start_date": b.get("start_date"),
                "finish_date": b.get("finish_date"),
                "percent_complete": b.get("percent_complete"),
                "is_current_critical_path": b.get("is_current_critical_path"),
                "starts_constraint_driven": b.get("starts_constraint_driven"),
                "wbs_sequence": b.get("wbs_sequence"),
                "activity_sequence": [a.get("task_name") for a in (b.get("activity_chain") or [])],
            }
            for b in (backbones.get("backbones") or [])
        ],
        "data_quality_note": backbones.get("data_quality_note"),
    }
    if instruction and str(instruction).strip():
        digest["user_instruction"] = str(instruction).strip()
    return digest


def investigate_from_path(xer_path: str | Path, *, k: int = 5) -> dict[str, Any]:
    snapshot = xc.snapshot_from_xer_path("schedule", xer_path)
    overview = project_overview(snapshot)
    backbones = longest_backbones(snapshot, k=k)
    return {"overview": overview, "backbones": backbones}
