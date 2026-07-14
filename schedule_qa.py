from __future__ import annotations

"""
Tier-2 schedule Q&A: a function-calling agent over a single parsed schedule.

Gemini is given tool functions that query the loaded `XerSnapshot` (find activities, look up logic,
trace driving paths, etc.) and answers from real data instead of guessing. The tools execute
client-side on the parsed schedule, so this runs in direct-Gemini mode (the key holder runs the loop).
"""

import json
from typing import Any, Callable

import narrative_engine as ne
import schedule_investigator as si
import xer_comparator as xc


QA_SYSTEM_INSTRUCTION = """You are a senior scheduler answering questions about ONE Primavera P6 schedule for a project team.

Rules:
- ALWAYS use the provided tools to look up real data before answering. Do not guess activity names, dates, float, or logic.
- Treat all float values as DAYS.
- If the tools return nothing relevant, say you could not find it in the schedule rather than inventing an answer.
- Be concise and concrete: name the activities, give the dates/float/status the tools return, and describe driving logic when relevant.
- You may call several tools to gather what you need before composing the final answer.
- Do not expose internal database IDs unless the user explicitly asks for the activity code.
"""


def _resolve(records: dict[str, dict[str, Any]], query: str) -> str | None:
    """Resolve a user-supplied activity code or name to an activity id (key in records)."""
    q = (query or "").strip()
    if not q:
        return None
    if q in records:
        return q
    ql = q.casefold()
    # exact name match first, then contains
    for aid, r in records.items():
        if str(r.get("task_name") or "").casefold() == ql:
            return aid
    for aid, r in records.items():
        if ql in str(r.get("task_name") or "").casefold():
            return aid
    return None


def _activity_brief(aid: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r = records.get(aid, {})
    return {
        "name": r.get("task_name"),
        "wbs": r.get("wbs_path"),
        "status": r.get("status"),
        "start": si._iso(r.get("start")),
        "finish": si._iso(r.get("finish")),
        "total_float_days": r.get("float_days"),
        "milestone": r.get("is_milestone"),
    }


def build_tools(snapshot: xc.XerSnapshot) -> list[Callable[..., str]]:
    """Build the tool callables (closing over the snapshot) for the function-calling agent."""
    records = si._activity_records(snapshot)
    preds_by_succ = si._preds_by_succ(snapshot)
    succs_by_pred: dict[str, list[dict[str, Any]]] = {}
    for succ_aid, links in preds_by_succ.items():
        for link in links:
            succs_by_pred.setdefault(link["pred_aid"], []).append(
                {"succ_aid": succ_aid, "relationship_type": link.get("relationship_type"), "lag_days": link.get("lag_days")}
            )

    def project_summary() -> str:
        """Return high-level facts about the whole project: name, schedule window, duration, activity count, percent complete, status breakdown, and the largest WBS areas. Call this for 'what is this project' or progress questions."""
        return json.dumps(si.project_overview(snapshot), default=str)

    def find_activities(keyword: str) -> str:
        """Find activities whose name contains the given keyword (case-insensitive). Returns up to 25 matches with name, WBS, status, dates, and total float (days). Use this to locate activities by topic, e.g. 'rebar', 'deck pour', 'inspection'."""
        kl = (keyword or "").casefold().strip()
        out = []
        for aid, r in records.items():
            if kl and kl in str(r.get("task_name") or "").casefold():
                out.append(_activity_brief(aid, records))
                if len(out) >= 25:
                    break
        return json.dumps({"match_count": len(out), "matches": out}, default=str)

    def get_activity(name_or_code: str) -> str:
        """Get full details for one activity by its name or activity code, including its immediate predecessors and successors with relationship type and lag (days)."""
        aid = _resolve(records, name_or_code)
        if not aid:
            return json.dumps({"found": False, "query": name_or_code})
        preds = [
            {"name": records.get(l["pred_aid"], {}).get("task_name"), "relationship_type": l.get("relationship_type"), "lag_days": l.get("lag_days")}
            for l in preds_by_succ.get(aid, [])
        ]
        succs = [
            {"name": records.get(l["succ_aid"], {}).get("task_name"), "relationship_type": l.get("relationship_type"), "lag_days": l.get("lag_days")}
            for l in succs_by_pred.get(aid, [])
        ]
        return json.dumps({"found": True, "activity": _activity_brief(aid, records), "predecessors": preds, "successors": succs}, default=str)

    def what_drives(name_or_code: str) -> str:
        """List the predecessors that drive the given activity (its immediate predecessors with relationship type and lag), and identify the single binding/driving predecessor."""
        aid = _resolve(records, name_or_code)
        if not aid:
            return json.dumps({"found": False, "query": name_or_code})
        driving_aid, constraint_driven = si._driving_predecessor(aid, records, preds_by_succ)
        preds = [
            {"name": records.get(l["pred_aid"], {}).get("task_name"), "relationship_type": l.get("relationship_type"), "lag_days": l.get("lag_days"), "is_driving": l["pred_aid"] == driving_aid}
            for l in preds_by_succ.get(aid, [])
        ]
        return json.dumps({
            "found": True,
            "activity": records[aid].get("task_name"),
            "driving_predecessor": (records.get(driving_aid, {}).get("task_name") if driving_aid else None),
            "start_is_constraint_driven": bool(constraint_driven),
            "predecessors": preds,
        }, default=str)

    def what_it_drives(name_or_code: str) -> str:
        """List the successors of the given activity (the work it drives), with relationship type and lag."""
        aid = _resolve(records, name_or_code)
        if not aid:
            return json.dumps({"found": False, "query": name_or_code})
        succs = [
            {"name": records.get(l["succ_aid"], {}).get("task_name"), "relationship_type": l.get("relationship_type"), "lag_days": l.get("lag_days")}
            for l in succs_by_pred.get(aid, [])
        ]
        return json.dumps({"found": True, "activity": records[aid].get("task_name"), "successors": succs}, default=str)

    def activities_in_wbs(area_keyword: str) -> str:
        """List activities whose WBS path contains the given area keyword (case-insensitive), e.g. 'Abutment', 'Pre-Construction', 'Zone 1'. Returns up to 30 with name, status, dates, and float."""
        kl = (area_keyword or "").casefold().strip()
        out = []
        for aid, r in records.items():
            if kl and kl in str(r.get("wbs_path") or "").casefold():
                out.append(_activity_brief(aid, records))
                if len(out) >= 30:
                    break
        return json.dumps({"match_count": len(out), "matches": out}, default=str)

    def longest_path_to(name_or_code: str) -> str:
        """Trace the driving (longest) path that leads into the given activity, from its earliest upstream driver down to it, using P6 dates and logic. Returns the ordered activity sequence."""
        aid = _resolve(records, name_or_code)
        if not aid:
            return json.dumps({"found": False, "query": name_or_code})
        path = si._trace_driving_path(aid, records, preds_by_succ)
        seq = [{"name": records[a].get("task_name"), "status": records[a].get("status"), "start": si._iso(records[a].get("start")), "finish": si._iso(records[a].get("finish"))} for a in path]
        return json.dumps({"found": True, "length": len(seq), "span_days": si._path_span_days(path, records), "sequence": seq}, default=str)

    def current_critical_path() -> str:
        """Return the project's current driving/critical backbone (the longest path whose remaining work is at/near zero float), as an ordered activity sequence."""
        bb = si.longest_backbones(snapshot)
        crit = next((b for b in bb.get("backbones", []) if b.get("is_current_critical_path")), None)
        if crit is None:
            crit = (bb.get("backbones") or [None])[0]
        if crit is None:
            return json.dumps({"found": False})
        return json.dumps({
            "found": True,
            "span_days": crit.get("span_days"),
            "percent_complete": crit.get("percent_complete"),
            "wbs_sequence": crit.get("wbs_sequence"),
            "activity_sequence": [a.get("task_name") for a in (crit.get("activity_chain") or [])],
        }, default=str)

    return [project_summary, find_activities, get_activity, what_drives, what_it_drives, activities_in_wbs, longest_path_to, current_critical_path]


def answer_question(
    snapshot: xc.XerSnapshot,
    question: str,
    *,
    model: str = ne.DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
) -> str:
    """
    Answer a free-form question about the loaded schedule using a function-calling agent.

    Requires direct Gemini access (the tools run locally on the parsed schedule).
    """
    if ne.genai is None:  # pragma: no cover
        raise RuntimeError("google-generativeai is not available; Q&A requires it.") from ne._GENAI_IMPORT_ERROR
    ne.configure_genai_client(api_key)
    model = ne.normalize_gemini_model(model)
    tools = build_tools(snapshot)
    model_obj = ne.genai.GenerativeModel(model_name=model, system_instruction=QA_SYSTEM_INSTRUCTION, tools=tools)
    chat = model_obj.start_chat(enable_automatic_function_calling=True)
    resp = chat.send_message(question)
    text = getattr(resp, "text", None)
    if not text:
        try:
            text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
        except Exception:
            text = str(resp)
    return str(text).strip()
