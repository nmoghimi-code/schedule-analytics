from __future__ import annotations

"""Microsoft Project XML (MSPDI) -> XER-compatible schedule tables.

The analysis engine consumes a small, P6-shaped set of pandas tables.  This
module translates Microsoft Project's documented XML interchange format into
that shape so the comparison, delay, overview, and Q&A code can stay shared.
It does not attempt to reschedule the project: exported Project dates, slack,
relationships, progress, and calendars remain authoritative.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd


MSPDI_NAMESPACE_FRAGMENT = "schemas.microsoft.com/project"
P6_XML_NAMESPACE_FRAGMENT = "xmlns.oracle.com/Primavera/P6"


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _namespace(tag: str) -> str:
    text = str(tag)
    return text[1:].split("}", 1)[0] if text.startswith("{") and "}" in text else ""


def _children(node: ET.Element | None, name: str) -> list[ET.Element]:
    if node is None:
        return []
    return [child for child in list(node) if _local_name(child.tag) == name]


def _child(node: ET.Element | None, name: str) -> ET.Element | None:
    for child in _children(node, name):
        return child
    return None


def _text(node: ET.Element | None, name: str, default: str = "") -> str:
    child = _child(node, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _bool(node: ET.Element | None, name: str) -> bool:
    return _text(node, name).strip().casefold() in {"1", "true", "yes", "y"}


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        return default if not text else float(text)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value)
    return default if number is None else int(number)


_DURATION_RE = re.compile(
    r"^(?P<sign>-)?P"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$",
    re.IGNORECASE,
)


def _duration_hours(value: str, *, hours_per_day: float) -> float | None:
    """Convert the ISO-8601 duration used by MSPDI to working hours."""
    text = str(value or "").strip()
    if not text:
        return None
    match = _DURATION_RE.match(text)
    if not match:
        return None
    parts = match.groupdict()
    hours = (
        float(parts.get("days") or 0.0) * float(hours_per_day)
        + float(parts.get("hours") or 0.0)
        + float(parts.get("minutes") or 0.0) / 60.0
        + float(parts.get("seconds") or 0.0) / 3600.0
    )
    return -hours if parts.get("sign") else hours


def detect_xml_schedule_format(xml_path: str | Path) -> str:
    """Return ``mspdi``, ``p6_api``, or ``unknown`` from the XML root."""
    path = Path(xml_path)
    try:
        for _event, root in ET.iterparse(path, events=("start",)):
            local = _local_name(root.tag)
            namespace = _namespace(root.tag)
            if local == "Project" and MSPDI_NAMESPACE_FRAGMENT.casefold() in namespace.casefold():
                return "mspdi"
            if local == "APIBusinessObjects" and P6_XML_NAMESPACE_FRAGMENT.casefold() in namespace.casefold():
                return "p6_api"
            # Some producers omit the namespace but retain the MSPDI structure.
            if local == "Project":
                return "mspdi"
            return "unknown"
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML file '{path.name}': {exc}") from exc
    return "unknown"


def _extended_attribute_names(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    container = _child(root, "ExtendedAttributes")
    for item in _children(container, "ExtendedAttribute"):
        field_id = _text(item, "FieldID")
        if not field_id:
            continue
        label = _text(item, "Alias") or _text(item, "FieldName")
        if label:
            result[field_id] = label
    return result


def _task_extended_values(task: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _children(task, "ExtendedAttribute"):
        field_id = _text(item, "FieldID")
        value = _text(item, "Value")
        if field_id and value:
            result[field_id] = value
    return result


def _activity_code(task: ET.Element, attribute_names: dict[str, str]) -> str:
    values = _task_extended_values(task)
    preferred_tokens = {"activity id", "activityid", "activity code", "task code"}
    for field_id, value in values.items():
        label = str(attribute_names.get(field_id, "")).strip().casefold().replace("_", " ")
        if label in preferred_tokens and value.strip():
            return value.strip()
    # UID is the most stable built-in key between successive Project saves.
    return _text(task, "UID") or _text(task, "ID") or _text(task, "WBS")


_CONSTRAINT_TYPES = {
    "0": "As Soon As Possible",
    "1": "As Late As Possible",
    "2": "Must Start On",
    "3": "Must Finish On",
    "4": "Start No Earlier Than",
    "5": "Start No Later Than",
    "6": "Finish No Earlier Than",
    "7": "Finish No Later Than",
}

_LINK_TYPES = {
    "0": "PR_FF",
    "1": "PR_FS",
    "2": "PR_SF",
    "3": "PR_SS",
    "FF": "PR_FF",
    "FS": "PR_FS",
    "SF": "PR_SF",
    "SS": "PR_SS",
}


def _status_code(task: ET.Element) -> str:
    percent = _number(_text(task, "PercentComplete"), 0.0) or 0.0
    if _text(task, "ActualFinish") or percent >= 100.0:
        return "TK_Complete"
    if _text(task, "ActualStart") or percent > 0.0:
        return "TK_Active"
    return "TK_NotStart"


def _task_dates(task: ET.Element) -> dict[str, str]:
    start = _text(task, "Start")
    finish = _text(task, "Finish")
    early_start = _text(task, "EarlyStart") or start
    early_finish = _text(task, "EarlyFinish") or finish
    late_start = _text(task, "LateStart")
    late_finish = _text(task, "LateFinish")
    actual_start = _text(task, "ActualStart")
    actual_finish = _text(task, "ActualFinish")
    return {
        "act_start_date": actual_start,
        "act_end_date": actual_finish,
        "start_date": start,
        "finish_date": finish,
        "restart_date": start if not actual_finish else "",
        "reend_date": finish if not actual_finish else "",
        "early_start_date": early_start,
        "early_end_date": early_finish,
        "late_start_date": late_start,
        "late_end_date": late_finish,
        # The current scheduled dates are the target dates for this exported snapshot.
        "target_start_date": start,
        "target_end_date": finish,
    }


def read_mspdi_tables(xml_path: str | Path) -> dict[str, pd.DataFrame]:
    """Read a Microsoft Project XML file and return XER-compatible tables."""
    path = Path(xml_path)
    detected = detect_xml_schedule_format(path)
    if detected == "p6_api":
        raise ValueError(
            f"'{path.name}' is an Oracle Primavera P6 XML export, not a Microsoft Project XML export. "
            "Open the schedule in Microsoft Project and save it as XML (MSPDI), or use the original XER file."
        )
    if detected != "mspdi":
        raise ValueError(
            f"'{path.name}' is not a recognized Microsoft Project XML (MSPDI) schedule."
        )

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid Microsoft Project XML '{path.name}': {exc}") from exc

    if _local_name(root.tag) != "Project":
        raise ValueError(f"Microsoft Project XML must have a <Project> root; found <{_local_name(root.tag)}>.")

    project_uid = _text(root, "UID") or "msp-project"
    project_name = _text(root, "Name") or _text(root, "Title") or path.stem
    status_date = _text(root, "StatusDate")
    minutes_per_day = _number(_text(root, "MinutesPerDay"), 480.0) or 480.0
    minutes_per_week = _number(_text(root, "MinutesPerWeek"), minutes_per_day * 5.0) or minutes_per_day * 5.0
    hours_per_day = minutes_per_day / 60.0

    project_rows = [
        {
            "proj_id": project_uid,
            "proj_short_name": project_name,
            "last_recalc_date": status_date,
            "data_date": status_date,
            "plan_start_date": _text(root, "StartDate"),
            "plan_end_date": _text(root, "FinishDate"),
            "clndr_id": _text(root, "CalendarUID"),
            "source_format": "mspdi_xml",
        }
    ]

    calendar_rows: list[dict[str, Any]] = []
    calendars = _child(root, "Calendars")
    for calendar in _children(calendars, "Calendar"):
        uid = _text(calendar, "UID")
        if not uid:
            continue
        calendar_rows.append(
            {
                "clndr_id": uid,
                "default_flag": "Y" if uid == _text(root, "CalendarUID") else "N",
                "clndr_name": _text(calendar, "Name") or f"Calendar {uid}",
                "proj_id": project_uid,
                "base_clndr_id": _text(calendar, "BaseCalendarUID"),
                "day_hr_cnt": hours_per_day,
                "week_hr_cnt": minutes_per_week / 60.0,
                "month_hr_cnt": hours_per_day * 20.0,
                "year_hr_cnt": hours_per_day * 250.0,
            }
        )
    if not calendar_rows:
        default_uid = _text(root, "CalendarUID") or "msp-default-calendar"
        calendar_rows.append(
            {
                "clndr_id": default_uid,
                "default_flag": "Y",
                "clndr_name": "Microsoft Project default calendar",
                "proj_id": project_uid,
                "base_clndr_id": "",
                "day_hr_cnt": hours_per_day,
                "week_hr_cnt": minutes_per_week / 60.0,
                "month_hr_cnt": hours_per_day * 20.0,
                "year_hr_cnt": hours_per_day * 250.0,
            }
        )

    attribute_names = _extended_attribute_names(root)
    tasks_container = _child(root, "Tasks")
    task_elements = _children(tasks_container, "Task")
    usable_tasks = [task for task in task_elements if _text(task, "UID") not in {"", "0"}]
    if not usable_tasks:
        raise ValueError(f"Microsoft Project XML '{path.name}' contains no usable tasks.")

    root_wbs_id = f"msp-root-{project_uid}"
    wbs_rows: list[dict[str, Any]] = [
        {
            "wbs_id": root_wbs_id,
            "proj_id": project_uid,
            "seq_num": 0,
            "status_code": "WS_Open",
            "wbs_short_name": project_name,
            "wbs_name": project_name,
            "parent_wbs_id": "",
        }
    ]

    # Track the most recent summary task at every outline level. MSPDI tasks are
    # exported in outline order, so this reconstructs the WBS hierarchy without
    # relying on localized outline-number delimiters.
    summary_at_level: dict[int, str] = {}
    task_wbs_id: dict[str, str] = {}
    seen_wbs: set[str] = {root_wbs_id}
    for sequence, task in enumerate(usable_tasks, start=1):
        uid = _text(task, "UID")
        level = max(1, _integer(_text(task, "OutlineLevel"), 1))
        for stale_level in [key for key in summary_at_level if key >= level]:
            summary_at_level.pop(stale_level, None)
        parent_wbs_id = summary_at_level.get(level - 1, root_wbs_id)
        if _bool(task, "Summary"):
            wbs_id = f"msp-summary-{uid}"
            task_wbs_id[uid] = wbs_id
            if wbs_id not in seen_wbs:
                seen_wbs.add(wbs_id)
                wbs_rows.append(
                    {
                        "wbs_id": wbs_id,
                        "proj_id": project_uid,
                        "seq_num": sequence,
                        "status_code": "WS_Open",
                        "wbs_short_name": _text(task, "WBS") or _text(task, "OutlineNumber"),
                        "wbs_name": _text(task, "Name") or f"Summary {uid}",
                        "parent_wbs_id": parent_wbs_id,
                    }
                )
            summary_at_level[level] = wbs_id
        else:
            task_wbs_id[uid] = parent_wbs_id

    task_rows: list[dict[str, Any]] = []
    valid_task_uids: set[str] = set()
    for task in usable_tasks:
        uid = _text(task, "UID")
        valid_task_uids.add(uid)
        summary = _bool(task, "Summary")
        milestone = _bool(task, "Milestone")
        duration_hours = _duration_hours(_text(task, "Duration"), hours_per_day=hours_per_day)
        remaining_hours = _duration_hours(_text(task, "RemainingDuration"), hours_per_day=hours_per_day)
        total_slack_tenths = _number(_text(task, "TotalSlack"))
        free_slack_tenths = _number(_text(task, "FreeSlack"))
        constraint_raw = _text(task, "ConstraintType")
        row: dict[str, Any] = {
            "task_id": uid,
            "proj_id": project_uid,
            "wbs_id": task_wbs_id.get(uid, root_wbs_id),
            "clndr_id": _text(task, "CalendarUID") or _text(root, "CalendarUID") or calendar_rows[0]["clndr_id"],
            "phys_complete_pct": _number(_text(task, "PhysicalPercentComplete"), _number(_text(task, "PercentComplete"), 0.0)),
            "msp_percent_complete": _number(_text(task, "PercentComplete"), 0.0),
            "msp_physical_percent_complete": _number(_text(task, "PhysicalPercentComplete")),
            "complete_pct_type": "CP_Phys" if _text(task, "PhysicalPercentComplete") else "CP_Drtn",
            "task_type": "TT_WBS" if summary else ("TT_FinMile" if milestone else "TT_Task"),
            "duration_type": _text(task, "Type") or "Microsoft Project",
            "status_code": _status_code(task),
            "task_code": _activity_code(task, attribute_names),
            "msp_uid": uid,
            "msp_id": _text(task, "ID"),
            "msp_wbs": _text(task, "WBS"),
            "outline_number": _text(task, "OutlineNumber"),
            "task_name": _text(task, "Name") or f"Task {uid}",
            "total_float_hr_cnt": None if total_slack_tenths is None else total_slack_tenths / 600.0,
            "free_float_hr_cnt": None if free_slack_tenths is None else free_slack_tenths / 600.0,
            "remain_drtn_hr_cnt": remaining_hours,
            "target_drtn_hr_cnt": duration_hours,
            "cstr_type": _CONSTRAINT_TYPES.get(constraint_raw, constraint_raw),
            "cstr_date": _text(task, "ConstraintDate"),
            "source_format": "mspdi_xml",
        }
        row.update(_task_dates(task))
        task_rows.append(row)

    relationship_rows: list[dict[str, Any]] = []
    relation_id = 1
    for successor in usable_tasks:
        successor_uid = _text(successor, "UID")
        for link in _children(successor, "PredecessorLink"):
            predecessor_uid = _text(link, "PredecessorUID")
            if predecessor_uid not in valid_task_uids or successor_uid not in valid_task_uids:
                continue
            type_raw = _text(link, "Type").upper()
            link_lag_tenths = _number(_text(link, "LinkLag"), 0.0) or 0.0
            relationship_rows.append(
                {
                    "task_pred_id": f"msp-rel-{relation_id}",
                    "task_id": successor_uid,
                    "pred_task_id": predecessor_uid,
                    "proj_id": project_uid,
                    "pred_proj_id": project_uid,
                    "pred_type": _LINK_TYPES.get(type_raw, "PR_FS"),
                    "lag_hr_cnt": link_lag_tenths / 600.0,
                    "comments": "",
                }
            )
            relation_id += 1

    return {
        "PROJECT": pd.DataFrame(project_rows),
        "TASK": pd.DataFrame(task_rows),
        "TASKPRED": pd.DataFrame(relationship_rows),
        "WBS": pd.DataFrame(wbs_rows),
        "CALENDAR": pd.DataFrame(calendar_rows),
    }
