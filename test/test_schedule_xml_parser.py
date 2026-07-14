from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import narrative_engine as ne
import schedule_xml_parser as sxp
import xer_comparator as xc


MSPDI_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<Project xmlns="http://schemas.microsoft.com/project/2007">
  <UID>77</UID>
  <Name>XML Adapter Test Project</Name>
  <StartDate>2026-01-05T08:00:00</StartDate>
  <FinishDate>2026-01-08T17:00:00</FinishDate>
  <StatusDate>2026-01-06T17:00:00</StatusDate>
  <CalendarUID>1</CalendarUID>
  <MinutesPerDay>480</MinutesPerDay>
  <MinutesPerWeek>2400</MinutesPerWeek>
  <ExtendedAttributes>
    <ExtendedAttribute>
      <FieldID>188743731</FieldID>
      <FieldName>Text1</FieldName>
      <Alias>Activity ID</Alias>
    </ExtendedAttribute>
  </ExtendedAttributes>
  <Calendars>
    <Calendar>
      <UID>1</UID>
      <Name>Standard</Name>
      <IsBaseCalendar>1</IsBaseCalendar>
      <BaseCalendarUID>-1</BaseCalendarUID>
    </Calendar>
  </Calendars>
  <Tasks>
    <Task>
      <UID>0</UID><ID>0</ID><Name>XML Adapter Test Project</Name><Summary>1</Summary>
    </Task>
    <Task>
      <UID>1</UID><ID>1</ID><Name>Construction</Name><WBS>1</WBS>
      <OutlineNumber>1</OutlineNumber><OutlineLevel>1</OutlineLevel><Summary>1</Summary>
      <Start>2026-01-05T08:00:00</Start><Finish>2026-01-08T17:00:00</Finish>
      <Duration>PT32H0M0S</Duration><RemainingDuration>PT24H0M0S</RemainingDuration>
      <TotalSlack>0</TotalSlack><PercentComplete>25</PercentComplete>
    </Task>
    <Task>
      <UID>2</UID><ID>2</ID><Name>Install envelope</Name><WBS>1.1</WBS>
      <OutlineNumber>1.1</OutlineNumber><OutlineLevel>2</OutlineLevel><Summary>0</Summary>
      <Start>2026-01-05T08:00:00</Start><Finish>2026-01-07T17:00:00</Finish>
      <EarlyStart>2026-01-05T08:00:00</EarlyStart><EarlyFinish>2026-01-07T17:00:00</EarlyFinish>
      <LateStart>2026-01-05T08:00:00</LateStart><LateFinish>2026-01-07T17:00:00</LateFinish>
      <Duration>PT24H0M0S</Duration><RemainingDuration>PT16H0M0S</RemainingDuration>
      <TotalSlack>0</TotalSlack><FreeSlack>0</FreeSlack><PercentComplete>33</PercentComplete>
      <ActualStart>2026-01-05T08:00:00</ActualStart><CalendarUID>1</CalendarUID>
      <ExtendedAttribute><FieldID>188743731</FieldID><Value>A-100</Value></ExtendedAttribute>
    </Task>
    <Task>
      <UID>3</UID><ID>3</ID><Name>Substantial Completion</Name><WBS>1.2</WBS>
      <OutlineNumber>1.2</OutlineNumber><OutlineLevel>2</OutlineLevel><Summary>0</Summary><Milestone>1</Milestone>
      <Start>2026-01-08T17:00:00</Start><Finish>2026-01-08T17:00:00</Finish>
      <EarlyStart>2026-01-08T17:00:00</EarlyStart><EarlyFinish>2026-01-08T17:00:00</EarlyFinish>
      <LateStart>2026-01-08T17:00:00</LateStart><LateFinish>2026-01-08T17:00:00</LateFinish>
      <Duration>PT0H0M0S</Duration><RemainingDuration>PT0H0M0S</RemainingDuration>
      <TotalSlack>0</TotalSlack><PercentComplete>0</PercentComplete><CalendarUID>1</CalendarUID>
      <ExtendedAttribute><FieldID>188743731</FieldID><Value>M-100</Value></ExtendedAttribute>
      <PredecessorLink><PredecessorUID>2</PredecessorUID><Type>1</Type><LinkLag>0</LinkLag><LagFormat>7</LagFormat></PredecessorLink>
    </Task>
  </Tasks>
</Project>
"""


class MicrosoftProjectXmlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "schedule.xml"
        self.path.write_text(MSPDI_SAMPLE, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_detect_and_normalize_mspdi(self) -> None:
        self.assertEqual(sxp.detect_xml_schedule_format(self.path), "mspdi")
        snapshot = xc.snapshot_from_schedule_path("current", self.path)

        self.assertEqual(snapshot.source_format, "mspdi_xml")
        self.assertEqual(snapshot.data_date.isoformat(), "2026-01-06T17:00:00")
        self.assertEqual(len(snapshot.task), 3)
        self.assertEqual(len(snapshot.taskpred), 1)
        self.assertEqual(snapshot.taskpred.iloc[0]["pred_type"], "PR_FS")

        milestone = snapshot.task[snapshot.task["task_code"] == "M-100"].iloc[0]
        self.assertEqual(milestone["task_type"], "TT_FinMile")
        self.assertEqual(milestone["wbs_path"], "XML Adapter Test Project / Construction")

    def test_existing_critical_path_engine_accepts_mspdi_snapshot(self) -> None:
        snapshot = xc.snapshot_from_schedule_path("current", self.path)
        result = xc.critical_path_to_target(snapshot, "M-100", near_critical_buffer_days=2)

        self.assertTrue(result["target_health"]["has_driving_logic"])
        self.assertIn("A-100", result["critical_trace_activity_ids"])
        self.assertIn("M-100", result["critical_trace_activity_ids"])

    def test_three_way_comparison_accepts_xml(self) -> None:
        baseline = xc.snapshot_from_schedule_path("baseline", self.path)
        last = xc.snapshot_from_schedule_path("last", self.path)
        current = xc.snapshot_from_schedule_path("current", self.path)
        result = xc.compare_three_way(
            baseline,
            last,
            current,
            variance_threshold=2,
            target_activity_id="M-100",
            look_ahead_horizon_days=30,
        )
        self.assertEqual(result["milestone"]["period_variance_days"], 0)
        self.assertEqual(result["critical_path_to_target"]["target_task_name"], "Substantial Completion")
        self.assertEqual(result["source_format"], "mspdi_xml")

    def test_xml_prompt_rules_are_appended_only_for_xml(self) -> None:
        with mock.patch.object(ne, "_run_gemini", return_value="ok") as run:
            ne.generate_narrative({"report_sections": {}})
            xer_instruction = run.call_args.args[0]

            ne.generate_narrative({"schedule_source": "mspdi_xml", "report_sections": {}})
            xml_instruction = run.call_args.args[0]

            ne.generate_project_overview({})
            xer_overview_instruction = run.call_args.args[0]
            ne.generate_project_overview({"schedule_source": "mspdi_xml"})
            xml_overview_instruction = run.call_args.args[0]

            ne.generate_handover_briefing({})
            xer_handover_instruction = run.call_args.args[0]
            ne.generate_handover_briefing({"schedule_source": "mspdi_xml"})
            xml_handover_instruction = run.call_args.args[0]

        self.assertEqual(xer_instruction, ne.SYSTEM_REPORT_GUIDELINES)
        self.assertEqual(xml_instruction, ne.SYSTEM_REPORT_GUIDELINES + ne.MSPDI_REPORT_GUIDELINES)
        self.assertEqual(xer_overview_instruction, ne.SYSTEM_PROJECT_OVERVIEW_GUIDELINES)
        self.assertEqual(
            xml_overview_instruction,
            ne.SYSTEM_PROJECT_OVERVIEW_GUIDELINES + ne.MSPDI_PROJECT_OVERVIEW_GUIDELINES,
        )
        self.assertEqual(xer_handover_instruction, ne.SYSTEM_HANDOVER_BRIEFING_GUIDELINES)
        self.assertEqual(
            xml_handover_instruction,
            ne.SYSTEM_HANDOVER_BRIEFING_GUIDELINES + ne.MSPDI_HANDOVER_BRIEFING_GUIDELINES,
        )


class SuppliedMicrosoftProjectXmlTests(unittest.TestCase):
    def test_same_status_date_uses_xml_only_safeguards(self) -> None:
        root = Path(__file__).parents[1] / "Sample XER test files" / "XML test"
        previous_path = root / "Construction Schedule w. 03.17.2026 Rev 1.xml"
        current_path = root / "Construction Schedule w. 04.28.2026.xml"
        if not previous_path.exists() or not current_path.exists():
            self.skipTest("Supplied Microsoft Project XML verification files are not present.")

        baseline = xc.snapshot_from_schedule_path("baseline", previous_path)
        previous = xc.snapshot_from_schedule_path("last", previous_path)
        current = xc.snapshot_from_schedule_path("current", current_path)
        result = xc.compare_three_way(
            baseline,
            previous,
            current,
            variance_threshold=5,
            target_activity_id="47",
            look_ahead_horizon_days=30,
        )

        trending = result["near_critical_trending"]
        self.assertEqual(trending["cutoff_current_days"], 5.0)
        self.assertEqual(trending["eroding_risk_count"], 0)
        self.assertIn("StatusDate did not advance", trending["erosion_assessment_warning"])

        look_ahead = result["look_ahead_window_analysis"]
        self.assertEqual(look_ahead["count"], 6)
        self.assertTrue(
            all("summary" not in str(item.get("task_type", "")).casefold() for item in look_ahead["items"])
        )

        progress = result["microsoft_project_progress_changes"]
        self.assertFalse(progress["period_assessment_available"])
        self.assertGreater(progress["count"], 0)

        digest = xc.get_ai_ready_digest(result)
        self.assertEqual(digest["schedule_source"], "mspdi_xml")
        self.assertNotIn("P6 total float", json.dumps(digest))
        period = digest["report_sections"]["section_2_strategic_progress_achievements"]["period_context"]
        self.assertFalse(period["period_assessment_available"])


class XerIsolationTests(unittest.TestCase):
    def test_neutral_loader_delegates_to_original_xer_loader(self) -> None:
        path = Path(__file__).parents[1] / "Sample XER test files" / "Wayne Bridge" / "OG.54209.2026.06.15.xer"
        if not path.exists():
            self.skipTest("Supplied XER verification file is not present.")

        direct = xc.snapshot_from_xer_path("current", path)
        neutral = xc.snapshot_from_schedule_path("current", path)
        self.assertEqual(direct.source_format, "xer")
        self.assertEqual(neutral.source_format, "xer")
        self.assertEqual(direct.data_date, neutral.data_date)
        for table_name in ["project", "task", "taskpred", "wbs", "calendar"]:
            pd.testing.assert_frame_equal(getattr(direct, table_name), getattr(neutral, table_name))

    def test_xer_comparison_and_digest_do_not_receive_xml_fields(self) -> None:
        root = Path(__file__).parents[1] / "Sample XER test files" / "Wayne Bridge"
        paths = [
            root / "BL.54209.2026.04.23.xer",
            root / "OG.54209.2026.06.08.xer",
            root / "OG.54209.2026.06.15.xer",
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("Supplied XER regression files are not present.")

        baseline, previous, current = [
            xc.snapshot_from_schedule_path(label, path)
            for label, path in zip(["baseline", "last", "current"], paths)
        ]
        result = xc.compare_three_way(
            baseline,
            previous,
            current,
            variance_threshold=5,
            target_activity_id="MILE-C-1160",
            look_ahead_horizon_days=30,
        )
        digest = xc.get_ai_ready_digest(result)

        self.assertNotIn("source_format", result)
        self.assertNotIn("microsoft_project_progress_changes", result)
        self.assertNotIn("schedule_source", digest)
        self.assertIn("P6 total float", json.dumps(digest))


class XmlFormatDetectionTests(unittest.TestCase):
    def test_supplied_primavera_xml_is_not_mspdi(self) -> None:
        fixtures = sorted(
            (Path(__file__).parents[1] / "Sample XER test files" / "XML test").glob("Universiade*.xml")
        )
        if not fixtures:
            self.skipTest("Supplied XML fixtures are not present.")
        self.assertEqual(sxp.detect_xml_schedule_format(fixtures[0]), "p6_api")
        with self.assertRaisesRegex(ValueError, "Primavera P6 XML"):
            xc.snapshot_from_schedule_path("schedule", fixtures[0])


if __name__ == "__main__":
    unittest.main()
