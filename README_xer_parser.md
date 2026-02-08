# XER Parser (Primavera P6)

`xer_parser.py` reads the `PROJECT`, `TASK`, `TASKPRED`, and `WBS` tables from a P6 `.XER` (tab-delimited) export, identifies the **Data Date**, finds the **least float** (including negatives), and flags **near-critical** activities relative to a target activity. It also maps `wbs_name` onto each task and adds `is_milestone`.

## Usage

```bash
python3 xer_parser.py /path/to/file.xer --target-activity-id A3000 --variance-threshold 8
```

- `--target-activity-id` can be an Activity ID (`task_code`) or internal `task_id`.
- `--variance-threshold` is in **days** (hour-based float columns are converted internally assuming an 8-hour day).

## WBS "Change" section helper

To reliably grab all activities under the highest-level WBS section containing the term `"change"` (including all child WBS nodes), use `activity_ids_in_change_wbs()` / `activity_ids_in_wbs()` from `xer_parser.py`.

## Gemini narrative (one-time setup)

- Create a local `.env` file next to `main.py` with: `GEMINI_API_KEY=...` (see `.env.example`). This avoids pasting the key every run.

## Build a Windows single-file EXE (GitHub Actions)

- Ensure the project is pushed to GitHub.
- The workflow file is `.github/workflows/build-windows-exe.yml`.
- Trigger it from GitHub: **Actions → Build Windows EXE (PyInstaller) → Run workflow**.
- Download the artifact `ScheduleAnalytics-windows-exe` and run `ScheduleAnalytics.exe`.

## Build a macOS single-file executable (GitHub Actions)

- The workflow file is `.github/workflows/build-macos-exe.yml`.
- Trigger it from GitHub: **Actions → Build macOS EXE (PyInstaller) → Run workflow**.
- Download the artifact for your Mac:
  - `ScheduleAnalytics-macos-13` (Intel Macs)
  - `ScheduleAnalytics-macos-14` (Apple Silicon Macs)
- Run: `./ScheduleAnalytics`
