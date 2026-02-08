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

## Narrative generation (recommended: per-user token via Proxy)

To avoid embedding a Gemini key in the desktop app, run the included proxy server (it holds the Gemini key) and give each user a **token**:

- Server setup:
  - Install server deps: `pip install -r requirements_server.txt`
  - Set Gemini key on the server: `export GEMINI_API_KEY=...`
  - Create a user token: `python3 narrative_proxy_server.py --create-token`
  - Add the printed `TOKEN_SHA256` to the server env var `NARRATIVE_TOKEN_HASHES` (comma-separated).
  - Run: `uvicorn narrative_proxy_server:app --host 0.0.0.0 --port 8080`
- Desktop app:
  - Set **Narrative Proxy URL** to your server (example: `http://localhost:8080`)
  - Paste the **User Token** you were given

With this setup, users do **not** need to enter a Gemini API key in the app.

## Build a Windows single-file EXE (GitHub Actions)

- Ensure the project is pushed to GitHub.
- The workflow file is `.github/workflows/build-windows-exe.yml`.
- To embed Gemini in the EXE for quick testing, add a GitHub Actions secret named `GEMINI_API_KEY` (repo **Settings → Secrets and variables → Actions**) and run the workflow. The build injects a `scheduleanalytics_secrets.py` at build time (the key is not committed), but note the EXE can still be reverse-engineered.
- Trigger it from GitHub: **Actions → Build Windows EXE (PyInstaller) → Run workflow**.
- Download the artifact `ScheduleAnalytics-windows-exe` and run `ScheduleAnalytics.exe`.

## Build a macOS single-file executable (GitHub Actions)

- The workflow file is `.github/workflows/build-macos-exe.yml`.
- Trigger it from GitHub: **Actions → Build macOS EXE (PyInstaller) → Run workflow**.
- Download the artifact for your Mac:
  - `ScheduleAnalytics-macos` (native architecture of the GitHub runner)
- Run: `./ScheduleAnalytics`

Note: Building an Intel (`x86_64`) macOS binary requires an Intel macOS runner or an Intel Mac for building. GitHub-hosted runner availability may vary over time.
