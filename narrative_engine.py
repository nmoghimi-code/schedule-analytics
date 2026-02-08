from __future__ import annotations

import json
import os
import sys
import warnings
import urllib.error
import urllib.request
from typing import Any, Mapping

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import google.generativeai as genai  # type: ignore

    _GENAI_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    _GENAI_IMPORT_ERROR = _e

try:
    import scheduleanalytics_secrets as _secrets  # type: ignore
except Exception:  # pragma: no cover
    _secrets = None  # type: ignore[assignment]


# Optional, insecure convenience for early testing only:
# If set, the app can generate narratives without requiring users to enter a key or set up a proxy.
# WARNING: Any key embedded in a desktop app can be extracted by end users. Do not use this for production.
EMBEDDED_GEMINI_API_KEY = ""


SYSTEM_REPORT_GUIDELINES = """You are a Senior Project Planner at EllisDon writing a proactive schedule narrative for an owner/GC report.
Your goal is to tell a clear, defensible story of project health and the “why” behind movement, using ONLY the provided JSON.

Persona & reasoning expectations:
- Synthesize cause-and-effect: when new change activities appear, explain how they logically flow into successor construction work.
- Distinguish physical work (e.g., installation, concrete, procurement, commissioning) from blockers/constraints (e.g., permits, third-party delays, seasonal/weather).

Rules:
- Zero-Invention: Use ONLY numbers/facts from the JSON input for dates and variances. Do not fabricate missing dates, float values, or reasons.
- If a required value is missing (null/empty), explicitly state “Not available in the provided data.”
- Treat all float values as DAYS.
- Whenever possible, group progress by physical area (e.g., Level/Floor, Structure/Zone, Detour/Area). If the JSON does not contain an explicit area field, infer grouping from available strings (wbs_name or activity/task names) conservatively; otherwise group by wbs_name.
- Accountability: If the JSON indicates “BY OTHERS” or a trade/subcontractor, explicitly attribute it. If trade data is not present, state it is not available.

Required narrative structure (use headings):

1) Executive Summary & Milestone Status
   - The bottom line: Lead with update_period.current_data_date and a clear statement of project health.
   - Milestone variance: Identify the “Substantial Performance” milestone using milestone_variance.target_activity_id and milestone_variance.current.task_name.
   - Quantified delta: State milestone_variance.period_variance_days (Current vs Last) and milestone_variance.total_variance_days (Current vs Baseline).
   - Primary driver: Summarize the single most influential driver of movement using change_impact and eroding_risks context. If causality cannot be supported from JSON, say so.

2) Strategic Progress & Achievements
   - Work accomplished: Summarize work_accomplished.activities in professional paragraphs.
   - Area-centric grouping: Group accomplishments by physical area if available; otherwise by wbs_name.
   - Quality & Safety: If the completed activities include clear indicators (e.g., “Hold Point”, “Inspection”, “QA/QC”, “Energization”), call them out as standalone control milestones; otherwise state not available.

3) Critical Path & Logic Flow
   - Path narrative: Use “stems from” and “flows through” language.
   - Use change_impact.critical_path_successor_summaries to describe how change-related new activities flow into successors.
   - Use change_impact.cross_wbs_alerts to explicitly name impacted departments/areas (e.g., Mechanical, Electrical) and describe the downstream effect.

4) Risks & Float Erosion (Look-Ahead)
   - Erosion velocity: Use eroding_risks.days_between, eroding_risks.least_float_current_days, and eroding_risks.items to highlight where float is eroding faster than time is passing.
   - Constraint management: If near-critical items are provided in the JSON (e.g., a list of near-critical activities/threshold), call them out as look-ahead constraints (e.g., Total Float < 10 days). If not provided, state not available.
   - Call out blocker-like items if they are explicitly indicated in names or summaries (e.g., permits, weather, third-party); do not invent constraints.

5) Mitigation & Recovery Strategy
   - Provide 2–3 tailored recovery recommendations (e.g., crashing, re-sequencing, overlapping/fast-tracking) specific to the areas/trades identified in Sections 3–4.
   - Keep recommendations practical and aligned with the identified drivers; do not recommend actions unrelated to the described path/risks.
"""

DOTENV_FILENAME = ".env"
API_KEY_ENV_VAR = "GEMINI_API_KEY"


def _read_dotenv_key(dotenv_path: str = DOTENV_FILENAME, env_var: str = API_KEY_ENV_VAR) -> str | None:
    """
    Minimal .env reader (no external dependency).
    Supports lines like:
      GEMINI_API_KEY=...
      GEMINI_API_KEY="..."
      GEMINI_API_KEY='...'
    """
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != env_var:
                    continue
                val = v.strip().strip("'").strip('"').strip()
                return val or None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _candidate_dotenv_paths(dotenv_filename: str = DOTENV_FILENAME) -> list[str]:
    paths: list[str] = []

    # 1) Current working directory
    paths.append(os.path.join(os.getcwd(), dotenv_filename))

    # 2) Directory containing this module (dev usage)
    try:
        paths.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), dotenv_filename))
    except Exception:
        pass

    # 3) Directory of the executable (PyInstaller / packaged usage)
    try:
        paths.append(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), dotenv_filename))
    except Exception:
        pass

    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def find_api_key() -> str | None:
    """
    Return API key from environment or local .env (if present), else None.
    """
    def normalize(value: str | None) -> str | None:
        v = (value or "").strip().strip("'").strip('"').strip()
        if not v:
            return None
        if v in {"YOUR_REAL_KEY_HERE", "your_key_here"}:
            return None
        return v

    embedded = normalize(EMBEDDED_GEMINI_API_KEY)
    if embedded:
        return embedded

    # Optional build-time injected module (recommended if you want an EXE with an embedded key
    # without committing the key to git history).
    try:
        embedded_mod = normalize(getattr(_secrets, "EMBEDDED_GEMINI_API_KEY", None) if _secrets else None)
        if embedded_mod:
            return embedded_mod
    except Exception:
        pass

    key = normalize(os.getenv(API_KEY_ENV_VAR))
    if key:
        return key
    for p in _candidate_dotenv_paths():
        k = normalize(_read_dotenv_key(p))
        if k:
            return k
    return None


def _get_api_key(api_key: str | None = None) -> str:
    key = (api_key or find_api_key() or "").strip()
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Set environment variable GEMINI_API_KEY, or create a local .env file "
            "with GEMINI_API_KEY=..., or pass api_key=... to generate_narrative()."
        )
    return key


def generate_narrative(data_digest: Mapping[str, Any], *, api_key: str | None = None, model: str = "gemini-2.5-flash") -> str:
    """
    Generate a schedule narrative using Gemini.

    Security note: API keys must NOT be hardcoded. Provide via GEMINI_API_KEY env var or api_key parameter.
    """
    key = _get_api_key(api_key)

    if genai is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'google-generativeai'. Install it with: pip install google-generativeai"
        ) from _GENAI_IMPORT_ERROR

    genai.configure(api_key=key)

    user_content = json.dumps(data_digest, indent=2, default=str)

    # google-generativeai supports system_instruction on GenerativeModel.
    model_obj = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_REPORT_GUIDELINES)
    resp = model_obj.generate_content(user_content)

    text = getattr(resp, "text", None)
    if not text:
        # Fallback for alternate response shapes.
        try:
            text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
        except Exception:
            text = str(resp)
    return str(text).strip()


def generate_narrative_via_proxy(
    data_digest: Mapping[str, Any],
    *,
    proxy_url: str,
    user_token: str,
    model: str,
    timeout_s: int = 120,
) -> str:
    """
    Call a backend proxy that holds the Gemini API key, authenticated with a per-user token.
    """
    url = proxy_url.strip().rstrip("/") + "/v1/narrative"
    token = user_token.strip()
    if not token:
        raise RuntimeError("Missing user token for narrative proxy.")

    body = json.dumps({"data_digest": data_digest, "model": model}, default=str).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Proxy error {e.code}: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach proxy: {e}") from e

    data = json.loads(raw)
    text = data.get("text")
    if not text:
        raise RuntimeError(f"Proxy response missing 'text': {data}")
    return str(text).strip()
