from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import time
import warnings
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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

LOG_NAME = "schedule_analytics"
_RECENT_LOG_LINES: deque[str] = deque(maxlen=400)
_WINDOWS_TLS_CONFIGURED = False


def _configure_windows_tls_trust() -> None:
    """
    On some Windows corporate networks, HTTPS is intercepted (MITM) with a custom root cert.

    urllib can succeed using the OS trust store, while requests/certifi fails with:
    SSLCertVerificationError: unable to get local issuer certificate

    Fix: use the Windows trust store for Python HTTPS (via truststore), with a fallback to certifi-win32.
    """
    global _WINDOWS_TLS_CONFIGURED
    if _WINDOWS_TLS_CONFIGURED:
        return
    _WINDOWS_TLS_CONFIGURED = True

    if platform.system() != "Windows":
        return

    logger = _get_logger()
    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
        logger.info("TLS trust: truststore injected into ssl")
        return
    except Exception as e:
        logger.info("TLS trust: truststore not available (%s)", str(e))

    try:
        import certifi_win32  # type: ignore  # noqa: F401

        logger.info("TLS trust: certifi_win32 loaded")
    except Exception as e:
        logger.info("TLS trust: certifi_win32 not available (%s)", str(e))


def _log_path() -> str:
    system = platform.system()
    if system == "Windows":
        base = os.getenv("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
        return os.path.join(base, "ScheduleAnalytics", "schedule_analytics.log")
    if system == "Darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "ScheduleAnalytics",
            "schedule_analytics.log",
        )
    return os.path.join(os.path.expanduser("~"), ".schedule_analytics", "schedule_analytics.log")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOG_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    class _RecentLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
            try:
                _RECENT_LOG_LINES.append(fmt.format(record))
            except Exception:
                pass

    logger.addHandler(_RecentLogHandler())
    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    except Exception:
        logger.addHandler(logging.NullHandler())
    return logger


def get_recent_logs(*, max_lines: int = 200) -> str:
    """
    Returns recent in-memory log lines (useful when users can't access log files).
    """
    try:
        n = max(1, min(int(max_lines), 400))
    except Exception:
        n = 200
    lines = list(_RECENT_LOG_LINES)[-n:]
    return "\n".join(lines)


def probe_gemini_connectivity(
    *,
    url: str = "https://generativelanguage.googleapis.com",
    timeout_s: int = 5,
) -> dict[str, Any]:
    """
    Lightweight network probe to help diagnose corporate proxies (e.g., Zscaler) without needing logs.

    This does NOT require an API key. It checks:
      - DNS resolution
      - HTTPS reachability and latency
      - Whether the response resembles a proxy block page
    """
    out: dict[str, Any] = {
        "url": url,
        "timeout_s": int(timeout_s),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "genai_transport_env": (os.getenv(GENAI_TRANSPORT_ENV_VAR) or ""),
        "genai_transport_selected": _select_genai_transport() or "default",
    }

    host = url.replace("https://", "").replace("http://", "").split("/", 1)[0]
    out["host"] = host

    # DNS
    t0 = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        ips = sorted({info[4][0] for info in infos if info and info[4]})
        out["dns_ok"] = True
        out["dns_s"] = round(time.perf_counter() - t0, 3)
        out["resolved_ips"] = ips[:10]
    except Exception as e:
        out["dns_ok"] = False
        out["dns_s"] = round(time.perf_counter() - t0, 3)
        out["dns_error"] = str(e)
        return out

    # HTTPS GET
    req = urllib.request.Request(url=url, method="GET", headers={"User-Agent": "ScheduleAnalytics/1.0"})
    t1 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out["http_ok"] = True
            out["http_s"] = round(time.perf_counter() - t1, 3)
            out["status"] = int(getattr(resp, "status", 0) or 0)
            headers = dict(resp.headers.items())
            out["server_header"] = headers.get("Server") or headers.get("server")
            # Read only a small sample to avoid big downloads.
            sample = resp.read(1024).decode("utf-8", errors="replace")
            out["body_sample"] = sample[:300]

            sample_lc = sample.casefold()
            out["likely_proxy_block"] = any(
                s in sample_lc
                for s in [
                    "zscaler",
                    "access denied",
                    "blocked",
                    "forbidden",
                    "your request was blocked",
                    "policy",
                ]
            )
    except urllib.error.HTTPError as e:
        out["http_ok"] = True
        out["http_s"] = round(time.perf_counter() - t1, 3)
        out["status"] = int(e.code)
        try:
            body = e.read(1024).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        out["body_sample"] = body[:300]
        body_lc = body.casefold()
        out["likely_proxy_block"] = any(s in body_lc for s in ["zscaler", "access denied", "blocked", "policy"])
    except Exception as e:
        out["http_ok"] = False
        out["http_s"] = round(time.perf_counter() - t1, 3)
        out["http_error"] = str(e)

    # API-path probe (no API key). Expect 401/403 if reachable.
    api_url = url.rstrip("/") + "/v1beta/models"
    req2 = urllib.request.Request(url=api_url, method="GET", headers={"User-Agent": "ScheduleAnalytics/1.0"})
    t2 = time.perf_counter()
    try:
        with urllib.request.urlopen(req2, timeout=timeout_s) as resp2:
            out["api_ok"] = True
            out["api_s"] = round(time.perf_counter() - t2, 3)
            out["api_status"] = int(getattr(resp2, "status", 0) or 0)
            out["api_body_sample"] = resp2.read(512).decode("utf-8", errors="replace")[:300]
    except urllib.error.HTTPError as e:
        out["api_ok"] = True
        out["api_s"] = round(time.perf_counter() - t2, 3)
        out["api_status"] = int(e.code)
        try:
            out["api_body_sample"] = e.read(512).decode("utf-8", errors="replace")[:300]
        except Exception:
            out["api_body_sample"] = ""
    except Exception as e:
        out["api_ok"] = False
        out["api_s"] = round(time.perf_counter() - t2, 3)
        out["api_error"] = str(e)

    # requests-based probe (mirrors what google-generativeai uses under the hood for REST).
    _configure_windows_tls_trust()
    try:
        import requests  # type: ignore

        t3 = time.perf_counter()
        r = requests.get(api_url, timeout=timeout_s, headers={"User-Agent": "ScheduleAnalytics/1.0"})
        out["requests_ok"] = True
        out["requests_s"] = round(time.perf_counter() - t3, 3)
        out["requests_status"] = int(getattr(r, "status_code", 0) or 0)
        out["requests_body_sample"] = (r.text or "")[:300]
    except Exception as e:
        out["requests_ok"] = False
        out["requests_error"] = str(e)

    return out


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
GENAI_TRANSPORT_ENV_VAR = "SCHEDULE_ANALYTICS_GENAI_TRANSPORT"


def _select_genai_transport() -> str | None:
    """
    Choose a transport for google-generativeai.

    Why: On some Windows networks, gRPC/HTTP2 can stall while plain REST works.
    Override with SCHEDULE_ANALYTICS_GENAI_TRANSPORT=grpc|rest|grpc_asyncio|auto.
    """
    raw = (os.getenv(GENAI_TRANSPORT_ENV_VAR) or "").strip().casefold()
    if raw in {"", "auto"}:
        return "rest" if platform.system() == "Windows" else None
    if raw in {"rest", "grpc", "grpc_asyncio"}:
        return raw
    return None


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


def _key_diagnostics() -> dict[str, Any]:
    def normalize(value: str | None) -> str | None:
        v = (value or "").strip().strip("'").strip('"').strip()
        if not v:
            return None
        if v in {"YOUR_REAL_KEY_HERE", "your_key_here"}:
            return None
        return v

    diag: dict[str, Any] = {}
    diag["embedded_constant_set"] = bool(normalize(EMBEDDED_GEMINI_API_KEY))
    diag["env_var_set"] = bool(normalize(os.getenv(API_KEY_ENV_VAR)))
    diag["dotenv_candidates"] = _candidate_dotenv_paths()
    diag["dotenv_found"] = []
    for p in diag["dotenv_candidates"]:
        try:
            if os.path.exists(p):
                diag["dotenv_found"].append(p)
        except Exception:
            continue
    diag["secrets_module_imported"] = bool(_secrets is not None)
    try:
        diag["secrets_module_key_set"] = bool(
            normalize(getattr(_secrets, "EMBEDDED_GEMINI_API_KEY", None) if _secrets else None)
        )
    except Exception:
        diag["secrets_module_key_set"] = False
    return diag


def _get_api_key(api_key: str | None = None) -> str:
    key = (api_key or find_api_key() or "").strip()
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Set environment variable GEMINI_API_KEY, or create a local .env file "
            "with GEMINI_API_KEY=..., or pass api_key=... to generate_narrative().\n\n"
            f"Diagnostics: {json.dumps(_key_diagnostics(), indent=2)}"
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

    logger = _get_logger()
    t0 = time.perf_counter()

    _configure_windows_tls_trust()

    transport = _select_genai_transport()
    try:
        if transport:
            genai.configure(api_key=key, transport=transport)
        else:
            genai.configure(api_key=key)
    except TypeError:
        # Backward-compat for older google-generativeai versions that don't accept transport=.
        genai.configure(api_key=key)
        transport = None

    # Keep payload compact to reduce request size / latency.
    user_content = json.dumps(data_digest, default=str)
    payload_bytes = len(user_content.encode("utf-8", errors="ignore"))
    logger.info("Gemini start model=%s transport=%s payload_bytes=%s", model, transport or "default", payload_bytes)

    # google-generativeai supports system_instruction on GenerativeModel.
    model_obj = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_REPORT_GUIDELINES)
    try:
        t_call = time.perf_counter()
        # Some environments ignore SDK-level timeouts (especially with gRPC/HTTP2).
        # Add a hard timeout so the UI doesn't hang indefinitely.
        request_timeout_s = 120
        hard_timeout_s = 150
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(model_obj.generate_content, user_content, request_options={"timeout": request_timeout_s})
            resp = fut.result(timeout=hard_timeout_s)
        logger.info(
            "Gemini ok model=%s call_s=%.3f total_s=%.3f",
            model,
            time.perf_counter() - t_call,
            time.perf_counter() - t0,
        )
    except FutureTimeoutError as e:
        logger.error(
            "Gemini timeout model=%s transport=%s total_s=%.3f",
            model,
            transport or "default",
            time.perf_counter() - t0,
        )
        raise RuntimeError(
            "Gemini call timed out (hard timeout). On some Windows networks, gRPC/HTTP2 stalls even when HTTPS works. "
            "Try setting environment variable SCHEDULE_ANALYTICS_GENAI_TRANSPORT=rest and retry."
        ) from e
    except Exception as e:
        msg = str(e)
        logger.error(
            "Gemini error model=%s transport=%s payload_bytes=%s total_s=%.3f err=%s",
            model,
            transport or "default",
            payload_bytes,
            time.perf_counter() - t0,
            msg,
        )
        if "429" in msg or "quota" in msg.lower():
            raise RuntimeError(
                "Gemini quota/rate-limit hit (HTTP 429). Try again later, or switch to a lower-tier model "
                "like gemini-2.5-flash, or reduce usage."
            ) from e
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            raise RuntimeError(
                "Gemini request timed out. Check your internet/firewall and try gemini-2.5-flash. "
                "If you are on a corporate network, the Gemini endpoint may be blocked."
            ) from e
        raise

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
    logger = _get_logger()
    t0 = time.perf_counter()
    url = proxy_url.strip().rstrip("/") + "/v1/narrative"
    token = user_token.strip()
    if not token:
        raise RuntimeError("Missing user token for narrative proxy.")

    body = json.dumps({"data_digest": data_digest, "model": model}, default=str).encode("utf-8")
    logger.info("Proxy start model=%s body_bytes=%s url=%s", model, len(body), url)
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
        logger.info("Proxy ok model=%s total_s=%.3f", model, time.perf_counter() - t0)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        logger.error("Proxy http_error code=%s total_s=%.3f body=%s", e.code, time.perf_counter() - t0, msg)
        raise RuntimeError(f"Proxy error {e.code}: {msg}") from e
    except urllib.error.URLError as e:
        logger.error("Proxy url_error total_s=%.3f err=%s", time.perf_counter() - t0, str(e))
        raise RuntimeError(f"Could not reach proxy: {e}") from e

    data = json.loads(raw)
    text = data.get("text")
    if not text:
        raise RuntimeError(f"Proxy response missing 'text': {data}")
    return str(text).strip()
