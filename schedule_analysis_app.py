from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
import platform

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

import xer_comparator as xc
import narrative_engine as ne
import delay_analyzer as da
import schedule_investigator as si
import schedule_qa as sqa


# Delay analysis traces the critical path from P6 total float; this small tolerance only controls how
# strictly the trace bridges calendar/constraint float gaps. It is not a user-facing near-critical setting.
DELAY_FLOAT_TOLERANCE_DAYS = 1

# Soft access gate for the Project Overview tab. NOTE: a hardcoded password is a deterrent only — it can
# be extracted from the source/EXE. Do not treat it as real security.
OVERVIEW_TAB_PASSWORD = "20091396"

try:
    import scheduleanalytics_build as _build  # type: ignore
except Exception:  # pragma: no cover
    _build = None  # type: ignore[assignment]


def _build_label() -> str:
    sha = (getattr(_build, "BUILD_SHA", "") if _build else "") or ""
    ts = (getattr(_build, "BUILD_TIME_UTC", "") if _build else "") or ""
    if sha:
        short = sha[:8]
        return f"{short} ({ts})" if ts else short
    return "dev"


def _config_path() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        return base / "ScheduleAnalytics" / "config.json"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "ScheduleAnalytics" / "config.json"
    return Path.home() / ".schedule_analytics" / "config.json"


def _load_config() -> dict[str, Any]:
    p = _config_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _save_config(cfg: dict[str, Any]) -> None:
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


class ScheduleAnalysisApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Schedule Analysis Tool (XER) — {_build_label()}")
        self.minsize(900, 780)

        self.baseline_path = tk.StringVar(value="")
        self.last_path = tk.StringVar(value="")
        self.current_path = tk.StringVar(value="")

        self.target_activity_id = tk.StringVar(value="A3000")
        self.near_critical_threshold = tk.StringVar(value="8")
        self.look_ahead_horizon = tk.StringVar(value="30")

        # Delay Analysis tab inputs (baseline + dynamic update slots).
        self.delay_baseline_path = tk.StringVar(value="")
        self.delay_target_activity_id = tk.StringVar(value="A3000")

        # Project Overview tab (single XER).
        self.overview_path = tk.StringVar(value="")

        self._last_compare_result: dict[str, Any] | None = None
        self._last_delay_result: dict[str, Any] | None = None
        self._last_overview_result: dict[str, Any] | None = None
        self._loaded_overview_snapshot: Any = None

        # Project Overview tab is gated behind a soft password until unlocked this session.
        self._overview_unlocked = False
        self._password_prompt_open = False

        cfg = _load_config()
        self.ai_model = tk.StringVar(value=ne.normalize_gemini_model(str(cfg.get("ai_model") or "")))
        self.proxy_url = tk.StringVar(value=str(cfg.get("proxy_url") or ""))
        self.user_token = tk.StringVar(value=str(cfg.get("user_token") or ""))

        self._build_ui()
        self._build_menu()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Check Network", command=self._check_network)
        tools.add_command(label="Diagnostics", command=self._show_diagnostics)
        tools.add_separator()
        tools.add_command(label="About", command=self._show_about)
        tools.add_separator()
        tools.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="Tools", menu=tools)

        self.config(menu=menubar)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(root)
        notebook.grid(row=0, column=0, sticky="nsew")

        narrative_tab = ttk.Frame(notebook, padding=10)
        delay_tab = ttk.Frame(notebook, padding=10)
        overview_tab = ttk.Frame(notebook, padding=10)
        notebook.add(narrative_tab, text="Narrative Generator")
        notebook.add(delay_tab, text="Delay Analysis")
        notebook.add(overview_tab, text="Project Overview")

        # Shared AI / connection settings live below the notebook so both tabs use them.
        ai = ttk.LabelFrame(root, text="AI / Connection (shared)", padding=10)
        ai.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ai.columnconfigure(1, weight=1)
        self._model_row(ai, 0, "AI Logic Engine:", self.ai_model)
        self._settings_row(ai, 1, "Narrative Proxy URL:", self.proxy_url)
        self._secret_row(ai, 2, "User Token:", self.user_token)

        self._build_narrative_tab(narrative_tab)
        self._build_delay_tab(delay_tab)
        self._build_overview_tab(overview_tab)

        # Gate the Project Overview tab behind a soft password.
        self.notebook = notebook
        self._overview_tab = overview_tab
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, _event: Any = None) -> None:
        if self._overview_unlocked or self._password_prompt_open:
            return
        try:
            selected = self.notebook.select()
        except Exception:
            return
        if selected != str(self._overview_tab):
            return

        self._password_prompt_open = True
        try:
            pw = simpledialog.askstring(
                "Project Overview — Locked",
                "Enter password to access the Project Overview tab:",
                show="•",
                parent=self,
            )
        finally:
            self._password_prompt_open = False

        if pw == OVERVIEW_TAB_PASSWORD:
            self._overview_unlocked = True
            return
        if pw is not None:  # None means the user cancelled
            messagebox.showerror("Project Overview — Locked", "Incorrect password.")
        # Send them back to the first tab.
        try:
            self.notebook.select(0)
        except Exception:
            pass

    def _build_narrative_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)  # Results pane expands; button row stays compact above it.

        files = ttk.LabelFrame(root, text="File Selection", padding=10)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)

        self._file_row(files, 0, "Baseline XER:", self.baseline_path)
        self._file_row(files, 1, "Last Update XER:", self.last_path)
        self._file_row(files, 2, "Current Update XER:", self.current_path)

        settings = ttk.LabelFrame(root, text="Settings", padding=10)
        settings.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        settings.columnconfigure(1, weight=1)
        self._settings_row(settings, 0, "Target Activity ID:", self.target_activity_id)
        self._settings_row(settings, 1, "Near-Critical Threshold (days):", self.near_critical_threshold)
        self._settings_row(settings, 2, "Look-Ahead Horizon (days):", self.look_ahead_horizon)

        run_row = ttk.Frame(root)
        run_row.grid(row=2, column=0, sticky="ew")
        run_row.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_row, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.run_button = ttk.Button(run_row, text="Run", command=self._run_analysis)
        self.run_button.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.narrative_button = ttk.Button(run_row, text="Generate Narrative Report", command=self._generate_narrative)
        self.narrative_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

        self.netcheck_button = ttk.Button(run_row, text="Check Network", command=self._check_network)
        self.netcheck_button.grid(row=0, column=3, sticky="e", padx=(8, 0))

        self.diag_button = ttk.Button(run_row, text="Diagnostics", command=self._show_diagnostics)
        self.diag_button.grid(row=0, column=4, sticky="e", padx=(8, 0))

        results_frame = ttk.LabelFrame(root, text="Results", padding=10)
        results_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results = ScrolledText(results_frame, wrap="none", height=16)
        self.results.grid(row=0, column=0, sticky="nsew")

        self._write_to(self.results, {"status": "ready", "hint": "Select three XER files, set inputs, then click Run."})

    def _build_delay_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)  # Results pane (row 4) expands; button row (row 3) stays compact above it.

        files = ttk.LabelFrame(root, text="File Selection", padding=10)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)

        self._file_row(files, 0, "Baseline XER:", self.delay_baseline_path)

        ttk.Label(files, text="Updates (chronological; auto-sorted by data date):").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        self.delay_updates_frame = ttk.Frame(files)
        self.delay_updates_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.delay_updates_frame.columnconfigure(1, weight=1)
        self.delay_update_paths = []
        # Start with two update slots.
        self._add_delay_update_row()
        self._add_delay_update_row()

        ttk.Button(files, text="+ Add Update", command=self._add_delay_update_row).grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )

        settings = ttk.LabelFrame(root, text="Settings", padding=10)
        settings.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        settings.columnconfigure(1, weight=1)
        self._settings_row(settings, 0, "Target Activity ID:", self.delay_target_activity_id)

        instr = ttk.LabelFrame(root, text="Report Instruction (optional)", padding=10)
        instr.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        instr.columnconfigure(0, weight=1)
        self.delay_instruction = ScrolledText(instr, wrap="word", height=3)
        self.delay_instruction.grid(row=0, column=0, sticky="ew")

        run_row = ttk.Frame(root)
        run_row.grid(row=3, column=0, sticky="ew")
        run_row.columnconfigure(0, weight=1)
        self.delay_status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_row, textvariable=self.delay_status_var).grid(row=0, column=0, sticky="w")
        self.delay_run_button = ttk.Button(run_row, text="Run Analysis", command=self._run_delay_analysis)
        self.delay_run_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.delay_report_button = ttk.Button(run_row, text="Generate Delay Report", command=self._generate_delay_report)
        self.delay_report_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

        results_frame = ttk.LabelFrame(root, text="Results", padding=10)
        results_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.delay_results = ScrolledText(results_frame, wrap="none", height=14)
        self.delay_results.grid(row=0, column=0, sticky="nsew")
        self._write_to(
            self.delay_results,
            {"status": "ready", "hint": "Select a baseline and at least two updates, set the target, then Run Analysis."},
        )

    def _add_delay_update_row(self) -> None:
        var = tk.StringVar(value="")
        self.delay_update_paths.append(var)
        self._rebuild_delay_update_rows()

    def _remove_delay_update_row(self, var: tk.StringVar) -> None:
        # Keep at least two update slots.
        if len(self.delay_update_paths) <= 2:
            return
        self.delay_update_paths = [v for v in self.delay_update_paths if v is not var]
        self._rebuild_delay_update_rows()

    def _rebuild_delay_update_rows(self) -> None:
        for child in self.delay_updates_frame.winfo_children():
            child.destroy()
        for i, var in enumerate(self.delay_update_paths):
            ttk.Label(self.delay_updates_frame, text=f"Update {i + 1} XER:").grid(
                row=i, column=0, sticky="w", padx=(0, 8), pady=3
            )
            ttk.Entry(self.delay_updates_frame, textvariable=var).grid(row=i, column=1, sticky="ew", pady=3)
            ttk.Button(self.delay_updates_frame, text="Browse", command=lambda v=var: self._browse_xer(v)).grid(
                row=i, column=2, sticky="e", padx=(8, 0), pady=3
            )
            remove_btn = ttk.Button(
                self.delay_updates_frame, text="✕", width=3, command=lambda v=var: self._remove_delay_update_row(v)
            )
            remove_btn.grid(row=i, column=3, sticky="e", padx=(6, 0), pady=3)
            if len(self.delay_update_paths) <= 2:
                remove_btn.configure(state="disabled")

    def _file_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)

        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)

        ttk.Button(parent, text="Browse", command=lambda: self._browse_xer(var)).grid(
            row=row, column=2, sticky="e", padx=(8, 0), pady=4
        )

    def _settings_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)

    def _secret_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var, show="•").grid(row=row, column=1, sticky="ew", pady=4)

    def _model_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(
            parent,
            textvariable=var,
            state="readonly",
            values=ne.GEMINI_MODEL_OPTIONS,
        )
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        normalized = ne.normalize_gemini_model(var.get())
        if normalized not in ne.GEMINI_MODEL_OPTIONS:
            normalized = ne.DEFAULT_GEMINI_MODEL
        var.set(normalized)

    def _browse_xer(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select XER File",
            filetypes=[("Primavera XER", "*.xer *.XER"), ("All Files", "*.*")],
        )
        if path:
            var.set(path)

    def _validate_inputs(self) -> dict[str, Any]:
        baseline = Path(self.baseline_path.get().strip())
        last = Path(self.last_path.get().strip())
        current = Path(self.current_path.get().strip())

        missing = [p for p in [baseline, last, current] if not p.as_posix() or not p.exists()]
        if missing:
            raise ValueError("One or more XER file paths are missing or invalid.")

        target = self.target_activity_id.get().strip()
        if not target:
            raise ValueError("Target Activity ID is required.")

        try:
            threshold = int(self.near_critical_threshold.get().strip())
        except ValueError as e:
            raise ValueError("Near-Critical Threshold must be an integer.") from e

        try:
            look_ahead = int(self.look_ahead_horizon.get().strip())
        except ValueError as e:
            raise ValueError("Look-Ahead Horizon must be an integer (days).") from e

        return {
            "baseline": baseline,
            "last": last,
            "current": current,
            "target_activity_id": target,
            "variance_threshold": threshold,
            "look_ahead_horizon_days": look_ahead,
        }

    def _run_analysis(self) -> None:
        try:
            inputs = self._validate_inputs()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        self._last_compare_result = None
        self.status_var.set("Running analysis...")
        self.run_button.configure(state="disabled")
        self.narrative_button.configure(state="disabled")
        self.configure(cursor="watch")
        self.update_idletasks()

        try:
            baseline = xc.snapshot_from_xer_path("baseline", inputs["baseline"])
            last = xc.snapshot_from_xer_path("last", inputs["last"])
            current = xc.snapshot_from_xer_path("current", inputs["current"])

            result = xc.compare_three_way(
                baseline,
                last,
                current,
                variance_threshold=inputs["variance_threshold"],
                target_activity_id=inputs["target_activity_id"],
                look_ahead_horizon_days=inputs["look_ahead_horizon_days"],
            )
            self._last_compare_result = result
            self._write_results(result)
            self.status_var.set("Analysis complete.")
            target_warning = (result.get("critical_path_to_target", {}) or {}).get("target_warning")
            if target_warning:
                messagebox.showwarning("Target Activity Check", target_warning)
        except Exception as e:
            messagebox.showerror("Run Error", str(e))
            self.status_var.set("Error.")
        finally:
            self.configure(cursor="")
            self.run_button.configure(state="normal")
            self.narrative_button.configure(state="normal")

    def _write_to(self, widget: ScrolledText, obj: Any) -> None:
        try:
            text = json.dumps(obj, indent=2, default=str)
        except Exception:
            text = str(obj)

        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _write_results(self, obj: Any) -> None:
        self._write_to(self.results, obj)

    def _check_network(self) -> None:
        self.status_var.set("Checking network...")
        self.netcheck_button.configure(state="disabled")

        def worker() -> None:
            try:
                info = ne.probe_gemini_connectivity(timeout_s=5)
                text = json.dumps(info, indent=2, default=str)
            except Exception as exc:
                info = {"error": str(exc)}
                text = json.dumps(info, indent=2, default=str)

            def done() -> None:
                self.netcheck_button.configure(state="normal")
                self.status_var.set("Network check complete.")
                self._write_results({"network_probe": info})
                messagebox.showinfo("Network Check", text)

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _show_diagnostics(self) -> None:
        self.status_var.set("Collecting diagnostics...")
        self.diag_button.configure(state="disabled")

        def worker() -> None:
            try:
                info = ne.probe_gemini_connectivity(timeout_s=5)
            except Exception as exc:
                info = {"error": str(exc)}

            try:
                key_diag = ne._key_diagnostics()  # type: ignore[attr-defined]
            except Exception as exc:
                key_diag = {"error": str(exc)}

            try:
                recent_logs = ne.get_recent_logs(max_lines=250)
            except Exception:
                recent_logs = ""

            payload = {
                "build": {
                    "label": _build_label(),
                    "sha": (getattr(_build, "BUILD_SHA", None) if _build else None),
                    "time_utc": (getattr(_build, "BUILD_TIME_UTC", None) if _build else None),
                },
                "app": {
                    "ai_model": self.ai_model.get().strip(),
                    "proxy_url_set": bool(self.proxy_url.get().strip()),
                    "user_token_set": bool(self.user_token.get().strip()),
                },
                "network_probe": info,
                "key_diagnostics": key_diag,
                "recent_logs": recent_logs,
            }

            text = json.dumps(payload, indent=2, default=str)

            def done() -> None:
                self.diag_button.configure(state="normal")
                self.status_var.set("Diagnostics ready.")

                win = tk.Toplevel(self)
                win.title("Diagnostics")
                win.minsize(900, 600)
                win.columnconfigure(0, weight=1)
                win.rowconfigure(1, weight=1)

                btns = ttk.Frame(win, padding=10)
                btns.grid(row=0, column=0, sticky="ew")
                btns.columnconfigure(0, weight=1)

                def copy_to_clipboard() -> None:
                    win.clipboard_clear()
                    win.clipboard_append(text)
                    win.update_idletasks()

                def save_document() -> None:
                    path = filedialog.asksaveasfilename(
                        title="Save Diagnostics",
                        defaultextension=".json",
                        filetypes=[("JSON", "*.json"), ("Text", "*.txt"), ("All Files", "*.*")],
                    )
                    if not path:
                        return
                    Path(path).write_text(text, encoding="utf-8")

                ttk.Button(btns, text="Copy to Clipboard", command=copy_to_clipboard).grid(
                    row=0, column=1, sticky="e", padx=(8, 0)
                )
                ttk.Button(btns, text="Save", command=save_document).grid(row=0, column=2, sticky="e", padx=(8, 0))

                box = ScrolledText(win, wrap="none")
                box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
                box.insert("1.0", text)
                box.configure(state="disabled")

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _show_about(self) -> None:
        import sys

        info = {
            "build": {
                "label": _build_label(),
                "sha": (getattr(_build, "BUILD_SHA", None) if _build else None),
                "time_utc": (getattr(_build, "BUILD_TIME_UTC", None) if _build else None),
            },
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "genai_transport_selected": getattr(ne, "_select_genai_transport", lambda: None)() or "default",
        }
        messagebox.showinfo("About", json.dumps(info, indent=2, default=str))

    def _on_close(self) -> None:
        try:
            cfg = _load_config()
            cfg.update(
                {
                    "proxy_url": self.proxy_url.get().strip(),
                    "user_token": self.user_token.get().strip(),
                    "ai_model": ne.normalize_gemini_model(self.ai_model.get()),
                }
            )
            _save_config(cfg)
        except Exception:
            pass
        self.destroy()

    def _generate_narrative(self) -> None:
        if not self._last_compare_result:
            messagebox.showerror("Generate Narrative", "Run the analysis first, then generate the narrative report.")
            return

        digest = xc.get_ai_ready_digest(self._last_compare_result)

        proxy_url = self.proxy_url.get().strip()
        token = self.user_token.get().strip()
        if bool(proxy_url) ^ bool(token):
            messagebox.showerror(
                "Generate Narrative",
                "To use the Narrative Proxy you must provide BOTH:\n"
                "  - Narrative Proxy URL\n"
                "  - User Token\n\n"
                "Or clear both fields to use direct Gemini (developer/test mode).",
            )
            return
        try:
            cfg = _load_config()
            cfg.update(
                {
                    "proxy_url": proxy_url,
                    "user_token": token,
                    "ai_model": ne.normalize_gemini_model(self.ai_model.get()),
                }
            )
            _save_config(cfg)
        except Exception:
            pass

        self.status_var.set("Generating narrative...")
        self.narrative_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        t0 = time.perf_counter()

        def worker() -> None:
            try:
                model = ne.normalize_gemini_model(self.ai_model.get())
                if proxy_url and token:
                    narrative = ne.generate_narrative_via_proxy(
                        digest,
                        proxy_url=proxy_url,
                        user_token=token,
                        model=model,
                    )
                else:
                    # Direct Gemini calls can take a while depending on model + network.
                    narrative = ne.generate_narrative(digest, model=model)
            except Exception as exc:
                msg = str(exc)
                elapsed = time.perf_counter() - t0
                self.after(0, lambda m=msg, s=elapsed: self._on_narrative_error(m, elapsed_s=s))
                return
            elapsed = time.perf_counter() - t0
            self.after(0, lambda s=elapsed: self._show_narrative_window(narrative, elapsed_s=s))

        threading.Thread(target=worker, daemon=True).start()

    def _on_narrative_error(self, msg: str, *, elapsed_s: float | None = None) -> None:
        if elapsed_s is None:
            self.status_var.set("Error.")
        else:
            self.status_var.set(f"Error after {elapsed_s:.1f}s.")
        self.narrative_button.configure(state="normal")
        self.run_button.configure(state="normal")
        messagebox.showerror("Narrative Error", msg)

    def _show_narrative_window(self, narrative: str, *, elapsed_s: float | None = None) -> None:
        if elapsed_s is None:
            self.status_var.set("Narrative ready.")
        else:
            self.status_var.set(f"Narrative ready ({elapsed_s:.1f}s).")
        self.narrative_button.configure(state="normal")
        self.run_button.configure(state="normal")

        win = tk.Toplevel(self)
        win.title("Narrative Report")
        win.minsize(900, 600)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(1, weight=1)

        btns = ttk.Frame(win, padding=10)
        btns.grid(row=0, column=0, sticky="ew")
        btns.columnconfigure(0, weight=1)

        def copy_to_clipboard() -> None:
            win.clipboard_clear()
            win.clipboard_append(narrative)
            win.update_idletasks()

        def save_document() -> None:
            path = filedialog.asksaveasfilename(
                title="Save Narrative Report",
                defaultextension=".txt",
                filetypes=[("Text Document", "*.txt"), ("Markdown", "*.md"), ("All Files", "*.*")],
            )
            if not path:
                return
            Path(path).write_text(narrative, encoding="utf-8")

        ttk.Button(btns, text="Copy to Clipboard", command=copy_to_clipboard).grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Button(btns, text="Save as Document", command=save_document).grid(row=0, column=2, sticky="e", padx=(8, 0))

        text = ScrolledText(win, wrap="word")
        text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        text.insert("1.0", narrative)
        text.configure(state="disabled")

    # ---------------- Delay Analysis tab ----------------

    def _validate_delay_inputs(self) -> dict[str, Any]:
        baseline = self.delay_baseline_path.get().strip()
        if not baseline or not Path(baseline).exists():
            raise ValueError("Baseline XER path is missing or invalid.")

        updates: list[str] = []
        for var in self.delay_update_paths:
            p = var.get().strip()
            if not p:
                continue
            if not Path(p).exists():
                raise ValueError(f"Update XER path does not exist:\n{p}")
            updates.append(p)
        if len(updates) < 2:
            raise ValueError("Provide at least two update XER files for delay analysis.")

        target = self.delay_target_activity_id.get().strip()
        if not target:
            raise ValueError("Target Activity ID is required.")

        return {"baseline": baseline, "updates": updates, "target": target}

    def _run_delay_analysis(self) -> None:
        try:
            inputs = self._validate_delay_inputs()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        self._last_delay_result = None
        self.delay_status_var.set("Running delay analysis...")
        self.delay_run_button.configure(state="disabled")
        self.delay_report_button.configure(state="disabled")
        self.configure(cursor="watch")
        self.update_idletasks()

        try:
            result = da.analyze_delays_from_paths(
                inputs["baseline"],
                inputs["updates"],
                target_activity_id=inputs["target"],
                variance_threshold=DELAY_FLOAT_TOLERANCE_DAYS,
            )
            self._last_delay_result = result
            self._write_to(self.delay_results, result)
            n_changed = sum(1 for t in result.get("transitions", []) if t.get("path_changed"))
            self.delay_status_var.set(
                f"Delay analysis complete. {result['settings']['update_count']} updates, {n_changed} path change(s)."
            )
            validation = result.get("target_validation", {}) or {}
            warnings = validation.get("warnings") or []
            if warnings:
                lead = (
                    "The target has no driving logic in one or more updates, so the delay story may be meaningless.\n\n"
                    if not validation.get("has_driving_logic_all_updates", True)
                    else ""
                )
                messagebox.showwarning("Target Activity Check", lead + "\n\n".join(warnings))
        except Exception as e:
            messagebox.showerror("Run Error", str(e))
            self.delay_status_var.set("Error.")
        finally:
            self.configure(cursor="")
            self.delay_run_button.configure(state="normal")
            self.delay_report_button.configure(state="normal")

    def _generate_delay_report(self) -> None:
        if not self._last_delay_result:
            messagebox.showerror("Generate Delay Report", "Run the delay analysis first, then generate the report.")
            return

        instruction = self.delay_instruction.get("1.0", "end").strip()
        digest = da.build_delay_digest(self._last_delay_result, instruction=instruction or None)

        proxy_url = self.proxy_url.get().strip()
        token = self.user_token.get().strip()
        if bool(proxy_url) ^ bool(token):
            messagebox.showerror(
                "Generate Delay Report",
                "To use the Narrative Proxy you must provide BOTH:\n"
                "  - Narrative Proxy URL\n"
                "  - User Token\n\n"
                "Or clear both fields to use direct Gemini (developer/test mode).",
            )
            return
        try:
            cfg = _load_config()
            cfg.update(
                {
                    "proxy_url": proxy_url,
                    "user_token": token,
                    "ai_model": ne.normalize_gemini_model(self.ai_model.get()),
                }
            )
            _save_config(cfg)
        except Exception:
            pass

        self.delay_status_var.set("Generating delay report...")
        self.delay_report_button.configure(state="disabled")
        self.delay_run_button.configure(state="disabled")
        t0 = time.perf_counter()

        def worker() -> None:
            try:
                model = ne.normalize_gemini_model(self.ai_model.get())
                if proxy_url and token:
                    report = ne.generate_delay_report_via_proxy(
                        digest,
                        proxy_url=proxy_url,
                        user_token=token,
                        model=model,
                        instruction=instruction or None,
                    )
                else:
                    report = ne.generate_delay_report(digest, instruction=instruction or None, model=model)
            except Exception as exc:
                msg = str(exc)
                elapsed = time.perf_counter() - t0
                self.after(0, lambda m=msg, s=elapsed: self._on_delay_error(m, elapsed_s=s))
                return
            elapsed = time.perf_counter() - t0
            self.after(0, lambda s=elapsed: self._show_delay_window(report, elapsed_s=s))

        threading.Thread(target=worker, daemon=True).start()

    def _on_delay_error(self, msg: str, *, elapsed_s: float | None = None) -> None:
        self.delay_status_var.set("Error." if elapsed_s is None else f"Error after {elapsed_s:.1f}s.")
        self.delay_report_button.configure(state="normal")
        self.delay_run_button.configure(state="normal")
        messagebox.showerror("Delay Report Error", msg)

    def _show_delay_window(self, report: str, *, elapsed_s: float | None = None) -> None:
        self.delay_status_var.set("Report ready." if elapsed_s is None else f"Report ready ({elapsed_s:.1f}s).")
        self.delay_report_button.configure(state="normal")
        self.delay_run_button.configure(state="normal")

        win = tk.Toplevel(self)
        win.title("Delay Analysis Report")
        win.minsize(900, 600)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(1, weight=1)

        btns = ttk.Frame(win, padding=10)
        btns.grid(row=0, column=0, sticky="ew")
        btns.columnconfigure(0, weight=1)

        def copy_to_clipboard() -> None:
            win.clipboard_clear()
            win.clipboard_append(report)
            win.update_idletasks()

        def save_document() -> None:
            path = filedialog.asksaveasfilename(
                title="Save Delay Analysis Report",
                defaultextension=".txt",
                filetypes=[("Text Document", "*.txt"), ("Markdown", "*.md"), ("All Files", "*.*")],
            )
            if not path:
                return
            Path(path).write_text(report, encoding="utf-8")

        ttk.Button(btns, text="Copy to Clipboard", command=copy_to_clipboard).grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Button(btns, text="Save as Document", command=save_document).grid(row=0, column=2, sticky="e", padx=(8, 0))

        text = ScrolledText(win, wrap="word")
        text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        text.insert("1.0", report)
        text.configure(state="disabled")

    # ---------------- Project Overview tab ----------------

    def _build_overview_tab(self, root: ttk.Frame) -> None:
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)  # Q&A pane takes all the spare vertical space.

        files = ttk.LabelFrame(root, text="File Selection", padding=10)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        self._file_row(files, 0, "XER File:", self.overview_path)

        instr = ttk.LabelFrame(root, text="Report Instruction (optional)", padding=10)
        instr.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        instr.columnconfigure(0, weight=1)
        self.overview_instruction = ScrolledText(instr, wrap="word", height=3)
        self.overview_instruction.grid(row=0, column=0, sticky="ew")

        run_row = ttk.Frame(root)
        run_row.grid(row=2, column=0, sticky="ew")
        run_row.columnconfigure(0, weight=1)
        self.overview_status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_row, textvariable=self.overview_status_var).grid(row=0, column=0, sticky="w")
        self.overview_report_button = ttk.Button(run_row, text="Generate Overview", command=self._generate_overview)
        self.overview_report_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.overview_briefing_button = ttk.Button(run_row, text="Handover Briefing", command=self._generate_handover)
        self.overview_briefing_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

        qa = ttk.LabelFrame(root, text="Ask about this schedule (direct Gemini key required)", padding=10)
        qa.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        qa.columnconfigure(0, weight=1)
        qa.rowconfigure(0, weight=1)
        self.overview_qa_transcript = ScrolledText(qa, wrap="word", height=18, state="disabled")
        self.overview_qa_transcript.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.overview_qa_question = tk.StringVar(value="")
        entry = ttk.Entry(qa, textvariable=self.overview_qa_question)
        entry.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        entry.bind("<Return>", lambda _e: self._ask_overview_question())
        self.overview_ask_button = ttk.Button(qa, text="Ask", command=self._ask_overview_question)
        self.overview_ask_button.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(8, 0))

    def _generate_overview(self) -> None:
        if not self._ensure_overview_loaded() or not self._last_overview_result:
            return

        instruction = self.overview_instruction.get("1.0", "end").strip()
        digest = si.build_investigation_digest(
            self._last_overview_result["overview"],
            self._last_overview_result["backbones"],
            instruction=instruction or None,
        )

        proxy_url = self.proxy_url.get().strip()
        token = self.user_token.get().strip()
        if bool(proxy_url) ^ bool(token):
            messagebox.showerror(
                "Generate Overview",
                "To use the Narrative Proxy you must provide BOTH the Proxy URL and User Token, "
                "or clear both to use direct Gemini.",
            )
            return

        self.overview_status_var.set("Generating overview...")
        self.overview_report_button.configure(state="disabled")
        self.overview_briefing_button.configure(state="disabled")
        t0 = time.perf_counter()

        def worker() -> None:
            try:
                model = ne.normalize_gemini_model(self.ai_model.get())
                if proxy_url and token:
                    report = ne.generate_narrative_via_proxy(digest, proxy_url=proxy_url, user_token=token, model=model)
                else:
                    report = ne.generate_project_overview(digest, instruction=instruction or None, model=model)
            except Exception as exc:
                msg = str(exc)
                elapsed = time.perf_counter() - t0
                self.after(0, lambda m=msg, s=elapsed: self._on_overview_error(m, elapsed_s=s))
                return
            elapsed = time.perf_counter() - t0
            self.after(0, lambda s=elapsed: self._show_overview_window(report, elapsed_s=s))

        threading.Thread(target=worker, daemon=True).start()

    def _append_qa(self, role: str, text: str) -> None:
        self.overview_qa_transcript.configure(state="normal")
        self.overview_qa_transcript.insert("end", f"{role}: {text}\n\n")
        self.overview_qa_transcript.see("end")
        self.overview_qa_transcript.configure(state="disabled")

    def _ask_overview_question(self) -> None:
        question = self.overview_qa_question.get().strip()
        if not question:
            return
        if not self._ensure_overview_loaded():
            return
        # Q&A runs the tool-calling agent locally, so it needs a direct Gemini key (not the proxy).
        proxy_url = self.proxy_url.get().strip()
        token = self.user_token.get().strip()
        if proxy_url and token and not ne.find_api_key():
            messagebox.showinfo(
                "Ask",
                "Q&A runs a local tool-calling agent and needs a direct Gemini key "
                "(GEMINI_API_KEY in environment or a .env file). The Narrative Proxy is not used for Q&A in this version.",
            )
            return

        self.overview_qa_question.set("")
        self._append_qa("You", question)
        self.overview_ask_button.configure(state="disabled")
        self.overview_status_var.set("Thinking...")
        snapshot = self._loaded_overview_snapshot
        model = ne.normalize_gemini_model(self.ai_model.get())

        def worker() -> None:
            try:
                answer = sqa.answer_question(snapshot, question, model=model)
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda m=msg: self._finish_qa(f"[Error] {m}"))
                return
            self.after(0, lambda a=answer: self._finish_qa(a))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_qa(self, answer: str) -> None:
        self._append_qa("Assistant", answer)
        self.overview_ask_button.configure(state="normal")
        self.overview_status_var.set("Ready.")

    def _ensure_overview_loaded(self) -> bool:
        """Load + analyze the selected XER if it hasn't been already, so Analyze is optional."""
        if self._loaded_overview_snapshot is not None:
            return True
        path = self.overview_path.get().strip()
        if not path or not Path(path).exists():
            messagebox.showerror("Project Overview", "Select a valid XER file first.")
            return False
        self.overview_status_var.set("Loading schedule...")
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            snap = xc.snapshot_from_xer_path("schedule", path)
            self._loaded_overview_snapshot = snap
            self._last_overview_result = {"overview": si.project_overview(snap), "backbones": si.longest_backbones(snap)}
            self.overview_status_var.set("Schedule loaded.")
            return True
        except Exception as e:
            messagebox.showerror("Project Overview", str(e))
            self.overview_status_var.set("Error.")
            return False
        finally:
            self.configure(cursor="")

    def _generate_handover(self) -> None:
        if not self._ensure_overview_loaded():
            return
        # The briefing sends the whole parsed schedule, so it runs against direct Gemini (like Q&A).
        if not ne.find_api_key():
            messagebox.showinfo(
                "Handover Briefing",
                "The handover briefing sends the full schedule and needs a direct Gemini key "
                "(GEMINI_API_KEY in environment or a .env file). It does not use the Narrative Proxy.",
            )
            return

        try:
            payload, est_tokens = si.build_handover_payload(self._loaded_overview_snapshot)
        except Exception as e:
            messagebox.showerror("Handover Briefing", str(e))
            return
        if est_tokens > si.RICH_OVERVIEW_TOKEN_LIMIT:
            messagebox.showwarning(
                "Handover Briefing",
                f"This schedule is large (~{est_tokens:,} tokens) and may exceed the model's context for a "
                "single-shot briefing. Use 'Generate Overview' (summarized) or the Q&A box instead.",
            )
            return

        instruction = self.overview_instruction.get("1.0", "end").strip()
        self.overview_status_var.set("Generating handover briefing...")
        self.overview_briefing_button.configure(state="disabled")
        self.overview_report_button.configure(state="disabled")
        t0 = time.perf_counter()

        def worker() -> None:
            try:
                model = ne.normalize_gemini_model(self.ai_model.get())
                report = ne.generate_handover_briefing(payload, instruction=instruction or None, model=model)
            except Exception as exc:
                msg = str(exc)
                elapsed = time.perf_counter() - t0
                self.after(0, lambda m=msg, s=elapsed: self._on_overview_error(m, elapsed_s=s))
                return
            elapsed = time.perf_counter() - t0
            self.after(0, lambda s=elapsed: self._show_overview_window(report, elapsed_s=s))

        threading.Thread(target=worker, daemon=True).start()

    def _on_overview_error(self, msg: str, *, elapsed_s: float | None = None) -> None:
        self.overview_status_var.set("Error." if elapsed_s is None else f"Error after {elapsed_s:.1f}s.")
        self.overview_report_button.configure(state="normal")
        self.overview_briefing_button.configure(state="normal")
        messagebox.showerror("Overview Error", msg)

    def _show_overview_window(self, report: str, *, elapsed_s: float | None = None) -> None:
        self.overview_status_var.set("Overview ready." if elapsed_s is None else f"Overview ready ({elapsed_s:.1f}s).")
        self.overview_report_button.configure(state="normal")
        self.overview_briefing_button.configure(state="normal")

        win = tk.Toplevel(self)
        win.title("Project Overview Report")
        win.minsize(900, 600)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(1, weight=1)

        btns = ttk.Frame(win, padding=10)
        btns.grid(row=0, column=0, sticky="ew")
        btns.columnconfigure(0, weight=1)

        def copy_to_clipboard() -> None:
            win.clipboard_clear()
            win.clipboard_append(report)
            win.update_idletasks()

        def save_document() -> None:
            p = filedialog.asksaveasfilename(
                title="Save Project Overview",
                defaultextension=".txt",
                filetypes=[("Text Document", "*.txt"), ("Markdown", "*.md"), ("All Files", "*.*")],
            )
            if p:
                Path(p).write_text(report, encoding="utf-8")

        ttk.Button(btns, text="Copy to Clipboard", command=copy_to_clipboard).grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Button(btns, text="Save as Document", command=save_document).grid(row=0, column=2, sticky="e", padx=(8, 0))

        text = ScrolledText(win, wrap="word")
        text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        text.insert("1.0", report)
        text.configure(state="disabled")


def main() -> None:
    app = ScheduleAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
