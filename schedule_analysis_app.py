from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any
import platform

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import xer_comparator as xc
import narrative_engine as ne


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
        self.title("Schedule Analysis Tool (XER)")
        self.minsize(900, 650)

        self.baseline_path = tk.StringVar(value="")
        self.last_path = tk.StringVar(value="")
        self.current_path = tk.StringVar(value="")

        self.target_activity_id = tk.StringVar(value="A3000")
        self.near_critical_threshold = tk.StringVar(value="8")
        self.look_ahead_horizon = tk.StringVar(value="30")

        self._last_compare_result: dict[str, Any] | None = None

        cfg = _load_config()
        self.ai_model = tk.StringVar(value=str(cfg.get("ai_model") or "gemini-3-flash-preview"))
        self.proxy_url = tk.StringVar(value=str(cfg.get("proxy_url") or ""))
        self.user_token = tk.StringVar(value=str(cfg.get("user_token") or ""))

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

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
        self._model_row(settings, 3, "AI Logic Engine:", self.ai_model)
        self._settings_row(settings, 4, "Narrative Proxy URL:", self.proxy_url)
        self._secret_row(settings, 5, "User Token:", self.user_token)

        run_row = ttk.Frame(root)
        run_row.grid(row=2, column=0, sticky="ew")
        run_row.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(run_row, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.run_button = ttk.Button(run_row, text="Run", command=self._run_analysis)
        self.run_button.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.narrative_button = ttk.Button(run_row, text="Generate Narrative Report", command=self._generate_narrative)
        self.narrative_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

        results_frame = ttk.LabelFrame(root, text="Results", padding=10)
        results_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results = ScrolledText(results_frame, wrap="none", height=20)
        self.results.grid(row=0, column=0, sticky="nsew")

        self._write_results(
            {
                "status": "ready",
                "hint": "Select three XER files, set inputs, then click Run.",
            }
        )

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
            values=[
                "gemini-3-flash-preview",
                "gemini-3-pro-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
            ],
        )
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        if not var.get().strip():
            var.set("gemini-3-flash-preview")

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
        except Exception as e:
            messagebox.showerror("Run Error", str(e))
            self.status_var.set("Error.")
        finally:
            self.configure(cursor="")
            self.run_button.configure(state="normal")
            self.narrative_button.configure(state="normal")

    def _write_results(self, obj: Any) -> None:
        try:
            text = json.dumps(obj, indent=2, default=str)
        except Exception:
            text = str(obj)

        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", text)
        self.results.configure(state="disabled")

    def _on_close(self) -> None:
        try:
            cfg = _load_config()
            cfg.update(
                {
                    "proxy_url": self.proxy_url.get().strip(),
                    "user_token": self.user_token.get().strip(),
                    "ai_model": self.ai_model.get().strip() or "gemini-3-flash-preview",
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
                    "ai_model": (self.ai_model.get().strip() or "gemini-3-flash-preview"),
                }
            )
            _save_config(cfg)
        except Exception:
            pass

        self.status_var.set("Generating narrative...")
        self.narrative_button.configure(state="disabled")
        self.run_button.configure(state="disabled")

        def worker() -> None:
            try:
                model = self.ai_model.get().strip() or "gemini-3-flash-preview"
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
                self.after(0, lambda m=msg: self._on_narrative_error(m))
                return
            self.after(0, lambda: self._show_narrative_window(narrative))

        threading.Thread(target=worker, daemon=True).start()

    def _on_narrative_error(self, msg: str) -> None:
        self.status_var.set("Error.")
        self.narrative_button.configure(state="normal")
        self.run_button.configure(state="normal")
        messagebox.showerror("Narrative Error", msg)

    def _show_narrative_window(self, narrative: str) -> None:
        self.status_var.set("Narrative ready.")
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


def main() -> None:
    app = ScheduleAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
