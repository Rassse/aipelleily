#!/usr/bin/env python3
"""
Tkinter GUI for the mortgage keyword -> Exa search script.

Put this file next to keyword_extractor.py and run:  python3 gui.py
The API key is read from EXA_API_KEY (or a .env file, if python-dotenv is installed).
Optional: qwen_queries.py (same folder) lets a local Qwen model write the queries.
"""

import json
import queue
import threading
import time
import tkinter as tk
from itertools import chain, zip_longest
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from keyword_extractor import (
    extract_keywords,
    find_csv_files,
    load_rows,
    looks_like_mortgage_csv,
)

CATEGORY_COLUMNS = ("event_type", "rejection_reason", "collateral_type", "property_type")
DEFAULT_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------- #
# Logic (no GUI code in here)
# --------------------------------------------------------------------------- #
def interleave(lists):
    """Round-robin merge so truncating the list doesn't drop whole columns."""
    return [k for k in chain.from_iterable(zip_longest(*lists)) if k]


def get_keywords(directory, recursive, top_n, include_phrases):
    files = find_csv_files(Path(directory).expanduser().resolve(), "*.csv", recursive)
    files = [p for p in files if looks_like_mortgage_csv(p)]
    if not files:
        raise ValueError("No mortgage CSV found in that directory.")

    per_source = []
    for path in files:
        result = extract_keywords(load_rows(path), top_n)
        for col in CATEGORY_COLUMNS:
            per_source.append([i["keyword"] for i in result["categorical_keywords"].get(col, [])])
        if include_phrases:
            per_source.append([i["keyword"] for i in result["top_phrases"]])

    merged = interleave(per_source)
    return list(dict.fromkeys(k.replace("_", " ") for k in merged))


def to_query(keyword):
    return keyword if "mortgage" in keyword.lower() else f"mortgage {keyword}"


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, root):
        self.root = root
        root.title("Mortgage keyword search (Exa)")
        root.geometry("820x640")
        root.minsize(640, 480)

        self.msgs = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.results = {}

        self.dir_var = tk.StringVar(value=str(DEFAULT_DIR))
        self.recursive_var = tk.BooleanVar(value=True)
        self.phrases_var = tk.BooleanVar(value=False)
        self.qwen_var = tk.BooleanVar(value=False)
        self.top_var = tk.IntVar(value=5)
        self.max_q_var = tk.IntVar(value=20)
        self.num_var = tk.IntVar(value=5)
        self.status_var = tk.StringVar(value="Ready.")

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.poll)

    def _build(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

        # Directory row
        ttk.Label(frm, text="Data directory:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.dir_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Browse...", command=self.browse).grid(row=0, column=2)

        # Options
        opts = ttk.LabelFrame(frm, text="Options", padding=8)
        opts.grid(row=1, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Checkbutton(opts, text="Search subfolders", variable=self.recursive_var).grid(
            row=0, column=0, sticky="w", padx=(0, 15))
        ttk.Label(opts, text="Keywords per list:").grid(row=0, column=1, sticky="e")
        ttk.Spinbox(opts, from_=1, to=50, width=5, textvariable=self.top_var).grid(row=0, column=2, padx=(4, 15))
        ttk.Label(opts, text="Max queries:").grid(row=0, column=3, sticky="e")
        ttk.Spinbox(opts, from_=1, to=200, width=5, textvariable=self.max_q_var).grid(row=0, column=4, padx=(4, 15))
        ttk.Label(opts, text="Results per query:").grid(row=0, column=5, sticky="e")
        ttk.Spinbox(opts, from_=1, to=25, width=5, textvariable=self.num_var).grid(row=0, column=6, padx=4)

        ttk.Checkbutton(
            opts,
            text="Include phrases from notes (may contain personal data, sent to Exa)",
            variable=self.phrases_var,
            command=self.on_phrases_toggle,
        ).grid(row=1, column=0, columnspan=7, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            opts,
            text="Use local Qwen server to write the queries (replaces keyword extraction; LM Studio/Ollama must be running)",
            variable=self.qwen_var,
        ).grid(row=2, column=0, columnspan=7, sticky="w", pady=(4, 0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=3, sticky="w")
        self.preview_btn = ttk.Button(btns, text="Preview queries", command=self.preview)
        self.preview_btn.pack(side="left")
        self.run_btn = ttk.Button(btns, text="Run search", command=self.run)
        self.run_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.save_btn = ttk.Button(btns, text="Save JSON...", command=self.save, state="disabled")
        self.save_btn.pack(side="left", padx=5)
        ttk.Button(btns, text="Clear log", command=self.clear_log).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.on_close).pack(side="left", padx=5)

        # Progress + status
        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 2))
        ttk.Label(frm, textvariable=self.status_var).grid(row=4, column=0, columnspan=3, sticky="w")

        # Log
        self.log = scrolledtext.ScrolledText(frm, wrap="word", state="disabled", font=("Menlo", 10))
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

    # ---- helpers ---------------------------------------------------------- #
    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def browse(self):
        chosen = filedialog.askdirectory(initialdir=self.dir_var.get() or str(Path.home()))
        if chosen:
            self.dir_var.set(chosen)

    def on_phrases_toggle(self):
        if self.phrases_var.get():
            ok = messagebox.askokcancel(
                "Privacy warning",
                "Note text can contain names, employers, amounts or addresses. "
                "Phrases will be sent to Exa, a third-party API.\n\n"
                "Use 'Preview queries' first to check exactly what would be sent. Continue?",
            )
            if not ok:
                self.phrases_var.set(False)

    def read_settings(self):
        try:
            return {
                "directory": self.dir_var.get().strip(),
                "recursive": self.recursive_var.get(),
                "top_n": max(1, self.top_var.get()),
                "max_queries": max(1, self.max_q_var.get()),
                "num_results": max(1, self.num_var.get()),
                "phrases": self.phrases_var.get(),
                "qwen": self.qwen_var.get(),
            }
        except tk.TclError:
            messagebox.showerror("Invalid input", "Numeric options must be whole numbers.")
            return None

    def build_queries(self, s):
        keywords = get_keywords(s["directory"], s["recursive"], s["top_n"], s["phrases"])
        return [to_query(k) for k in keywords][: s["max_queries"]]

    # ---- actions ---------------------------------------------------------- #
    def preview(self):
        self.start_job(search=False)

    def run(self):
        self.start_job(search=True)

    def start_job(self, search):
        s = self.read_settings()
        if not s:
            return
        self.results = {}
        self.stop_event.clear()
        self.progress.configure(maximum=1, value=0)
        self.run_btn.configure(state="disabled")
        self.preview_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal" if search else "disabled")
        self.status_var.set("Preparing queries...")
        self.worker = threading.Thread(
            target=self.job_worker, args=(s, search), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Stopping after the current query...")

    def save(self):
        if not self.results:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="search_results.json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            Path(path).write_text(
                json.dumps(self.results, indent=2, ensure_ascii=False), encoding="utf-8")
            self.status_var.set(f"Saved to {path}")

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()

    # ---- background thread ------------------------------------------------ #
    def job_worker(self, s, do_search):
        """Runs off the UI thread; talks to the UI only through self.msgs."""
        # Step 1: build the queries (optionally with local Qwen, which is slow)
        try:
            if s["qwen"]:
                self.msgs.put(("status", "Asking the local Qwen server (can take a while "
                                         "on the first request)..."))
                from qwen_queries import queries_from_directory
                queries = queries_from_directory(
                    s["directory"], s["recursive"], s["max_queries"])
            else:
                queries = self.build_queries(s)
        except Exception as e:
            self.msgs.put(("fatal", f"Could not build queries: {e}"))
            return

        self.msgs.put(("queries", queries, do_search))
        if not queries:
            self.msgs.put(("log", "No usable queries were produced."))
            self.msgs.put(("preview_done", 0))
            return
        if not do_search:
            self.msgs.put(("preview_done", len(queries)))
            return

        # Step 2: search Exa
        try:
            from exa_py import Exa
            exa = Exa()  # reads EXA_API_KEY from the environment
        except Exception as e:
            self.msgs.put(("fatal", f"Could not start Exa: {e}"))
            return

        for i, query in enumerate(queries, 1):
            if self.stop_event.is_set():
                break
            try:
                res = exa.search(
                    query, type="auto", num_results=s["num_results"],
                    contents={"highlights": True})
                items = [{"url": r.url, "highlights": r.highlights} for r in res.results]
                self.results[query] = items
                self.msgs.put(("result", query, items))
            except Exception as e:
                self.msgs.put(("log", f"Skipping '{query}': {e}"))
            self.msgs.put(("progress", i))
            time.sleep(0.5)  # be gentle with rate limits

        self.msgs.put(("done", self.stop_event.is_set()))

    # ---- UI thread: drain queue ------------------------------------------ #
    def poll(self):
        try:
            while True:
                msg = self.msgs.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self.write(msg[1])
                elif kind == "result":
                    self.write(f"\n=== {msg[1]} ===")
                    for item in msg[2]:
                        self.write(f"{item['url']}  {item['highlights']}")
                elif kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "queries":
                    self.write(f"--- {len(msg[1])} queries ---")
                    for q in msg[1]:
                        self.write(f"  {q}")
                    self.progress.configure(maximum=max(1, len(msg[1])), value=0)
                    if msg[2] and msg[1]:
                        self.status_var.set(f"Running {len(msg[1])} queries...")
                elif kind == "preview_done":
                    self.finish()
                    self.status_var.set(f"Preview: {msg[1]} queries.")
                elif kind == "progress":
                    self.progress.configure(value=msg[1])
                elif kind == "fatal":
                    self.finish()
                    self.status_var.set("Failed.")
                    messagebox.showerror("Error", msg[1])
                elif kind == "done":
                    self.finish()
                    n = len(self.results)
                    self.status_var.set(
                        f"{'Stopped' if msg[1] else 'Done'}: {n} queries returned results.")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def finish(self):
        self.run_btn.configure(state="normal")
        self.preview_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.save_btn.configure(state="normal" if self.results else "disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()