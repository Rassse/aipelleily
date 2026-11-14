#!/usr/bin/env python3
"""
Local Qwen step: turn mortgage event data into safe, generic search queries
BEFORE anything is sent to Exa.

The model runs on your machine behind a local OpenAI-compatible server
(LM Studio or Ollama), so the raw notes never leave it. Its output is
still checked by plain code (is_safe) because an LLM is not a reliable
anonymizer on its own.

Setup (pick one, then load a Qwen model in it and start its local server):
    LM Studio : Developer tab -> start server   (default http://localhost:1234/v1)
    Ollama    : ollama serve                    (use http://localhost:11434/v1)

Optional environment variables:
    QWEN_BASE_URL   server URL, default http://localhost:1234/v1
    QWEN_MODEL      model id; if unset, the first model the server lists is used
    QWEN_API_KEY    only if your server wants one (default "local")

Usage:
    python3 qwen_queries.py ./data          # print the queries it would send
"""

import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from keyword_extractor import (
    find_csv_files,
    load_rows,
    looks_like_mortgage_csv,
)

LABEL_COLUMNS = ("event_type", "rejection_reason", "collateral_type", "property_type")
NOTES_PER_EVENT = 3        # sample notes shown to the model per event type
NOTE_MAX_CHARS = 200


# --------------------------------------------------------------------------- #
# Local model server (OpenAI-compatible: LM Studio, Ollama, vLLM, ...)
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:1234/v1")
_model_name_cache = os.environ.get("QWEN_MODEL", "")


def _request(path, payload=None, timeout=900):
    """GET if payload is None, otherwise POST JSON. Local generation can be slow."""
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("QWEN_API_KEY", "local"),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _model_name():
    global _model_name_cache
    if not _model_name_cache:
        models = _request("/models", timeout=30).get("data", [])
        if not models:
            raise RuntimeError("The server lists no models. Load a Qwen model first.")
        _model_name_cache = models[0]["id"]
    return _model_name_cache


def ask(prompt, max_tokens=2048):
    """Send one text-only prompt, return the answer without any <think> block."""
    try:
        resp = _request("/chat/completions", {
            "model": _model_name(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        })
    except OSError as e:  # URLError, connection refused, timeouts
        raise RuntimeError(
            f"Could not reach the local model server at {BASE_URL}. "
            f"Is it running with a model loaded? ({e})"
        ) from e
    text = resp["choices"][0]["message"].get("content") or ""
    # If the model "thinks" first, keep only what comes after the thinking.
    return text.split("</think>")[-1].strip()


# --------------------------------------------------------------------------- #
# Safety filter (plain code, not the LLM)
# --------------------------------------------------------------------------- #
ALLOWED_CAPS = {"Finland", "Finnish", "EU", "Euribor", "Suomi", "ECB"}


def is_safe(query):
    if re.search(r"\d|@|https?:|www\.", query):
        return False
    words = query.split()
    if not (2 <= len(words) <= 10):
        return False
    # capitalised words after the first one look like names -> reject
    for w in words[1:]:
        if w[:1].isupper() and w not in ALLOWED_CAPS:
            return False
    return True


# --------------------------------------------------------------------------- #
# Build prompt, ask the model, validate
# --------------------------------------------------------------------------- #
def collect(directory, recursive=True):
    files = find_csv_files(Path(directory).expanduser().resolve(), "*.csv", recursive)
    files = [p for p in files if looks_like_mortgage_csv(p)]
    if not files:
        raise ValueError("No mortgage CSV found.")

    labels = defaultdict(set)
    notes = defaultdict(list)  # event_type -> sample notes
    for path in files:
        for row in load_rows(path):
            for col in LABEL_COLUMNS:
                v = (row.get(col) or "").strip()
                if v:
                    labels[col].add(v)
            ev = (row.get("event_type") or "unknown").strip()
            note = (row.get("note") or "").strip()
            if note and len(notes[ev]) < NOTES_PER_EVENT and note not in notes[ev]:
                notes[ev].append(note[:NOTE_MAX_CHARS])
    return labels, notes


def build_prompt(labels, notes, max_queries):
    label_block = "\n".join(f"- {c}: {', '.join(sorted(v))}" for c, v in labels.items())
    note_block = "\n".join(
        f"[{ev}] " + " | ".join(ns) for ev, ns in notes.items()
    )
    return f"""You help research the mortgage domain on the public web.

Below are internal labels and a few sample notes from a mortgage event log.

LABELS
{label_block}

SAMPLE NOTES (internal, confidential)
{note_block}

Write at most {max_queries} short web search queries (3-8 words each) that would find
useful public information about the TOPICS these labels and notes cover, such as
regulations, typical rejection reasons, collateral rules, or process steps in Finland.

Hard rules:
- Generic topics only. NO names, companies, addresses, ids, dates, amounts or any
  digits, and do not copy phrases from the notes.
- Plain English, lowercase.
- Reply with ONLY a JSON array of strings and nothing else."""


def parse_json_list(text):
    match = re.search(r"\[.*\]", text, flags=re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in data if isinstance(q, str)]


def queries_from_directory(directory, recursive=True, max_queries=20):
    labels, notes = collect(directory, recursive)
    raw = parse_json_list(ask(build_prompt(labels, notes, max_queries)))
    safe = [q for q in raw if is_safe(q)]
    return list(dict.fromkeys(safe))[:max_queries]


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data"
    for q in queries_from_directory(folder):
        print(q)