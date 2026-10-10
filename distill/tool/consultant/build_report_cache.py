#!/usr/bin/env python3
"""Build the MedGemma report tool cache for the medboard module.

Scans MedGemma inference outputs (fixed multi-slice -> findings/impression/
followup reports) across 5 organ dirs and flattens them into a replay cache
with the same key convention as the other medboard/vista3d caches:

    {patient_id}::list_available_reports
    {patient_id}::get_report::{organ}::{report_type}     # findings|impression|followup

Report text is formatted and truncated so a single tool response stays under
the RL/agent tool-response budget (~2048 chars).

Usage:
    python distill/tool/consultant/build_report_cache.py \
        [--patient-ids 1005896,1006424] [--organs liver,wholebody] \
        [--output /mnt/faster_nas/benyou.mys/medixrl/data/medgemma_report_cache/tool_cache.json]
"""
import argparse
import json
import os
import re
import sys

OSS_ROOT = "/root/h100_data_transfer_oss/huakun.ych/minye/inference_standalone"

# organ -> (report subdir name); sequence subdirs are auto-detected.
SOURCES = {
    "brain": "output_brain_merge_0512/medgemma27b_brain_reports",
    "heart": "output_heart/medgemma27b_heart_reports",
    "liver": "output_liver/medgemma27b_liver_reports",
    "pancreas": "output_pancreas/medgemma27b_pancreas_reports",
    "wholebody": "output_wholebody/medgemma27b_wholebody_report",
}

REPORT_TYPES = ("findings", "impression", "followup")
MAX_TEXT_CHARS = 1200


def _clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


# ICD-10 code token: letter + 2 digits (+ optional .digits), e.g. I10, E78.0.
# Scrubbed from free text because MedGemma may echo the cohort's coded
# diagnoses, which are the question-bank answers themselves.
_ICD_TOKEN = re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,2})?\b")


def mask_diagnosis_codes(text):
    return _ICD_TOKEN.sub("[code]", text)


def format_findings(doc):
    parts = [f"[MedGemma findings | {doc.get('anatomy', '?')} | {doc.get('modality', '?')}]"]
    if doc.get("interval_change"):
        parts.append("Interval change: " + _clip(doc["interval_change"], 400))
    return mask_diagnosis_codes(_clip("\n".join(parts), MAX_TEXT_CHARS))


def format_impression(doc):
    parts = [f"[MedGemma impression | {doc.get('anatomy', '?')} | {doc.get('modality', '?')}]"]
    if doc.get("impression"):
        parts.append("Impression: " + _clip(doc["impression"], 400))
    return mask_diagnosis_codes(_clip("\n".join(parts), MAX_TEXT_CHARS))


def format_followup(doc):
    rep = doc.get("report") or {}
    parts = [f"[MedGemma follow-up | {doc.get('anatomy', '?')} | {doc.get('modality', '?')}]"]
    parts.append(f"Compared: {doc.get('visit1_source', '?')} vs {doc.get('visit2_source', '?')}")
    if rep.get("comparison_summary"):
        parts.append("Summary: " + _clip(rep["comparison_summary"], 400))
    return mask_diagnosis_codes(_clip("\n".join(parts), MAX_TEXT_CHARS))


FORMATTERS = {
    "findings": format_findings,
    "impression": format_impression,
    "followup": format_followup,
}


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def process_patient_dir(pid, organ, seq, pdir, cache):
    """Add report entries for one patient/organ/sequence dir. Returns #reports."""
    docs = {}
    for rt in REPORT_TYPES:
        doc = _load_json(os.path.join(pdir, f"{rt}_report.json"))
        if doc:
            docs[rt] = doc
    if not docs:
        return 0

    list_key = f"{pid}::list_available_reports"
    entry = cache.setdefault(list_key, {"text": "[Available MedGemma reports]\n", "images": []})
    entry["text"] += f"- {organ} ({seq}): " + ", ".join(sorted(docs)) + "\n"

    for rt, doc in docs.items():
        key = f"{pid}::get_report::{organ}::{rt}"
        if key in cache:
            # Same patient+organ across sequences: append a seq-labelled variant.
            cache[key]["text"] = _clip(
                cache[key]["text"] + f"\n\n--- sequence {seq} ---\n" + FORMATTERS[rt](doc),
                MAX_TEXT_CHARS + 400)
        else:
            cache[key] = {"text": FORMATTERS[rt](doc), "images": []}
    return len(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oss-root", default=OSS_ROOT)
    ap.add_argument("--organs", default=",".join(SOURCES),
                    help="comma-separated subset of organs")
    ap.add_argument("--patient-ids", default="",
                    help="comma-separated filter; empty = all (slow on OSS mount)")
    ap.add_argument("--output",
                    default="/mnt/faster_nas/benyou.mys/medixrl/data/medgemma_report_cache/tool_cache.json")
    args = ap.parse_args()

    wanted = set(x.strip() for x in args.patient_ids.split(",") if x.strip()) or None
    cache = {}
    n_patients = n_reports = 0

    for organ in args.organs.split(","):
        organ = organ.strip()
        if organ not in SOURCES:
            print(f"[warn] unknown organ {organ}, skip", file=sys.stderr)
            continue
        report_root = os.path.join(args.oss_root, SOURCES[organ])
        if not os.path.isdir(report_root):
            print(f"[warn] missing {report_root}", file=sys.stderr)
            continue
        seqs = [d for d in sorted(os.listdir(report_root))
                if os.path.isdir(os.path.join(report_root, d))]
        for seq in seqs:
            seq_dir = os.path.join(report_root, seq)
            pids = sorted(d for d in os.listdir(seq_dir)
                          if d.isdigit() and (wanted is None or d in wanted))
            print(f"[build] {organ}/{seq}: {len(pids)} patients", flush=True)
            for pid in pids:
                n_rep = process_patient_dir(pid, organ, seq, os.path.join(seq_dir, pid), cache)
                if n_rep:
                    n_reports += n_rep
                    n_patients += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cache, f, ensure_ascii=False)
    n_list = sum(1 for k in cache if k.endswith("::list_available_reports"))
    print(f"done: {n_list} patients with reports, {n_reports} report docs, "
          f"{len(cache)} cache keys -> {args.output}")


if __name__ == "__main__":
    main()
