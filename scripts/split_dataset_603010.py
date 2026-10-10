#!/usr/bin/env python3
"""Deterministic 40/50/10 patient split (paper Appendix C) for the open release (stdlib only).

One salt, three mutually exclusive bands covering every patient:
    h = md5("split603010:<seed>:<pid>") % 100   (salt name is historical)
        h <  40          -> SFT   (0.4)
        40 <= h < 90     -> RL    (0.5)
        h >= 90          -> TEST  (0.1)

Per trajectory dir this writes <dir>/split_<seed>/:
    sft_kept.jsonl     {"patient_id","question_idx"} - ALL questions of SFT patients
    rl_patients.txt    RL patient ids (one per line)
    rl_kept.jsonl      {"patient_id","question_idx"} - up to --mpp questions per RL
                       patient, chosen by an md5-stable shuffle (reproducible on
                       any machine, unlike python's randomized hash())
    test_patients.txt  TEST patient ids (one per line)

Ship this script together with the raw trajectory dirs: anyone can re-derive
the exact same split from the raw data alone, no extra metadata needed.

Usage:
    python scripts/split_dataset_603010.py \
        --trajectory-dir /path/vista3d_traj_0811fix \
        --trajectory-dir /path/medpanel_v2_trajectory_0811fix \
        --trajectory-dir /path/medboard_trajectory_0820_full \
        --seed 20260926 --mpp 4
"""
import argparse
import hashlib
import json
import os
import random


def band(pid: str, seed: int) -> int:
    return int(hashlib.md5(f"split603010:{seed}:{pid}".encode()).hexdigest(), 16) % 100


def mpp_questions(questions, pid: str, seed: int, mpp: int):
    if mpp <= 0 or len(questions) <= mpp:
        return questions
    h = int(hashlib.md5(f"mpp:{seed}:{pid}".encode()).hexdigest(), 16)
    qs = list(questions)
    random.Random(h).shuffle(qs)
    return qs[:mpp]


def qidx(q):
    return q.get("question_idx", q.get("index", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-dir", action="append", required=True,
                    help="repeatable; dir containing dense_<pid>.json files")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--sft-frac", type=float, default=0.4)
    ap.add_argument("--rl-frac", type=float, default=0.5)
    ap.add_argument("--mpp", type=int, default=4,
                    help="questions kept per RL patient (0 = all)")
    args = ap.parse_args()

    sft_hi = int(round(args.sft_frac * 100))
    rl_hi = sft_hi + int(round(args.rl_frac * 100))
    assert 0 < sft_hi < rl_hi < 100, (sft_hi, rl_hi)

    for d in args.trajectory_dir:
        out = os.path.join(d, f"split_{args.seed}")
        os.makedirs(out, exist_ok=True)
        n_sft = n_rl = n_test = q_sft = q_rl = 0
        with open(f"{out}/sft_kept.jsonl", "w") as fs, \
             open(f"{out}/rl_kept.jsonl", "w") as fr, \
             open(f"{out}/rl_patients.txt", "w") as pr, \
             open(f"{out}/test_patients.txt", "w") as pt:
            names = sorted(f for f in os.listdir(d)
                           if f.startswith("dense_") and f.endswith(".json"))
            for fname in names:
                pid = fname[len("dense_"):-len(".json")]
                with open(os.path.join(d, fname)) as fh:
                    data = json.load(fh)
                pid_s = str(data.get("patient_id", pid))
                qs = data.get("questions", [])
                h = band(pid, args.seed)
                if h < sft_hi:
                    n_sft += 1
                    for q in qs:
                        fs.write(json.dumps({"patient_id": pid_s, "question_idx": qidx(q)}) + "\n")
                        q_sft += 1
                elif h < rl_hi:
                    n_rl += 1
                    pr.write(pid + "\n")
                    for q in mpp_questions(qs, pid, args.seed, args.mpp):
                        fr.write(json.dumps({"patient_id": pid_s, "question_idx": qidx(q)}) + "\n")
                        q_rl += 1
                else:
                    n_test += 1
                    pt.write(pid + "\n")
        print(f"{os.path.basename(d)}: SFT {n_sft} patients / {q_sft} questions, "
              f"RL {n_rl} patients / {q_rl} questions, TEST {n_test} patients "
              f"(bands {sft_hi}/{rl_hi - sft_hi}/{100 - rl_hi}, seed {args.seed})")


if __name__ == "__main__":
    main()
