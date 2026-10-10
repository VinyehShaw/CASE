#!/usr/bin/env python3
"""Generate the SFT kept-manifest (patient_id, question_idx) for a trajectory dir,
covering the full SFT/RL POOL (all patients EXCEPT the held-out TEST bucket).

Pairs with prepare_mixed_tool_rl_data.py --test-frac/--test-seed: both use the
same _is_test hash (salt "test:<seed>:<pid>"), so SFT (this manifest) + RL
(prepare --split rl) together consume exactly the non-TEST pool, and the TEST
bucket stays held out for final evaluation.

Usage:
    python scripts/gen_sft_kept_manifest.py \
        --trajectory-dir /mnt/faster_nas/benyou.mys/dataset/vista3d_traj_0811fix \
        --test-frac 0.1 --test-seed 20260812 \
        --out /mnt/faster_nas/benyou.mys/dataset/vista3d_traj_0811fix/sft_pool_kept.jsonl
"""
import argparse
import hashlib
import json
import os


def _is_test(pid, test_frac, test_seed):
    if not test_frac or test_frac <= 0:
        return False
    h = int(hashlib.md5(f"test:{test_seed}:{pid}".encode()).hexdigest(), 16)
    return h % 100 < int(test_frac * 100)


def _in_train_pool(pid, patient_frac, patient_seed, patient_min_frac=None):
    """Patient-level sampling over the non-TEST pool. Independent "train:" salt
    so it never collides with the TEST bucket. patient_frac>=1.0 keeps everyone.
    patient_min_frac selects the band [min, frac) of the same hash, so e.g.
    --patient-frac 0.6 --patient-min-frac 0.4 yields exactly the patients in
    the 0.6 set that are NOT in the 0.4 set (deterministic superset property).
    Same seed + frac as prepare_mixed_tool_rl_data.py => SFT and RL pick the
    exact same patient list."""
    if patient_frac is None or patient_frac >= 1.0:
        return True
    h = int(hashlib.md5(f"train:{patient_seed}:{pid}".encode()).hexdigest(), 16)
    b = h % 100
    lo = int((patient_min_frac or 0.0) * 100)
    return lo <= b < int(patient_frac * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-dir", required=True)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--test-seed", type=int, default=20260812)
    ap.add_argument("--patient-frac", type=float, default=1.0,
                    help="keep this fraction of non-TEST patients (1.0 = all)")
    ap.add_argument("--patient-min-frac", type=float, default=None,
                    help="keep the hash band [min, frac) instead of [0, frac); "
                         "e.g. --patient-frac 0.6 --patient-min-frac 0.4 gives "
                         "exactly the NEW patients when extending 0.4 -> 0.6")
    ap.add_argument("--patient-seed", type=int, default=20260812,
                    help="seed for patient-level sampling; same value as the RL "
                         "prepare script makes SFT/RL use the same patient list")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_pat = n_q = n_test = n_fracdrop = 0
    dense_files = sorted(
        f for f in os.listdir(args.trajectory_dir)
        if f.startswith("dense_") and f.endswith(".json")
    )
    with open(args.out, "w") as out:
        for fname in dense_files:
            pid = fname[len("dense_"):-len(".json")]
            if _is_test(pid, args.test_frac, args.test_seed):
                n_test += 1
                continue
            if not _in_train_pool(pid, args.patient_frac, args.patient_seed,
                                  args.patient_min_frac):
                n_fracdrop += 1
                continue
            with open(os.path.join(args.trajectory_dir, fname)) as f:
                data = json.load(f)
            patient_id = str(data.get("patient_id", pid))
            n_pat += 1
            for q in data.get("questions", []):
                out.write(json.dumps(
                    {"patient_id": patient_id,
                     "question_idx": q.get("question_idx", q.get("index", 0))}
                ) + "\n")
                n_q += 1

    print(f"{args.trajectory_dir}: kept {n_pat} patients / {n_q} questions "
          f"(excluded {n_test} TEST, {n_fracdrop} by patient-frac) -> {args.out}")


if __name__ == "__main__":
    main()
