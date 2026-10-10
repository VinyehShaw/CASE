#!/bin/bash
# Mixed-data evaluation for SFT checkpoint (vista3d + medpanel_v2)
# Splits val patients: shard 0 → vista3d agent, shard 1 → medpanel_v2 agent
# 8 GPUs (tensor-parallel-size=8), CUDA graphs enabled (enforce_eager not set)
# tool-call-parser=qwen3_xml (matches Qwen3.5 chat template <function=...> format)
# Model autonomously decides imaging modalities via LangGraph agent tools
# Results stored on NAS at /mnt/faster_nas/benyou.mys/medixrl/0801/eval_sft/
#
# Usage:
#   bash scripts/run_mixed_eval.sh
#
# Prerequisites:
#   - vLLM not already running on port 8000
#   - Val parquet at /tmp/mys/dataset/opd_rl_all_0.9/val_dir/train.parquet (526 patients, test set)
#   - Checkpoint at /mnt/faster_nas/benyou.mys/medixrl/0801/sft_single_turn
set -e

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-/mnt/faster_nas/benyou.mys/medixrl/0801/sft_single_turn}"
MODEL_NAME="${MODEL_NAME:-qwen3-mixed-sft}"
PORT="${PORT:-8000}"
VAL_PARQUET="${VAL_PARQUET:-/mnt/faster_nas/benyou.mys/medixrl/data/val_opd_rl_all_0.9/train.parquet}"
VISTA3D_DIR="/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d"
VISTA3D_DIR_V2="/mnt/faster_nas/benyou.mys/dataset/ukb_dixon_vista3d_v2"
MRI_DIR="/mnt/faster_nas/benyou.mys/dataset/ukb_mri_all"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/faster_nas/benyou.mys/medixrl/0801/eval_sft}"
VISTA3D_OUTPUT="${OUTPUT_BASE}/vista3d"
MEDPANEL_OUTPUT="${OUTPUT_BASE}/medpanel_v2"
MEDBOARD_OUTPUT="${OUTPUT_BASE}/medboard"
MEDBOARD_REPORT_CACHE="${MEDBOARD_REPORT_CACHE:-/mnt/faster_nas/benyou.mys/medixrl/data/medgemma_report_cache/tool_cache_full_masked.json}"
SKILL_STORE="${SKILL_STORE:-/mnt/faster_nas/benyou.mys/medixrl/data/medboard_skill_store/skill_store.jsonl}"
SKILL_ACCUM_DIR="${SKILL_ACCUM_DIR:-/mnt/faster_nas/benyou.mys/medixrl/data/medboard_skill_store/harvest_accum}"
AUTO_BUILD_SKILL="${AUTO_BUILD_SKILL:-1}"

# ── Self-evolving skill memory (round-aware, per OUTPUT_BASE) ────────────────
# Round 1 (cold): OUTPUT_BASE/skill_store/ is empty → eval runs with the default
#   seed store; pass SKILL_STORE=<empty file> for a truly skill-less cold start.
#   Either way the run writes skills_*.jsonl into OUTPUT_BASE/medboard/.
# Round 2+: pass PREV_EVAL_DIR=<previous OUTPUT_BASE>. The harness aggregates
#   that run's harvested skills (build_skill_store_v2: leak filter + organ×topic
#   buckets + jaccard dedup) into OUTPUT_BASE/skill_store/skill_store_merged.jsonl
#   and uses it for this round. An explicitly set SKILL_STORE env always wins.
SKILL_STORE_EXPLICIT="${SKILL_STORE_EXPLICIT:-}"
if [ -n "${SKILL_STORE}" ] && [ "${SKILL_STORE}" != "/mnt/faster_nas/benyou.mys/medixrl/data/medboard_skill_store/skill_store.jsonl" ]; then
    SKILL_STORE_EXPLICIT=1
fi
SKILL_ROUND_DIR="${OUTPUT_BASE}/skill_store"
SKILL_ROUND_STORE="${SKILL_ROUND_DIR}/skill_store_merged.jsonl"
if [[ -z "${SKILL_STORE_EXPLICIT}" && -n "${PREV_EVAL_DIR:-}" ]] && \
   [[ ! -f "${SKILL_ROUND_STORE}" && -d "${PREV_EVAL_DIR}/medboard" ]] && \
   ls "${PREV_EVAL_DIR}/medboard"/skills_*.jsonl >/dev/null 2>&1; then
    mkdir -p "${SKILL_ROUND_DIR}"
    echo "[skill-round] aggregating harvested skills from ${PREV_EVAL_DIR}/medboard"
    /opt/conda/envs/medix2/bin/python \
        distill/tool/consultant/build_skill_store_v2.py \
        --harvest-dir "${PREV_EVAL_DIR}/medboard" \
        --output "${SKILL_ROUND_STORE}" \
        --stats || true
fi
if [[ -z "${SKILL_STORE_EXPLICIT}" && -f "${SKILL_ROUND_STORE}" ]]; then
    SKILL_STORE="${SKILL_ROUND_STORE}"
    echo "[skill-round] using self-evolved store: ${SKILL_ROUND_STORE}"
fi

MAX_PATIENTS_PER_AGENT="${MAX_PATIENTS_PER_AGENT:-100}"
MAX_API_CONCURRENT=50
MAX_PATIENT_WORKERS=10
MEDBOARD_PATIENT_WORKERS="${MEDBOARD_PATIENT_WORKERS:-20}"
MEDBOARD_QUESTION_WORKERS="${MEDBOARD_QUESTION_WORKERS:-20}"
TP_SIZE="${TP_SIZE:-8}"

# Which modules to evaluate: all | comma list of vista3d,medpanel,medboard.
# Single module => that module runs the FULL bank (no shard split).
MODULES="${MODULES:-all}"

# ── CUDA 13 library paths (torch 2.11.0+cu130) ─────────────────────────────
export CUDA_HOME=/opt/conda/envs/medix2
export LD_LIBRARY_PATH="/opt/conda/envs/medix2/lib:/opt/conda/envs/medix2/lib/python3.10/site-packages/nvidia/cu13/lib:/opt/conda/envs/medix2/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/opt/conda/envs/medix2/bin:${PATH}"
# FlashInfer JIT needs CUDA headers; nvcc resolves CUDA_HOME include/ which lacks them
export CPATH="/opt/conda/envs/medix2/lib/python3.10/site-packages/nvidia/cu13/include:${CPATH:-}"
# Skip FlashInfer JIT entirely (same as training): native sampler + triton GDN prefill
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Per-LLM-call response cap for all three agents. RL models write longer Phase-4/5
# responses; the old 2000/3000 defaults truncated final answers mid-sentence (None answers).
export MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-4096}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
# sitecustomize patches vLLM at import time (incl. Patch 9: JSON fallback for
# the built-in qwen3_xml parser, needed by Qwen3-VL checkpoints)
export PYTHONPATH="/mnt/faster_nas/benyou.mys/git/ezdistill/scripts/vllm_patch:${PYTHONPATH:-}"

# ── Start vLLM server ───────────────────────────────────────────────────────
mkdir -p "${OUTPUT_BASE}"
echo "Starting vLLM server on port ${PORT}..."
vllm serve "${CHECKPOINT}" \
    --served-model-name "${MODEL_NAME}" \
    --port ${PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN:-16384} \
    --dtype bfloat16 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 32}' \
    --gpu-memory-utilization "${GPU_UTIL:-0.9}" \
    --gdn-prefill-backend "${GDN_PREFILL_BACKEND}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    > "${OUTPUT_BASE}/vllm_server.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for vLLM to be ready..."

# Wait for vLLM to be ready (up to 30 minutes; first run JIT-compiles GDN kernels)
for i in $(seq 1 "${WAIT_ITERS:-180}"); do
    if curl -s http://localhost:${PORT}/v1/models | grep -q "${MODEL_NAME}"; then
        echo "vLLM is ready!"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "vLLM process died! Check ${OUTPUT_BASE}/vllm_server.log"
        exit 1
    fi
    echo "  Waiting... (${i}/60)"
    sleep 10
done

# Verify vLLM is up
if ! curl -s http://localhost:${PORT}/v1/models | grep -q "${MODEL_NAME}"; then
    echo "vLLM failed to start within 10 minutes"
    exit 1
fi

# ── Run vista3d eval (shard 0/2, or full bank when selected alone) ──────────
if [[ "${MODULES}" == "all" || "${MODULES}" == *vista3d* ]]; then
# When both vista3d and medpanel run, split the bank in halves; alone => full bank.
V_SHARD_ARGS=(--shard-id 0 --num-shards 1)
if [[ "${MODULES}" == "all" || "${MODULES}" == *medpanel* ]]; then V_SHARD_ARGS=(--shard-id 0 --num-shards 2); fi
V_MODE="full bank"
[[ ${V_SHARD_ARGS[1]} == 2 ]] && V_MODE="shard 0/2"
echo ""
echo "========================================"
echo "Running vista3d eval (${V_MODE}, ${MAX_PATIENTS_PER_AGENT} patients cap)"
echo "========================================"
rm -rf "${VISTA3D_OUTPUT}"
OPENAI_BASE_URL=http://localhost:${PORT}/v1 \
OPENAI_API_KEY=empty \
/opt/conda/envs/medix2/bin/python -m distill.tool.anatomist.run_batch \
    --parquet "${VAL_PARQUET}" \
    --vista3d-dir "${VISTA3D_DIR}" \
    --vista3d-dir-v2 "${VISTA3D_DIR_V2}" \
    --output "${VISTA3D_OUTPUT}" \
    --model "${MODEL_NAME}" \
    --max-patients ${MAX_PATIENTS_PER_AGENT} \
    "${V_SHARD_ARGS[@]}" \
    --max-api-concurrent ${MAX_API_CONCURRENT} \
    --max-patient-workers ${MAX_PATIENT_WORKERS} \
    2>&1 | tee "${OUTPUT_BASE}/vista3d_eval.log"
fi

# ── Run medpanel_v2 eval (shard 1/2) ──────────────────────────────────────
if [[ "${MODULES}" == "all" || "${MODULES}" == *medpanel* ]]; then
M_SHARD_ARGS=(--shard-id 1 --num-shards 2)
if [[ "${MODULES}" != "all" && "${MODULES}" != *vista3d* ]]; then M_SHARD_ARGS=(--shard-id 0 --num-shards 1); fi
echo ""
echo "========================================"
echo "Running medpanel_v2 eval (${MAX_PATIENTS_PER_AGENT} patients cap)"
echo "========================================"
rm -rf "${MEDPANEL_OUTPUT}"
OPENAI_BASE_URL=http://localhost:${PORT}/v1 \
OPENAI_API_KEY=empty \
/opt/conda/envs/medix2/bin/python -m distill.tool.radiologist.run_batch \
    --parquet "${VAL_PARQUET}" \
    --mri-dir "${MRI_DIR}" \
    --vista3d-dir "${VISTA3D_DIR}" \
    --vista3d-dir-v2 "${VISTA3D_DIR_V2}" \
    --output "${MEDPANEL_OUTPUT}" \
    --model "${MODEL_NAME}" \
    --max-patients ${MAX_PATIENTS_PER_AGENT} \
    "${M_SHARD_ARGS[@]}" \
    --max-api-concurrent ${MAX_API_CONCURRENT} \
    --max-patient-workers ${MAX_PATIENT_WORKERS} \
    2>&1 | tee "${OUTPUT_BASE}/medpanel_v2_eval.log"
fi

# ── Run medboard eval (full, same agent+tools as recording) ─────────────────
if [[ "${MODULES}" == "all" || "${MODULES}" == *medboard* ]]; then
echo ""
echo "========================================"
echo "Running medboard eval (full, ${MAX_PATIENTS_PER_AGENT} patients)"
echo "========================================"
rm -rf "${MEDBOARD_OUTPUT}"
MEDBOARD_LIVE_SKILL=1 \
/opt/conda/envs/medix2/bin/python -m distill.tool.consultant.run_batch \
    --question-bank "${VAL_PARQUET}" \
    --max-patients ${MAX_PATIENTS_PER_AGENT} \
    --max-questions 0 \
    --model "${MODEL_NAME}" \
    --api-key empty \
    --base-url "http://localhost:${PORT}/v1" \
    --report-cache "${MEDBOARD_REPORT_CACHE}" \
    --skill-store "${SKILL_STORE}" \
    --output-dir "${MEDBOARD_OUTPUT}" \
    --max-api-concurrent ${MAX_API_CONCURRENT} \
    --patient-workers ${MEDBOARD_PATIENT_WORKERS} \
    --question-workers ${MEDBOARD_QUESTION_WORKERS} \
    2>&1 | tee "${OUTPUT_BASE}/medboard_eval.log"
fi

# ── Combined summary ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Combined Mixed Eval Results"
echo "========================================"
export VISTA3D_OUTPUT MEDPANEL_OUTPUT MEDBOARD_OUTPUT
/opt/conda/envs/medix2/bin/python3 << 'PYEOF'
import json, glob, os
from collections import defaultdict

vista3d_dir = os.environ.get("VISTA3D_OUTPUT", "/mnt/faster_nas/benyou.mys/medixrl/0801/eval_sft/vista3d")
medpanel_dir = os.environ.get("MEDPANEL_OUTPUT", "/mnt/faster_nas/benyou.mys/medixrl/0801/eval_sft/medpanel_v2")
medboard_dir = os.environ.get("MEDBOARD_OUTPUT", "/mnt/faster_nas/benyou.mys/medixrl/0801/eval_sft/medboard")

# ── Question-type taxonomy (matched on question text + option count) ─────────
def qtype(qtxt, nopt, gt):
    t = (qtxt or "").lower()
    if "cancer" in t:
        return "Q1 cancer(Y/N)"
    if "diabetes" in t:
        return "Q2 diabetes(Y/N)"
    if "cardiovascular" in t:
        return "Q3-multi cardio" if nopt == 5 else "Q3-single cardio"
    if "respiratory or allergic" in t:
        return "Q4-multi resp" if nopt == 6 else "Q4-single resp"
    if "disease systems" in t:
        return "Q5-multi ICD-system"
    if "icd-10" in t and ("personal history" in t or "family history" in t):
        return "Q6 ICD-history(Y/N)"
    if "hypertension" in t or "essential" in t:
        return "Q7 hypertension(Y/N)"
    if "icd-10" in t:
        return "Q5-single ICD-cond"
    return "other"

QTYPE_ORDER = [
    "Q1 cancer(Y/N)", "Q2 diabetes(Y/N)",
    "Q3-multi cardio", "Q3-single cardio",
    "Q4-multi resp", "Q4-single resp",
    "Q5-multi ICD-system", "Q5-single ICD-cond",
    "Q6 ICD-history(Y/N)", "Q7 hypertension(Y/N)",
    "other",
]

def norm(s):
    return ",".join(sorted(x.strip().upper() for x in str(s).replace(" ", ",").split(",") if x.strip()))

stats = {}
type_stats = {}  # {module: {qtype: [correct, total]}}
for label, ddir in [("vista3d", vista3d_dir), ("medpanel_v2", medpanel_dir), ("medboard", medboard_dir)]:
    n_correct = n_total = 0
    n_patients = 0
    ts = defaultdict(lambda: [0, 0])
    for f in sorted(glob.glob(os.path.join(ddir, "dense_*.json"))):
        try:
            with open(f) as fh:
                data = json.load(fh)
            n_patients += 1
            for q in data.get("questions", []):
                gt = q.get("ground_truth") or q.get("solution")
                if not gt:
                    continue
                n_total += 1
                ans = q.get("answer", "")
                ok = norm(ans) == norm(gt)
                if ok:
                    n_correct += 1
                qt = qtype(q.get("question", ""), len(q.get("options", {}) or {}), gt)
                ts[qt][1] += 1
                if ok:
                    ts[qt][0] += 1
        except Exception as e:
            print(f"  WARN: error reading {f}: {e}")

    acc = n_correct / n_total * 100 if n_total > 0 else 0
    stats[label] = {"patients": n_patients, "correct": n_correct, "total": n_total, "acc": acc}
    type_stats[label] = ts
    print(f"  {label:15s}: {n_patients} patients, {n_correct}/{n_total} correct, {acc:.1f}%")

all_correct = sum(s["correct"] for s in stats.values())
all_total = sum(s["total"] for s in stats.values())
all_acc = all_correct / all_total * 100 if all_total > 0 else 0
print(f"  {'COMBINED':15s}: {all_correct}/{all_total} correct, {all_acc:.1f}%")

# ── Per-question-type breakdown ─────────────────────────────────────────────
modules = [m for m in ("vista3d", "medpanel_v2", "medboard") if stats.get(m, {}).get("total")]
if len(modules) >= 1:
    print("")
    print("Per-Question-Type Accuracy")
    print(f"  {'type':22s}" + "".join(f"{m.split('_')[0][:9]:>12s}" for m in modules) + f"{'ALL':>12s}")
    for qt in QTYPE_ORDER:
        row = f"  {qt:22s}"
        any_total = False
        for m in modules:
            c, t = type_stats[m].get(qt, [0, 0])
            any_total = any_total or t > 0
            row += f"{(c / t * 100):>11.1f}%" if t else f"{'-':>12s}"
        tc = sum(type_stats[m].get(qt, [0, 0])[0] for m in modules)
        tt = sum(type_stats[m].get(qt, [0, 0])[1] for m in modules)
        row += f"{(tc / tt * 100):>11.1f}%" if tt else f"{'-':>12s}"
        if any_total:
            print(row)
    print(f"  {'':22s}" + "".join(f"{'n=' + str(stats[m]['total']):>12s}" for m in modules))
PYEOF

echo ""
echo "--- vista3d log tail ---"
grep -E "Accuracy|Correct|Questions|Patients" "${OUTPUT_BASE}/vista3d_eval.log" 2>/dev/null | tail -5 || true
echo ""
echo "--- medpanel_v2 log tail ---"
grep -E "accuracy|Done" "${OUTPUT_BASE}/medpanel_v2_eval.log" 2>/dev/null | tail -5 || true

# ── Cleanup ─────────────────────────────────────────────────────────────────
echo ""
echo "Stopping vLLM server (PID ${VLLM_PID})..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true
echo "Done. Results in ${OUTPUT_BASE}/"
