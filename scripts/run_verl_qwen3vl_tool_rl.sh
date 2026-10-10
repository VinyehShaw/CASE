#!/bin/bash
# Base RL launcher: GRPO + OPSD multi-turn tool RL on Qwen3-VL (the patched
# verl that ships inside the conda env; FSDP actor + async vLLM rollout).
# The model autonomously calls tools (view_image, compare_visits, ...) during
# rollout; mock tools replay pre-computed results from cache for fast,
# deterministic execution.
#
# Usually invoked via run/05_rl_train.sh (which sets MODEL_PATH / TOOL_CONFIG /
# TRAIN_FILES / OPENAI_API_KEY); can also be used standalone.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/out}"
mkdir -p "$OUT_ROOT/generated" "${OUT_ROOT}/logs"

# ---- Interpreter + CUDA 13 library paths (torch 2.11.0+cu130) ---------------
if [ -z "${PY:-}" ]; then
    for c in /opt/conda/envs/case/bin/python /opt/conda/envs/medix2/bin/python; do
        [ -x "$c" ] && PY="$c" && break
    done
fi
: "${PY:?conda env not found; install it (bash run/00_env.sh) or use the docker image}"
PY_PREFIX="$(dirname "$(dirname "$PY")")"
export CUDA_HOME="$PY_PREFIX"
export LD_LIBRARY_PATH="$PY_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$PY_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$PY_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include:${CPATH:-}"
export LIBRARY_PATH="$PY_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$PY_PREFIX/lib64:${LIBRARY_PATH:-}"
export PATH="$PY_PREFIX/bin:${PATH}"
export PYTHONPATH="$REPO_ROOT/scripts/vllm_patch:${PYTHONPATH:-}"
# Let scripts/vllm_patch prefer THIS package's distill copy in every process.
export CASE_CODE_ROOT="$REPO_ROOT"

# ---- API credentials for the causal-judge reward (required) -----------------
: "${OPENAI_API_KEY:?export OPENAI_API_KEY=<dashscope key> first}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0 NCCL_ALGO=Ring NCCL_SOCKET_PORT_RANGE=40000-50000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export FLA_USE_FAST_OPS=1
# Ray forces OMP_NUM_THREADS=1 for actor tasks that do not request num_cpus
# (ray/_private/utils.py) and verl's WorkerDict does not override it, which
# would leave the CPU-offloaded AdamW step single-threaded over 9.4B params
# (minutes per step). Ray only forces the value when unset, so this wins.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

RUN_TS="${RUN_TS:-$(date '+%Y%m%d_%H%M%S')}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUT_ROOT/tensorboard/grpo_qwen3vl_tool_rl_${RUN_TS}}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/grpo_qwen3vl_tool_rl_${RUN_TS}.log"

# ---- Private Ray temp dir ----------------------------------------------------
# The default /tmp/ray holds one machine-wide `ray_current_cluster` marker, so
# on a shared node another user's ray.init() could discover this cluster (and
# vice versa), which either fails with a Ray/Python version mismatch or tears
# down the raylet mid-run. A private temp dir keeps the two clusters apart.
RAY_TMP_DIR="${RAY_TMP_DIR:-/tmp/ray_$(id -un)}"

# A shell-exported RAY_ADDRESS would name someone else's cluster. `address=local`
# already wins over it, but dropping it here keeps the intent unambiguous.
unset RAY_ADDRESS

ray_owners=$(
    for pid in $(pgrep -f 'ray/core/src/ray/raylet/raylet' 2>/dev/null || true); do
        argv=$(tr '\0' '\n' < "/proc/${pid}/cmdline" 2>/dev/null || true)
        exe=$(printf '%s\n' "${argv}" | head -1)
        # Skip shells that merely mention the pattern in their own command text.
        case "${exe}" in */raylet) ;; *) continue ;; esac
        tmp=$(printf '%s\n' "${argv}" | sed -n 's/^--temp_dir=//p' | head -1)
        echo "pid ${pid}  env=${exe%%/lib/python*}  temp_dir=${tmp:-?}"
    done
) || true

if [ -n "${ray_owners}" ]; then
    echo "[preflight] Ray cluster(s) already running on this node:"
    printf '%s\n' "${ray_owners}" | sed 's/^/[preflight]     /'
    echo "[preflight]   We start our own under ${RAY_TMP_DIR} (address=local), so we never join theirs."
    echo "[preflight]   Do NOT run 'ray stop': it kills raylets by process name, taking their job down too."
    echo "[preflight]   A line already showing temp_dir=${RAY_TMP_DIR} is one of OUR runs - still alive, or"
    echo "[preflight]   crashed leaving orphans that hold GPU memory. Check 'nvidia-smi' before starting."
else
    echo "[preflight] No other Ray cluster on this node."
fi

# ---- Tool config (default: mixed 3+3 replay tools) ---------------------------
TOOL_CONFIG="${TOOL_CONFIG:-$REPO_ROOT/distill/tool/mixed_tool_config.yaml}"

# ---- Batch sizes (env-overridable for quick debug runs) ----------------------
# verl requires train_batch_size >= ppo_mini_batch_size (both counted in
# prompts), so MINI_BS follows TRAIN_BS unless set explicitly.
TRAIN_BS="${TRAIN_BS:-16}"
MINI_BS="${MINI_BS:-${TRAIN_BS}}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PPO_MBS="${PPO_MBS:-2}"
# The log-prob forward materializes logits of LOGP_MBS x seq_len x 248320 in
# bf16; LOGP_MBS=4 needs 15.16 GiB and OOMs on an 80GB card next to vLLM.
LOGP_MBS="${LOGP_MBS:-1}"

# ---- Throughput knobs ---------------------------------------------------------
# Dynamic bsz packs sequences up to a token budget per forward instead of
# running one short sequence at a time. It only speeds up the training-side
# forwards; the multi-turn vLLM generation cost is untouched, and that is what
# dominates an agentic step.
DYNAMIC_BSZ="${DYNAMIC_BSZ:-true}"
PPO_MAX_TOKEN="${PPO_MAX_TOKEN:-12288}"
LOGP_MAX_TOKEN="${LOGP_MAX_TOKEN:-12288}"
ENTROPY_CKPT="${ENTROPY_CKPT:-true}"
# Rollout and training share these GPUs. A GPU-resident optimizer frees the
# AdamW step from CPU but then competes with the vLLM KV cache, so these two
# have to be tuned together.
OPT_OFFLOAD="${OPT_OFFLOAD:-true}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-true}"
GPU_UTIL="${GPU_UTIL:-0.60}"

# ---- Data (no image column; images are fetched dynamically via tool calls) ----
TRAIN_FILES="${TRAIN_FILES:-['$OUT_ROOT/rl/verl_train_tool.parquet']}"
VAL_FILES="${VAL_FILES:-'$OUT_ROOT/rl/verl_train_tool.parquet'}"

# ---- Run -----------------------------------------------------------------------
cd "$REPO_ROOT"

"$PY" scripts/run_verl_tool_rl_wrapper.py \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init._temp_dir="${RAY_TMP_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS:-8} trainer.total_epochs=1 \
    trainer.save_freq=25 \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR:-$OUT_ROOT/${RUN_TS}/rl_qwen3vl_tool/}" \
    trainer.project_name=medixrl trainer.experiment_name=grpo_qwen3vl_tool_rl_${RUN_TS} \
    trainer.logger='[console,tensorboard]' trainer.max_actor_ckpt_to_keep=0 \
    trainer.test_freq=-1 trainer.val_before_train=false \
    actor_rollout_ref.model.path="${MODEL_PATH:-$OUT_ROOT/sft}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MBS}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN}" \
    actor_rollout_ref.actor.entropy_checkpointing="${ENTROPY_CKPT}" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPT_OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=0.7 actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.response_length="${RESP_LEN:-6144}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-12288}" \
    actor_rollout_ref.rollout.enforce_eager=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_num_seqs=96 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGP_MBS}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGP_MAX_TOKEN}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGP_MBS}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOGP_MAX_TOKEN}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_xml \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS:-8}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    algorithm.adv_estimator=grpo algorithm.gamma=1.0 algorithm.lam=1.0 \
    algorithm.kl_ctrl.kl_coef=0.05 \
    +algorithm.causal_kl_coef="${CAUSAL_KL_COEF:-2.0}" \
    +algorithm.causal_kl_method="${CAUSAL_KL_METHOD:-bucket_iv}" \
    +algorithm.causal_kl_topk="${CAUSAL_KL_TOPK:-128}" \
    +algorithm.causal_kl_topk_chunk="${CAUSAL_KL_TOPK_CHUNK:-6}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length=4096 data.max_response_length="${RESP_LEN:-6144}" \
    data.train_batch_size="${TRAIN_BS}" \
    custom_reward_function.path="./distill/opd_rl/verl_opsd/reward_v5_anatomist.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.insight_alpha=0.3 \
    +custom_reward_function.reward_kwargs.causal_judge_beta=0.6 \
    +custom_reward_function.reward_kwargs.format_weight=0.01 \
    +custom_reward_function.reward_kwargs.judge_mode=api \
    +custom_reward_function.reward_kwargs.judge_model=kimi-k2.6 \
    +custom_reward_function.reward_kwargs.judge_max_concurrent=150 \
    +custom_reward_function.reward_kwargs.judge_batch_size=4 \
    +custom_reward_function.reward_kwargs.judge_max_workers=40 \
    +custom_reward_function.reward_kwargs.insight_threshold=0.4 \
    +custom_reward_function.reward_kwargs.insight_topk=2 \
    +custom_reward_function.reward_kwargs.insight_min_range=0.02 \
    +custom_reward_function.reward_kwargs.insight_correct_bonus=0.5 \
    +custom_reward_function.reward_kwargs.insight_wrong_penalty=0.3 \
    +reward_model.reward_kwargs.reward_max_concurrent=500 \
    reward_manager.name=batch \
    2>&1 | tee "${LOG_FILE}"
