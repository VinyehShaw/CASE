<h1 align="center">From Given to Gathered Evidence: Agentic Learning for Longitudinal Medical Reasoning</h1>

<h3 align="center"><img src="assets/logo_512.png" height="28" style="vertical-align:middle" alt="logo"> CASE — <b>C</b>linical <b>A</b>gents for <b>S</b>eeking <b>E</b>vidence</h3>

<!--
<p align="center">
  Minye Shao, Chaohui Yu, Yixuan Wu, Fan Wang, Ling Shao, Yang Long
  <br>
  Many thanks to contributors: Jing Wang, Hangjie Yuan, Shang Liu
</p>
-->

<p align="center">
  <a href="https://vinyehshaw.github.io/case-project/"><img src="assets/badge_project.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2609.39566"><img src="https://img.shields.io/badge/arXiv-2609.39566-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/CASE-Bench/CASE-8B-RL"><img src="https://img.shields.io/badge/%F0%9F%A4%97-CASE--8B-ffd21e.svg" alt="CASE-8B model"></a>
  <a href="https://huggingface.co/datasets/CASE-Bench/CASE_CondaEnv_Tar"><img src="https://img.shields.io/badge/-Env-ffd21e.svg" alt="Conda env"></a>
  <a href="https://huggingface.co/datasets/CASE-Bench/CASE-Agent-Traj-And-Bench-Partial"><img src="https://img.shields.io/badge/-Data-ffd21e.svg" alt="Dataset"></a>
  <a href="https://hub.docker.com/r/vinyeh/case"><img src="https://img.shields.io/badge/Docker-Env-2496ED.svg?logo=docker&logoColor=white" alt="Docker"></a>
</p>

CASE trains a compact vision–language policy (Qwen3-VL-8B) to answer longitudinal
clinical questions by **gathering evidence itself**: it decides which visit, organ,
MRI sequence, slice and specialist tool to inspect before answering. Training is
two-stage:

1. **RSFT** – rejection-sampling fine-tuning on screened teacher trajectories.
2. **Agentic RL** – GRPO on the policy's own tool-use rollouts, with a rubric-judge
   reward (RLAIF) and an on-policy privileged self-distillation penalty
   (**OPSD**, causal-KL over the Phase-4 insight).

One shared checkpoint is evaluated as three clinician roles:

| Role        | What it does                                        | Tools (budget)   | Internal name in data/code |
|-------------|-----------------------------------------------------|------------------|----------------------------|
| Anatomist   | Vista3D organ segmentation / volume change          | 3 tools (4 calls)  | `vista3d`     |
| Radiologist | Cross-visit MRI view comparison                     | 3 tools (6 calls)  | `medpanel_v2` |
| Consultant  | Multi-source synthesis: images, volumes, MedGemma reports, skill memory | 10 tools (12 calls) | `medboard` |

> Some CLI flags and cache directories keep historical identifiers
> (`--vista3d-dir`, `vista3d_tool_cache`, …) because they are part of the recorded
> data layout. Python modules use the role names
> (`distill.tool.anatomist`, `distill.tool.radiologist`, `distill.tool.consultant`).

---

## Contents

- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Benchmark](#benchmark)
- [Dataset](#dataset)
- [Model weights](#model-weights)
- [Pipeline](#pipeline)
- [Environment quirks](#environment-quirks)
- [Not included](#not-included)
- [License](#license)

---

## Repository layout

```
.
├── run/                          # pipeline entry points (start here)
│   ├── _common.sh                #   shared path / interpreter resolution (sourced)
│   ├── 00_env.sh                 #   install conda env from tar (non-Docker route)
│   ├── 01_split.sh               #   patient-level split
│   ├── 02_sft_build_cache.sh     #   tokenized RSFT cache
│   ├── 03_sft_train.sh           #   RSFT (accelerate + DeepSpeed ZeRO-2)
│   ├── 04_rl_prepare.sh          #   RL parquet + tool config
│   ├── 05_rl_train.sh            #   GRPO + OPSD tool RL (verl)
│   └── 06_eval.sh                #   3-role evaluation (vLLM serve + runners)
├── distill/                      # core package
│   ├── tool/
│   │   ├── anatomist/            #   Vista3D tools, real + mock (RL replay)
│   │   ├── radiologist/          #   MRI view / compare tools, real + mock
│   │   ├── consultant/           #   reports, skill memory, LangGraph runner
│   │   ├── common/               #   shared tool + trajectory utilities
│   │   └── *_tool_config.yaml    #   per-role and mixed tool registries for verl
│   └── opd_rl/
│       ├── reward*.py            #   answer / format / rubric-judge rewards
│       └── verl_opsd/            #   causal-KL (OPSD), causal judge, RL data prep
├── scripts/                      # trainers, launchers, filters, eval runners
│   └── vllm_patch/               #   tool-call parser + vLLM runtime patches
├── configs/accelerate_config/    # accelerate / DeepSpeed config for RSFT
└── env_release/                  # conda-env tar installer
```

---

## Environment

All published numbers were produced in the environment below (8 × H100-80GB).

### Option A — Conda-env tar from Hugging Face (recommended)

This is the exact conda env behind all published runs, packed as a tar
(`case_env_0904.tar`). Download it from
[🤗 CASE-Bench/CASE_CondaEnv_Tar](https://huggingface.co/datasets/CASE-Bench/CASE_CondaEnv_Tar) and extract it. You don't need
to build anything or resolve dependencies.

```bash
# 1. download into env_release/ (run/00_env.sh looks there by default)
pip install -U huggingface_hub
huggingface-cli download CASE-Bench/CASE_CondaEnv_Tar case_env_0904.tar \
    --repo-type dataset --local-dir env_release

# 2. extract + sanity-check (installs /opt/conda/envs/case,
#    adds a medix2 -> case compat symlink, prints torch/vLLM/verl versions)
bash run/00_env.sh
# or with an explicit path / prefix:
#   ENV_TAR=/path/to/case_env_0904.tar bash run/00_env.sh
#   bash env_release/install_env_tar.sh /path/to/case_env_0904.tar /your/prefix

# 3. use it
conda activate /opt/conda/envs/case        # or let run/_common.sh resolve PY
export DATA_ROOT=/path/to/data OUT_ROOT=$PWD/out
```

### Option B — Docker

Image: [`vinyeh/case:0904`](https://hub.docker.com/r/vinyeh/case). The image
contains the same env, so skip `run/00_env.sh`.

```bash
docker pull vinyeh/case:0904

# on a GPU machine
docker run --gpus all -it --shm-size=64g \
    -v "$PWD":/code \
    -v /path/to/data:/data \
    vinyeh/case:0904 bash

# inside the container
cd /code
export DATA_ROOT=/data OUT_ROOT=/code/out
```

### Contents of both routes

| Component | Version |
|-----------|---------|
| Python | 3.10.20 |
| PyTorch | 2.11.0 (CUDA 13.0) |
| transformers | 5.14.1 |
| vLLM | 0.26.0 |
| Ray | 2.55.1 |
| LangGraph | 1.2.9 (multi-turn tool-agent orchestration) |
| verl | 0.7.0 (rebuilt for agentic RL training: agent-loop state machine and related adaptations, based on upstream verl 0.7.0) |
| DeepSpeed | 0.19.2 |
| Triton | 3.7.1 |
| others | flash-attn, flash-linear-attention (FLA), FlashInfer |

Requirements:

- 8 GPUs by default (RSFT `NPROC=8`, RL `N_GPUS=8`).
- `gcc` at runtime (Triton compiles its driver shim once per machine). The Docker
  image already has it; on bare metal, run `apt-get install -y gcc libc6-dev`.
- The **Qwen3-VL-8B-Instruct** base weights (download separately).
- An OpenAI-compatible API key for the rubric judge during RL (`OPENAI_API_KEY`,
  optionally `OPENAI_BASE_URL`; DashScope by default).

---

## Benchmark

> **Data availability.** In line with UK Biobank access conditions (Application
> 603483), the de-identified materials — agent execution trajectories, RL
> prompts, benchmark banks and example preview images — are openly available as
> [🤗 CASE-Agent-Traj-And-Bench-Partial](https://huggingface.co/datasets/CASE-Bench/CASE-Agent-Traj-And-Bench-Partial). Data containing sensitive
> personal information cannot be redistributed here: it is returned to UK Biobank
> through [Returning Derived Data to UK Biobank](#returning-derived-data-to-uk-biobank),
> submitted via the [AMS platform](https://ams.ukbiobank.ac.uk/), where it is
> managed and accessed.

The **CASE benchmark** tests whether an agent can find evidence on its own,
rather than answer from evidence it is handed.

| Item | Value |
|------|-------|
| Questions | 50,401 multiple-choice questions from ICD-10-coded, doctor-diagnosed conditions |
| Participants | 4,739 (participant-disjoint splits) |
| Held-out test set | 4,850 questions / 455 participants (14,135 agent–question evaluations over 3 roles) |
| Task strata | 10 strata over 6 clinical domains; binary / single-select / multi-select |
| Evidence per patient | baseline + follow-up MRI (median interval 2.26 y), 5 organ regions, 7 sub-modalities, Vista3D measurements, MedGemma-27B reports |
| Metric | first-pass exact answer-set accuracy (Micro / Macro over the 10 strata) |

Each question links to a patient-specific tool environment. The agent decides which
visit, organ, sequence, slice and tool to query. Evaluation code is `run/06_eval.sh`
(see [Pipeline](#pipeline)), and the assets it reads are:

```
DATA_ROOT/
├── splits/{anatomist,radiologist,consultant}/split_20260926/test_patients.txt
└── eval_assets/                     # sources for the real (non-mock) tools
        ukb_dixon_vista3d/  ukb_dixon_vista3d_v2/  ukb_mri_all/
        medgemma_report_cache/  medboard_skill_store/
```

### Returning Derived Data to UK Biobank

As UK Biobank's access conditions require, the full CASE benchmark and its derived
data are being **returned to UK Biobank** — submitted through the
[Access Management System (AMS)](https://ams.ukbiobank.ac.uk/) — and cannot be
redistributed here. Once the return is accepted, approved researchers can access
it through:

- the [UK Biobank Showcase](https://biobank.ndph.ox.ac.uk/showcase/) → *Returns* (Returned Datasets Catalogue);
- the [UK Biobank Research Analysis Platform (UKB-RAP)](https://ukbiobank.dnanexus.com/), where it can be used directly under an approved application.

We will add the return ID and direct link here once they are assigned.

---

## Dataset

The training corpus is 147,434 role-conditioned trajectories recorded from the teacher
(Qwen3.8-Max) inside the same tool environment. They are filtered by the
evidence-linkage rubric (RSFT) and then used together with RL replay caches.

| Release | Content | Where |
|---------|---------|-------|
| Partial (public) | De-identified trajectories, RL prompts, benchmark banks, example preview images | [🤗 CASE-Agent-Traj-And-Bench-Partial](https://huggingface.co/datasets/CASE-Bench/CASE-Agent-Traj-And-Bench-Partial) |
| Full (approved researchers) | All trajectories, splits and RL tool caches | via the [UK Biobank AMS](https://ams.ukbiobank.ac.uk/) (see [Returning Derived Data to UK Biobank](#returning-derived-data-to-uk-biobank)) |

In the partial release, per UK Biobank requirements, participant identifiers are
replaced by uniform anonymous IDs (`case0001`–`case4739`; the mapping is withheld)
and raw imaging is not included. Contents:

- `trajectories/{Anatomist_agent,Radiologist_agent,Consultant_agent}/` — recorded
  agent trajectories (`agent_caseNNNN.jsonl`) and per-patient annotations
  (`dense_caseNNNN.json`);
- `rl/verl_train_tool.parquet` — RL training prompts, one row per
  patient–question (30,604 rows);
- `benchmark/` — evaluation question banks, each with its participant ID list;
- `images_examples/` — rendered preview slices for the benchmark set
  (4,938 images, figure headers cropped).

Expected layout under `DATA_ROOT` (default: `<repo>/data`):

```
DATA_ROOT/
├── sft/trajectories/{anatomist,radiologist,consultant}/        # recorded teacher trajectories
├── splits/{anatomist,radiologist,consultant}/split_20260926/
│       sft_kept.jsonl  rl_patients.txt  rl_kept.jsonl  test_patients.txt
└── rl/tool_env/                                                # RL replay caches
        vista3d_tool_cache/  medpanel_v2_tool_cache/  medboard_skill_store/
        medgemma_0820/  medgemma_report_cache/
```

If your copy uses a different layout, set these environment variables (defaults in
`run/_common.sh`, `run/05_rl_train.sh`, `run/06_eval.sh`):

| Variable | Purpose |
|----------|---------|
| `DATA_ROOT`, `OUT_ROOT` | dataset / output roots |
| `PY` | Python interpreter (auto-resolved to the conda env) |
| `TRAJ_ANATOMIST`, `TRAJ_RADIOLOGIST`, `TRAJ_CONSULTANT` | per-role trajectory dirs |
| `SPLITS`, `SPLIT_ANATOMIST`, `SPLIT_RADIOLOGIST`, `SPLIT_CONSULTANT` | split dirs (e.g. legacy `splits/vista3d/...`) |
| `EVAL_ASSETS` | evaluation asset root |
| `VISTA3D_TOOL_CACHE_DIR`, `MEDPANEL_V2_TOOL_CACHE_DIR`, `MEDGEMMA_IMAGE_BASE` | tool caches |

---

## Model weights

Checkpoints are on [🤗 CASE-Bench/models](https://huggingface.co/CASE-Bench/models).
All are fine-tuned from Qwen3-VL-8B-Instruct.

| Model | Description |
|-------|-------------|
| [CASE-8B-RL](https://huggingface.co/CASE-Bench/CASE-8B-RL) | Final model (RSFT + GRPO with RLAIF and OPSD) |
| [CASE-8B-RSFT](https://huggingface.co/CASE-Bench/CASE-8B-RSFT) | After stage 1 (RSFT); starting point for RL |
| [CASE-8B-ABL-OPSD](https://huggingface.co/CASE-Bench/CASE-8B-ABL-OPSD) | Ablation |
| [CASE-8B-ABL-RLAIF](https://huggingface.co/CASE-Bench/CASE-8B-ABL-RLAIF) | Ablation |

```bash
huggingface-cli download CASE-Bench/CASE-8B-RL --local-dir ckpt/CASE-8B-RL
CHECKPOINT=ckpt/CASE-8B-RL bash run/06_eval.sh
```

---

## Pipeline

Run every stage from the repository root. Stages read from `DATA_ROOT` and write to
`OUT_ROOT`.

| Stage | Command | Output |
|-------|---------|--------|
| 0. Env (conda tar) | `bash run/00_env.sh` | `/opt/conda/envs/case` |
| 1. Split | `bash run/01_split.sh` | `splits/` |
| 2. RSFT cache | `bash run/02_sft_build_cache.sh` | `$OUT_ROOT/sft_cache` |
| 3. RSFT train | `bash run/03_sft_train.sh` | `$OUT_ROOT/sft` |
| 4. RL data | `bash run/04_rl_prepare.sh` | `$OUT_ROOT/rl/verl_train_tool.parquet` |
| 5. RL train | `bash run/05_rl_train.sh` | `$OUT_ROOT/<run_ts>/rl_qwen3vl_tool/` |
| 6. Eval | `CHECKPOINT=<ckpt> bash run/06_eval.sh` | `$OUT_ROOT/eval/` |

### 1. Split

Patient-level split, md5-stable with seed `20260926`. Buckets (40/50/10, paper Appendix C): `[0,40)` RSFT,
`[40,90)` RL, `[90,100)` test. You only need this to re-derive the lists from raw
trajectories, because the released dataset ships the split files.

### 2–3. RSFT

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
bash run/02_sft_build_cache.sh     # always rebuilds the cache
bash run/03_sft_train.sh           # reuses the cache if its manifest matches
```

Training data is the **already-filtered** trajectory set (`sft_kept.jsonl`):
recorded teacher trajectories with at least one tool call and a parseable
Phase-4 insight, kept when their Kimi K2.6 evidence-linkage rubric score is
≥ 0.4 (paper Eq. 3). Filtering is offline (`scripts/filter_trajectories_causal.py`,
`scripts/run_causal_filter_*.sh`) and the released splits ship the result, so
`run/02–03` simply fine-tune on it. File and directory names keep the `sft`
prefix for compatibility.

The three roles' trajectories are interleaved, and the loss covers assistant tokens
only. Training updates the full model, including the ViT and merger. Run `run/02`
once, then rerun `run/03` as often as you like. It prints `Cache valid - skipping build`.

### 4–5. RL (GRPO + OPSD)

```bash
export MODEL_PATH=$OUT_ROOT/sft                # or a specific RSFT checkpoint
export OPENAI_API_KEY=<judge api key>          # rubric judge (Kimi K2.6 by default)
bash run/04_rl_prepare.sh
bash run/05_rl_train.sh
```

Main knobs: `TRAIN_BS`, `ROLLOUT_N`, `MAX_TURNS`, `GPU_UTIL`, `CAUSAL_KL_COEF`.
Logs go to `$OUT_ROOT/logs/grpo_*.log`. During RL, tools replay precomputed caches
instead of running Vista3D or MedGemma, so rollouts need no extra GPU work.

### 6. Evaluation

```bash
CHECKPOINT=/path/to/ckpt bash run/06_eval.sh
```

This builds the test bank, starts a vLLM server, runs the Anatomist, Radiologist
and Consultant runners against the real tool implementations, and summarizes
first-pass exact-match accuracy. Knobs: `PORT`, `TP_SIZE`, `MODEL_NAME`,
`VAL_PARQUET`, `OUTPUT_BASE`.

---

## Environment quirks

The scripts already handle these:

- **FlashInfer JIT is unavailable** in the shipped env (no CUDA toolkit). The scripts
  export `VLLM_USE_FLASHINFER_SAMPLER=0` and `FLA_CACHE_RESULTS=0`, and the eval
  server runs with `--enforce-eager` (the same eager mode the RL rollout uses).
- **Triton** compiles its driver shim with the system `gcc` on first use.
- **Ray** starts under a private `--temp_dir`, isolated from other jobs on a shared node.

---

## Not included

- Raw UK Biobank images and processed trajectories (see [Benchmark](#benchmark) and [Dataset](#dataset)).
- verl source code. The patched `verl 0.7.0` ships inside the Docker image / conda env
  as `site-packages`, which is the exact tree behind the published numbers.
- Base Qwen3-VL-8B-Instruct weights (CASE checkpoints are on
  [Hugging Face](#model-weights)) and plotting scripts.

---

<!--
## Citation

```bibtex
@article{shao2026case,
  title   = {From Given to Gathered Evidence: Agentic Learning for Longitudinal Medical Reasoning},
  author  = {Shao, Minye and Yu, Chaohui and Wu, Yixuan and Wang, Fan and Shao, Ling and Long, Yang},
  journal = {arXiv preprint arXiv:2609.39566},
  year    = {2026}
}
```
-->

## License

Code is released for research use. The released dataset (trajectories, RL
prompts, benchmark banks, example images) is for non-commercial academic use
only — commercial use and redistribution are prohibited. Derived data follows
UK Biobank terms of access (Application 603483); do not redistribute UK
Biobank-derived data outside an approved project.
