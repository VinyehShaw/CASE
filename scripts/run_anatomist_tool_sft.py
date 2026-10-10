"""Vista3D tool-calling SFT — trains Qwen3-VL to autonomously call vista3d tools.

Reads agent_*.jsonl (raw LangGraph tool-call trajectories) produced by vista3d_run_batch.py.

Data format per sample (OpenAI chat with tools):
    system   — SYSTEM_PROMPT (+ tools schema injected by chat template)
    user     — patient info + question + options
    assistant — <tool_call> vista3d_compare_visits(organ)      [LOSS]
    tool     — <tool_response> volume text                      [no loss]
    user     — overlay images (V1/V2 per station)               [no loss]
    assistant — 5-phase final answer                            [LOSS]

Loss is computed ONLY on assistant messages (tool_call + final answer);
system/user/tool messages are masked with -100. Boundary detection uses
incremental chat-template rendering (special tokens prevent BPE drift).

Usage:
    python scripts/run_anatomist_tool_sft.py \
        --trajectory-dir outputs/vista3d_trajectory_tool_100 \
        --output /mnt/faster_nas/benyou.mys/medixrl/checkpoints/vista3d_tool_sft \
        --max-steps 30
"""

import os
import sys

# Preload NVRTC builtins for CUDA 13.0 (Qwen3.5 linear attention JIT)
_nvrtc_dir = "/opt/conda/envs/medix2/lib/python3.10/site-packages/nvidia/cu13/lib"
if os.path.isdir(_nvrtc_dir):
    _cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{_nvrtc_dir}:{_cur}"
    import ctypes
    for _lib in ["libnvrtc-builtins.so.13.0", "libnvrtc.so.13"]:
        _p = os.path.join(_nvrtc_dir, _lib)
        if os.path.exists(_p):
            ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
import io
import json
import glob
import shutil
import base64
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# OpenAI tool schemas — must match the @tool definitions in vista3d_tool.py
VISTA3D_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vista3d_segment",
            "description": "Get Vista3D segmentation volume for a specific organ, broken down per body station.",
            "parameters": {
                "type": "object",
                "properties": {"organ": {"type": "string"}},
                "required": ["organ"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vista3d_compare_stations",
            "description": "Compare an organ's volume and morphology between dixon_8_W (thorax) and dixon_12_W (abdomen).",
            "parameters": {
                "type": "object",
                "properties": {"organ": {"type": "string"}},
                "required": ["organ"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vista3d_compare_visits",
            "description": "Compare an organ's volume between V1 (baseline) and V2 (follow-up) MRI visits.",
            "parameters": {
                "type": "object",
                "properties": {"organ": {"type": "string"}},
                "required": ["organ"],
            },
        },
    },
]


def load_agent_dataset(trajectory_dir: str, only_correct: bool = False) -> list:
    """Load agent_*.jsonl files; return list of template-ready message lists.

    Images are kept as file PATHS only — actually loaded lazily in
    LazyVista3DDataset.__getitem__ to keep host memory flat (the previous
    eager-load version exhausted RAM: 9360 samples x PIL + pixel_values
    lists x 8 ranks).
    Entries without any assistant tool_calls are skipped.
    """
    files = sorted(glob.glob(os.path.join(trajectory_dir, "agent_*.jsonl")))
    if not files:
        print(f"No agent_*.jsonl files found in {trajectory_dir}")
        return []

    samples = []
    skipped_no_tool = skipped_correct = skipped_no_image = 0
    for fpath in files:
        for line in open(fpath, "r", encoding="utf-8"):
            entry = json.loads(line)
            if only_correct and not entry.get("correct"):
                skipped_correct += 1
                continue
            msgs = entry.get("messages", [])
            if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs):
                skipped_no_tool += 1
                continue

            # Convert to template-ready messages (PIL Images in image blocks).
            # NOTE: processor.apply_chat_template(tokenize=True) requires every
            # message content to be a list of typed blocks (strings crash it).
            ready = []
            for m in msgs:
                role = m.get("role")
                content = m.get("content")
                if isinstance(content, str):
                    blocks = [{"type": "text", "text": content}]
                else:
                    blocks = []
                    for b in content:
                        if b.get("type") == "text":
                            blocks.append({"type": "text", "text": b.get("text", "")})
                        elif b.get("type") == "image":
                            ref = b.get("image", "")
                            img_path = os.path.join(trajectory_dir, ref)
                            if not os.path.exists(img_path):
                                img_path = os.path.join(trajectory_dir, "images", ref)
                            if os.path.exists(img_path):
                                blocks.append({"type": "image_path", "path": img_path})
                out = {"role": role, "content": blocks}
                if m.get("tool_calls"):
                    out["tool_calls"] = m["tool_calls"]
                ready.append(out)

            # No-image samples skip the vision tower — gradient buckets then
            # differ across ranks and the DeepSpeed allreduce deadlocks.
            if not any(b.get("type") == "image_path" for m in ready for b in m["content"]):
                skipped_no_image += 1
                continue

            samples.append({"messages": ready, "patient_id": entry.get("patient_id"), "question_idx": entry.get("question_idx")})

    print(f"Loaded {len(samples)} tool-call samples from {len(files)} agent files "
          f"(skipped: {skipped_no_tool} no-tool-call, {skipped_correct} incorrect, {skipped_no_image} no-image)")
    return samples


def materialize_images(msgs: list, resize: int) -> list:
    """Normalise message content for the processor.

    - String content → [{"type": "text", "text": ...}] (safety net)
    - {"type": "image_path", "path": ...} → {"type": "image", "image": PIL}
    - tool_call.function.arguments JSON string → dict (Qwen3.5 template uses |items)
    """
    import json as _json
    from PIL import Image

    out_msgs = []
    for m in msgs:
        content = m["content"]
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = []
            for b in content:
                if not isinstance(b, dict):
                    blocks.append(b)
                    continue
                if b.get("type") == "image_path":
                    img = Image.open(b["path"]).convert("RGB")
                    if resize:
                        img = img.resize((resize, resize))
                    blocks.append({"type": "image", "image": img})
                else:
                    blocks.append(b)
        out = {"role": m["role"], "content": blocks}
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                if "function" in tc:
                    fn = dict(tc["function"])
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        fn["arguments"] = _json.loads(args)
                    tc["function"] = fn
                tcs.append(tc)
            out["tool_calls"] = tcs
        out_msgs.append(out)
    return out_msgs


def tokenize_with_spans(sample: dict, processor, max_length: int, resize: int = 512):
    """Tokenize full conversation; labels only on assistant-message spans.

    Boundaries via incremental rendering:
      start of assistant i = len(template(msgs[:i], add_generation_prompt=True))
      end of assistant i   = len(template(msgs[:i+1]))
    """
    msgs = materialize_images(sample["messages"], resize)

    full = processor.apply_chat_template(
        msgs, tools=VISTA3D_TOOLS, tokenize=True, return_dict=True, return_tensors="pt",
    )
    input_ids = full["input_ids"][0]
    if input_ids.shape[0] > max_length:
        return None

    labels = [-100] * input_ids.shape[0]
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        start_out = processor.apply_chat_template(
            msgs[:i], tools=VISTA3D_TOOLS, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        end_out = processor.apply_chat_template(
            msgs[:i + 1], tools=VISTA3D_TOOLS, tokenize=True, return_dict=True, return_tensors="pt",
        )
        start = start_out["input_ids"].shape[1]
        end = end_out["input_ids"].shape[1]
        if end <= input_ids.shape[0] and start < end:
            labels[start:end] = input_ids[start:end].tolist()

    if all(l == -100 for l in labels):
        return None

    entry = {
        "input_ids": input_ids.tolist(),
        "labels": labels,
        "attention_mask": full["attention_mask"][0].tolist(),
    }
    if "pixel_values" in full:
        entry["pixel_values"] = full["pixel_values"]
    if "image_grid_thw" in full:
        entry["image_grid_thw"] = full["image_grid_thw"]
    return entry


_PROC = None


def _cache_worker_init(model_path: str):
    global _PROC
    import torch
    from transformers import AutoProcessor
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    _PROC = AutoProcessor.from_pretrained(model_path)


def _cache_one(args):
    idx, sample, cache_dir, max_length, resize = args
    import torch
    out_path = os.path.join(cache_dir, f"sample_{idx:06d}.pt")
    if os.path.exists(out_path):
        return 1
    try:
        entry = tokenize_with_spans(sample, _PROC, max_length, resize)
    except Exception as e:
        print(f"  Skip sample {idx}: {type(e).__name__}: {e}")
        return 0
    if entry is None:
        return 0
    if "pixel_values" in entry:
        entry["pixel_values"] = entry["pixel_values"].to(torch.bfloat16)
    torch.save(entry, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    return 1


def build_cache(
    trajectory_dir: str,
    model_path: str,
    cache_dir: str,
    max_length: int,
    resize: int,
    only_correct: bool,
    workers: int,
):
    """One-time parallel pass: tokenize every sample and save .pt files.

    Resumable — existing sample_*.pt files are skipped on re-run.
    """
    from multiprocessing import Pool

    samples = load_agent_dataset(trajectory_dir, only_correct)
    if not samples:
        print("No samples to cache. Exiting.")
        return
    os.makedirs(cache_dir, exist_ok=True)

    tasks = [(i, s, cache_dir, max_length, resize) for i, s in enumerate(samples)]
    saved = 0
    with Pool(workers, initializer=_cache_worker_init, initargs=(model_path,)) as pool:
        for n_done, ok in enumerate(pool.imap_unordered(_cache_one, tasks, chunksize=4), 1):
            saved += ok
            if n_done % 200 == 0:
                print(f"  cached {n_done}/{len(tasks)} ({saved} saved)")
    manifest = {
        "num_saved": saved,
        "max_length": max_length,
        "resize": resize,
        "only_correct": only_correct,
    }
    mpath = os.path.join(cache_dir, "manifest.json")
    with open(mpath + ".tmp", "w") as f:
        json.dump(manifest, f)
    os.replace(mpath + ".tmp", mpath)
    print(f"Cache complete: {saved}/{len(tasks)} samples saved to {cache_dir}")


def cache_is_valid(cache_dir: str, max_length: int, resize: int, only_correct: bool) -> bool:
    """Cache is usable only if manifest exists and build params match."""
    mpath = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(mpath):
        return False
    try:
        m = json.load(open(mpath))
    except Exception:
        return False
    return (m.get("num_saved", 0) > 0
            and m.get("max_length") == max_length
            and m.get("resize") == resize
            and m.get("only_correct") == only_correct)


def _get_length_worker(f):
    import torch
    # Only len(input_ids) is needed; a plain load would pull the ~14MB
    # pixel_values too, i.e. ~646GB of NAS reads across the whole cache.
    # mmap keeps the big tensors on disk and measured 20-30x faster per file.
    try:
        e = torch.load(f, weights_only=True, map_location="cpu", mmap=True)
    except Exception:
        e = torch.load(f, weights_only=True, map_location="cpu")
    return len(e["input_ids"])


def build_lengths(cache_dir: str):
    """One-time pass: read len(input_ids) from every sample_*.pt, write lengths.json.

    Uses multiprocessing for parallel NAS reads.
    """
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(cache_dir, "sample_*.pt")))
    # NAS metadata latency dominates this scan, so oversubscribe the CPUs.
    n_workers = int(os.environ.get("SFT_LENGTH_WORKERS", min(64, (os.cpu_count() or 16) * 2)))
    print(f"  scanning {len(files)} cached samples with {n_workers} workers...")
    with Pool(n_workers) as pool:
        lengths = []
        for i, length in enumerate(pool.imap(_get_length_worker, files, chunksize=32)):
            lengths.append(length)
            if (i + 1) % 1000 == 0:
                print(f"  lengths {i+1}/{len(files)}")
    lpath = os.path.join(cache_dir, "lengths.json")
    with open(lpath + ".tmp", "w") as f:
        json.dump(lengths, f)
    os.replace(lpath + ".tmp", lpath)
    print(f"lengths.json written ({len(lengths)} entries)")


def train(
    trajectory_dir: str,
    model_path: str,
    output_dir: str,
    max_length: int = 8192,
    batch_size: int = 2,
    grad_accum: int = 2,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    resize: int = 512,
    max_steps: int = None,
    save_steps: int = 100,
    save_total_limit: int = 2,
    resume: bool = False,
    only_correct: bool = False,
    cache_dir: str = None,
    cache_workers: int = 16,
    max_samples: int = None,
    local_cache_dir: str = None,
):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments

    torch.backends.cudnn.enabled = False
    # cudnn.enabled does NOT gate the cuDNN SDPA backend; it fails with
    # "No valid execution plans built" on the ViT's variable-length sequences.
    torch.backends.cuda.enable_cudnn_sdp(False)

    # Auto-build cache: if cache_dir is set, build it on rank 0 (skipping when
    # already valid); other ranks wait for the manifest. Params-changed rebuilds
    # from scratch (stale .pt files are removed first). Default-on behavior —
    # just pass --cache-dir every run and the script figures out the rest.
    if cache_dir:
        import time
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        manifest_existed = os.path.exists(os.path.join(cache_dir, "manifest.json"))
        if cache_is_valid(cache_dir, max_length, resize, only_correct):
            print(f"Cache valid — skipping build ({cache_dir})")
        elif rank == 0:
            if manifest_existed:
                print("Cache params changed — rebuilding from scratch")
                for f in glob.glob(os.path.join(cache_dir, "sample_*.pt")):
                    os.remove(f)
                try:
                    os.remove(os.path.join(cache_dir, "manifest.json"))
                except OSError:
                    pass
            print(f"Building cache (rank 0, {cache_workers} workers)...")
            build_cache(trajectory_dir, model_path, cache_dir, max_length, resize,
                        only_correct, cache_workers)
        else:
            mpath = os.path.join(cache_dir, "manifest.json")
            print(f"[rank {rank}] waiting for rank 0 to finish cache build...")
            while not os.path.exists(mpath):
                time.sleep(30)

        # lengths.json for group_by_length (rank 0 builds, others wait)
        lpath = os.path.join(cache_dir, "lengths.json")
        if not os.path.exists(lpath):
            if rank == 0:
                print("Building lengths.json (rank 0)...")
                build_lengths(cache_dir)
            else:
                while not os.path.exists(lpath):
                    time.sleep(10)

    print(f"Loading processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    if cache_dir:
        cache_files = sorted(glob.glob(os.path.join(cache_dir, "sample_*.pt")))
        if not cache_files:
            print(f"No sample_*.pt files found in {cache_dir}. "
                  f"Run with --build-cache first. Exiting.")
            return

        if max_samples is not None and len(cache_files) > max_samples:
            import random
            n_total = len(cache_files)
            idxs = list(range(n_total))
            random.seed(42)
            random.shuffle(idxs)
            idxs = sorted(idxs[:max_samples])
            cache_files = [cache_files[i] for i in idxs]
            print(f"Subsampled cache to {max_samples}/{n_total} files")

        if local_cache_dir and cache_files:
            import atexit
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            ready_marker = os.path.join(local_cache_dir, "_ready")
            if rank == 0:
                os.makedirs(local_cache_dir, exist_ok=True)
                local_files = []
                local_lengths = []
                lpath_src = os.path.join(cache_dir, "lengths.json")
                all_lengths = json.load(open(lpath_src)) if os.path.exists(lpath_src) else None
                all_files_sorted = sorted(glob.glob(os.path.join(cache_dir, "sample_*.pt")))
                name_to_len = {}
                if all_lengths and len(all_lengths) == len(all_files_sorted):
                    name_to_len = {os.path.basename(f): all_lengths[i] for i, f in enumerate(all_files_sorted)}
                print(f"[rank 0] Copying {len(cache_files)} files to local disk ({local_cache_dir})...")
                for i, src in enumerate(cache_files):
                    dst = os.path.join(local_cache_dir, os.path.basename(src))
                    shutil.copy2(src, dst)
                    local_files.append(dst)
                    if name_to_len:
                        _l = name_to_len.get(os.path.basename(src))
                        if _l is None:  # stale lengths.json -> drop lengths entirely
                            name_to_len = {}
                            local_lengths = []
                        else:
                            local_lengths.append(_l)
                    if (i + 1) % 1000 == 0:
                        print(f"  copied {i+1}/{len(cache_files)}")
                local_lpath = os.path.join(local_cache_dir, "lengths.json")
                with open(local_lpath, "w") as f:
                    json.dump(local_lengths, f)
                with open(ready_marker, "w") as f:
                    f.write("ready")
                print(f"[rank 0] Local cache ready: {len(local_files)} files at {local_cache_dir}")
                def _cleanup_local():
                    shutil.rmtree(local_cache_dir, ignore_errors=True)
                    print(f"Cleaned up local cache: {local_cache_dir}")
                atexit.register(_cleanup_local)
            else:
                import time
                print(f"[rank {rank}] Waiting for rank 0 to copy local cache...")
                while not os.path.exists(ready_marker):
                    time.sleep(10)
            local_files_sorted = sorted(glob.glob(os.path.join(local_cache_dir, "sample_*.pt")))
            cache_files = local_files_sorted

        class CachedVista3DDataset(torch.utils.data.Dataset):
            def __init__(self, files):
                import time

                self.files = files
                self.lengths = None
                cache_dir_ = os.path.dirname(files[0])
                lpath = os.path.join(cache_dir_, "lengths.json")
                all_files = sorted(glob.glob(os.path.join(cache_dir_, "sample_*.pt")))
                all_lengths = json.load(open(lpath)) if os.path.exists(lpath) else None
                # lengths.json can go stale when the cache is rebuilt/extended.
                # Rebuild it once (fast mmap scan) instead of degrading to
                # no-lengths: without lengths the grouped sampler full-loads
                # every sample to measure lengths (prohibitively slow on NAS).
                if all_lengths is None or len(all_lengths) != len(all_files):
                    rank_ = int(os.environ.get("RANK", "0"))
                    if rank_ == 0:
                        print("lengths.json stale or missing - rebuilding (mmap scan)...")
                        build_lengths(cache_dir_)
                    else:
                        while not (os.path.exists(lpath)
                                   and len(json.load(open(lpath))) == len(all_files)):
                            time.sleep(10)
                    all_lengths = json.load(open(lpath))
                if len(all_lengths) == len(all_files):
                    name_to_len = {os.path.basename(f): all_lengths[i] for i, f in enumerate(all_files)}
                    lens = [name_to_len.get(os.path.basename(f)) for f in files]
                    if all(l is not None for l in lens):
                        self.lengths = lens

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                return torch.load(self.files[idx], weights_only=True)

        dataset = CachedVista3DDataset(cache_files)
        n_samples = len(cache_files)
        print(f"Cached dataset ready: {n_samples} pre-tokenized samples from {cache_dir}")
    else:
        samples = load_agent_dataset(trajectory_dir, only_correct)
        if not samples:
            print("No samples to train on. Exiting.")
            return

        class LazyVista3DDataset(torch.utils.data.Dataset):
            """Tokenizes on demand — keeps host RAM flat across 8 ranks.

            Over-long / fully-masked samples fall through to the next index so
            __getitem__ always returns a valid entry.
            """

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                for offset in range(len(self.data)):
                    j = (idx + offset) % len(self.data)
                    try:
                        entry = tokenize_with_spans(self.data[j], processor, max_length, resize)
                    except Exception as e:
                        print(f"  Skip sample {j}: {type(e).__name__}: {e}")
                        entry = None
                    if entry is not None:
                        return entry
                raise RuntimeError("No tokenizable sample found in dataset")

        dataset = LazyVista3DDataset(samples)
        n_samples = len(samples)
        print(f"Lazy dataset ready: {n_samples} samples (tokenized on the fly)")

    print(f"Loading model: {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    image_token_id = getattr(model.config, "image_token_id", None)

    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        pixel_values, image_grid_thw = [], []

        pad_id = tokenizer.pad_token_id or 0
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            if "pixel_values" in item:
                pixel_values.append(item["pixel_values"])
            if "image_grid_thw" in item:
                image_grid_thw.append(item["image_grid_thw"])

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if image_token_id is not None:
            result["mm_token_type_ids"] = (result["input_ids"] == image_token_id).long()
        if pixel_values:
            result["pixel_values"] = torch.cat(pixel_values, dim=0).to(torch.bfloat16)
        if image_grid_thw:
            result["image_grid_thw"] = torch.cat(image_grid_thw, dim=0).to(torch.long)
        return result

    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(output_dir, "tensorboard")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        logging_steps=5,
        save_steps=save_steps,
        gradient_checkpointing=True,
        save_total_limit=save_total_limit if save_total_limit > 0 else None,
        report_to="tensorboard",
        bf16=True,
        max_steps=max_steps or -1,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        ddp_timeout=7200,
    )

    class ChatTemplateSaveCallback(TrainerCallback):
        """Copy the VLM processor file set into every checkpoint after save.

        HF Trainer saves none of the processor files for VLMs, so a checkpoint
        loaded later fails with "does not have a chat template" (agent loop)
        or "Can't load image processor ... preprocessor_config.json" (verl
        rollout). Copying the full set in makes each checkpoint usable as a
        processor.
        """

        _FILES = [
            "chat_template.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ]

        def __init__(self, source_dir: str):
            self._src_dir = source_dir

        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt_dir):
                return
            for name in self._FILES:
                src = os.path.join(self._src_dir, name)
                dst = os.path.join(ckpt_dir, name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copyfile(src, dst)

    class Vista3DTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is not None and loss.dim() > 0:
                loss = loss.mean()
            return (loss, outputs) if return_outputs else loss

        def _get_train_sampler(self, *args, **kwargs):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            lengths = getattr(self.train_dataset, "lengths", None)
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=None,
            )

    trainer = Vista3DTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[ChatTemplateSaveCallback(model_path)],
    )

    print(f"Starting training: {n_samples} samples, {num_epochs} epochs")
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*")) if resume else []
    if resume and ckpts:
        latest = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
        print(f"Resuming from {latest}")
        trainer.train(resume_from_checkpoint=True)
    else:
        if resume:
            print(f"--resume set but no checkpoint-* under {output_dir}; starting fresh")
        trainer.train()

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vista3D Tool-Calling SFT")
    parser.add_argument("--trajectory-dir", type=str, required=True,
                        help="Directory containing agent_*.jsonl files")
    parser.add_argument("--model", type=str,
                        default="/mnt/faster_nas/benyou.mys/medixrl/checkpoints/sft_mm_v3/checkpoint-709")
    parser.add_argument("--output", type=str,
                        default="/mnt/faster_nas/benyou.mys/medixrl/checkpoints/vista3d_tool_sft_full")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=100,
                        help="checkpoint every N steps; a full agentic epoch is ~1425 steps "
                             "at several minutes each, so 500 would mean no checkpoint for a day")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="keep only the N most recent checkpoint-*; older ones are deleted")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint-* in --output")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override num_epochs with fixed step count (for testing)")
    parser.add_argument("--only-correct", action="store_true",
                        help="Train only on questions the teacher answered correctly")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Auto-built tokenized cache: built on rank 0 if missing/stale, "
                             "skipped if valid. Params changed (max-length/resize/only-correct) "
                             "triggers a full rebuild.")
    parser.add_argument("--build-cache", action="store_true",
                        help="Pre-tokenize all samples into --cache-dir and exit (run with plain python, not accelerate)")
    parser.add_argument("--cache-workers", type=int, default=16,
                        help="Parallel workers for cache build")
    args = parser.parse_args()

    if args.build_cache:
        if not args.cache_dir:
            parser.error("--build-cache requires --cache-dir")
        build_cache(
            trajectory_dir=args.trajectory_dir,
            model_path=args.model,
            cache_dir=args.cache_dir,
            max_length=args.max_length,
            resize=args.resize,
            only_correct=args.only_correct,
            workers=args.cache_workers,
        )
        sys.exit(0)

    train(
        trajectory_dir=args.trajectory_dir,
        model_path=args.model,
        output_dir=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        resize=args.resize,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume=args.resume,
        only_correct=args.only_correct,
        cache_dir=args.cache_dir,
        cache_workers=args.cache_workers,
    )
