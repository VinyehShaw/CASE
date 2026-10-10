"""Auto-loaded by Python's site module in every process (including Ray workers).

Patches eight incompatibilities via lazy import hooks:

1. vLLM 0.26.0 WorkerWrapperBase.__init__ — verl passes vllm_config kwarg;
   vLLM 0.26.0 moved that to init_worker(all_kwargs).

2. vLLM 0.26.0 WorkerWrapperBase.execute_method — removed in 0.26.0; verl
   calls it for init_device / execute_model / sample_tokens.

3. Qwen3VL get_rope_index — transformers 5.x requires mm_token_type_ids and
   calls self.get_vision_position_ids. When verl binds get_rope_index to the
   processor (not the model), neither is available. Patch computes
   mm_token_type_ids from input_ids and binds get_vision_position_ids.

4. qwen3_xml tool parser — registered when verl's tool_parser module loads,
   so multi_turn.format=qwen3_xml resolves in Ray workers too.

5. AgentLoopBase.apply_chat_template — normalizes empty image/video lists to
   None, so text-only tool responses (images=[]) don't crash the HF processor.

6. Qwen3_5Model.forward — Qwen3.5's vision tower is FSDP-wrapped and its
   forward is guarded by `if pixel_values is not None`. In multi-turn tool RL
   some ranks receive text-only microbatches and skip the vision tower's
   allgathers, desynchronizing collectives and deadlocking compute_log_prob.
   This patch forces every rank to execute a dummy vision pass when no real
   images are present, keeping the collective sequence identical across ranks.

Hooks both builtins.__import__ AND importlib.import_module because transformers
uses the latter for lazy imports, bypassing __import__.
"""
import builtins
import importlib
import os
import sys

# Triton persists compiled kernels *and* FLA autotune results (FLA 0.5.2 passes
# cache_results=True) under TRITON_CACHE_DIR. Keep it stable and shared so every
# rank and every later run reuses both.
#
# An earlier revision sharded this per PID to dodge autotune lock contention.
# That defeated reuse completely: the dir name embedded os.getpid() so it changed
# every run, and since it read TRITON_CACHE_DIR as its own root and appended to
# it, children inherited an already-sharded value and nested further — each of
# the 8 workers re-benchmarked every GDN kernel from scratch, every run.
# setdefault keeps an externally supplied value and makes the nesting
# impossible to reintroduce.
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))

# Sharing the Triton cache dir also enables Triton's autotune *disk* cache
# (FLA passes cache_results=True), and that path takes a file lock while
# benchmarking. Under FSDP that lock deadlocks: one rank benchmarks holding the
# lock while the others block on it instead of entering the NCCL collective the
# first rank will need. It shows up as a hang with no traceback -- two ranks at
# 100% GPU, the rest idle. Disabling just the autotune disk cache keeps
# compiled-kernel reuse and lets ranks autotune in parallel on their own GPUs.
# Must precede the fla.utils._config import, which reads this via os.getenv.
os.environ.setdefault("FLA_CACHE_RESULTS", "0")

_orig_import = builtins.__import__
_orig_import_module = importlib.import_module

_ROPE_PATCH_TARGETS = {
    "transformers.models.qwen3_vl",
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
}

def _try_patch_qwen3vl(module):
    """Patch Qwen3VLModel.get_rope_index if the module has it."""
    mod_name = getattr(module, "__name__", "")
    if mod_name not in _ROPE_PATCH_TARGETS:
        return
    # Use __dict__ to avoid triggering transformers' lazy __getattr__
    if "Qwen3VLModel" not in module.__dict__:
        return
    _Q3VL = module.__dict__["Qwen3VLModel"]
    if getattr(_Q3VL, "_rope_compat_patched", False):
        return
    import types as _types
    _orig_rope = _Q3VL.get_rope_index
    _orig_gvp = _Q3VL.get_vision_position_ids

    def _compat_get_rope_index(
        self, input_ids, mm_token_type_ids=None,
        image_grid_thw=None, video_grid_thw=None,
        attention_mask=None, **kwargs
    ):
        if mm_token_type_ids is None:
            import torch as _torch
            image_token_id = getattr(self, "image_token_id", None)
            if image_token_id is None and hasattr(self, "config"):
                image_token_id = getattr(self.config, "image_token_id", None)
            video_token_id = getattr(self, "video_token_id", None)
            if video_token_id is None and hasattr(self, "config"):
                video_token_id = getattr(self.config, "video_token_id", None)
            mm_token_type_ids = _torch.zeros_like(input_ids, dtype=_torch.int)
            if image_token_id is not None:
                mm_token_type_ids[input_ids == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2

        if not hasattr(self, "get_vision_position_ids"):
            self.get_vision_position_ids = _types.MethodType(_orig_gvp, self)

        return _orig_rope(
            self, input_ids, mm_token_type_ids,
            image_grid_thw, video_grid_thw, attention_mask, **kwargs
        )

    _Q3VL.get_rope_index = _compat_get_rope_index
    _Q3VL._rope_compat_patched = True


def _try_patch_agent_loop(module):
    """Patch agent_loop classes for two issues:

    5. AgentLoopBase.apply_chat_template — normalizes empty image/video lists
       to None, so text-only tool responses (images=[]) don't crash the HF
       processor.

    7. AgentLoopWorkerBase._compute_multi_modal_inputs — pop mm_token_type_ids.
       This key has shape [1, seq_len] and seq_len varies per sample;
       torch.cat in extract_multi_modal_inputs fails when micro batch > 1.
       The agent loop already precomputes position_ids via get_rope_index,
       and the model only uses mm_token_type_ids inside
       compute_3d_position_ids which is skipped when position_ids is provided
       (always the case here).  So dropping it is safe and unblocks batched
       log_prob.
    """
    mod_name = getattr(module, "__name__", "")
    if mod_name != "verl.experimental.agent_loop.agent_loop":
        return

    # Patch 5: empty image/video list normalization (on AgentLoopBase)
    if "AgentLoopBase" in module.__dict__:
        _ALB = module.__dict__["AgentLoopBase"]
        if not getattr(_ALB, "_img_compat_patched", False):
            _orig_apply = _ALB.apply_chat_template
            _render_dbg = {"n": 0}

            async def _patched_apply(self, messages, tools=None, images=None, videos=None, remove_system_prompt=False):
                if images is not None and len(images) == 0:
                    images = None
                if videos is not None and len(videos) == 0:
                    videos = None
                ids = await _orig_apply(
                    self, messages, tools=tools, images=images, videos=videos,
                    remove_system_prompt=remove_system_prompt,
                )
                # Patch 11 (diagnostic): rendered image-token count vs image sizes
                if images and _render_dbg["n"] < 12:
                    try:
                        img_tok_id = getattr(self.processor, "image_token_id", None)
                        cnt = sum(1 for t in ids if t == img_tok_id) if img_tok_id is not None else -1
                        sizes = [(getattr(im, "width", -1), getattr(im, "height", -1)) for im in images][:4]
                        _render_dbg["n"] += 1
                        print(f"[PatchDebug][render] n_images={len(images)} sizes={sizes} "
                              f"n_image_tokens={cnt} total_ids={len(ids)}", flush=True)
                    except Exception as _e:
                        print(f"[PatchDebug][render] dbg failed: {_e}", flush=True)
                return ids

            _ALB.apply_chat_template = _patched_apply
            _ALB._img_compat_patched = True

    # Patch 15: get_rope_index StopIteration → clean sample-level failure.
    # Sequence image-segment count can exceed image_grid_thw rows after ~1.6M
    # tool turns (observed once, step 1295); upstream get_rope_index does a
    # bare next() on the grid iterator and kills the whole 128-trajectory
    # batch. Strategy: wrap _compute_position_ids — on StopIteration return a
    # text-only position_ids fallback and flag the CURRENT asyncio task via a
    # ContextVar (worker coroutines interleave, so an instance attribute would
    # leak between trajectories). The postprocess wrapper reads the flag and,
    # if set, zeroes the returned response_mask (no policy gradient) and
    # forces reward_score = -2.0 (failure tier; the async reward result for
    # this trajectory is discarded). Success path is untouched.
    if "AgentLoopWorkerBase" in module.__dict__:
        _AWB = module.__dict__["AgentLoopWorkerBase"]
        if not getattr(_AWB, "_rope_stopiter_patched", False):
            import contextvars as _cv15
            import torch as _torch15

            _rope_degraded = _cv15.ContextVar("rope_degraded", default=False)

            _orig_cpi = _AWB._compute_position_ids
            _orig_postproc = _AWB._agent_loop_postprocess

            def _patched_cpi(self, input_ids, attention_mask, multi_modal_inputs):
                try:
                    return _orig_cpi(self, input_ids, attention_mask, multi_modal_inputs)
                except (StopIteration, TypeError) as _e:
                    # StopIteration: image-segment count exceeds image_grid_thw
                    # rows (iterator exhausted). TypeError: image segments exist
                    # but image_grid_thw is None ('NoneType' is not an iterator).
                    # Same family of placeholder/grid mismatch — degrade this
                    # trajectory instead of killing the batch.
                    _rope_degraded.set((_e, input_ids, attention_mask, multi_modal_inputs))
                    # Text-only sequential positions; the vision dims are moot
                    # because response_mask gets zeroed for this trajectory.
                    valid = attention_mask[0].bool()
                    pos = _torch15.ones((1, len(input_ids[0])), dtype=_torch15.long)
                    pos[0, valid] = _torch15.arange(valid.sum().item())
                    return pos.unsqueeze(0)

            def _p15_dump(self, exc, input_ids, attention_mask, mmi, output):
                """Dump forensics for one placeholder/grid mismatch, then
                decide: same patient+question recurring => data; spread out
                and correlated with wild outputs => model/sampling."""
                import json as _json
                import os as _os
                import time as _time
                try:
                    img_tok = getattr(self.processor, "image_token_id", None)
                    ids = input_ids.reshape(-1)
                    n_img_tok = int((ids == img_tok).sum()) if img_tok is not None else -1
                    # count contiguous image-token segments
                    segs = 0
                    prev = False
                    for t in ids.tolist():
                        cur = (img_tok is not None and t == img_tok)
                        if cur and not prev:
                            segs += 1
                        prev = cur
                    grid = mmi.get("image_grid_thw") if mmi else None
                    n_grid = int(grid.shape[0]) if grid is not None else 0
                    imgs = (output.multi_modal_data or {}).get("images") or []
                    n_msg = len(getattr(output, "messages", []) or [])
                    n_resp = len(getattr(output, "response_mask", []) or [])
                    extra = getattr(output, "extra_fields", {}) or {}
                    raw_prompt = str(extra.get("raw_prompt", ""))[:200]
                    # last few messages (tool rounds) — shapes tell whether the
                    # model emitted a malformed multi-image call
                    tail_msgs = []
                    for m_ in (getattr(output, "messages", []) or [])[-6:]:
                        c = m_.get("content") if isinstance(m_, dict) else None
                        if isinstance(c, list):
                            kinds = [p_.get("type") for p_ in c if isinstance(p_, dict)]
                            tail_msgs.append(f"{m_.get('role')}:[{','.join(map(str, kinds))}]")
                        else:
                            tail_msgs.append(f"{m_.get('role')}:{str(c)[:60]!r}")
                    line = (
                        f"[Patch15-DUMP] exc={type(exc).__name__} "
                        f"img_token_count={n_img_tok} img_segments={segs} "
                        f"grid_rows={n_grid} images_listed={len(imgs)} "
                        f"n_messages={n_msg} resp_len={n_resp} turns={getattr(output,'num_turns',-1)} "
                        f"raw_prompt={raw_prompt!r} tail={tail_msgs}"
                    )
                    print(line, flush=True)
                    dump_dir = "/tmp/patch15_dumps"
                    _os.makedirs(dump_dir, exist_ok=True)
                    fn = _os.path.join(dump_dir, "traj_%d_%d.json" % (int(_time.time()), id(output) % 100000))
                    with open(fn, "w") as f_:
                        _json.dump({
                            "exc": type(exc).__name__,
                            "img_token_count": n_img_tok, "img_segments": segs,
                            "grid_rows": n_grid, "images_listed": len(imgs),
                            "num_turns": getattr(output, "num_turns", -1),
                            "raw_prompt": extra.get("raw_prompt", ""),
                            "extra_fields_keys": list(extra.keys()),
                            "messages": [
                                {k: (str(v)[:500] if not isinstance(v, list) else
                                     [str(x)[:200] for x in v])
                                 for k, v in m_.items()}
                                for m_ in (getattr(output, "messages", []) or [])
                            ] if isinstance(getattr(output, "messages", None), list) else [],
                        }, f_, ensure_ascii=False, indent=1, default=str)
                    print(f"[Patch15-DUMP] full trajectory saved to {fn}", flush=True)
                except Exception as _dump_e:
                    print(f"[Patch15-DUMP] dump failed: {_dump_e}", flush=True)

            async def _patched_postproc(self, output, **kwargs):
                tok = _rope_degraded.set(None)
                try:
                    result = await _orig_postproc(self, output, **kwargs)
                finally:
                    degraded = _rope_degraded.get()
                    _rope_degraded.reset(tok)
                if degraded is not None:
                    exc, ids, mask, mmi = degraded
                    self._p15_dump(exc, ids, mask, mmi, output)
                    n_img = len((output.multi_modal_data or {}).get("images") or [])
                    print(
                        f"[Patch15][rope-mismatch] degraded 1 trajectory: "
                        f"exc={type(exc).__name__} images={n_img} "
                        f"-> mask zeroed, reward=-2.0",
                        flush=True,
                    )
                    # No token counts as model-decided → no policy gradient.
                    result.response_mask = _torch15.zeros_like(result.response_mask)
                    # Failure tier (same as r_correct unextractable); rm_scores
                    # tensor is rebuilt from this in _postprocess.
                    result.reward_score = -2.0
                return result

            _AWB._compute_position_ids = _patched_cpi
            _AWB._agent_loop_postprocess = _patched_postproc
            _AWB._p15_dump = _p15_dump
            _AWB._rope_stopiter_patched = True
    # (on AgentLoopWorkerBase, where _compute_multi_modal_inputs is defined)
    if "AgentLoopWorkerBase" in module.__dict__:
        _ALW = module.__dict__["AgentLoopWorkerBase"]
        if not getattr(_ALW, "_mmi_compat_patched", False):
            _orig_cmmi = _ALW._compute_multi_modal_inputs
            _collate_dbg = {"n": 0}

            def _patched_cmmi(self, output, input_ids):
                mmi = _orig_cmmi(self, output, input_ids)
                mmi.pop("mm_token_type_ids", None)
                # Patch 11 (diagnostic): collate-side expectation vs input_ids
                if _collate_dbg["n"] < 20:
                    try:
                        imgs = (output.multi_modal_data or {}).get("images") or []
                        img_tok_id = getattr(self.processor, "image_token_id", None)
                        ids = input_ids.reshape(-1)
                        actual = int((ids == img_tok_id).sum()) if img_tok_id is not None else -1
                        gthw = mmi.get("image_grid_thw")
                        _collate_dbg["n"] += 1
                        print(f"[PatchDebug][collate] n_images={len(imgs)} mmi_keys={sorted(mmi.keys())} "
                              f"grid_thw={gthw.tolist() if gthw is not None else None} "
                              f"actual_image_tokens={actual} seq_len={ids.shape[0]}", flush=True)
                    except Exception as _e:
                        print(f"[PatchDebug][collate] dbg failed: {_e}", flush=True)
                return mmi

            _ALW._compute_multi_modal_inputs = _patched_cmmi
            _ALW._mmi_compat_patched = True


def _try_patch_extract_mmi(module):
    """Patch verl.utils.model.extract_multi_modal_inputs (Patch 8).

    Safety net for Patch 7: even if mm_token_type_ids survives into the
    batch (different code path, patch not loaded, etc.), this patch
    catches the torch.cat failure and keeps values as a list.  The model
    forward ignores mm_token_type_ids when position_ids is provided
    (always the case in agent-loop RL), so a list value is harmless.
    """
    mod_name = getattr(module, "__name__", "")
    if mod_name != "verl.utils.model":
        return
    if "extract_multi_modal_inputs" not in module.__dict__:
        return
    _func = module.__dict__["extract_multi_modal_inputs"]
    if getattr(_func, "_cat_compat_patched", False):
        return

    import torch as _torch

    def _patched_extract(batch_data, indices=None):
        collected = {}
        has_image_bound = False

        sel = batch_data
        if indices is not None:
            sel = [batch_data[i] for i in indices if i < len(batch_data)]

        for inputs in sel:
            d = inputs.data if hasattr(inputs, "data") and not isinstance(inputs, dict) else inputs
            if d is None:
                continue
            if "image_bound" in d:
                has_image_bound = True
            for key, value in d.items():
                if value is not None:
                    collected.setdefault(key, []).append(value)

        multi_modal_inputs = {}
        for key, values in collected.items():
            if has_image_bound:
                multi_modal_inputs[key] = values
            else:
                try:
                    multi_modal_inputs[key] = _torch.cat(values, dim=0)
                except RuntimeError:
                    multi_modal_inputs[key] = values
        return multi_modal_inputs

    module.__dict__["extract_multi_modal_inputs"] = _patched_extract
    _patched_extract._cat_compat_patched = True


def _try_patch_qwen35_vision_sync(module):
    """Patch Qwen3_5Model.forward so every rank exercises the vision tower.

    Qwen3.5's vision blocks are FSDP-wrapped.  In multi-turn tool RL a rank may
    receive a text-only microbatch while another rank receives images; the text
    rank skips the `if pixel_values is not None` branch and therefore skips the
    vision tower's FSDP allgathers.  The collective sequence then diverges and
    compute_log_prob deadlocks.

    This wrapper, when pixel_values is None, computes a single dummy image and
    calls the original get_image_features (which runs self.visual and therefore
    triggers the vision collectives) before delegating to the original forward.
    The dummy embeddings are discarded.  All ranks now issue identical vision
    collectives at the same logical point.
    """
    mod_name = getattr(module, "__name__", "")
    if mod_name != "transformers.models.qwen3_5.modeling_qwen3_5":
        return
    if "Qwen3_5Model" not in module.__dict__:
        return
    _Q35M = module.__dict__["Qwen3_5Model"]
    if getattr(_Q35M, "_vision_sync_patched", False):
        return
    _orig_forward = _Q35M.forward
    _orig_get_image_features = _Q35M.get_image_features

    def _patched_forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        ref = input_ids if input_ids is not None else inputs_embeds

        if pixel_values is None:
            import torch as _torch
            vc = getattr(self.config, "vision_config", None)
            patch_size = getattr(vc, "patch_size", 16) or 16
            temporal_patch = getattr(vc, "temporal_patch_size", 2) or 2
            in_channels = getattr(vc, "in_channels", 3) or 3
            # 448 is the standard input size for Qwen3.5 vision.
            grid = 448 // patch_size
            # The vision tower takes a flat patch sequence, not an image:
            # forward() unpacks `seq_len, _ = hidden_states.size()` and
            # patch_embed views it as (-1, C, temporal, patch, patch).
            dummy_pixels = _torch.zeros(
                grid * grid,
                in_channels * temporal_patch * patch_size * patch_size,
                dtype=_torch.float32,
                device=ref.device,
            )
            dummy_grid = _torch.tensor(
                [[1, grid, grid]],
                dtype=_torch.long,
                device=ref.device,
            )
            # Trigger the vision tower's FSDP all-gather; discard the output.
            # Deliberately not guarded: skipping this on a subset of ranks
            # desynchronizes the collective sequence and hangs FSDP, which is
            # far harder to debug than a loud shape error.
            #
            # kwargs from the text forward are NOT forwarded: the vision path
            # feeds them to get_vision_position_ids/bilinear_indices/cu_seqlens,
            # which then derive positions from the real batch instead of this
            # dummy grid, and rotary_pos_emb ends up sized for a different grid.
            _orig_get_image_features(
                self, dummy_pixels, dummy_grid, return_dict=True
            )

        return _orig_forward(
            self,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs
        )

    _Q35M.forward = _patched_forward
    _Q35M._vision_sync_patched = True


def _try_patch_qwen3_xml_json(module):
    """Patch 9: JSON fallback for vLLM's built-in qwen3_xml engine parser.

    The Qwen3-VL chat template trains the model to emit JSON objects
    ({"name": ..., "arguments": {...}}) inside the tool-call wrapper tags,
    but Qwen3EngineToolParser only understands the <function=...> XML form
    and returns zero tool calls for JSON output — which makes eval agents
    silently treat every tool round as a final answer. Wrap
    extract_tool_calls: when the original finds nothing, scan the
    tool-call blocks for JSON objects and build ToolCall entries from them.
    """
    cls = getattr(module, "Qwen3EngineToolParser", None)
    if cls is None or getattr(cls, "_json_fallback_patched", False):
        return

    import json as _json
    import re as _re

    _tc_s = chr(60) + "tool_call" + chr(62)
    _tc_e = chr(60) + chr(47) + "tool_call" + chr(62)
    _block_re = _re.compile(_re.escape(_tc_s) + r"(.*?)" + _re.escape(_tc_e), _re.DOTALL)
    _json_re = _re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")

    _orig = cls.extract_tool_calls

    def _patched_extract(self, model_output, request):
        result = _orig(self, model_output, request)
        if result.tool_calls:
            return result
        try:
            from vllm.entrypoints.openai.engine.protocol import (
                ExtractedToolCallInformation,
                FunctionCall,
                ToolCall,
            )
        except Exception:
            return result
        calls = []
        for block in _block_re.findall(model_output):
            for jm in _json_re.finditer(block):
                try:
                    obj = _json.loads(jm.group(0))
                except Exception:
                    continue
                if not (isinstance(obj, dict) and obj.get("name")):
                    continue
                args = obj.get("arguments") or {}
                if not isinstance(args, dict):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {}
                calls.append(ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=str(obj["name"]).strip(),
                        arguments=_json.dumps(args, ensure_ascii=False),
                    ),
                ))
        if not calls:
            return result
        content = _block_re.sub("", model_output)
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=calls, content=content
        )

    cls.extract_tool_calls = _patched_extract
    cls._json_fallback_patched = True


def _try_patch_tool_loop_multi_image(module):
    """Patch 10: one image placeholder per image in multi-image tool responses.

    Upstream ToolAgentLoop._handle_processing_tools_state appends a single
    {"type": "image"} content item even when tool_response.image is a list
    of N images, while all N images are added to pixel_values. The rendered
    sequence then has 1 image-pad run but N images' features, and the
    training forward crashes with "Image features and image tokens do not
    match". This replacement inserts one placeholder per image.
    """
    mod_name = getattr(module, "__name__", "")
    if mod_name != "verl.experimental.agent_loop.tool_agent_loop":
        return
    cls = module.__dict__.get("ToolAgentLoop")
    if cls is None or getattr(cls, "_multi_image_patched", False):
        return

    import asyncio as _asyncio

    AgentState = module.__dict__["AgentState"]
    simple_timer = module.__dict__["simple_timer"]
    build_gpt_oss_tool_response_text = module.__dict__.get("build_gpt_oss_tool_response_text")
    logger = module.__dict__.get("logger")

    async def _patched_handle_processing_tools_state(self, agent_data):
        add_messages = []
        new_images_this_turn = []

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await _asyncio.gather(*tasks)

        for tool_response, tool_reward, _ in responses:
            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    # Patch 10: one placeholder per image (upstream adds only one).
                    n_imgs = len(tool_response.image) if isinstance(tool_response.image, list) else 1
                    content.extend({"type": "image"} for _ in range(n_imgs))
                if tool_response.video:
                    n_vids = len(tool_response.video) if isinstance(tool_response.video, list) else 1
                    content.extend({"type": "video"} for _ in range(n_vids))
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_response.image:
                if isinstance(tool_response.image, list):
                    for img in tool_response.image:
                        if img is not None:
                            new_images_this_turn.append(img)
                else:
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)

            if tool_response.video:
                if logger is not None:
                    logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            if logger is not None:
                logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn,
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    cls._handle_processing_tools_state = _patched_handle_processing_tools_state
    cls._multi_image_patched = True


def _try_patch_qwen3vl_embeds_debug(module):
    """Patch 12 (diagnostic): dump shapes when _get_input_embeds raises the
    image token/feature mismatch, so one crash yields definitive evidence."""
    if getattr(module, "__name__", "") != "verl.models.transformers.qwen3_vl":
        return
    orig = module.__dict__.get("_get_input_embeds")
    if orig is None or getattr(orig, "_dbg_wrapped", False):
        return

    def _wrapped(model, input_ids, attention_mask=None, pixel_values=None,
                 pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None):
        try:
            return orig(model, input_ids, attention_mask, pixel_values,
                        pixel_values_videos, image_grid_thw, video_grid_thw)
        except ValueError as e:
            try:
                img_id = getattr(model.config, "image_token_id", None)
                n_tok = int((input_ids == img_id).sum()) if img_id is not None else -1
                pv = getattr(pixel_values, "shape", None)
                g = image_grid_thw.tolist() if image_grid_thw is not None else None
                print(f"[PatchDebug][embeds] input_ids={tuple(input_ids.shape)} "
                      f"image_tokens={n_tok} pixel_values={pv} grid_thw={g} "
                      f"grid_prod={(image_grid_thw.prod(dim=1).tolist()) if image_grid_thw is not None else None}",
                      flush=True)
            except Exception as e2:
                print(f"[PatchDebug][embeds] dbg failed: {e2}", flush=True)
            raise

    _wrapped._dbg_wrapped = True
    module._get_input_embeds = _wrapped


def _try_patch_qwen3vl_unpack(module):
    """Patch 13: verl's _unpack_visual_out reads last_hidden_state, but newer
    transformers return BaseModelOutputWithDeepstackFeatures where
    last_hidden_state is PRE-merger (patches) and pooler_output is the
    POST-merger embeds (the ones that match image-pad token counts). Without
    this, _get_input_embeds sees 4x too many features (merge_size**2) and
    raises 'Image features and image tokens do not match'."""
    if getattr(module, "__name__", "") != "verl.models.transformers.qwen3_vl":
        return
    orig = module.__dict__.get("_unpack_visual_out")
    if orig is None or getattr(orig, "_unpack_patched", False):
        return

    def _wrapped(vision_out):
        if not isinstance(vision_out, tuple):
            merged = getattr(vision_out, "pooler_output", None)
            if merged is not None:
                return merged, getattr(vision_out, "deepstack_features", None)
        return orig(vision_out)

    _wrapped._unpack_patched = True
    module._unpack_visual_out = _wrapped


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _orig_import(name, globals, locals, fromlist, level)
    mod_name = getattr(module, "__name__", "")

    # Patch 1 & 2: vLLM WorkerWrapperBase
    if mod_name == "vllm.v1.worker.worker_base" and "WorkerWrapperBase" in module.__dict__:
        _WWB = module.__dict__["WorkerWrapperBase"]
        if not getattr(_WWB, "_vllm_compat_patched", False):
            _orig_init = _WWB.__init__
            def _patched_init(self, rpc_rank=0, global_rank=None, vllm_config=None, **kwargs):
                _orig_init(self, rpc_rank=rpc_rank, global_rank=global_rank)

            def _execute_method(self, method, *args, **kwargs):
                if callable(method):
                    return method(*args, **kwargs)
                func = getattr(self, method)
                return func(*args, **kwargs)

            _WWB.__init__ = _patched_init
            _WWB.execute_method = _execute_method
            _WWB._vllm_compat_patched = True

    # Patch 3: Qwen3VL get_rope_index (via __import__ path)
    _try_patch_qwen3vl(module)

    # Patch 5: AgentLoopBase.apply_chat_template empty image/video normalization
    _try_patch_agent_loop(module)

    # Patch 6: Qwen3_5Model.forward vision-tower collective sync
    _try_patch_qwen35_vision_sync(module)

    # Patch 8: extract_multi_modal_inputs cat-failure safety net
    _try_patch_extract_mmi(module)

    # Patch 4: Register qwen3_xml tool parser when verl tool_parser is imported
    if mod_name == "verl.experimental.agent_loop.tool_parser":
        try:
            import qwen3_xml_parser  # noqa: F401
        except Exception:
            pass  # verl not fully loaded yet, parser registered by wrapper

    # Patch 9: JSON fallback for vLLM built-in qwen3_xml engine parser
    if mod_name == "vllm.tool_parsers.qwen3_engine_tool_parser":
        _try_patch_qwen3_xml_json(module)

    # Patch 10: one image placeholder per image in multi-image tool responses
    _try_patch_tool_loop_multi_image(module)

    # Patch 12: diagnostic dump at the qwen3_vl embed mismatch site
    _try_patch_qwen3vl_embeds_debug(module)

    # Patch 13: use pooler_output (post-merger) from visual tower output
    _try_patch_qwen3vl_unpack(module)

    return module


def _patched_import_module(name, package=None):
    module = _orig_import_module(name, package)
    # Patch 3: Qwen3VL get_rope_index (via importlib path — transformers lazy imports)
    _try_patch_qwen3vl(module)
    # Patch 5: AgentLoopBase.apply_chat_template empty image/video normalization
    _try_patch_agent_loop(module)
    # Patch 6: Qwen3_5Model.forward vision-tower collective sync
    _try_patch_qwen35_vision_sync(module)
    # Patch 8: extract_multi_modal_inputs cat-failure safety net
    _try_patch_extract_mmi(module)
    # Patch 9: JSON fallback for vLLM built-in qwen3_xml engine parser
    _try_patch_qwen3_xml_json(module)
    # Patch 10: one image placeholder per image in multi-image tool responses
    _try_patch_tool_loop_multi_image(module)
    # Patch 12: diagnostic dump at the qwen3_vl embed mismatch site
    _try_patch_qwen3vl_embeds_debug(module)
    # Patch 13: use pooler_output (post-merger) from visual tower output
    _try_patch_qwen3vl_unpack(module)
    return module


# Patch 14: make this code package's distill the canonical source for
# causal_kl (and friends). A pip-installed copy in site-packages or a stale
# local snapshot can shadow repo edits and cause silent behavior differences.
def _prefer_repo_distill():
    try:
        # CASE_CODE_ROOT is exported by the run/*.sh scripts; otherwise derive
        # the repo root relative to this file (<repo>/scripts/vllm_patch/).
        repo = os.environ.get("CASE_CODE_ROOT") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ck = os.path.join(repo, "distill", "opd_rl", "verl_opsd", "causal_kl.py")
        if not os.path.isfile(ck):
            return
        if repo not in sys.path:
            sys.path.insert(0, repo)
        # If another stale copy of this repo tree exists (point CASE_STALE_REPO
        # at it), pre-import the repo copy now so it is cached in sys.modules
        # and wins regardless.
        stale_dir = os.environ.get("CASE_STALE_REPO")
        if stale_dir and os.path.isdir(stale_dir):
            import importlib
            m = importlib.import_module("distill.opd_rl.verl_opsd.causal_kl")
            src = getattr(m, "__file__", "") or ""
            if not src.startswith(repo):
                print(f"[vllm_patch] WARNING: causal_kl loaded from {src}, not repo", flush=True)
            else:
                print(f"[vllm_patch] Patch 14: causal_kl pinned to repo copy ({src})", flush=True)
    except Exception as e:
        print(f"[vllm_patch] Patch 14 preload failed: {e}", flush=True)


_prefer_repo_distill()

builtins.__import__ = _patched_import
importlib.import_module = _patched_import_module


# Patch 15b: late-binding safety net. The import hook above can fire while
# verl's agent_loop module is still executing (e.g. an in-module import
# re-enters the hook): at that moment AgentLoopBase is defined but
# AgentLoopWorkerBase is not, so Patch 5 applies and Patch 15 silently skips —
# and the cached module never re-triggers the hook. This left some Ray
# workers unpatched (observed: worker with PatchDebug output crashed bare).
# Fix: whenever ANY verl module finishes importing, re-check the agent_loop
# module and apply Patch 15 if it is now complete but still unpatched.
def _p15b_latebind(module):
    try:
        mod_name = getattr(module, "__name__", "")
        if not mod_name.startswith("verl"):
            return
        import sys as _sys
        al = _sys.modules.get("verl.experimental.agent_loop.agent_loop")
        if al is None or "AgentLoopWorkerBase" not in al.__dict__:
            return
        _AWB = al.__dict__["AgentLoopWorkerBase"]
        if getattr(_AWB, "_rope_stopiter_patched", False):
            return
        _try_patch_agent_loop(al)
        if getattr(_AWB, "_rope_stopiter_patched", False):
            print("[vllm_patch] Patch 15b: late-bound onto already-loaded "
                  "agent_loop module", flush=True)
    except Exception as e:
        print(f"[vllm_patch] Patch 15b latebind failed: {e}", flush=True)


_orig_import_for_15b = builtins.__import__ if not getattr(builtins.__import__, "_p15b", False) else None


def _patched_import_15b(name, globals=None, locals=None, fromlist=(), level=0):
    module = _patched_import(name, globals, locals, fromlist, level)
    _p15b_latebind(module)
    return module


_patched_import_15b._p15b = True
builtins.__import__ = _patched_import_15b


# Patch 15c: bulletproof last resort. 15b's import-hook trigger can be too
# late: agent_loop is often among the LAST verl modules imported, so no later
# verl import ever fires and the half-init window never gets repaired (this
# is exactly what happened at the 15:28 run: 5/8 workers patched, the
# crashing one was in the unpatched 3). Fix: also hook AgentLoopManager.__init__
# — the single entry point that CREATES every AgentLoopWorker. Before any
# worker exists, we force-apply Patch 15 if still missing. This runs in the
# TaskRunner process (which creates the workers via Ray actors) AND in any
# process that instantiates the manager, so the class-level patch is verified
# right before first use. (Ray actor workers re-import in their own process;
# those still rely on 15/15b above — 15c closes the creator-side gap.)
def _p15c_hook_manager_init(module):
    try:
        if getattr(module, "__name__", "") != "verl.experimental.agent_loop.agent_loop":
            return
        ALM = module.__dict__.get("AgentLoopManager")
        if ALM is None or getattr(ALM, "_p15c_patched", False):
            return
        _orig_init = ALM.__init__

        def _patched_init(self, *a, **k):
            # Force-apply Patch 15 (and friends) right before workers spawn.
            import sys as _sys
            al = _sys.modules.get("verl.experimental.agent_loop.agent_loop")
            if al is not None:
                _p15b_latebind(al)  # reuse the completeness-aware applier
            return _orig_init(self, *a, **k)

        ALM.__init__ = _patched_init
        ALM._p15c_patched = True
        print("[vllm_patch] Patch 15c: AgentLoopManager.__init__ hooked for "
              "pre-spawn Patch 15 verification", flush=True)
    except Exception as e:
        print(f"[vllm_patch] Patch 15c hook failed: {e}", flush=True)


# Fire 15c both via the import hook (below) and immediately for modules
# that are ALREADY loaded when sitecustomize runs late (e.g. inside a
# Ray worker that re-executes sitecustomize after verl import began).
_orig_import_15b_prev = builtins.__import__


def _patched_import_15c(name, globals=None, locals=None, fromlist=(), level=0):
    module = _patched_import_15b(name, globals, locals, fromlist, level)
    _p15c_hook_manager_init(module)
    return module


_patched_import_15c._p15c = True
builtins.__import__ = _patched_import_15c

# If sitecustomize is imported late (after verl started loading), the manager
# module may already be in sys.modules — hook it right now.
import sys as _sys_15c
for _mod_15c in list(_sys_15c.modules.values()):
    _p15c_hook_manager_init(_mod_15c)
    _p15b_latebind(_mod_15c)
