"""Wrapper for verl multi-turn tool RL — patches vLLM 0.26.0 compatibility.

Patches two incompatibilities on WorkerWrapperBase:
1. __init__ — verl passes vllm_config kwarg; vLLM 0.26.0 moved that to
   init_worker(all_kwargs). Patch __init__ to accept and ignore the kwarg.
2. execute_method — verl calls inference_engine.execute_method(method, ...)
   for init_device / execute_model / sample_tokens, but vLLM 0.26.0 removed
   this method. Re-add it as a simple getattr dispatcher.

sitecustomize.py handles both patches in Ray worker processes via a lazy
__import__ hook. This wrapper is the fallback for the main process.

Usage (from run_verl_mixed_tool_rl.sh):
    python scripts/run_verl_tool_rl_wrapper.py [hydra args...]
"""

from vllm.v1.worker.worker_base import WorkerWrapperBase

if not getattr(WorkerWrapperBase, "_vllm_compat_patched", False):
    _orig_init = WorkerWrapperBase.__init__

    def _patched_init(self, rpc_rank=0, global_rank=None, vllm_config=None, **kwargs):
        _orig_init(self, rpc_rank=rpc_rank, global_rank=global_rank)

    def _execute_method(self, method, *args, **kwargs):
        if callable(method):
            return method(*args, **kwargs)
        func = getattr(self, method)
        return func(*args, **kwargs)

    WorkerWrapperBase.__init__ = _patched_init
    WorkerWrapperBase.execute_method = _execute_method
    WorkerWrapperBase._vllm_compat_patched = True

# Register qwen3_xml tool parser (model generates XML, not hermes JSON)
import qwen3_xml_parser  # noqa: F401  (vllm_patch dir in PYTHONPATH)

from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
