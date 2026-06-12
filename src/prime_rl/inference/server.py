import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    # kv-recall / markovian-recall: apply the validated engine-side recall
    # stack (soft-pin, lazy publish, reload keep-cpu, eager CPU archive).
    # setdefault so explicit per-var overrides in the launch env win.
    if config.kv_mode is not None:
        from kv_eviction.modes import engine_env_for_mode

        for key, value in engine_env_for_mode(
            config.kv_mode, recall_max_spans=config.kv_recall_max_spans
        ).items():
            os.environ.setdefault(key, value)


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
