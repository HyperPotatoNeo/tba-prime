import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
import uuid
import wandb

from zeroband.inference import envs

import numpy as np
import pyarrow.parquet as pq
import requests
import torch

from huggingface_hub import snapshot_download

from zeroband.tasks import *
from zeroband.inference.config import Config as InferenceConfig
from zeroband.utils.pydantic_config import parse_argv
from zeroband.training.mp import EnvWrapper
from zeroband.utils.utils import clean_exit
from zeroband.inference.logger import setup_logger

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

def log_process(config, num_processes):
    # If the current step does not get any files for an hour; exit the wandb loop
    # If the current step gets partial files for 30 minutes; log and move to next step
    # Start the logging with some delay

    time.sleep(300)

    step = 0
    wandb.init(
        project=config.logger.wandb.project,
        config=OmegaConf.to_container(config, resolve=True)
    )

    start_time = time.time()
    while True:
        if time.time() - start_time > 3600:
            break

        metric_path = Path(config.path.metric_path) / f"step_{step}"
        if not os.path.exists(metric_path):
            time.sleep(60)
            continue

        files = list(metric_path.glob("*.json"))
        if len(files) < num_processes:
            time.sleep(60)
            continue

        final_dict = dict()
        for file in files:
            with open(file, 'r') as f:
                metrics = json.load(f)
                for key, value in metrics.items():
                    if key not in final_dict:
                        final_dict[key] = value / num_processes
                    else:
                        final_dict[key] += value / num_processes

        print(f'Logging step: {step}')
        wandb.log(final_dict, step=step)
        step += 1
        start_time = time.time()

@clean_exit
def inference(config: InferenceConfig):
    import zeroband.vllm_08_shim # FIX?
    from vllm import SamplingParams, TokensPrompt
    from zeroband.inference.parquet import get_parquet_table
    from zeroband.inference.rewards import compute_vllm_rewards

    from zeroband.inference.utils import (
        setup_model,
        filter_data_by_prompt_length,
        reload_checkpoints_from_range,
        reload_model_weights,
        format_prompts,
    )

    # Rank and device utils
    dp_rank = int(os.environ.get("DP_RANK", 0))
    device = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    print(f'Process loaded with rank {dp_rank} with GPU: {device}')

    # Logging utils
    logger = setup_logger(config.logger.log, parallel_config=config.parallel, dp_rank=dp_rank)
    logger.info("Starting inference")

    # Pre-download the model weights
    logger.info(f"Downloading model weights for {config.model.name}")
    start_time = time.time()
    snapshot_download(config.model.name)
    logger.success(f"Downloaded model weights in {time.time() - start_time:.2f}s")

    # Initialize model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model.name} tensor_parallel_size={config.parallel.tp} seed={config.seed})")
    start_time = time.time()
    llm = setup_model(config.model, tp=config.parallel.tp, seed=config.seed)
    tokenizer = llm.get_tokenizer()
    logger.success(f"Initialized model and tokenizer in {time.time() - start_time:.2f}s")

    # Initialize dataset
    logger.info(f"Initializing dataset (name={config.data.name}, split={config.data.train_split})")
    start_time = time.time()
    dataset_class = instantiate(config.data.dataset, tokenizer=tokenizer)
    dataset = dataset_class.dataset[config.data.train_split]
    logger.success(f"Initialized dataset with {len(dataset):,} problems in {time.time() - start_time:.2f}s")

    # Set dataset seeds for shuffling
    seed = config.seed + int(os.environ.get("DP_RANK", 0))
    generator = np.random.default_rng(seed)

    # Optionally, filter out prompts that are too long
    if config.data.max_prompt_len:
        logger.info(f"Filtering out prompts with more than {config.data.max_prompt_len} tokens")
        start_time = time.time()
        dataset = filter_data_by_prompt_length(dataset, config.data.max_prompt_len, tokenizer)
        logger.success(f"Filtered long prompts in {time.time() - start_time:.2f}s - {len(dataset)} samples remaining")

    # Optionally, filter dataset for samples within difficulty range
    if config.data.difficulty_filtering:
        logger.info(
            f"Filtering dataset for difficulty in [{config.data.difficulty_filtering.min_solve_rate}, {config.data.difficulty_filtering.max_solve_rate}]"
        )
        dataset = dataset.filter(
            lambda x: x[config.data.difficulty_filtering.solve_rate_field] >= config.data.difficulty_filtering.min_solve_rate
            and x[config.data.difficulty_filtering.solve_rate_field] <= config.data.difficulty_filtering.max_solve_rate
        )

    # Initialize sampling parameters
    logger.info(f"Initializing sampling parameters ({config.sampling})")
    sampling_params = SamplingParams(**config.sampling)

    ckpt_step = 0
    step = 0

    if config.reload:
        # Find the latest checkpoint in the path and reload it, if error then don't reload
        try:
            checkpoints = [int(str(r).split('_')[-1]) for r in Path(config.path.ckpt_path).glob('*') if os.path.exists(os.path.join(config.path.ckpt_path, r))]
            ckpt_step = max(checkpoints)
            step = ckpt_step
            reload_model_weights(llm, os.path.join(config.path.ckpt_path, f'step_{ckpt_step}/model.safetensors'))
            logger.info(f"Resuming from step {ckpt_step}")
        except:
            logger.info(f"Could not reload from checkpoints, starting fresh training")

    logger.info(
        f"Batch size: {config.batch_size}, Number of rollouts: {config.sampling.n}"
    )

    while True:
        # If currently async, try reloading a more recent checkpoint
        if step != ckpt_step:
            llm, ckpt_step = reload_checkpoints_from_range(llm, config.path.ckpt_path, curr_step=ckpt_step, last_step=step, first_step=step-config.async_level, waiting=False)

        if step - ckpt_step > config.async_level:
            logger.warning(
                f"Hit async level ({config.async_level}) because inference step {step} is {step - ckpt_step} steps ahead of checkpoint step {ckpt_step}. Trying to reload model weights from {config.path.ckpt_path}"
            )
            llm, ckpt_step = reload_checkpoints_from_range(llm, config.path.ckpt_path, curr_step=ckpt_step, last_step=step, first_step=step - config.async_level, waiting=True)

        logger.info(f"Inference step {step} (Checkpoint step: {ckpt_step})")

        # Randomly sample indices
        indices = generator.choice(len(dataset), config.batch_size, replace=False)
        if seed is not None:
            sampling_params.seed = seed + step * 1_000_000  # 1M is needed to avoid collision from sampling.n

        problems = dataset.select(indices)
        prompts = [item["prompt"] for item in problems]
        ground_truths = [item["ground_truth"] for item in problems]

        # Get tokenized prompts as BatchEncoding
        tokenized_prompts = dataset_class.format_prompts(
            prompts,
        )

        generate_start_time = time.time()
        token_prompts: list[TokensPrompt] = [TokensPrompt(prompt_token_ids=prompt_token_ids) for prompt_token_ids in tokenized_prompts]
        request_outputs = llm.generate(token_prompts, sampling_params, use_tqdm=config.use_tqdm)
        generation_time = time.time() - generate_start_time

        # Compute performance metrics
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        batch_tokens_per_second = batch_tokens / generation_time
        batch_samples_per_minute = config.batch_size / generation_time * 60
        batch_avg_seq_length = batch_tokens / (config.batch_size * config.sampling.n)

        logger.info(
            f"Batch throughput: {batch_tokens_per_second:.2f} tokens/sec, {batch_samples_per_minute:.2f} samples/min ({batch_tokens} tokens in {generation_time:.2f}s, avg seq len: {batch_avg_seq_length:.1f})"
        )

        # Print first example of prompt and completion
        first_prompt = tokenizer.decode(request_outputs[0].prompt_token_ids)
        first_completion = tokenizer.decode(request_outputs[0].outputs[0].token_ids)
        logger.info(f'Showing example of prompt and completion\nPrompt: {first_prompt}\nCompletion: {first_completion}\nTrue Answer: {ground_truths[0]}')

        # Compute and log rewards and advantages
        logger.info("Computing rewards and advantages")
        start_time = time.time()
        request_rewards = compute_vllm_rewards(request_outputs, ground_truths, dataset_class.reward_fn, config.rewards)
        batch_reward = sum(sum(r.reward for r in req.rewards) for req in request_rewards) / (config.batch_size * config.sampling.n)
        logger.info(f"Average reward of the batch: {batch_reward:.2f} | Computed in {time.time() - start_time} time")

        metrics = {
            "infer/progress/batch_tokens": batch_tokens,
            "infer/performance/batch_tokens_per_second": batch_tokens_per_second,
            "infer/performance/batch_per_minute": batch_samples_per_minute,
            "infer/performance/batch_avg_seq_length": batch_avg_seq_length,
            "infer/rewards/batch_reward": batch_reward,
        }

        if sampling_params.seed is not None:
            sampling_seeds = [sampling_params.seed + i for i in range(sampling_params.n)] * config.batch_size
        else:
            sampling_seeds = [None] * (config.batch_size * config.sampling.n)

        # Get parquet table
        table = get_parquet_table(
            request_outputs,
            request_rewards,
            prompts,
            ckpt_step,
            problems,
            enable_logprobs=config.sampling.logprobs is not None,
            seeds=sampling_seeds,
            temperature=sampling_params.temperature,
        )

        # Save outputs to parquet file
        step_path = Path(config.path.rollout_path) / f"step_{step}"
        step_path.mkdir(parents=True, exist_ok=True)
        save_path = step_path / f"{uuid.uuid4()}.parquet"
        logger.info(f"Saving batch outputs to {save_path}")
        pq.write_table(table, save_path)

        # Save metrics to metric file
        metric_path = Path(config.path.metric_path) / f"step_{step}"
        metric_path.mkdir(parents=True, exist_ok=True)
        metric_path = metric_path / f"{uuid.uuid4()}.json"
        with open(metric_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Increment step
        step += 1

        if config.max_steps is not None and step > config.max_steps:
            logger.info(f"Reached max steps {config.max_steps}, stopping inference")
            break

    logger.success(f"Inference finished!")

@hydra.main(version_base=None, config_path="../../configs", config_name="inference")
def main(config: DictConfig) -> list[mp.Process]:
    processes = []
    print(OmegaConf.to_yaml(config))

    # Optionally, clean the rollout path
    if config.clean_paths:
        print(f"Cleaning existing paths")
        shutil.rmtree(config.path.metric_path, ignore_errors=True)
        shutil.rmtree(config.path.rollout_path, ignore_errors=True)

    if config.parallel.dp > 1:
        if config.parallel.tp == "auto":
            assert torch.cuda.device_count() % config.parallel.dp == 0, "Number of GPUs must be divisible by DP"
            config.parallel.tp = torch.cuda.device_count() // config.parallel.dp

        gpu_ids = envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.parallel.tp] for i in range(0, len(gpu_ids), config.parallel.tp)]

        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            print(f'Process rank: {rank} with GPU: {",".join(map(str, gpu_ids))}')
            env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "DP_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, env), args=(config,))
            processes.append(process)
    else:
        if config.parallel.tp == "auto":
            config.parallel.tp = torch.cuda.device_count()

        processes.append(mp.Process(target=inference, args=(config,)))

    processes.append(mp.Process(target=log_process, args=(config, len(processes))))

    # Start all processes
    print(f'Running {len(processes) - 1} processes')
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

if __name__ == "__main__":
    # Set spawn method before any other multiprocessing code
    mp.set_start_method("spawn")
    main()

    config = parse_argv(InferenceConfig)

    if config.rl and config.rl.step_endpoint is not None:
        current_step = requests.get(config.rl.step_endpoint).json()
        assert isinstance(current_step, int), "Current step must be an integer"

    # Maybe start shardcast downloader
    from zeroband.inference import envs as inference_envs

    if inference_envs.SHARDCAST_SERVERS is not None:
        assert config.rl is not None, "RL config is required when SHARDCAST_SERVERS is set"
        from zeroband.inference.shardcast_downloader import run_main_bg

        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.rl.ckpt_path,
            config.rl.async_level + 1,
            # TODO: maybe +1 because we most likely won't download the current step in time?
            # We could deadlock though.
            max(current_step - config.rl.async_level, 1),
        )
    else:
        shardcast_process = None

    try:
        main(config)

    finally:
        if shardcast_process is not None:
            import os
            import signal

            # SIGTERM is not working, so we use SIGKILL
            os.kill(shardcast_process.pid, signal.SIGKILL)
            shardcast_process.join()