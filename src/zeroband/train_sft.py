import logging
import os
import shutil
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import shardcast
import torch
import torch.distributed as dist
import torch.distributed.tensor
from jaxtyping import Float
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torch.utils.data import DataLoader
from zeroband.training import envs
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.config import Config as TrainingConfig
from zeroband.training.data import BatchOutput, DatasetOutput, get_dataloader, packed_batch
from zeroband.training.logger import setup_logger
from zeroband.training.loss import entropy_loss, grpo_loss, kl_penalty, selective_log_softmax, grpo_loss_tb
from zeroband.training.utils import (
    MetricsAverager,
    PerfCounter,
    apply_ac_ckpt,
    copy_model_to_cpu,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.training.config import ClippingConfig, GRPOVariantsConfig, KlCovConfig, RatioConfig, TBConfig
from zeroband.training.world_info import WorldInfo, get_world_info
from zeroband.utils.models import ModelType, get_model_and_tokenizer
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import wandb
from functools import partial

def sft_format(example, dataset, max_seq_len):
    prompt_tokens = torch.tensor(dataset.format_prompts([example['prompt']])[0])
    answer_tokens = torch.tensor(dataset.format_prompts_with_answers([example['prompt']], [example['ground_truth']])[0])
    loss_mask = torch.cat([torch.zeros(len(prompt_tokens)), torch.ones(len(answer_tokens) - len(prompt_tokens))], dim=0).int()
    padding_len = max_seq_len - len(answer_tokens)
    if padding_len > 0:
        answer_tokens = torch.cat([answer_tokens, torch.full((padding_len,), fill_value=dataset.tokenizer.pad_token_id, dtype=answer_tokens.dtype)], dim=0)
        loss_mask = torch.cat([loss_mask, torch.full((padding_len,), fill_value=0, dtype=prompt_tokens.dtype)], dim=0)
    return {
        'prompt_tokens': prompt_tokens[:max_seq_len].unsqueeze(0),
        'answer_tokens': answer_tokens[:max_seq_len].unsqueeze(0),
        'loss_mask': loss_mask[:max_seq_len].unsqueeze(0),
    }

def collate_fn(batch):
    return {
        'input_ids': torch.cat([torch.tensor(example['answer_tokens']) for example in batch], dim=0),
        'loss_mask': torch.cat([torch.tensor(example['loss_mask']) for example in batch], dim=0),
    }

def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank


def get_logprobs(model: ModelType, input_ids: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


@clean_exit
@hydra.main(version_base=None, config_path="../../configs/", config_name="train_sft")
def train(config: DictConfig):    
    # Multiprocessing and Torch / Cuda Details
    if "ZERO_BAND_DEV" not in os.environ:
        torch_log.setLevel(logging.CRITICAL)
    world_info = get_world_info()

    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    # Set up logging
    logger = setup_logger(config.logger.log, world_info)
    if world_info.rank == 0:
        wandb.init(
            project=config.logger.wandb.project,
            config=OmegaConf.to_container(config, resolve=True)
        )

    # Optionally, clean the checkpoint path
    if config.clean_paths and config.path.ckpt_path is not None:
        logger.info(f"Cleaning checkpoint paths {config.path.ckpt_path}")
        shutil.rmtree(config.path.ckpt_path, ignore_errors=True)

    logger.info(f"Start training on {world_info.world_size} rank(s)")

    model, tokenizer = get_model_and_tokenizer(config.model.name, config.attn_impl)
    perf_counter = PerfCounter(window_size=min(10, 2), model=model, seq_len=config.data_seq_len)

    if config.liger_qwen:
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            model=model,
        )

    # Shard the current model
    apply_ac_ckpt(model, 1)
    apply_fsdp(model, config.reshard_after_forward)

    optimizer = instantiate(config.optimizer.optim, params=model.parameters())
    loss_fn = instantiate(config.loss)

    # [TODO]
    # Edit based on checkpoint loading
    # total_samples = config.start_total_samples if config.start_total_samples is not None else 0
    start_step = 0
    training_progress = TrainingProgress(total_tokens=0, step=start_step, total_samples=0)

    if config.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

    # [TODO]
    # # Logic to resume training from a checkpoint
    # if config.reload:

    # if config.ckpt.resume:
    #     logger.info(f"loading checkpoint from {config.ckpt.resume}")
    #     load_checkpoint_fsdp_state(model, [optimizer], training_progress, config.ckpt.resume)

    dataset_class = instantiate(config.data.dataset, tokenizer=tokenizer)
    dataset = dataset_class.dataset[config.data.train_split]
    dataset = dataset.map(partial(sft_format, dataset=dataset_class, max_seq_len=config.data_seq_len))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    
    training_progress = TrainingProgress(total_tokens=0, step=0, total_samples=0)
    metric_averager = MetricsAverager()

    # Add labels to the dataset as additional column
    
    # Compute loss masks for the dataset

    logger.info("Starting training loop")
    
    for epochs in range(1000):
        for batch in dataloader:
            metric_averager = MetricsAverager()

            input_ids = batch["input_ids"].to("cuda")
            loss_mask = batch["loss_mask"].to("cuda")
            
            print(tokenizer.decode(input_ids[0]))

            # Model Forward
            logprobs = get_logprobs(model, input_ids)
            
            # Compute loss
            loss = -(logprobs * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum()
            
            # Update metric_averager
            metric_averager.update(f'train/loss', loss.detach().clone())

            # Delete Batch Information
            del batch, logprobs, input_ids, loss_mask

            # Backward
            loss.backward()

            del loss

            # Sync all metrics
            metric_averager.sync()

            # Sync loss and gradient computation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_norm_clip).full_tensor()
            optimizer.step()
            optimizer.zero_grad()
            training_progress.step += 1

            if world_info.rank == 0:
                metrics = {'train/grad_norm': grad_norm.detach().clone()}
                for key, value in metric_averager.items():
                    metrics[key] = value.item()
                wandb.log(metrics, step=training_progress.step)

            del grad_norm

            if config.path.ckpt_path is not None and training_progress.step % config.interval_rollout == 0:
                path = Path(config.path.ckpt_path) / f"step_{training_progress.step}"
                save_ckpt_for_rollout(model, tokenizer, path)

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("Training finished!")

if __name__ == "__main__":
    train()