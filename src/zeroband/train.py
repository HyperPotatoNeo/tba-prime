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

def check_config(config):
    return
    # Check gradient accumulation is consistent with logZ computation
    assert config.sampling.n % config.micro_bs == 0, str(
        f"The micro batch size ({config.micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )
    
    # Check mini-batching consistent with batch size and number of processes / GPUs
    assert config.batch_size % config.micro_bs == 0
    assert batch_size % world_info.world_size == 0

def get_local_batch_size(batch_size: int, micro_bs: int, data_workers: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size


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


def get_logprobs(model: ModelType, input_ids: torch.Tensor, position_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids, position_ids=position_ids).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


@clean_exit
@hydra.main(version_base=None, config_path="../../configs/", config_name="train")
def train(config: DictConfig):
    check_config(config)
    
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

    local_batch_size = get_local_batch_size(config.batch_size, config.micro_bs, config.num_data_workers, world_info)
    model, tokenizer = get_model_and_tokenizer(config.model.name, config.attn_impl)

    perf_counter = PerfCounter(window_size=min(10, 2 * config.step_per_rollout), model=model, seq_len=config.data_seq_len)

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

    # Shard the reference model
    model_reference, _ = get_model_and_tokenizer(config.model.name, config.attn_impl)
    apply_fsdp(model_reference, config.reshard_after_forward)

    # Shard the current model used for only logprob computation
    model_for_logprob_only, _ = get_model_and_tokenizer(config.model.name, config.attn_impl)
    apply_fsdp(model_for_logprob_only, config.reshard_after_forward)

    optimizer = instantiate(config.optimizer.optim, params=model.parameters())
    loss_fn = instantiate(config.loss)

    # [TODO]
    # Edit based on checkpoint loading
    # total_samples = config.start_total_samples if config.start_total_samples is not None else 0
    start_step = 0
    training_progress = TrainingProgress(total_tokens=0, step=start_step, total_samples=0)

    if config.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model
        model_reference = torch.compile(model_reference) if not TYPE_CHECKING else model_reference
        model_for_logprob_only = torch.compile(model_for_logprob_only) if not TYPE_CHECKING else model_for_logprob_only

    logger.info(f"memory before model reference offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    offloaded_reference = offload_model_to_cpu(model_reference)
    logger.info(f"memory after model reference offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    logger.info(f"memory before model for logprob offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    offloaded_model = offload_model_to_cpu(model_for_logprob_only)
    logger.info(f"memory after model for logprob offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # [TODO]
    # # Logic to resume training from a checkpoint
    # if config.reload:

    # if config.ckpt.resume:
    #     logger.info(f"loading checkpoint from {config.ckpt.resume}")
    #     load_checkpoint_fsdp_state(model, [optimizer], training_progress, config.ckpt.resume)

    # step_count_init = (
    #     config.start_rollout_step if config.start_rollout_step is not None else training_progress.step // config.optim.step_per_rollout
    # )
    step_count_init = training_progress.step // config.step_per_rollout

    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        local_batch_size=local_batch_size,
        batch_size=config.batch_size * config.step_per_rollout,
        data_path=config.path.rollout_path,
        num_workers=config.num_data_workers,
        step_count_init=step_count_init,
    )
    train_dataloader_iterator = iter(train_dataloader)

    previous_ckpt_rollout = []

    logger.info("Starting training loop")

    while True:
        time_start = time.time()
        total_time_data_loading = 0
        total_time_packing = 0
        log_Z = dict()

        # Pre-compute the logprobs with the model before updating it
        with torch.no_grad():
            wake_up_model_from_cpu(model_reference, offloaded_reference)
            wake_up_model_from_cpu(model_for_logprob_only, offloaded_model)

            data: list[list[BatchOutput]] = []

            for rollout_step in range(config.step_per_rollout):
                if rollout_step not in log_Z:
                    log_Z[rollout_step] = []

                logger.debug(f"Start rollout step {rollout_step} / {config.step_per_rollout}")

                time_data_loading = time.time()
                # Get a batch of rollouts
                batch_rollout: list[DatasetOutput] = next(train_dataloader_iterator)
                total_time_data_loading += time.time() - time_data_loading

                time_packing = time.time()
                batch_packed = packed_batch(
                    batch_rollout, config.data_seq_len, tokenizer.pad_token_id, config.micro_bs, config.collate_mode
                )
                total_time_packing += time.time() - time_packing

                num_grad_acc_steps = len(batch_packed)

                for grad_acc_step in range(num_grad_acc_steps):
                    batch = batch_packed[grad_acc_step]

                    logger.debug(f"Grad Accumulation: {grad_acc_step} / {num_grad_acc_steps}, Batch: {batch['input_ids'].shape}")
                    input_ids = batch["input_ids"].to("cuda")

                    # Re-compute logprobs
                    per_token_logps = get_logprobs(model_for_logprob_only, input_ids, batch["position_ids"], batch["temperature"])
                    batch["logprobs"] = per_token_logps.to("cpu")

                    # Get reference logprobs
                    per_token_logps_reference = get_logprobs(model_reference, input_ids, batch["position_ids"], batch["temperature"])
                    batch["ref_logprobs"] = per_token_logps_reference.to("cpu")
                    
                    log_Z[rollout_step].append(batch["rewards"] + config.loss.beta * (batch["loss_mask"][:, 1:]*(batch["ref_logprobs"] - batch["logprobs"])).sum(1))
                
                data.append(batch_packed)
                log_Z[rollout_step] = torch.stack(log_Z[rollout_step]).view(-1, config.sampling.n)
                log_Z[rollout_step] = log_Z[rollout_step].mean(dim=-1, keepdim=True).repeat(1, config.sampling.n).view(-1, config.micro_bs)

            # If we don't manually reshard the the embed and lm head will conflict with the offloading because they will stay unshard until backward which we never call
            reshard_module(model_reference)
            offloaded_reference = offload_model_to_cpu(model_reference)

            # Here we sepcifically don't save the tensor offloaded, they are alreay consumed and we will never use it again.
            # This avoids having to make sure we don't keep too much tensor offloaded in cpu memory
            reshard_module(model_for_logprob_only)
            offload_model_to_cpu(model_for_logprob_only)

            dataloader = iter(data)
            total_time = time.time() - time_start
            total_time_logprob = total_time - total_time_data_loading - total_time_packing
            logger.info(f"Time Metrics | Data Loading: {total_time_data_loading:.2f} seconds, Packing: {total_time_packing:.2f} seconds, LogProbs: {total_time_logprob:.2f} seconds, Total: {total_time:.2f} seconds")

        # After packing relevant data, now we do training
        for rollout_step in range(config.step_per_rollout):
            logger.debug(f"Start training rollout step {rollout_step} / {config.step_per_rollout}")
            metric_averager = MetricsAverager()

            rollout_data = next(dataloader)
            num_grad_acc_steps = len(rollout_data)

            # Gradient Accumulation based training step
            for grad_acc_step in range(num_grad_acc_steps):
                batch = rollout_data[grad_acc_step]
                logger.debug(f"Training Grad Accumulation: {grad_acc_step} / {num_grad_acc_steps}, Batch: {batch['input_ids'].shape}")

                input_ids = batch["input_ids"].to("cuda")
                if config.normalize_batch_to_token_count:
                    max_tokens = int(sum(batch["seq_lens"]))
                else:
                    max_tokens = input_ids.shape[0] * input_ids.shape[1]

                loss_mask = batch["loss_mask"]

                # Update general metrics
                for rewards in batch["rewards"]:
                    metric_averager.update("train/rewards/batch_reward", rewards)
                for seq_lens in batch["seq_lens"]:
                    metric_averager.update("train/lengths/seq_len", seq_lens)
                
                # Model Forward
                logits = model(
                    input_ids=input_ids, position_ids=batch["position_ids"]
                ).logits.contiguous()

                # Gather arguments for loss computation 
                advantages = batch["advantages"].to("cuda")
                loss_mask = loss_mask.to("cuda")
                original_logprobs = batch["logprobs"].to("cuda")
                ref_logp = batch["ref_logprobs"].to("cuda")

                # Loss
                loss_metrics = loss_fn(
                    logits,
                    input_ids,
                    batch["rewards"].to("cuda"),
                    advantages,
                    log_Z[rollout_step][grad_acc_step].to("cuda"),
                    original_logprobs,
                    ref_logp,
                    loss_mask,
                    None,
                    batch["temperature"],
                    max_tokens,
                )
                loss = loss_metrics['total_loss']

                for key, value in loss_metrics.items():
                    metric_averager.update(f'train/losses/{key}', value.detach().clone())

                loss = loss / num_grad_acc_steps
                inputs_ids_shape = input_ids.shape

                # Delete Batch Information
                del batch, logits, input_ids, advantages, loss_mask, original_logprobs

                # Backward
                loss.backward()

                del loss, loss_metrics

            # Sync all metrics
            metric_averager.sync()

            # Sync loss and gradient computation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_norm_clip).full_tensor()
            optimizer.step()
            optimizer.zero_grad()

            training_progress.step += 1

            # Checkpoint the model after a gradient update
            if config.path.ckpt_path is not None and training_progress.step % config.step_per_rollout == 0:
                logger.debug("Saving rollout ckpt")
                save_step = training_progress.step // config.step_per_rollout
                path = Path(config.path.ckpt_path) / f"step_{save_step}"
                previous_ckpt_rollout.append(path)
                time_rollout_ckpt = time.time()
                save_ckpt_for_rollout(model, tokenizer, path)
                time_rollout_ckpt = time.time() - time_rollout_ckpt

                time_rollout_delete = time.time()
                if len(previous_ckpt_rollout) > config.max_async_level:
                    path_to_delete = previous_ckpt_rollout.pop(0)
                    ckpt_step = int(str(path_to_delete).split("_")[-1])

                    should_keep = config.interval_rollout is not None and ckpt_step % config.interval_rollout == 0
                    if path_to_delete.exists() and not should_keep:
                        logger.info(f"Removing past rollout ckpt at {path_to_delete}")
                        shutil.rmtree(path_to_delete, ignore_errors=True)
                time_rollout_delete = time.time() - time_rollout_delete

            # Compute some auxiliary metrics
            inner_lr = [group["lr"] for group in optimizer.param_groups][0]
            token_per_gpu = inputs_ids_shape[0] * inputs_ids_shape[1] * num_grad_acc_steps
            new_tokens = world_info.world_size * token_per_gpu
            perf_counter.count_tokens(new_tokens)
            training_progress.total_tokens += new_tokens
            training_progress.total_samples += config.batch_size
            padding_proportion = (config.data_seq_len - metric_averager["train/lengths/seq_len"].item() - 1) / config.data_seq_len

            avg_log_Z = log_Z[rollout_step].mean().to('cuda')
            dist.all_reduce(avg_log_Z, op=dist.ReduceOp.SUM)
            avg_log_Z /= world_info.world_size

            metrics = {
                "train/losses/grad_norm": grad_norm.item(),
                "train/rollout_step": rollout_step,
                "train/inner_lr": inner_lr,
                "train/total_tokens": training_progress.total_tokens,
                "train/total_samples": training_progress.total_samples,
                "train/lengths/padding_proportion": padding_proportion,
                "train/rewards/log_Z": avg_log_Z,
            }
            for key, value in metric_averager.items():
                metrics[key] = value.item()

            log = (
                f"step: {training_progress.step}, "
                f"rollout_step: {training_progress.step // config.step_per_rollout}, "
                f"loss: {metric_averager['train/losses/total_loss'].item():.4f}, "
                f"sample_reward: {metric_averager['train/rewards/batch_reward'].item():.4f}, "
                f"avg_log_Z: {avg_log_Z.item():.4f}, "
            )

            del grad_norm

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                tokens_per_second_per_gpu = tokens_per_second / world_info.world_size
                mfu = perf_counter.get_mfu()
                metrics.update(
                    {
                        "train/perf/tokens_per_second": tokens_per_second,
                        "train/perf/tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                        "train/perf/mfu": mfu,
                    }
                )

                log += f", tokens_per_second: {tokens_per_second:.2f}, tokens_per_second_per_gpu: {tokens_per_second_per_gpu:.2f}, mfu: {mfu:.2f}"

            if world_info.rank == 0:
                wandb.log(metrics, step=training_progress.step)

            logger.info(log)

        reshard_module(model_for_logprob_only)
        offloaded_model = copy_model_to_cpu(model)

        time_rollout_step = time.time() - time_start
        logger.success(f"Finished training step {training_progress.step} in {time_rollout_step:.2f}s")

        if config.max_steps is not None and training_progress.step > config.max_steps:
            logger.info(f"Reached max steps {config.max_steps}, stopping training")
            break

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("Training finished!")

if __name__ == "__main__":
    train()