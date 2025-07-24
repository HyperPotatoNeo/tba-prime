# Import environment before any other imports
# ruff: noqa
import time

# Import environment before any other imports
# ruff: noqa: I001
from pathlib import Path

from huggingface_hub import snapshot_download

import zeroband.vllm_08_shim
from zeroband.eval.logger import setup_logger
from zeroband.eval.utils import run_benchmark
from zeroband.inference.utils import reload_checkpoint, setup_model
from zeroband.utils.utils import clean_exit
import wandb
import re

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

def extract_step_num(path):
    match = re.search(r'step_(\d+)', path.name)
    return int(match.group(1)) if match else float('inf')

@clean_exit
@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(config):
    # Initialize the logger
    print(OmegaConf.to_yaml(config))

    wandb.init(
        project=config.logger.wandb.project,
        config=OmegaConf.to_container(config, resolve=True)
    )

    logger = setup_logger(config.logger.log)
    logger.info("Starting evaluation")

    # Pre-download the model weights
    logger.info(f"Downloading model weights for {config.model.name}")
    start_time = time.time()
    snapshot_download(config.model.name)
    logger.success(f"Downloaded model weights in {time.time() - start_time:.2f}s")

    # Initializing the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model} tensor_parallel_size={config.parallel.tp} seed={config.seed})")
    start_time = time.time()
    llm = setup_model(config.model, tp=config.parallel.tp, seed=config.seed)
    logger.success(f"Initialized model and tokenizer in {time.time() - start_time:.2f}s")

    # Run benchmarks on base model
    dataset_class = instantiate(config.data.dataset, tokenizer=llm.get_tokenizer())

    logger.info(f"Running evals on base model {config.model.name}")
    metrics = run_benchmark(llm, dataset_class, config.data, config.model, config.sampling, step=0, seed=config.seed, use_tqdm=config.use_tqdm)
    wandb.log(metrics, step=0)

    # If specified, run online evaluation
    checkpoints = sorted(Path(config.path.ckpt_path).glob('*'), key=extract_step_num)
    for checkpoint in checkpoints:
        logger.info(f"Loading checkpoint {checkpoint}")
        step = int(str(checkpoint).split('_')[-1])
        llm = reload_checkpoint(llm, config.path.ckpt_path, step)
        logger.info(f"Running evals for checkpoint step {step}")
        metrics = run_benchmark(llm, dataset_class, config.data, config.model, config.sampling, step, seed=config.seed, use_tqdm=config.use_tqdm)
        wandb.log(metrics, step=step)

    logger.info("Evaluation finished!")

if __name__ == "__main__":
    main()
