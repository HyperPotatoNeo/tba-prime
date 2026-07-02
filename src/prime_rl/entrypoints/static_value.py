from __future__ import annotations

import getpass
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from subprocess import Popen

import tomli_w

from prime_rl.configs.static_value import StaticValueConfig
from prime_rl.configs.trainer import ClassificationValueLossConfig, LinearSchedulerConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title

STATIC_VALUE_TOML = "static_value.toml"


class StaticValueRunner:
    def __init__(self, config: StaticValueConfig):
        self.config = config
        self.config.repo_dir = self.config.repo_dir.resolve()
        self.config.output_dir = self.config.output_dir.resolve()
        self.config_dir = self.config.output_dir / "configs"
        self.data_dir = self.config.output_dir / "data"
        self.log_dir = self.config.output_dir / "logs"
        self.value_dir = self.config.output_dir / "value"
        self.value_eval_dir = self.config.output_dir / "value_eval"
        self.diagnostics_dir = self.config.output_dir / "diagnostics"
        self.logger = setup_logger("info")
        self.child_procs: list[Popen] = []
        self.inference_procs: list[Popen] = []
        self.inference_nodes: list[str] = []
        self.inference_configs: list[Path] = []
        self.remote_user = getpass.getuser()

    @property
    def wandb_project(self) -> str | None:
        return self.config.wandb.project if self.config.wandb is not None else None

    @property
    def run_name(self) -> str:
        if self.config.wandb is not None and self.config.wandb.name:
            return self.config.wandb.name
        return self.config.output_dir.name

    @property
    def max_completion_tokens(self) -> int:
        return self.config.sampling.max_completion_tokens or max(self.config.model.seq_len - 512, 1)

    def run(self) -> None:
        self._prepare_output()
        self._write_resolved_config()
        if self.config.dry_run:
            self._dry_run()
            return

        stages = (
            ["collect_train", "train_value", "collect_eval", "predict", "diagnostics"]
            if self.config.stage == "all"
            else [self.config.stage]
        )
        try:
            for stage in stages:
                self.logger.info(f"Starting static-value stage: {stage}")
                getattr(self, stage)()
        finally:
            self._cleanup_child_processes()
            self.stop_inference()

    def collect_train(self) -> None:
        nodes = self._slurm_nodes()
        self.start_inference(nodes)
        try:
            train_groups = self.config.data.train_groups
            node0_groups = (train_groups + 1) // 2
            node1_groups = train_groups - node0_groups
            specs = [
                (
                    nodes[0],
                    "train-node0",
                    self.data_dir / "train_node0",
                    node0_groups,
                    self.config.data.train_offset,
                    "node0:",
                ),
                (
                    nodes[1],
                    "train-node1",
                    self.data_dir / "train_node1",
                    node1_groups,
                    self.config.data.train_offset + node0_groups,
                    "node1:",
                ),
            ]
            train_inputs = [output_dir / "rollouts.jsonl" for _, _, output_dir, groups, _, _ in specs if groups > 0]
            procs = [
                self._start_collect(
                    node=node,
                    name=name,
                    output_dir=output_dir,
                    group_id_prefix=prefix,
                    num_train_prompts=groups,
                    num_train_groups=groups,
                    num_val_prompts=0,
                    num_test_prompts=0,
                    train_offset=offset,
                )
                for node, name, output_dir, groups, offset, prefix in specs
                if groups > 0
            ]
            self._wait_all(procs)
            self._merge_rollouts(
                self.data_dir / "train_rollouts.jsonl",
                self.config.data.train_episodes,
                train_inputs,
            )
        finally:
            self.stop_inference()

    def train_value(self) -> None:
        args = [
            *self._value_train_args(
                rollouts=self.data_dir / "train_rollouts.jsonl",
                output_dir=self.value_dir,
                steps=self.config.train.steps,
            ),
            "--skip-predict",
        ]
        if self.config.value_function.init_checkpoint is not None:
            args += ["--load-value-checkpoint", str(self.config.value_function.init_checkpoint)]
        self._run_value_torchrun(args, "value_train")

    def collect_eval(self) -> None:
        nodes = self._slurm_nodes()
        self.start_inference(nodes)
        try:
            procs = [
                self._start_collect(
                    node=nodes[0],
                    name="val",
                    output_dir=self.data_dir / "eval_val",
                    group_id_prefix="val:",
                    num_train_prompts=0,
                    num_train_groups=0,
                    num_val_prompts=self.config.data.val_groups,
                    num_test_prompts=0,
                    val_offset=self.config.data.eval_val_offset,
                ),
                self._start_collect(
                    node=nodes[1],
                    name="test",
                    output_dir=self.data_dir / "eval_test",
                    group_id_prefix="test:",
                    num_train_prompts=0,
                    num_train_groups=0,
                    num_val_prompts=0,
                    num_test_prompts=self.config.data.test_groups,
                    test_offset=self.config.data.eval_test_offset,
                ),
            ]
            self._wait_all(procs)
            self._merge_rollouts(
                self.data_dir / "eval_rollouts.jsonl",
                self.config.data.eval_episodes,
                [self.data_dir / "eval_val" / "rollouts.jsonl", self.data_dir / "eval_test" / "rollouts.jsonl"],
            )
        finally:
            self.stop_inference()

    def predict(self) -> None:
        args = [
            "--predict-only",
            "--load-value-checkpoint",
            str(self.value_dir / "value_checkpoint"),
            *self._value_train_args(
                rollouts=self.data_dir / "train_rollouts.jsonl",
                output_dir=self.value_eval_dir,
                steps=0,
                predict_rollouts=self.data_dir / "eval_rollouts.jsonl",
            ),
            "--predict-splits",
            "val",
            "test",
        ]
        self._run_value_torchrun(args, "value_predict")

    def diagnostics(self) -> None:
        common = self._local_env()
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "experiments.static_value_diagnostics.compute_offline_diagnostics",
            "--predictions-dir",
            str(self.value_eval_dir),
            "--output-dir",
            str(self.diagnostics_dir),
            "--group-size",
            str(self.config.data.group_size),
            "--group-sizes",
            *[str(size) for size in self.config.diagnostics.group_sizes],
            "--rho-step",
            str(self.config.diagnostics.rho_step),
            "--sensitivity-draws",
            str(self.config.diagnostics.sensitivity_draws),
            "--position-bucket-edges",
            *[str(edge) for edge in self.config.diagnostics.position_bucket_edges],
            "--seed",
            str(self.config.diagnostics.seed),
            *self._wandb_args("diagnostics"),
        ]
        self._run_local(cmd, self.log_dir / "diagnostics.log", env=common)
        cmd = [
            "uv",
            "run",
            "--with",
            "matplotlib",
            "python",
            "-m",
            "experiments.static_value_diagnostics.plot_offline_diagnostics",
            "--diagnostics-dir",
            str(self.diagnostics_dir),
            *self._wandb_args("plots"),
        ]
        self._run_local(cmd, self.log_dir / "plots.log", env=common)

    def start_inference(self, nodes: list[str]) -> None:
        self.stop_inference()
        for idx, node in enumerate(nodes[: self.config.deployment.num_nodes]):
            config_path = self._write_inference_config(idx)
            log_path = self.log_dir / f"inference_node{idx}.log"
            cmd = ["uv", "run", "inference", "@", str(config_path)]
            proc = self._popen_remote(node, cmd, log_path, cuda_visible_devices=self._all_gpus())
            self.inference_procs.append(proc)
            self.inference_nodes.append(node)
            self.inference_configs.append(config_path)
        for idx, node in enumerate(nodes[: self.config.deployment.num_nodes]):
            self._wait_for_inference(node, self.log_dir / f"inference_node{idx}.log")

    def stop_inference(self) -> None:
        for proc in self.inference_procs:
            proc.terminate()
        for proc in self.inference_procs:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        for node, config_path in zip(self.inference_nodes, self.inference_configs, strict=False):
            remote = f"pkill -u {shlex.quote(self.remote_user)} -f {shlex.quote(str(config_path))} || true"
            subprocess.run(["ssh", node, remote], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.inference_procs.clear()
        self.inference_nodes.clear()
        self.inference_configs.clear()

    def _prepare_output(self) -> None:
        for path in (
            self.config.output_dir,
            self.config_dir,
            self.data_dir,
            self.log_dir,
            self.value_dir,
            self.value_eval_dir,
            self.diagnostics_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._load_wandb_env()

    def _write_resolved_config(self) -> Path:
        path = self.config_dir / STATIC_VALUE_TOML
        data = self.config.model_dump(exclude={"dry_run"}, exclude_none=True, mode="json")
        with path.open("wb") as f:
            tomli_w.dump(data, f)
        self.logger.info(f"Wrote resolved static-value config to {path}")
        return path

    def _write_inference_config(self, idx: int) -> Path:
        path = self.config_dir / f"inference_node{idx}.toml"
        model = {"name": self.config.model.name, "max_model_len": self.config.model.seq_len}
        if self.config.model.trust_remote_code is not None:
            model["trust_remote_code"] = self.config.model.trust_remote_code
        if self.config.model.chat_template is not None:
            model["chat_template"] = self.config.model.chat_template
        data = {
            "gpu_memory_utilization": self.config.inference.gpu_memory_utilization,
            "server": {"host": "0.0.0.0", "port": self.config.inference.port},
            "model": model,
            "parallel": {"tp": self.config.inference.tp, "dp": self.config.inference.dp},
            "vllm_extra": {"generation_config": "vllm"},
        }
        with path.open("wb") as f:
            tomli_w.dump(data, f)
        return path

    def _dry_run(self) -> None:
        nodes = self._slurm_nodes(required=False)
        self.logger.success(
            "Dry run complete. "
            f"output_dir={self.config.output_dir} nodes={nodes or '<from SLURM at runtime>'} "
            f"train_episodes={self.config.data.train_episodes} eval_episodes={self.config.data.eval_episodes}"
        )

    def _slurm_nodes(self, *, required: bool = True) -> list[str]:
        if override := os.environ.get("STATIC_VALUE_NODES"):
            nodes = [node.strip() for node in override.split(",") if node.strip()]
        else:
            nodelist = os.environ.get("SLURM_JOB_NODELIST")
            if not nodelist:
                if required:
                    raise RuntimeError("SLURM_JOB_NODELIST is not set; run staged static-value diagnostics under Slurm")
                return []
            result = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"scontrol show hostnames failed: {result.stderr.strip()}")
            nodes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(nodes) < self.config.deployment.num_nodes:
            raise RuntimeError(
                f"static-value diagnostics require {self.config.deployment.num_nodes} nodes, got {len(nodes)}"
            )
        return nodes[: self.config.deployment.num_nodes]

    def _train_node(self) -> str:
        return self._slurm_nodes()[self.config.deployment.train_node_index]

    def _start_collect(
        self,
        *,
        node: str,
        name: str,
        output_dir: Path,
        group_id_prefix: str,
        num_train_prompts: int,
        num_train_groups: int,
        num_val_prompts: int,
        num_test_prompts: int,
        train_offset: int | None = None,
        val_offset: int | None = None,
        test_offset: int | None = None,
    ) -> Popen:
        cmd = [
            "uv",
            "run",
            "--with",
            "reasoning-gym",
            "python",
            "-m",
            "experiments.static_value_diagnostics.collect_static_rollouts",
            "--base-url",
            f"http://127.0.0.1:{self.config.inference.port}/v1",
            "--output-dir",
            str(output_dir),
            "--dataset-path",
            str(self.config.data.dataset_path),
            "--model",
            self.config.model.name,
            "--api-key-var",
            "VLLM_API_KEY",
            "--group-size",
            str(self.config.data.group_size),
            "--group-id-prefix",
            group_id_prefix,
            "--num-train-prompts",
            str(num_train_prompts),
            "--num-train-groups",
            str(num_train_groups),
            "--num-val-prompts",
            str(num_val_prompts),
            "--num-test-prompts",
            str(num_test_prompts),
            "--seed",
            str(self.config.data.seed),
            "--temperature",
            str(self.config.sampling.temperature),
            "--top-p",
            str(self.config.sampling.top_p),
            "--top-k",
            str(self.config.sampling.top_k),
            "--min-p",
            str(self.config.sampling.min_p),
            "--max-completion-tokens",
            str(self.max_completion_tokens),
            "--renderer-pool-size",
            str(self.config.sampling.renderer_pool_size),
            "--max-concurrent-groups",
            str(self.config.sampling.max_concurrent_groups),
            "--max-retries",
            str(self.config.sampling.max_retries),
            "--group-max-attempts",
            str(self.config.sampling.group_max_attempts),
            *self._wandb_args(name),
        ]
        if train_offset is not None:
            cmd += ["--train-offset", str(train_offset)]
        if val_offset is not None:
            cmd += ["--val-offset", str(val_offset)]
        if test_offset is not None:
            cmd += ["--test-offset", str(test_offset)]
        return self._popen_remote(node, cmd, self.log_dir / f"collect_{name}.log")

    def _value_train_args(
        self,
        *,
        rollouts: Path,
        output_dir: Path,
        steps: int,
        predict_rollouts: Path | None = None,
    ) -> list[str]:
        loss = self.config.value_function.loss
        scheduler = self.config.value_function.scheduler
        assert isinstance(loss, ClassificationValueLossConfig)
        assert isinstance(scheduler, LinearSchedulerConfig)
        args = [
            "--rollouts",
            str(rollouts),
            "--output-dir",
            str(output_dir),
            "--model",
            self.config.model.name,
            "--seq-len",
            str(self.config.model.seq_len),
            "--steps",
            str(steps),
            "--global-batch-size",
            str(self.config.train.global_batch_size),
            "--group-size",
            str(self.config.data.group_size),
            "--updates-per-batch",
            str(self.config.value_function.warmup_updates_per_batch),
            "--lr",
            str(self.config.value_function.optim.lr),
            "--weight-decay",
            str(self.config.value_function.optim.weight_decay),
            "--max-norm",
            str(self.config.value_function.optim.max_norm),
            "--warmup-steps",
            str(scheduler.warmup_steps),
            "--min-lr",
            str(scheduler.min_lr),
            "--gamma",
            str(self.config.value_function.gamma),
            "--gae-lambda",
            str(self.config.value_function.gae_lambda),
            "--seed",
            str(self.config.data.seed),
            "--reward-range",
            str(loss.reward_range[0]),
            str(loss.reward_range[1]),
            "--n-bins",
            str(loss.n_bins),
            "--loss-weight",
            str(self.config.value_function.loss_weight),
            "--attn",
            self.config.model.attn,
            "--impl",
            self.config.model.impl,
            "--optimization-dtype",
            self.config.model.optimization_dtype,
            "--reduce-dtype",
            self.config.model.reduce_dtype,
            "--dp-replicate",
            str(self.config.model.dp_replicate),
            "--cp",
            str(self.config.model.cp),
            "--ep",
            str(self.config.model.ep),
            "--dist-timeout-seconds",
            str(self.config.train.dist_timeout_seconds),
            "--log-every",
            str(self.config.train.log_every),
            "--mfu-model-params",
            str(self.config.train.mfu_model_params),
            "--mfu-peak-tflops-per-gpu",
            str(self.config.train.mfu_peak_tflops_per_gpu),
            *self._wandb_args("value" if steps > 0 else "predict"),
        ]
        if self.config.train.micro_batch_tokens is not None:
            args += ["--micro-batch-tokens", str(self.config.train.micro_batch_tokens)]
        if predict_rollouts is not None:
            args += ["--predict-rollouts", str(predict_rollouts)]
        if self.config.model.compile is None:
            args.append("--disable-compile")
        if self.config.model.ac is None:
            args.append("--disable-ac")
        if self.config.model.ac_offloading is None:
            args.append("--disable-ac-offloading")
        if not self.config.model.reshard_after_forward:
            args.append("--disable-reshard-after-forward")
        if not self.config.model.optim_cpu_offload:
            args.append("--disable-optim-cpu-offload")
        if self.config.model.trust_remote_code:
            args.append("--trust-remote-code")
        return args

    def _value_nodes(self) -> list[str]:
        if self.config.train.num_nodes == 1:
            return [self._train_node()]
        return self._slurm_nodes()[: self.config.train.num_nodes]

    def _rdzv_port(self) -> int:
        job_id = os.environ.get("SLURM_JOB_ID")
        return 20000 + (int(job_id) % 20000 if job_id and job_id.isdigit() else int(time.time()) % 20000)

    def _run_value_torchrun(self, args: list[str], log_stem: str) -> None:
        nodes = self._value_nodes()
        common = [
            "uv",
            "run",
            "torchrun",
            "--nproc-per-node",
            str(self.config.inference.gpus_per_node),
        ]
        if len(nodes) == 1:
            cmd = [
                *common,
                "--standalone",
                "-m",
                "experiments.static_value_diagnostics.train_static_value",
                *args,
            ]
            self._run_remote(nodes[0], cmd, self.log_dir / f"{log_stem}.log", cuda_visible_devices=self._all_gpus())
            return

        rdzv_id = f"static-value-{os.environ.get('SLURM_JOB_ID', int(time.time()))}-{log_stem}"
        rdzv_endpoint = f"{nodes[0]}:{self._rdzv_port()}"
        procs = []
        for node_rank, node in enumerate(nodes):
            cmd = [
                *common,
                "--nnodes",
                str(len(nodes)),
                "--node-rank",
                str(node_rank),
                "--rdzv-backend",
                "c10d",
                "--rdzv-endpoint",
                rdzv_endpoint,
                "--rdzv-id",
                rdzv_id,
                "-m",
                "experiments.static_value_diagnostics.train_static_value",
                *args,
            ]
            log_name = f"{log_stem}.log" if node_rank == 0 else f"{log_stem}_node{node_rank}.log"
            procs.append(self._popen_remote(node, cmd, self.log_dir / log_name, cuda_visible_devices=self._all_gpus()))
        self._wait_all(procs)

    def _wandb_args(self, suffix: str) -> list[str]:
        if self.config.wandb is None:
            return []
        return ["--wandb-project", self.config.wandb.project, "--wandb-run-name", f"{self.run_name}-{suffix}"]

    def _merge_rollouts(self, output_path: Path, expected_rows: int, inputs: list[Path]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        with output_path.open("w", encoding="utf-8") as out:
            for path in inputs:
                if not path.exists() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"missing rollout shard: {path}")
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                        rows += 1
        if rows != expected_rows:
            raise RuntimeError(f"merged {output_path} has {rows} rows, expected {expected_rows}")
        self.logger.info(f"Merged {rows} rollout rows into {output_path}")

    def _wait_for_inference(self, node: str, log_path: Path) -> None:
        deadline = time.monotonic() + self.config.inference.ready_timeout_seconds
        url = f"http://{node}:{self.config.inference.port}/v1/models"
        headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"}
        while time.monotonic() < deadline:
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=5) as response:
                    if 200 <= response.status < 300:
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(10)
        tail = self._tail(log_path)
        raise RuntimeError(f"inference server on {node} did not become ready; log tail:\n{tail}")

    def _wait_all(self, procs: list[Popen]) -> None:
        first_error: tuple[Popen, int] | None = None
        for proc in procs:
            code = proc.wait()
            if code != 0 and first_error is None:
                first_error = (proc, code)
        if first_error is not None:
            raise RuntimeError(f"process failed with exit code {first_error[1]}")

    def _popen_remote(
        self,
        node: str,
        cmd: list[str],
        log_path: Path,
        *,
        cuda_visible_devices: str | None = None,
    ) -> Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        remote = self._remote_command(cmd, cuda_visible_devices=cuda_visible_devices)
        log_file = log_path.open("w", encoding="utf-8")
        stdin = subprocess.PIPE if os.environ.get("WANDB_API_KEY") else None
        proc = Popen(["ssh", node, remote], stdin=stdin, stdout=log_file, stderr=log_file, text=True)
        if proc.stdin is not None:
            proc.stdin.write(os.environ["WANDB_API_KEY"] + "\n")
            proc.stdin.close()
        self.child_procs.append(proc)
        return proc

    def _run_remote(
        self,
        node: str,
        cmd: list[str],
        log_path: Path,
        *,
        cuda_visible_devices: str | None = None,
    ) -> None:
        proc = self._popen_remote(node, cmd, log_path, cuda_visible_devices=cuda_visible_devices)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"remote command on {node} failed with exit code {code}; log tail:\n{self._tail(log_path)}")

    def _run_local(self, cmd: list[str], log_path: Path, *, env: dict[str, str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(cmd, cwd=self.config.repo_dir, env=env, stdout=log_file, stderr=log_file)
        if proc.returncode != 0:
            raise RuntimeError(f"local command failed with exit code {proc.returncode}; log tail:\n{self._tail(log_path)}")

    def _remote_command(self, cmd: list[str], *, cuda_visible_devices: str | None) -> str:
        exports = self._remote_env(cuda_visible_devices)
        read_wandb = (
            "IFS= read -r WANDB_API_KEY; export WANDB_API_KEY"
            if os.environ.get("WANDB_API_KEY")
            else "true"
        )
        source_env = (
            "for env_file in \"$HOME/kv-eviction/.env\" \"$HOME/compaction-rl/.env\" \"$HOME/.env\"; do "
            "if [ -z \"${WANDB_API_KEY:-}\" ] && [ -f \"$env_file\" ]; then set -a; source \"$env_file\"; set +a; fi; "
            "done"
        )
        return " && ".join(
            [
                "set -euo pipefail",
                "module unload darshan >/dev/null 2>&1 || true",
                exports,
                read_wandb,
                source_env,
                f"cd {shlex.quote(str(self.config.repo_dir))}",
                shlex.join(cmd),
            ]
        )

    def _remote_env(self, cuda_visible_devices: str | None) -> str:
        env = self._base_env()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        return f"export {assignments}" if assignments else "true"

    def _local_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self._base_env())
        if os.environ.get("WANDB_API_KEY"):
            env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
        return env

    def _base_env(self) -> dict[str, str]:
        home = os.environ.get("HOME", str(Path.home()))
        path = os.environ.get("PATH", "")
        home_bin = str(Path(home) / ".local" / "bin")
        if home_bin not in path.split(":"):
            path = f"{home_bin}:{path}" if path else home_bin
        env = {
            "HOME": home,
            "PATH": path,
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path(home) / ".cache" / "uv")),
            "VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "EMPTY"),
            "PYTHONPATH": str(self.config.repo_dir),
            "PYTHONUNBUFFERED": "1",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600"),
            **self.config.env_vars,
        }
        if self.config.wandb is not None:
            if self.config.wandb.entity is not None:
                env["WANDB_ENTITY"] = self.config.wandb.entity
            if self.config.wandb.offline:
                env["WANDB_MODE"] = "offline"
        return env

    def _load_wandb_env(self) -> None:
        if os.environ.get("WANDB_API_KEY"):
            return
        home = Path(os.environ.get("HOME", str(Path.home())))
        for path in (home / "kv-eviction/.env", home / "compaction-rl/.env", home / ".env"):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.removeprefix("export ").split("=", 1)
                if key.strip() == "WANDB_API_KEY":
                    os.environ["WANDB_API_KEY"] = value.strip().strip("'\"")
                    return

    def _all_gpus(self) -> str:
        return ",".join(str(i) for i in range(self.config.inference.gpus_per_node))

    def _cleanup_child_processes(self) -> None:
        for proc in self.child_procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.child_procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        self.child_procs.clear()

    @staticmethod
    def _tail(path: Path, limit: int = 120) -> str:
        if not path.exists():
            return f"<missing log {path}>"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-limit:])


def static_value(config: StaticValueConfig) -> None:
    runner = StaticValueRunner(config)
    runner.run()


def main() -> None:
    set_proc_title("StaticValue")
    try:
        static_value(cli(StaticValueConfig))
    except Exception as exc:
        setup_logger("info").error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
