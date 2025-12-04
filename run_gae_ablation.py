import os 
import subprocess
import datetime
import json
import csv
from pathlib import Path
from typing import Dict, List, Any

# =======================
# 1. Common config pieces
# =======================

# Base command components that are shared across experiments.
BASE_CMD = [
    "python",
    "-m",
    "main",
    "+name=testing",
    "dataset=ts_exchange",
    "algorithm=df_prediction",
    "experiment=exp_prediction",
    'wandb.mode=disabled',
]

BASE_OVERRIDES = [
    # Task
    "experiment.tasks=['training_schedule_matrix']",

    # Pretrained checkpoint
    'load="outputs/2025-11-07/05-10-26/checkpoints/epoch=13-step=1400.ckpt"',

    # Diffusion / environment sizing
    "algorithm.diffusion.sampling_timesteps=10",
    "algorithm.chunk_size=10",
    "algorithm.schedule_matrix.build=True",
    "algorithm.schedule_matrix.actions=5", 
    "algorithm.schedule_matrix.positive_only=False",
    "algorithm.schedule_matrix.max_roller=10",
    "algorithm.schedule_matrix.rollout_multiple=1",

    # Reward: MSE-based, stepwise + differenced + denoise bonus
    "algorithm.schedule_matrix.step_reward=True",
    "algorithm.schedule_matrix.difference_step_reward=True",
    "algorithm.schedule_matrix.denoise_reward=True",
    "algorithm.schedule_matrix.denoise_bonus=0.1",

    # I'll set GAE to be on for default
    "algorithm.schedule_matrix.use_gae=True",

    # Entropy regularization 
    "algorithm.schedule_matrix.entropy_beta=0.05",

    # Training config
    "experiment.training_schedule_matrix.epochs=5",
    "experiment.training_schedule_matrix.train_batch_size=512",
]



# Where to store ablation metadata
ABLATION_CSV = Path("ablation_runs.csv")

# GPU to use by default
DEFAULT_CUDA_DEVICE = "7"


# =====================================
# 2. Define your GAE ablation experiments
# =====================================

EXPERIMENTS = [
    # =========================
    # Stage A: GAE lambda sweep
    # =========================
    {
        "name": "lam0.80_v0.5_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.80",
            "algorithm.schedule_matrix.value_coef=0.5",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },
    {
        "name": "lam0.90_v0.5_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.90",
            "algorithm.schedule_matrix.value_coef=0.5",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },
    {
        "name": "lam0.95_v0.5_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.95",
            "algorithm.schedule_matrix.value_coef=0.5",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },
    {
        "name": "lam0.99_v0.5_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.99",
            "algorithm.schedule_matrix.value_coef=0.5",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },

    # ==============================
    # Stage B: value loss coefficient
    # (around lambda = 0.95)
    # ==============================
    {
        "name": "lam0.95_v0.10_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.95",
            "algorithm.schedule_matrix.value_coef=0.10",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },
    {
        "name": "lam0.95_v0.50_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.95",
            "algorithm.schedule_matrix.value_coef=0.50",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },
    {
        "name": "lam0.95_v1.00_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.95",
            "algorithm.schedule_matrix.value_coef=1.00",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },

    # ==========================
    # Stage C: entropy beta sweep
    # (lambda = 0.95, v = 0.5)
    # ==========================
    {
        "name": "lam0.95_v0.5_beta0.01",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.95",
            "algorithm.schedule_matrix.value_coef=0.5",
            "algorithm.schedule_matrix.entropy_beta=0.01",
        ],
    },
    {
        "name": "lam0.95_v0.5_beta0.05",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.95",
            "algorithm.schedule_matrix.value_coef=0.5",
            "algorithm.schedule_matrix.entropy_beta=0.05",
        ],
    },
    {
        "name": "lam0.95_v0.5_beta0.10",
        "cuda": "7",
        "overrides": [
            "algorithm.schedule_matrix.lam=0.95",
            "algorithm.schedule_matrix.value_coef=0.5",
            "algorithm.schedule_matrix.entropy_beta=0.10",
        ],
    },
]




# ============================
# 3. Helper functions
# ============================

def ensure_csv_header(path: Path):
    """Create the CSV with header if it doesn't exist yet."""
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_dir",
            "name",
            "timestamp",
            "cuda_device",
            "command",
            "overrides_json",
        ])


def build_run_dir(base_dir: Path = Path("outputs")) -> Path:
    """Create a timestamped run directory like outputs/YYYY-MM-DD/HH-MM-SS."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    run_dir = base_dir / date_str / time_str
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_command(run_dir: Path, exp: Dict[str, Any]) -> List[str]:
    """
    Build the full `python -m main ...` command for a given experiment.
    We override hydra.run.dir so that images and logs share this directory.
    """
    cmd = list(BASE_CMD)
    cmd.extend(BASE_OVERRIDES)

    # Experiment-specific overrides (GAE hyperparams, entropy, etc.)
    exp_overrides = exp.get("overrides", [])
    cmd.extend(exp_overrides)

    # Override Hydra run dir so it matches our timestamped directory
    hydra_override = f"hydra.run.dir={run_dir.as_posix()}"
    cmd.append(hydra_override)

    return cmd


def run_experiment(exp: Dict[str, Any]):
    """Run a single experiment, save stdout/stderr, and log metadata."""
    # 1. Make deterministic run dir
    run_dir = build_run_dir()

    # 2. Build command
    cmd = build_command(run_dir, exp)
    cmd_str = " ".join(cmd)

    # 3. Prepare logging
    log_path = run_dir / "run.log"
    print(f"\n=== Running experiment: {exp['name']} ===")
    print(f"Run dir: {run_dir}")
    print(f"Command: {cmd_str}")
    print(f"Logging to: {log_path}")

    # 4. Set CUDA device
    cuda_device = exp.get("cuda", DEFAULT_CUDA_DEVICE)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # 5. Run the process and tee stdout/stderr to file
    with log_path.open("w") as log_file:
        process = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    # 6. Append metadata to CSV
    ensure_csv_header(ABLATION_CSV)
    with ABLATION_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_dir.as_posix(),
            exp["name"],
            datetime.datetime.now().isoformat(),
            cuda_device,
            cmd_str,
            json.dumps(exp.get("overrides", [])),
        ])

    print(f"=== Finished experiment: {exp['name']} (return code {process.returncode}) ===")


# ============================
# 4. Main
# ============================

def main():
    for exp in EXPERIMENTS:
        run_experiment(exp)


if __name__ == "__main__":
    main()
