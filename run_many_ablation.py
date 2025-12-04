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


# changed to new base overrides
BASE_OVERRIDES = [
    # Task
    "experiment.tasks=['training_schedule_matrix']",

    # Pretrained checkpoint
    'load="outputs/2025-11-07/05-10-26/checkpoints/epoch=13-step=1400.ckpt"',

    # Env size (fixed in all ablations)
    "algorithm.diffusion.sampling_timesteps=10",
    "algorithm.chunk_size=-1",
    "algorithm.schedule_matrix.build=True",
    "algorithm.schedule_matrix.max_roller=-1",

    # Action space default (some ablations override actions)
    "algorithm.schedule_matrix.actions=3",
    "algorithm.schedule_matrix.positive_only=False",

    # RL horizon base
    "algorithm.schedule_matrix.rollout_multiple=1",
    "algorithm.schedule_matrix.use_gae=False",   # will override to True in GAE ablations

    # Training config (fixed)
    "experiment.training_schedule_matrix.epochs=3",
    "experiment.training_schedule_matrix.train_batch_size=512",
]

# Where to store ablation metadata
ABLATION_CSV = Path("ablation_runs_many.csv")

# GPU to use by default
DEFAULT_CUDA_DEVICE = "7"


# ==========================================
# 2. Build EXPERIMENTS for Ablations 0–3
# ==========================================

EXPERIMENTS: List[Dict[str, Any]] = []

# -------------
# Ablation 0:
# REINFORCE reward function ablation
# -------------
# Fixed (in addition to BASE_OVERRIDES):
#   actions=3, positive_only=False   (already in BASE_OVERRIDES)
#   use_gae=False                    (already in BASE_OVERRIDES)
#   rollout_multiple=1               (already in BASE_OVERRIDES)
#
# Vary:
#   entropy_beta in [0.05, 0.1]
#   step_reward in [True, False]
#   difference_step_reward in [True, False]  (if step_reward=False, must be False)
#   denoise_reward in [True, False]
#   denoise_bonus in [0.01, 0.1, 0.25]  (if denoise_reward=False, should be 0.0)

ab0_entropy_betas = [0.05, 0.10]
ab0_step_reward_opts = [True, False]
ab0_diff_opts = [True, False]
ab0_denoise_opts = [True, False]
ab0_denoise_bonuses = [0.01, 0.1, 0.25]

for beta in ab0_entropy_betas:
    for step_reward in ab0_step_reward_opts:
        for diff in ab0_diff_opts:
            # Enforce: if step_reward is False, diff must be False
            if not step_reward and diff:
                continue

            for denoise_reward in ab0_denoise_opts:
                if denoise_reward:
                    bonuses = ab0_denoise_bonuses
                else:
                    bonuses = [0.0]

                for db in bonuses:
                    name = (
                        f"ab0_REINFORCE_"
                        f"beta{beta:.2f}_"
                        f"step{int(step_reward)}_"
                        f"diff{int(diff)}_"
                        f"denoise{int(denoise_reward)}_"
                        f"db{db:g}"
                    )

                    overrides = [
                        f"algorithm.schedule_matrix.entropy_beta={beta}",
                        f"algorithm.schedule_matrix.step_reward={str(step_reward)}",
                        f"algorithm.schedule_matrix.difference_step_reward={str(diff)}",
                        f"algorithm.schedule_matrix.denoise_reward={str(denoise_reward)}",
                        f"algorithm.schedule_matrix.denoise_bonus={db}",
                        # Explicitly keep no GAE
                        "algorithm.schedule_matrix.use_gae=False",
                    ]
                    EXPERIMENTS.append(
                        {
                            "name": name,
                            "overrides": overrides,
                        }
                    )

# -------------
# Ablation 1:
# REINFORCE, actions=3, positive_only=False
# -------------
# Fixed:
#   step_reward=True
#   difference_step_reward=True
#   denoise_reward=True
#   use_gae=False
#
# Vary:
#   entropy_beta in [0.05, 0.1]
#   denoise_bonus in [0.01, 0.1, 0.25]

ab1_entropy_betas = [0.05, 0.10]
ab1_denoise_bonuses = [0.01, 0.1, 0.25]

for beta in ab1_entropy_betas:
    for db in ab1_denoise_bonuses:
        name = f"ab1_REINFORCE_a3_beta{beta:.2f}_db{db:g}"
        overrides = [
            "algorithm.schedule_matrix.use_gae=False",
            "algorithm.schedule_matrix.step_reward=True",
            "algorithm.schedule_matrix.difference_step_reward=True",
            "algorithm.schedule_matrix.denoise_reward=True",
            f"algorithm.schedule_matrix.entropy_beta={beta}",
            f"algorithm.schedule_matrix.denoise_bonus={db}",
        ]
        EXPERIMENTS.append(
            {
                "name": name,
                "overrides": overrides,
            }
        )

# -------------
# Ablation 2:
# GAE, actions=3, positive_only=False
# -------------
# Fixed:
#   step_reward=True
#   difference_step_reward=True
#   denoise_reward=True
#   use_gae=True
#   rollout_multiple=1  (already in base)
#
# Vary:
#   entropy_beta in [0.01, 0.1]
#   denoise_bonus in [0.01, 0.1, 0.25]
#   gamma in [0.95, 0.99]
#   lam in [0.9]
#   value_coef in [0.25, 0.5]

ab2_entropy_betas = [0.01, 0.10]
ab2_denoise_bonuses = [0.01, 0.1, 0.25]
ab2_gammas = [0.95, 0.99]
ab2_lams = [0.9]
ab2_value_coefs = [0.25, 0.5]

for beta in ab2_entropy_betas:
    for db in ab2_denoise_bonuses:
        for gamma in ab2_gammas:
            for lam in ab2_lams:
                for vcoef in ab2_value_coefs:
                    name = (
                        f"ab2_GAE_a3_beta{beta:.2f}_db{db:g}_"
                        f"gamma{gamma:.2f}_lam{lam:.2f}_v{vcoef:.2f}"
                    )
                    overrides = [
                        "algorithm.schedule_matrix.use_gae=True",
                        "algorithm.schedule_matrix.step_reward=True",
                        "algorithm.schedule_matrix.difference_step_reward=True",
                        "algorithm.schedule_matrix.denoise_reward=True",
                        f"algorithm.schedule_matrix.entropy_beta={beta}",
                        f"algorithm.schedule_matrix.denoise_bonus={db}",
                        f"algorithm.schedule_matrix.gamma={gamma}",
                        f"algorithm.schedule_matrix.lam={lam}",
                        f"algorithm.schedule_matrix.value_coef={vcoef}",
                    ]
                    EXPERIMENTS.append(
                        {
                            "name": name,
                            "overrides": overrides,
                        }
                    )

# -------------
# Ablation 3:
# Action Space ablation using best GAE result
# -------------
# Fixed:
#   positive_only=False
#   use_gae=True
#   step_reward=True
#   difference_step_reward=True
#   denoise_reward=True
#   entropy_beta=0.05
#   denoise_bonus=0.1
#   gamma=0.95
#   lam=0.9
#   value_coef=0.5
#
# Vary:
#   actions in [3, 5, 7]

ab3_actions = [3, 5, 7]

for acts in ab3_actions:
    name = f"ab3_GAE_actions{acts}_best"
    overrides = [
        f"algorithm.schedule_matrix.actions={acts}",
        "algorithm.schedule_matrix.positive_only=False",
        "algorithm.schedule_matrix.use_gae=True",
        "algorithm.schedule_matrix.step_reward=True",
        "algorithm.schedule_matrix.difference_step_reward=True",
        "algorithm.schedule_matrix.denoise_reward=True",
        "algorithm.schedule_matrix.entropy_beta=0.05",
        "algorithm.schedule_matrix.denoise_bonus=0.1",
        "algorithm.schedule_matrix.gamma=0.95",
        "algorithm.schedule_matrix.lam=0.9",
        "algorithm.schedule_matrix.value_coef=0.5",
    ]
    EXPERIMENTS.append(
        {
            "name": name,
            "overrides": overrides,
        }
    )



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
