"""
Generate controlled and uncontrolled trajectories for KoopmanRL environments.

Hyperparameters for SAKC and SKVI are auto-loaded from:
    configurations/<algo>_<env_slug>_hparams.json

Any flag passed explicitly on the CLI overrides the config file value.

Example usage:

    # SAKC on FluidFlow — seed and num-trajectories come from config
    python -m koopmanrl_utils.movies.generate_trajectories \
        --env_id FluidFlow-v0 \
        --algo sakc \
        --chkpt_timestamp 1732368170 \
        --chkpt_step 50000 \
        --emit_dat

    # LQR baseline on DoubleWell — no config, all defaults
    python -m koopmanrl_utils.movies.generate_trajectories \
        --env_id DoubleWell-v0 \
        --algo lqr \
        --baseline_algo zero \
        --num_trajectories 1
"""

import json
import os
import time
import warnings
from typing import Optional

import gym
import numpy as np
import torch
from tap import Tap

from koopmanrl.environments import DoubleWell, FluidFlow, Lorenz  # noqa: F401 — registers gym envs
from koopmanrl.utils import create_folder
from koopmanrl_utils.movies.algo_policies import LQR, SAKC, SKVI
from koopmanrl_utils.movies.default_policies import RandomPolicy, ZeroPolicy
from koopmanrl_utils.movies.env_enum import EnvEnum
from koopmanrl_utils.movies.generator import Generator

# ---------------------------------------------------------------------------
# Environment slug mapping (LinearSystem excluded from plotting targets)
# ---------------------------------------------------------------------------

ENV_SLUG: dict[str, str] = {
    "FluidFlow-v0": "fluid_flow",
    "Lorenz-v0": "lorenz",
    "DoubleWell-v0": "double_well",
}

SUPPORTED_ENVS = set(ENV_SLUG.keys())


# ---------------------------------------------------------------------------
# Argument class
# ---------------------------------------------------------------------------


class Args(Tap):
    env_id: str = "FluidFlow-v0"
    """Gym environment ID. One of: FluidFlow-v0, Lorenz-v0, DoubleWell-v0."""

    seed: Optional[int] = None
    """RNG seed. If omitted, loaded from config file; falls back to 123."""

    torch_deterministic: bool = True
    """Set torch.backends.cudnn.deterministic."""

    cuda: bool = True
    """Enable CUDA if available."""

    num_trajectories: Optional[int] = None
    """Number of trajectories. If omitted, loaded from config (num-paths)."""

    num_steps: Optional[int] = None
    """Steps per trajectory. If omitted, loaded from config (num-steps-per-path)."""

    # Algorithm selection
    algo: str = "sakc"
    """Main policy algorithm: zero | random | lqr | skvi | sakc."""

    baseline_algo: str = "zero"
    """Baseline policy algorithm: zero | random | lqr."""

    # RL evaluation hyperparameters (not stored in config files)
    gamma: float = 0.99
    """Discount factor gamma."""

    alpha: float = 1.0
    """Entropy regularization coefficient."""

    num_actions: int = 101
    """Discrete action grid size for SKVI."""

    # Checkpoint arguments (required for skvi / sakc)
    chkpt_timestamp: Optional[int] = None
    """Unix timestamp of the training run whose checkpoint to load."""

    chkpt_step: Optional[int] = None
    """Step number for SAKC checkpoint."""

    chkpt_epoch: Optional[int] = None
    """Epoch number for SKVI checkpoint."""

    koopman_model_name: Optional[str] = None
    """Saved Koopman tensor model name (required for SKVI)."""

    # Config override
    config_file: Optional[str] = None
    """Explicit path to hparams JSON. Auto-resolved from algo + env-id if omitted."""

    # Output
    output_dir: str = "video_frames"
    """Root directory for output."""

    emit_dat: bool = False
    """Also write trajectory data as .dat files for TikZ ingestion."""


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

_ALGO_HAS_CONFIG = {"sakc", "skvi"}


def resolve_config_path(algo: str, env_id: str) -> str:
    slug = ENV_SLUG[env_id]
    return os.path.join("configurations", f"{algo}_{slug}_hparams.json")


def load_and_apply_config(args: Args) -> Args:
    """
    Load the hparams JSON for the chosen algo/env and fill in any Args field
    that was not explicitly set on the CLI (i.e. still None).
    Skips config loading for zero/random/lqr — no config files exist for them.
    """
    if args.algo not in _ALGO_HAS_CONFIG:
        return args

    config_path = args.config_file or resolve_config_path(args.algo, args.env_id)

    if not os.path.exists(config_path):
        warnings.warn(
            f"Config file not found at '{config_path}'; using CLI defaults only.",
            stacklevel=2,
        )
        return args

    with open(config_path) as f:
        cfg = json.load(f)

    if args.seed is None:
        args.seed = cfg.get("seed", 123)
    if args.num_trajectories is None:
        args.num_trajectories = cfg.get("num-paths", 1)
    if args.num_steps is None:
        args.num_steps = cfg.get("num-steps-per-path", None)

    print(f"Loaded config from '{config_path}'")
    return args


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------


def build_policy(algo: str, args: Args, envs, device):
    if algo == "zero":
        return ZeroPolicy(is_2d=True, name="Zero Policy")
    if algo == "random":
        return RandomPolicy(envs.envs[0], name="Random Policy")
    if algo == "lqr":
        return LQR(args=args, envs=envs, name="LQR")
    if algo == "skvi":
        assert args.chkpt_timestamp and args.chkpt_epoch and args.koopman_model_name, (
            "SKVI requires --chkpt-timestamp, --chkpt-epoch, and --koopman-model-name"
        )
        return SKVI(
            args=args,
            envs=envs,
            saved_koopman_model_name=args.koopman_model_name,
            trained_model_start_timestamp=args.chkpt_timestamp,
            chkpt_epoch_number=args.chkpt_epoch,
            device=device,
            name="SKVI",
        )
    if algo == "sakc":
        assert args.chkpt_timestamp and args.chkpt_step, (
            "SAKC requires --chkpt-timestamp and --chkpt-step"
        )
        return SAKC(
            args=args,
            envs=envs,
            is_value_based=False,
            is_koopman=True,
            chkpt_timestamp=args.chkpt_timestamp,
            chkpt_step_number=args.chkpt_step,
            device=device,
            name="SAKC",
        )
    raise ValueError(f"Unknown algorithm '{algo}'. Choose from: zero, random, lqr, skvi, sakc.")


# ---------------------------------------------------------------------------
# .dat export
# ---------------------------------------------------------------------------


def write_dat(path: str, data: np.ndarray, col_names: list[str]) -> None:
    """
    Write *data* (shape: steps × cols) to a tab-separated .dat file with a
    #-prefixed header row that PGFPlots can consume with \\addplot table.
    """
    header = "# " + "\t".join(col_names)
    np.savetxt(path, data, fmt="%.6g", delimiter="\t", header=header, comments="")


def export_dat(folder: str, prefix: str, trajectories: np.ndarray, actions: np.ndarray, costs: np.ndarray) -> None:
    """
    Write trajectory, action, and cost arrays for a single policy as .dat files.
    Each file contains all trajectories stacked with a blank separator line between them.
    """
    state_dim = trajectories.shape[2]
    state_cols = [f"x{i}" for i in range(state_dim)]
    action_dim = actions.shape[2]
    action_cols = [f"a{i}" for i in range(action_dim)]

    traj_rows, act_rows, cost_rows = [], [], []

    for traj_idx in range(trajectories.shape[0]):
        steps = trajectories.shape[1]
        step_col = np.arange(steps).reshape(-1, 1)

        traj_data = np.hstack([step_col, trajectories[traj_idx]])
        act_data = np.hstack([step_col, actions[traj_idx]])
        cost_data = np.hstack([step_col, costs[traj_idx].reshape(-1, 1)])

        traj_rows.append(traj_data)
        act_rows.append(act_data)
        cost_rows.append(cost_data)

    write_dat(
        os.path.join(folder, f"{prefix}_trajectories.dat"),
        np.vstack(traj_rows),
        ["step"] + state_cols,
    )
    write_dat(
        os.path.join(folder, f"{prefix}_actions.dat"),
        np.vstack(act_rows),
        ["step"] + action_cols,
    )
    write_dat(
        os.path.join(folder, f"{prefix}_costs.dat"),
        np.vstack(cost_rows),
        ["step", "cost"],
    )


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def reset_seed(seed: int, torch_deterministic: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = Args().parse_args()

    # Validate environment
    if args.env_id not in SUPPORTED_ENVS:
        raise ValueError(
            f"Environment '{args.env_id}' is not a supported plotting target. "
            f"Choose from: {sorted(SUPPORTED_ENVS)}"
        )

    # Load config and fill in None fields
    args = load_and_apply_config(args)

    # Apply remaining defaults for fields that config didn't supply either
    if args.seed is None:
        args.seed = 123
    if args.num_trajectories is None:
        args.num_trajectories = 1

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    reset_seed(args.seed, args.torch_deterministic)

    # Build vectorised environment
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])

    # Build policies
    main_policy = build_policy(args.algo, args, envs, device)
    baseline_policy = build_policy(args.baseline_algo, args, envs, device)
    zero_policy = ZeroPolicy(is_2d=True, name="Zero Policy")

    # Generate trajectories
    zero_gen = Generator(args, envs, zero_policy)
    main_gen = Generator(args, envs, main_policy)
    baseline_gen = Generator(args, envs, baseline_policy)

    reset_seed(args.seed, args.torch_deterministic)
    zero_traj, zero_act, zero_cost = zero_gen.generate_trajectories(
        args.num_trajectories, args.num_steps
    )
    reset_seed(args.seed, args.torch_deterministic)
    main_traj, main_act, main_cost = main_gen.generate_trajectories(
        args.num_trajectories, args.num_steps
    )
    reset_seed(args.seed, args.torch_deterministic)
    baseline_traj, baseline_act, baseline_cost = baseline_gen.generate_trajectories(
        args.num_trajectories, args.num_steps
    )

    print("Completed generating trajectories.")

    # Verify shared initial conditions
    assert np.array_equal(zero_traj[0, 0], main_traj[0, 0]) and np.array_equal(
        main_traj[0, 0], baseline_traj[0, 0]
    ), "Trajectories have different initial conditions — check your RNG seed."

    # Prepare output folder
    curr_time = int(time.time())
    output_folder = os.path.join(args.output_dir, f"{args.env_id}_{curr_time}")
    create_folder(output_folder)

    # Save .npy files
    np.save(os.path.join(output_folder, "zero_policy_trajectories.npy"), zero_traj)
    np.save(os.path.join(output_folder, "zero_policy_actions.npy"), zero_act)
    np.save(os.path.join(output_folder, "zero_policy_costs.npy"), zero_cost)

    np.save(os.path.join(output_folder, "main_policy_trajectories.npy"), main_traj)
    np.save(os.path.join(output_folder, "main_policy_actions.npy"), main_act)
    np.save(os.path.join(output_folder, "main_policy_costs.npy"), main_cost)

    np.save(os.path.join(output_folder, "baseline_policy_trajectories.npy"), baseline_traj)
    np.save(os.path.join(output_folder, "baseline_policy_actions.npy"), baseline_act)
    np.save(os.path.join(output_folder, "baseline_policy_costs.npy"), baseline_cost)

    metadata = {
        "env_id": args.env_id,
        "main_policy_name": main_policy.name,
        "baseline_policy_name": baseline_policy.name,
        "zero_policy_name": zero_policy.name,
    }
    np.save(os.path.join(output_folder, "metadata.npy"), metadata, allow_pickle=True)

    print(f"Saved .npy files to '{output_folder}'")

    # Optionally write .dat files
    if args.emit_dat:
        export_dat(output_folder, "zero_policy", zero_traj, zero_act, zero_cost)
        export_dat(output_folder, "main_policy", main_traj, main_act, main_cost)
        export_dat(output_folder, "baseline_policy", baseline_traj, baseline_act, baseline_cost)
        print(f"Saved .dat files to '{output_folder}'")


if __name__ == "__main__":
    main()
