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
from koopmanrl_utils.movies.algo_policies import LQR, SAC, SAKC, SKVI
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
    """Main policy algorithm: zero | random | lqr | skvi | sakc | sac."""

    baseline_algo: str = "zero"
    """Baseline policy algorithm: zero | random | lqr."""

    # RL evaluation hyperparameters (not stored in config files)
    gamma: float = 0.99
    """Discount factor gamma."""

    alpha: float = 1.0
    """Entropy regularization coefficient."""

    num_actions: int = 101
    """Discrete action grid size for SKVI."""

    # SKVI-specific Koopman tensor hyperparameters (loaded from config when algo=skvi)
    state_order: Optional[int] = None
    """Monomial order for the state observable. Loaded from SKVI config if omitted (default: 2)."""

    action_order: Optional[int] = None
    """Monomial order for the action observable. Loaded from SKVI config if omitted (default: 2)."""

    skvi_lr: Optional[float] = None
    """Learning rate for the SKVI value function. Loaded from SKVI config if omitted (default: 1e-3)."""

    skvi_koopman_num_paths: Optional[int] = None
    """Paths used to build the Koopman tensor for SKVI. Loaded from config (num-paths) if omitted.
    Kept separate from --num_trajectories so the Koopman tensor can be reconstructed with its
    original training parameters regardless of how many plotting trajectories are requested."""

    skvi_koopman_num_steps: Optional[int] = None
    """Steps-per-path used to build the Koopman tensor for SKVI. Loaded from config
    (num-steps-per-path) if omitted. Kept separate from --num_steps for the same reason."""

    regressor: str = "ols"
    """Regression method for Koopman tensor fitting: ols | sindy | rrr | ridge."""

    # Checkpoint arguments (required for skvi / sakc)
    chkpt_timestamp: Optional[str] = None
    """Folder suffix of the checkpoint directory. For SAKC: '{seed}_{unix_timestamp}' (e.g. '1_1768954004'). For SKVI: '{seed}_{unix_timestamp}' (e.g. '1_1768953873')."""

    chkpt_step: Optional[int] = None
    """Step number for SAKC checkpoint."""

    chkpt_epoch: Optional[int] = None
    """Epoch number for SKVI checkpoint."""

    # Config override
    config_file: Optional[str] = None
    """Explicit path to hparams JSON. Auto-resolved from algo + env-id if omitted."""

    # Output
    output_dir: str = "video_frames"
    """Root directory for output."""

    run_label: Optional[str] = None
    """Optional label used as the output sub-folder name instead of a unix timestamp.
    E.g. 'sakc_fluid_flow_5000steps' produces 'video_frames/FluidFlow-v0_sakc_fluid_flow_5000steps/'."""

    emit_dat: bool = False
    """Also write trajectory data as .dat files for TikZ ingestion."""

    vector_field_resolution: int = 20
    """Grid points per axis for the DoubleWell deterministic drift vector field (resolution² points total)."""


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
    # SKVI Koopman tensor hyperparameters
    if args.state_order is None:
        args.state_order = cfg.get("state-order", 2)
    if args.action_order is None:
        args.action_order = cfg.get("action-order", 2)
    if args.skvi_lr is None:
        args.skvi_lr = cfg.get("learning-rate", 1e-3)
    # Koopman tensor data parameters — intentionally separate from num_trajectories/num_steps
    if args.skvi_koopman_num_paths is None:
        args.skvi_koopman_num_paths = cfg.get("num-paths", 100)
    if args.skvi_koopman_num_steps is None:
        args.skvi_koopman_num_steps = cfg.get("num-steps-per-path", 300)

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
        assert args.chkpt_timestamp and args.chkpt_epoch, (
            "SKVI requires --chkpt_timestamp and --chkpt_epoch"
        )
        return SKVI(
            args=args,
            envs=envs,
            trained_model_start_timestamp=args.chkpt_timestamp,
            chkpt_epoch_number=args.chkpt_epoch,
            device=device,
            name="SKVI",
            koopman_num_paths=args.skvi_koopman_num_paths,
            koopman_num_steps=args.skvi_koopman_num_steps,
        )
    if algo == "sakc":
        assert args.chkpt_timestamp and args.chkpt_step, (
            "SAKC requires --chkpt_timestamp and --chkpt_step"
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
    if algo == "sac":
        assert args.chkpt_timestamp and args.chkpt_step, (
            "SAC requires --chkpt_timestamp and --chkpt_step"
        )
        return SAC(
            args=args,
            envs=envs,
            chkpt_timestamp=args.chkpt_timestamp,
            chkpt_step_number=args.chkpt_step,
            device=device,
            name="SAC",
        )
    raise ValueError(f"Unknown algorithm '{algo}'. Choose from: zero, random, lqr, skvi, sakc, sac.")


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
# DoubleWell vector field
# ---------------------------------------------------------------------------


def _double_well_drift_field(env, resolution: int):
    """
    Evaluate the DoubleWell deterministic drift (u=0, no diffusion) on a
    resolution×resolution grid that covers the full state_range.

    Returns flat 1-D arrays (x, y, dx, dy) where dx/dy are the continuous-time
    drift components  ẋ = 4x − 4x³,  ẏ = −2y.
    """
    xs = np.linspace(env.state_range[0], env.state_range[1], resolution)
    ys = np.linspace(env.state_range[0], env.state_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.stack([X.flatten(), Y.flatten()], axis=1)

    zero_action = np.zeros(env.action_dim)
    f_u = env.continuous_f(action=zero_action)
    drifts = np.array([f_u(0.0, pt) for pt in grid_pts])  # (N, 2)

    return X.flatten(), Y.flatten(), drifts[:, 0], drifts[:, 1]


def export_double_well_vector_field(env, folder: str, resolution: int, emit_dat: bool) -> None:
    """
    Compute and persist the DoubleWell deterministic drift vector field.

    Always writes:
        double_well_vector_field.npy  — shape (resolution², 6), columns
                                        [x, y, dx, dy, dxn, dyn] where
                                        (dx, dy) are raw continuous-time drift
                                        and (dxn, dyn) are L2-normalised.

    When emit_dat is True also writes:
        double_well_vector_field.dat  — same columns, tab-separated with
                                        a #-prefixed header for PGFPlots
                                        \\addplot table ingestion.
    """
    x, y, dx, dy = _double_well_drift_field(env, resolution)
    norms = np.sqrt(dx**2 + dy**2).clip(min=1e-10)
    dxn, dyn = dx / norms, dy / norms

    vf = np.stack([x, y, dx, dy, dxn, dyn], axis=1)
    npy_path = os.path.join(folder, "double_well_vector_field.npy")
    np.save(npy_path, vf)
    print(f"Saved double_well_vector_field.npy to '{folder}'")

    if emit_dat:
        dat_path = os.path.join(folder, "double_well_vector_field.dat")
        write_dat(dat_path, vf, ["x", "y", "dx", "dy", "dxn", "dyn"])
        print(f"Saved double_well_vector_field.dat to '{folder}'")


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
    folder_suffix = args.run_label if args.run_label is not None else str(int(time.time()))
    output_folder = os.path.join(args.output_dir, f"{args.env_id}_{folder_suffix}")
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

    # DoubleWell: emit deterministic drift vector field at the end of the run
    if args.env_id == "DoubleWell-v0":
        export_double_well_vector_field(
            envs.envs[0],
            output_folder,
            args.vector_field_resolution,
            args.emit_dat,
        )

    # Optionally write .dat files
    if args.emit_dat:
        export_dat(output_folder, "zero_policy", zero_traj, zero_act, zero_cost)
        export_dat(output_folder, "main_policy", main_traj, main_act, main_cost)
        export_dat(output_folder, "baseline_policy", baseline_traj, baseline_act, baseline_cost)
        print(f"Saved .dat files to '{output_folder}'")


if __name__ == "__main__":
    main()
