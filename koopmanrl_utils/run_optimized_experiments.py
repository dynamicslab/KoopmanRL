"""Run the experiments behind the episodic-return figures of the paper.

Every algorithm of the comparison is run on every benchmark, once per seed, for 50,000 environment steps:

    lqr     koopmanrl.linear_quadratic_regulator         linear quadratic regulator
    sac_q   koopmanrl.sac_continuous_action              soft actor-critic, Q-function form
    sac_v   koopmanrl.value_based_sac_continuous_action  soft actor-critic with a value network
    skvi    koopmanrl.soft_koopman_value_iteration       soft Koopman value iteration, tuned configuration
    sakc    koopmanrl.soft_actor_koopman_critic          soft actor Koopman-critic, tuned configuration

SKVI and SAKC take their hyperparameters from `configurations/<algorithm>_<benchmark>_hparams.json`; the three
baselines run with the defaults of their scripts. Each run writes a TensorBoard log with the return of every episode
(`charts/episodic_return`), which is the raw data of the figures. `koopmanrl_utils/EPISODIC_RETURNS.md` describes how
to turn these logs into the tables the figures are drawn from.

Seeds
    `SEEDS` lists the seeds of the SKVI and SAKC runs of the paper. Its baseline runs were made when the baseline
    scripts drew a seed below 1000 at random on every start; that seed is the third field of the name of each run
    folder. By default the baselines are run here with the seeds of `SEEDS`. To repeat baseline runs whose seeds are
    known, pass them with `--seeds` together with `--algorithms` and `--environments`.

Output
    The algorithms write relative to their working directory, which is `--output_dir` (default
    `episodic_returns_results/`, not tracked by git):

        <output_dir>/runs/<run name>/        TensorBoard logs of lqr, sac_q and sac_v
        <output_dir>/runs/SKVI/<run name>/   TensorBoard logs of skvi
        <output_dir>/runs/SAKC/<run name>/   TensorBoard logs of sakc
        <output_dir>/saved_models/           checkpoints
        <output_dir>/logs/                   console output, one file per run

    Use a directory that holds no other runs: the processing scripts take every run they find in a folder.

Usage (from the repository root):

    uv run -m koopmanrl_utils.run_optimized_experiments                      # everything, one run at a time
    uv run -m koopmanrl_utils.run_optimized_experiments --num_workers 8      # eight runs at a time
    uv run -m koopmanrl_utils.run_optimized_experiments --algorithms lqr sac_q --environments Lorenz-v0
    uv run -m koopmanrl_utils.run_optimized_experiments --dry_run            # print the commands only

The runs of one algorithm on one benchmark are made one after the other, and `--num_workers` runs several such
sequences at a time, with Ray. The largest run, SKVI on the fluid flow, takes about 3.5 GB of memory. A run that
fails does not stop the others; the failed runs are listed at the end, and their incomplete TensorBoard logs have to
be removed before the logs are processed.

Reproducibility
    Every run is seeded. In our checks (runs of a few thousand steps on the linear system), two runs of LQR, SAC (Q),
    SAC (V) or SKVI with the same seed, on the same machine and library versions, logged the same returns. Two such
    runs of SAKC usually did not: their returns agree until training starts and drift apart afterwards. The
    random-agent data and their dictionary features are the same bit for bit in both runs, but the least-squares
    solution that identifies the Koopman tensor differs from process to process in its last digits, with one
    numerical thread as with several, and the critic of SAKC is trained through that tensor. A rerun of SAKC
    therefore agrees with an earlier run in distribution, not run by run.

    The returns also depend on the number of numerical threads: runs of SAC (Q) with one seed repeated exactly with
    one thread and exactly with two, but the two settings gave different returns once training had started. This
    script leaves the number of threads to the environment (OMP_NUM_THREADS), so it has to be the same for two runs
    to agree. On another machine, floating-point rounding can change the returns of every algorithm.
"""

import os
import subprocess
import sys
from typing import Optional

from tap import Tap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configurations")

# benchmark: name of the benchmark in the configuration files
ENVIRONMENTS = {
    "LinearSystem-v0": "linear_system",
    "FluidFlow-v0": "fluid_flow",
    "Lorenz-v0": "lorenz",
    "DoubleWell-v0": "double_well",
}

# algorithm: module that runs it
ALGORITHMS = {
    "lqr": "koopmanrl.linear_quadratic_regulator",
    "sac_q": "koopmanrl.sac_continuous_action",
    "sac_v": "koopmanrl.value_based_sac_continuous_action",
    "skvi": "koopmanrl.soft_koopman_value_iteration",
    "sakc": "koopmanrl.soft_actor_koopman_critic",
}
TUNED = ("skvi", "sakc")  # algorithms with a configuration file per benchmark

# benchmark: seeds of the SKVI and SAKC runs of the paper
SEEDS = {
    "LinearSystem-v0": [
        4430,
        2738,
        9700,
        3478,
        8578,
        3602,
        1228,
        1749,
        6687,
        7659,
        3362,
        521,
        435,
        8229,
        762,
        7154,
        1045,
        4754,
        2936,
        868,
        2574,
        3169,
        4605,
        7,
    ],
    "FluidFlow-v0": [
        5412,
        3839,
        5062,
        3776,
        9127,
        3910,
        9604,
        5458,
        1745,
        5575,
        7601,
        2447,
        1584,
        3289,
        8699,
        5437,
        3771,
        1065,
        4787,
        9253,
        7844,
        7922,
        4050,
        6517,
        6597,
    ],
    "Lorenz-v0": [
        8801,
        8207,
        7115,
        9370,
        6503,
        5442,
        1053,
        7904,
        5611,
        1635,
        2064,
        41,
        7644,
        1427,
        8573,
        1779,
        9355,
        169,
        3786,
        6957,
        4788,
        5900,
        3158,
        8953,
        6504,
    ],
    "DoubleWell-v0": [
        5991,
        5243,
        5581,
        726,
        2549,
        6408,
        6146,
        159,
        6382,
        1078,
        500,
        4000,
        9761,
        8178,
        8623,
        8866,
        9135,
        3642,
        9150,
        1106,
        5549,
        5202,
        6617,
        8294,
        469,
    ],
}


class ArgumentParser(Tap):
    environments: list[str] = list(ENVIRONMENTS)  # benchmarks to run
    algorithms: list[str] = list(ALGORITHMS)  # algorithms to run: lqr, sac_q, sac_v, skvi, sakc
    seeds: list[int] = []  # seeds to run on every benchmark (default: the seeds of SEEDS)
    total_timesteps: Optional[int] = None  # environment steps per run (default: 50,000, as in the paper)
    output_dir: str = "episodic_returns_results"  # working directory of the runs
    num_workers: int = 1  # number of runs made at the same time
    dry_run: bool = False  # print the commands without running them


def config_file(algorithm: str, env_id: str) -> str:
    """Path of the tuned configuration of SKVI or SAKC on a benchmark."""
    return os.path.join(CONFIG_DIR, f"{algorithm}_{ENVIRONMENTS[env_id]}_hparams.json")


def command(algorithm: str, env_id: str, seed: int, total_timesteps: Optional[int] = None) -> list[str]:
    """Command line of one run."""
    cmd = [sys.executable, "-m", ALGORITHMS[algorithm]]
    if algorithm in TUNED:
        cmd.append(f"--config_file={config_file(algorithm, env_id)}")
    cmd += [f"--env_id={env_id}", f"--seed={seed}"]
    if total_timesteps is not None:
        cmd.append(f"--total_timesteps={total_timesteps}")
    return cmd


def build_lanes(
    environments: list[str],
    algorithms: list[str],
    seeds: Optional[list[int]] = None,
    total_timesteps: Optional[int] = None,
) -> list[list[tuple[str, list[str]]]]:
    """Runs grouped into lanes, one per algorithm and benchmark; a run is (name of its log file, command line).

    The runs of a lane are made one after the other. The checkpoint folder of a baseline run is named after the
    second in which the run starts, so two runs of one baseline on one benchmark must not start together.
    """
    lanes = []
    for env_id in environments:
        for algorithm in algorithms:
            lanes.append(
                [
                    (f"{algorithm}__{env_id}__{seed}", command(algorithm, env_id, seed, total_timesteps))
                    for seed in (seeds or SEEDS[env_id])
                ]
            )
    return lanes


def run_lane(lane: list[tuple[str, list[str]]], output_dir: str) -> list[str]:
    """Make the runs of a lane one after the other, in `output_dir`, and return the log files of those that failed."""
    failed = []
    for name, cmd in lane:
        log_file = os.path.join(output_dir, "logs", f"{name}.log")
        print(f"Running: {' '.join(cmd)} > {log_file}", flush=True)
        with open(log_file, "w") as f:
            if subprocess.run(cmd, cwd=output_dir, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
                failed.append(log_file)
    return failed


def run_lanes(lanes: list[list[tuple[str, list[str]]]], output_dir: str, num_workers: int = 1) -> None:
    """Run the lanes in `output_dir`, `num_workers` at a time, and raise at the end if a run failed."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    if num_workers <= 1:
        failed = [run_lane(lane, output_dir) for lane in lanes]
    else:
        import ray

        ray.init(num_cpus=num_workers, include_dashboard=False)
        try:
            remote_lane = ray.remote(num_cpus=1)(run_lane)
            failed = ray.get([remote_lane.remote(lane, output_dir) for lane in lanes])
        finally:
            ray.shutdown()

    failed = [log_file for lane in failed for log_file in lane]
    if failed:
        raise RuntimeError(
            f"{len(failed)} run(s) failed; their TensorBoard logs are incomplete and have to be removed before the"
            " logs are processed. Console output of the failed runs:\n" + "\n".join(failed)
        )


def check_choices(given: list[str], known, what: str) -> None:
    unknown = [name for name in given if name not in known]
    if unknown:
        raise ValueError(f"Unknown {what}: {', '.join(unknown)}. Choose from: {', '.join(known)}.")


def main() -> None:
    args = ArgumentParser().parse_args()
    check_choices(args.environments, ENVIRONMENTS, "environments")
    check_choices(args.algorithms, ALGORITHMS, "algorithms")

    lanes = build_lanes(args.environments, args.algorithms, args.seeds, args.total_timesteps)
    if args.dry_run:
        for lane in lanes:
            for _, cmd in lane:
                print(" ".join(cmd))
        return
    run_lanes(lanes, args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()
