"""Run the experiments behind the two ablation figures of the paper.

The default campaign is 1,440 runs of 50,000 environment steps (2 algorithms x 4 benchmarks x 36 grid points x
5 seeds) and takes several days of computing time. `--dry_run` prints the commands without running them, and
"Running a part" below describes how to select a part.

Each ablation sweeps two hyperparameters of one algorithm over a 6 x 6 grid, on every benchmark, with five seeds per
grid point:

    skvi   koopmanrl.soft_koopman_value_iteration   --num_actions           71, 81, 91, 101, 111, 121
                                                    --num_training_epochs   75, 100, 125, 150, 175, 200
    sakc   koopmanrl.soft_actor_koopman_critic      --v_lr                  0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05
                                                    --q_lr                  0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05

`--num_actions` is the number of actions of the grid SKVI chooses from. `--v_lr` is the learning rate of the optimiser
of the value network of SAKC and `--q_lr` the learning rate of the optimiser of its Q-networks; the figure of the
paper labels the `v_lr` axis "Value Network Learning Rate" and the `q_lr` axis "Policy Network Learning Rate".

Each run writes a TensorBoard log with the return of every episode (`charts/episodic_return`), which is the raw data
of the figures. `koopmanrl_utils/ABLATIONS.md` describes how to turn these logs into the tables the surfaces are drawn
from.

What is run
    The two grids and the seeds are listed in `configurations/ablations.json`, which the Snakemake workflow of
    `workflow/` reads as well; that workflow calls this script once per run. The modules of the two algorithms,
    their tuned configurations and the benchmarks are those of `run_optimized_experiments`, which reads them from
    `configurations/episodic_returns.json`.

Hyperparameters that are not swept
    Every run is started with the tuned configuration of its benchmark,
    `configurations/<algorithm>_<benchmark>_hparams.json`, and with the two swept hyperparameters, the benchmark and
    the seed on the command line, which take priority over the file. Everything else is the tuned value of the file
    or, where the file has no entry, the default of the script. The paper states that the hyperparameters that are
    not swept are fixed at the tuned values. Each run writes all of its hyperparameters into its TensorBoard log
    (text `hyperparameters`).

    The data frames of the ablations of the paper are published at
    https://huggingface.co/datasets/dynamicslab/KoopmanRL-v2 (`data/ablation_skvi/<benchmark>.json` and
    `data/ablation_sakc/<benchmark>.json`). Their runs were made in May 2025, and the data frames do not record the
    hyperparameters that are not swept. It can therefore not be checked from this repository that those runs used
    the configurations that are in `configurations/` today.

    The published data frame of SKVI on the double well holds repeated runs: the file is about twice the size of the
    data frames of SKVI on the fluid flow and on the Lorenz system (578,112 bytes against 293,079 and 269,127), and
    grid points with two runs per seed were found in it, the two runs started 16.6 hours apart in each of the five
    cases seen. The processing script counts every run of a grid point. From the size of the file we infer, without
    having counted the runs, that most or all grid points were run twice; the table of the paper for this benchmark
    then pools about twice as many runs per grid point as five seeds give. This launcher makes one run per grid
    point and seed, so a rerun does not compute the same estimate for SKVI on the double well.

Seeds
    `SEEDS` lists the five seeds of the published data frames, which are used at every grid point.

Output
    The algorithms write relative to their working directory, which is `--output_dir` (default `ablation_results/`,
    not tracked by git):

        <output_dir>/runs/SKVI/<run name>/   TensorBoard logs of skvi
        <output_dir>/runs/SAKC/<run name>/   TensorBoard logs of sakc
        <output_dir>/saved_models/           checkpoints
        <output_dir>/logs/                   console output, one file per run

    Use a directory that holds no other runs: the processing scripts take every run they find in a folder.

Usage (from the repository root):

    uv run -m koopmanrl_utils.run_ablations --dry_run            # print the 1,440 commands only
    uv run -m koopmanrl_utils.run_ablations --num_workers 8      # everything, eight runs at a time
    uv run -m koopmanrl_utils.run_ablations --algorithms sakc --environments LinearSystem-v0 DoubleWell-v0

The runs that share algorithm, benchmark and seed are made one after the other, and `--num_workers` runs several
such sequences at a time, with Ray; the default campaign has 40 of them, with 36 runs each. A run of SKVI takes a
few GB of memory: we measured about 2.5 GB on the linear system with 71 actions, and about 3.5 GB on the fluid flow
with the tuned configuration; the runs of the ablation on the fluid flow and the Lorenz system were not measured. A
run that fails does not stop the others; the failed runs are listed at the end, and their incomplete TensorBoard
logs have to be removed before the logs are processed.

Running a part
    `--num_actions`, `--num_training_epochs`, `--v_lr` and `--q_lr` take the values of the grid that are to be run
    (default: all six). With `--algorithms`, `--environments` and `--seeds` they select a part of the campaign, for a
    test or to make the runs that are missing after an interruption:

        uv run -m koopmanrl_utils.run_ablations --algorithms skvi --environments LinearSystem-v0 --seeds 1 21 \
            --num_actions 71 81 --num_training_epochs 75 --total_timesteps 2200 --output_dir ablation_results/test

    The example is a test with short runs, which is why it has an output directory of its own: among the runs of the
    campaign its runs would be the incomplete runs that have to be kept out of the processing. Values that are not
    on the grid of the paper are refused, because the processing scripts read that grid only.
    A grid point that is run twice with the same seed gives two logs, and both are counted when the logs are
    processed; remove the log of an interrupted run before it is repeated.

Reproducibility
    The section of this name in `run_optimized_experiments` applies: every run is seeded, and two runs with one seed
    log the same returns on the same machine, with the same library versions and the same number of numerical threads.
"""

import itertools
import json
import os
import sys
from typing import Optional

from tap import Tap

from koopmanrl_utils.run_optimized_experiments import ALGORITHMS as MODULES
from koopmanrl_utils.run_optimized_experiments import (
    CONFIG_DIR,
    ENVIRONMENTS,
    check_choices,
    config_file,
    run_lanes,
)

# What is run: the two grids and the seeds. The Snakemake workflow reads the same file.
with open(os.path.join(CONFIG_DIR, "ablations.json")) as f:
    ABLATIONS = json.load(f)["ablations"]

# algorithm: {swept command-line flag: its values in the paper}; the first flag is the first column of the tables
GRIDS = {name: algorithm["grid"] for name, algorithm in ABLATIONS["algorithms"].items()}

# seeds of the published data frames, used at every grid point
SEEDS = ABLATIONS["seeds"]


class ArgumentParser(Tap):
    environments: list[str] = list(ENVIRONMENTS)  # benchmarks to run
    algorithms: list[str] = list(GRIDS)  # ablations to run: skvi, sakc
    seeds: list[int] = list(SEEDS)  # seeds to run at every grid point
    num_actions: list[int] = list(GRIDS["skvi"]["num_actions"])  # skvi: numbers of actions to run
    num_training_epochs: list[int] = list(GRIDS["skvi"]["num_training_epochs"])  # skvi: numbers of epochs to run
    v_lr: list[float] = list(GRIDS["sakc"]["v_lr"])  # sakc: learning rates of the value network to run
    q_lr: list[float] = list(GRIDS["sakc"]["q_lr"])  # sakc: learning rates of the Q-networks to run
    total_timesteps: Optional[int] = None  # environment steps per run (default: 50,000, as in the paper)
    output_dir: str = "ablation_results"  # working directory of the runs
    num_workers: int = 1  # number of runs made at the same time
    dry_run: bool = False  # print the commands without running them


def command(algorithm: str, env_id: str, seed: int, point: tuple, total_timesteps: Optional[int] = None) -> list[str]:
    """Command line of one run; `point` holds the values of the two swept flags, in the order of `GRIDS`."""
    cmd = [
        sys.executable,
        "-m",
        MODULES[algorithm],
        f"--config_file={config_file(algorithm, env_id)}",
        f"--env_id={env_id}",
        f"--seed={seed}",
    ]
    cmd += [f"--{flag}={value}" for flag, value in zip(GRIDS[algorithm], point)]
    if total_timesteps is not None:
        cmd.append(f"--total_timesteps={total_timesteps}")
    return cmd


def build_lanes(
    environments: list[str],
    algorithms: list[str],
    seeds: Optional[list[int]] = None,
    grids: Optional[dict[str, dict[str, list]]] = None,
    total_timesteps: Optional[int] = None,
) -> list[list[tuple[str, list[str]]]]:
    """Runs grouped into lanes, one per algorithm, benchmark and seed; a run is (name of its log file, command line).

    The runs of a lane are made one after the other. The checkpoint folder of a run is named after the benchmark,
    the seed and the second in which the folder name is set, not after the grid point:
    `saved_models/SKVI/<benchmark>/skvi_chkpts_<seed>_<second>` (set when the SKVI policy is built, after the Koopman
    tensor) and `saved_models/SAKC/<benchmark>/sakc_chkpts_<seed>_<second>` (set at the start of the run). Two grid
    points of one lane that ran at the same time could therefore write into the same folder.
    """
    grids = grids or GRIDS
    lanes = []
    for env_id in environments:
        for algorithm in algorithms:
            for seed in seeds or SEEDS:
                lanes.append(
                    [
                        (
                            f"{algorithm}__{env_id}__{point[0]}__{point[1]}__{seed}",
                            command(algorithm, env_id, seed, point, total_timesteps),
                        )
                        for point in itertools.product(*grids[algorithm].values())
                    ]
                )
    return lanes


def main() -> None:
    args = ArgumentParser().parse_args()
    check_choices(args.environments, ENVIRONMENTS, "environments")
    check_choices(args.algorithms, GRIDS, "algorithms")
    grids = {algorithm: {flag: getattr(args, flag) for flag in GRIDS[algorithm]} for algorithm in GRIDS}
    for algorithm in GRIDS:
        for flag, values in GRIDS[algorithm].items():
            check_choices([str(v) for v in grids[algorithm][flag]], [str(v) for v in values], f"values of --{flag}")

    lanes = build_lanes(args.environments, args.algorithms, args.seeds, grids, args.total_timesteps)
    if args.dry_run:
        for lane in lanes:
            for _, cmd in lane:
                print(" ".join(cmd))
        return
    run_lanes(lanes, args.output_dir, args.num_workers)


if __name__ == "__main__":
    main()
