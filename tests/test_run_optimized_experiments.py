import json
import os
import sys

import pytest

from koopmanrl_utils import run_optimized_experiments as launcher


def test_every_algorithm_runs_on_every_benchmark_with_every_seed():
    lanes = launcher.build_lanes(list(launcher.ENVIRONMENTS), list(launcher.ALGORITHMS))
    assert len(lanes) == len(launcher.ENVIRONMENTS) * len(launcher.ALGORITHMS)
    assert sum(len(lane) for lane in lanes) == len(launcher.ALGORITHMS) * sum(len(s) for s in launcher.SEEDS.values())
    names = [name for lane in lanes for name, _ in lane]
    assert len(set(names)) == len(names)
    for seeds in launcher.SEEDS.values():
        assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("env_id", list(launcher.ENVIRONMENTS))
@pytest.mark.parametrize("algorithm", launcher.TUNED)
def test_tuned_algorithms_use_the_configuration_of_the_benchmark(algorithm, env_id):
    cmd = launcher.command(algorithm, env_id, seed=3)
    (config,) = [arg.split("=", 1)[1] for arg in cmd if arg.startswith("--config_file=")]
    with open(config) as f:
        assert json.load(f)["env-id"] == env_id
    assert cmd[:3] == [sys.executable, "-m", launcher.ALGORITHMS[algorithm]]
    assert "--seed=3" in cmd


@pytest.mark.parametrize("algorithm", ["lqr", "sac_q", "sac_v"])
def test_baselines_run_with_their_defaults_and_the_given_seed(algorithm):
    cmd = launcher.command(algorithm, "Lorenz-v0", seed=5, total_timesteps=400)
    assert cmd == [
        sys.executable,
        "-m",
        launcher.ALGORITHMS[algorithm],
        "--env_id=Lorenz-v0",
        "--seed=5",
        "--total_timesteps=400",
    ]


def test_given_seeds_replace_the_seeds_of_the_paper():
    lanes = launcher.build_lanes(["Lorenz-v0", "DoubleWell-v0"], ["lqr"], seeds=[1, 2])
    assert [[name for name, _ in lane] for lane in lanes] == [
        ["lqr__Lorenz-v0__1", "lqr__Lorenz-v0__2"],
        ["lqr__DoubleWell-v0__1", "lqr__DoubleWell-v0__2"],
    ]


def test_runs_are_made_in_the_output_directory_and_failures_are_reported_at_the_end(tmp_path):
    fail = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]
    succeed = [sys.executable, "-c", "open('made_here.txt', 'w').close()"]
    with pytest.raises(RuntimeError, match="1 run"):
        launcher.run_lanes([[("first", fail), ("second", succeed)]], str(tmp_path))
    assert (tmp_path / "made_here.txt").exists()  # the run after the failed one was still made, in the output dir
    assert (tmp_path / "logs" / "first.log").read_text().strip() == "boom"
    assert os.path.exists(tmp_path / "logs" / "second.log")
