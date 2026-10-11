import json
import sys

import pytest

from koopmanrl_utils import run_ablations as launcher
from koopmanrl_utils.run_optimized_experiments import ALGORITHMS, ENVIRONMENTS


def dry_run(monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", ["run_ablations", "--dry_run", *arguments])
    launcher.main()
    return capsys.readouterr().out.splitlines()


def test_default_campaign_is_1440_runs_in_40_lanes():
    lanes = launcher.build_lanes(list(ENVIRONMENTS), list(launcher.GRIDS))
    assert len(lanes) == len(ENVIRONMENTS) * len(launcher.GRIDS) * len(launcher.SEEDS) == 40
    assert all(len(lane) == 36 for lane in lanes)
    names = [name for lane in lanes for name, _ in lane]
    commands = [" ".join(cmd) for lane in lanes for _, cmd in lane]
    assert len(set(names)) == len(set(commands)) == 1440
    assert launcher.SEEDS == [1, 21, 41, 61, 81]
    for grid in launcher.GRIDS.values():
        assert [len(set(values)) for values in grid.values()] == [6, 6]


def test_dry_run_prints_the_commands_of_the_default_campaign(monkeypatch, capsys):
    lines = dry_run(monkeypatch, capsys)
    assert len(lines) == 1440
    for algorithm in launcher.GRIDS:
        assert sum(f" -m {ALGORITHMS[algorithm]} " in line for line in lines) == 720


@pytest.mark.parametrize("env_id", list(ENVIRONMENTS))
@pytest.mark.parametrize(
    "algorithm, point, swept",
    [
        ("skvi", (71, 200), ["--num_actions=71", "--num_training_epochs=200"]),
        ("sakc", (0.0001, 0.05), ["--v_lr=0.0001", "--q_lr=0.05"]),
    ],
)
def test_run_takes_the_tuned_configuration_and_sets_the_two_swept_flags(algorithm, point, swept, env_id):
    cmd = launcher.command(algorithm, env_id, seed=21, point=point)
    assert cmd[:3] == [sys.executable, "-m", ALGORITHMS[algorithm]]
    name, config = cmd[3].split("=", 1)
    assert name == "--config_file"
    with open(config) as f:
        assert json.load(f)["env-id"] == env_id
    assert cmd[4:] == [f"--env_id={env_id}", "--seed=21", *swept]
    assert launcher.command(algorithm, env_id, 21, point, total_timesteps=400)[-1] == "--total_timesteps=400"


def test_runs_that_share_algorithm_benchmark_and_seed_are_in_one_lane():
    # the checkpoint folder of a run is named after these three and a time in seconds, not after the grid point
    lanes = launcher.build_lanes(list(ENVIRONMENTS), list(launcher.GRIDS))
    keys = [{(cmd[2], cmd[4], cmd[5]) for _, cmd in lane} for lane in lanes]
    assert all(len(key) == 1 for key in keys)
    assert len(set.union(*keys)) == len(lanes)


def test_given_grid_values_and_seeds_replace_the_defaults():
    grids = {"skvi": {"num_actions": [71, 81], "num_training_epochs": [75]}}
    lanes = launcher.build_lanes(["DoubleWell-v0"], ["skvi"], seeds=[1, 21], grids=grids)
    assert [[name for name, _ in lane] for lane in lanes] == [
        ["skvi__DoubleWell-v0__71__75__1", "skvi__DoubleWell-v0__81__75__1"],
        ["skvi__DoubleWell-v0__71__75__21", "skvi__DoubleWell-v0__81__75__21"],
    ]
    assert lanes[1][1][1][-3:] == ["--seed=21", "--num_actions=81", "--num_training_epochs=75"]


def test_part_of_each_grid_from_the_command_line(monkeypatch, capsys):
    lines = dry_run(
        monkeypatch,
        capsys,
        *["--environments", "LinearSystem-v0", "--seeds", "1", "21", "--total_timesteps", "400"],
        *["--num_actions", "71", "81", "--num_training_epochs", "75", "100"],
        *["--v_lr", "1e-4", "0.05", "--q_lr", "5e-4"],
    )
    assert len(lines) == 2 * 4 + 2 * 2
    skvi = [line.split()[-3:-1] for line in lines if "soft_koopman_value_iteration" in line]
    assert skvi[:4] == [
        ["--num_actions=71", "--num_training_epochs=75"],
        ["--num_actions=71", "--num_training_epochs=100"],
        ["--num_actions=81", "--num_training_epochs=75"],
        ["--num_actions=81", "--num_training_epochs=100"],
    ]
    # the learning rates are passed on as Python prints them, which is how SAKC writes them into the run name
    sakc = [line.split()[-3:-1] for line in lines if "soft_actor_koopman_critic" in line]
    assert sakc[:2] == [["--v_lr=0.0001", "--q_lr=0.0005"], ["--v_lr=0.05", "--q_lr=0.0005"]]
    assert all(line.endswith("--total_timesteps=400") for line in lines)


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--v_lr", "0.002"], "Unknown values of --v_lr: 0.002"),
        (["--num_actions", "100"], "Unknown values of --num_actions: 100"),
        (["--algorithms", "lqr"], "Unknown algorithms: lqr"),
        (["--environments", "Lorenz"], "Unknown environments: Lorenz"),
    ],
)
def test_values_off_the_grid_and_unknown_names_are_refused(arguments, message, monkeypatch, capsys):
    with pytest.raises(ValueError, match=message):
        dry_run(monkeypatch, capsys, *arguments)
