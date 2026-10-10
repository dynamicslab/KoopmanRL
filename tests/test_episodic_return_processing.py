import json

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from koopmanrl_utils.dataframe_creator import run_name_fields
from tests.utils import run_module

# Names of run folders as the five algorithms write them.
RUN_NAMES = {
    "linear_quadratic_regulator": "FluidFlow-v0__linear_quadratic_regulator__88__1765986001",
    "sac_continuous_action": "FluidFlow-v0__sac_continuous_action__417__1765986000",
    "value_based_sac_continuous_action": "FluidFlow-v0__value_based_sac_continuous_action__3__1765986002",
    "soft_koopman_value_iteration": "FluidFlow-v0__soft_koopman_value_iteration__101__125__5412__1765986900",
    "soft_actor_koopman_critic": (
        "FluidFlow-v0__soft_actor_koopman_critic__5412__0.009423359172870875__0.0017865746944645956__1765986950"
    ),
}
SEEDS = {
    "linear_quadratic_regulator": 88,
    "sac_continuous_action": 417,
    "value_based_sac_continuous_action": 3,
    "soft_koopman_value_iteration": 5412,
    "soft_actor_koopman_critic": 5412,
}


@pytest.mark.parametrize("algorithm", list(RUN_NAMES))
def test_seed_and_time_are_read_from_every_layout_of_the_run_name(algorithm):
    fields = run_name_fields(RUN_NAMES[algorithm])
    assert fields["environment"] == "FluidFlow-v0"
    assert fields["rl_algorithm"] == algorithm
    assert fields["seed"] == SEEDS[algorithm]
    assert fields["time"] == int(RUN_NAMES[algorithm].rsplit("__", 1)[1])


def test_learning_rates_are_read_from_the_run_name_of_sakc():
    fields = run_name_fields(RUN_NAMES["soft_actor_koopman_critic"])
    assert (fields["v_lr"], fields["q_lr"]) == (0.009423359172870875, 0.0017865746944645956)


@pytest.mark.parametrize("algorithm", ["soft_actor_koopman_critic", "soft_koopman_value_iteration"])
def test_data_frame_of_episodic_returns_from_tensorboard_logs(algorithm, tmp_path):
    writer = SummaryWriter(str(tmp_path / "runs" / RUN_NAMES[algorithm]))
    for episode in range(3):
        writer.add_scalar("charts/episodic_return", -10.0 * episode, 1999 + 2000 * episode)
    writer.close()

    result = run_module(
        "koopmanrl_utils.dataframe_creator",
        [
            "--mode=Episodic_Returns",
            "--system=FluidFlow-v0",
            f"--rl_algo={algorithm}",
            f"--target_dir={tmp_path / 'runs'}",
            f"--storage_dir={tmp_path}",
            "--output_file=frame.json",
        ],
    )
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "frame.json") as f:
        (run,) = json.load(f).values()
    assert run["seed"] == SEEDS[algorithm]
    assert run["steps"] == [1999, 3999, 5999]
    assert run["episodic_returns"] == [0.0, -10.0, -20.0]


def test_data_frame_lists_the_runs_in_the_order_of_their_names(tmp_path):
    names = [f"FluidFlow-v0__linear_quadratic_regulator__{seed}__1765986001" for seed in (30, 4, 100, 7, 55)]
    for name in names:
        writer = SummaryWriter(str(tmp_path / "runs" / name))
        writer.add_scalar("charts/episodic_return", -1.0, 1999)
        writer.close()

    result = run_module(
        "koopmanrl_utils.dataframe_creator",
        [
            "--mode=Episodic_Returns",
            "--system=FluidFlow-v0",
            "--rl_algo=linear_quadratic_regulator",
            f"--target_dir={tmp_path / 'runs'}",
            f"--storage_dir={tmp_path}",
            "--output_file=frame.json",
        ],
    )
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "frame.json") as f:
        assert list(json.load(f)) == sorted(names)


@pytest.mark.parametrize("output_name", ["table.csv", "table.dat"])
def test_table_of_episodic_returns(output_name, tmp_path):
    # eight runs: the inter-quartile mean is the mean of the four middle returns of a step
    first = [-108.0, -107.0, -106.0, -105.0, -104.0, -103.0, -102.0, -101.0]
    last = [-1.0, -2.0, -30.0, -4.0, -5.0, -6.0, -7.0, -800.0]
    frame = {f"run_{seed}": {"episodic_returns": [first[seed], last[seed]], "steps": [1999, 3999]} for seed in range(8)}
    with open(tmp_path / "frame.json", "w") as f:
        json.dump(frame, f)

    args = [
        f"--root_dir={tmp_path}",
        "--data_frame=frame",
        f"--output_dir={tmp_path}",
        f"--output_name={output_name}",
        "--deterministic_bootstrap",
    ]
    tables = []
    for _ in range(2):
        result = run_module("koopmanrl_utils.process_episodic_returns", args)
        assert result.returncode == 0, result.stderr
        tables.append((tmp_path / output_name).read_text())
    assert tables[0] == tables[1]  # the seeded bootstrap gives the same band on every call

    # both formats have a header row that pgfplots can take the names of the columns from
    delimiter = "," if output_name.endswith(".csv") else " "
    header, *rows = tables[0].splitlines()
    assert header == delimiter.join(
        ["timesteps", "episodic_returns", "lower_confidence_bound", "upper_confidence_bound"]
    )
    values = np.array([row.split(delimiter) for row in rows], dtype=float)
    assert list(values[:, 0]) == [1999.0, 3999.0]
    assert list(values[:, 1]) == [-104.5, -5.5]  # means of -106 ... -103 and of -4 ... -7
    assert np.all(values[:, 2] <= values[:, 1]) and np.all(values[:, 1] <= values[:, 3])


def test_data_frame_without_a_matching_run_is_an_error(tmp_path):
    writer = SummaryWriter(str(tmp_path / "runs" / RUN_NAMES["soft_actor_koopman_critic"]))
    writer.add_scalar("charts/episodic_return", -1.0, 1999)
    writer.close()

    result = run_module(
        "koopmanrl_utils.dataframe_creator",
        [
            "--mode=Episodic_Returns",
            "--system=FluidFlow-v0",
            "--rl_algo=sakc",  # not the name the algorithm has in the run folders
            f"--target_dir={tmp_path / 'runs'}",
            f"--storage_dir={tmp_path}",
            "--output_file=frame.json",
        ],
    )
    assert result.returncode != 0
    assert "No run of FluidFlow-v0" in result.stderr
    assert not (tmp_path / "frame.json").exists()
