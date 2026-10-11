import itertools
import json
import os

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from koopmanrl_utils import run_ablations as launcher
from tests.utils import PROJECT_ROOT, run_module

MODES = {"skvi": "SKVI_Ablations", "sakc": "SAKC_Ablations"}
COLUMNS = {
    "skvi": ["num_actions", "num_training_epochs", "episodic_return"],
    "sakc": ["v_lr", "q_lr", "episodic_return"],
}


def run_name(algorithm, env_id, seed, point, time):
    """Name of the run folder as the algorithm writes it for a command of the launcher."""
    # the algorithms read the two swept flags from the command line and print them into the name
    first, second = [arg.split("=", 1)[1] for arg in launcher.command(algorithm, env_id, seed, point)[-2:]]
    if algorithm == "skvi":
        return f"{env_id}__soft_koopman_value_iteration__{int(first)}__{int(second)}__{seed}__{time}"
    return f"{env_id}__soft_actor_koopman_critic__{seed}__{float(first)}__{float(second)}__{time}"


def table_order(algorithm):
    """Grid points in the order of the rows of the table."""
    first, second = launcher.GRIDS[algorithm].values()
    if algorithm == "sakc":  # as in the tables of the paper: v_lr from the smallest, q_lr from the largest
        second = second[::-1]
    return list(itertools.product(first, second))


def write_log(folder, returns, episode_length=2000):
    writer = SummaryWriter(str(folder))
    for episode, value in enumerate(returns):
        writer.add_scalar("charts/episodic_return", value, episode_length * (episode + 1) - 1)
    writer.close()


def read_table(path):
    header, *rows = path.read_text().splitlines()
    delimiter = "," if path.suffix == ".csv" else None
    return header, [row.split(delimiter) for row in rows]


@pytest.mark.parametrize("algorithm", list(MODES))
def test_every_run_of_the_launcher_reaches_the_data_frame_and_the_table(algorithm, tmp_path):
    points = table_order(algorithm)
    final_return = {point: -10.0 * index - 1.0 for index, point in enumerate(points)}
    for index, point in enumerate(points):
        name = run_name(algorithm, "DoubleWell-v0", 21, point, 1765986000 + index)
        write_log(tmp_path / "runs" / name, [-5000.0, final_return[point]])
    # a run on another benchmark in the same folder is left out
    write_log(tmp_path / "runs" / run_name(algorithm, "Lorenz-v0", 21, points[0], 1765987000), [-1.0, -2.0])

    result = run_module(
        "koopmanrl_utils.dataframe_creator",
        [
            f"--mode={MODES[algorithm]}",
            "--system=DoubleWell-v0",
            f"--target_dir={tmp_path / 'runs'}",
            f"--storage_dir={tmp_path}",
            "--output_file=frame.json",
        ],
    )
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "frame.json") as f:
        frame = json.load(f)
    assert len(frame) == 36
    first, second = launcher.GRIDS[algorithm]
    for index, point in enumerate(points):
        run = frame[run_name(algorithm, "DoubleWell-v0", 21, point, 1765986000 + index)]
        assert (run[first], run[second]) == point
        assert (run["environment"], run["seed"], run["time"]) == ("DoubleWell-v0", 21, 1765986000 + index)
        assert run["steps"] == [1999, 3999]
        assert run["episodic_returns"] == [-5000.0, final_return[point]]

    result = run_module(
        f"koopmanrl_utils.process_{algorithm}_ablations",
        [f"--root_dir={tmp_path}", "--data_frame=frame", f"--output_dir={tmp_path}", "--output_name=table.csv"],
    )
    assert result.returncode == 0, result.stderr
    header, rows = read_table(tmp_path / "table.csv")
    assert header == ",".join(COLUMNS[algorithm])
    assert {(float(x), float(y)): float(z) for x, y, z in rows} == final_return


@pytest.mark.parametrize("algorithm", list(MODES))
@pytest.mark.parametrize("output_name", ["table.csv", "table.dat"])
def test_table_of_an_ablation(algorithm, output_name, tmp_path):
    first, second = launcher.GRIDS[algorithm]
    points = table_order(algorithm)
    frame = {}
    for index, point in enumerate(points):
        # last returns of the five seeds: the inter-quartile mean drops the smallest and the largest
        for seed, offset in zip(launcher.SEEDS, [0.0, 1.0, 2.0, 3.0, 100.0]):
            frame[f"run_{index}_{seed}"] = {
                first: point[0],
                second: point[1],
                "seed": seed,
                "episodic_returns": [-7000.0, -10.0 * index + offset],
                "steps": [1999, 3999],
            }
    (tmp_path / "frames").mkdir()
    (tmp_path / "tables").mkdir()
    with open(tmp_path / "frames" / "frame.json", "w") as f:
        json.dump(frame, f)

    # the output directory is given relative to the directory the script is started from
    result = run_module(
        f"koopmanrl_utils.process_{algorithm}_ablations",
        [
            f"--root_dir={tmp_path / 'frames'}",
            "--data_frame=frame",
            f"--output_dir={os.path.relpath(tmp_path / 'tables', PROJECT_ROOT)}",
            f"--output_name={output_name}",
        ],
    )
    assert result.returncode == 0, result.stderr

    rows = ["%.4f %.4f %.4f" % (*point, -10.0 * index + 2.0) for index, point in enumerate(points)]
    if output_name.endswith(".csv"):
        expected = "\n".join([",".join(COLUMNS[algorithm])] + [row.replace(" ", ",") for row in rows]) + "\n"
    else:  # the layout of the tables of the paper: header `x y z`, an empty line after every block of six rows
        blocks = ["\n".join(rows[start : start + 6]) + "\n" for start in range(0, 36, 6)]
        expected = "x y z\n" + "\n".join(blocks) + "\n"
    assert (tmp_path / "tables" / output_name).read_text() == expected
    if algorithm == "sakc":  # the two smallest learning rates keep their value at four decimals
        assert rows[5].startswith("0.0001 0.0001 ") and rows[4].startswith("0.0001 0.0005 ")
        assert sorted({float(row.split()[0]) for row in rows}) == launcher.GRIDS["sakc"]["v_lr"]


def test_grid_points_without_runs_are_nan_in_the_table(tmp_path):
    frame = {
        f"run_{seed}": {"num_actions": 81, "num_training_epochs": 100, "episodic_returns": [-3.0 - seed]}
        for seed in (1, 21)
    }
    with open(tmp_path / "frame.json", "w") as f:
        json.dump(frame, f)
    result = run_module(
        "koopmanrl_utils.process_skvi_ablations",
        [f"--root_dir={tmp_path}", "--data_frame=frame", f"--output_dir={tmp_path}", "--output_name=table.csv"],
    )
    assert result.returncode == 0, result.stderr
    _, rows = read_table(tmp_path / "table.csv")
    assert len(rows) == 36
    values = np.array(rows, dtype=float)
    assert np.isnan(values[:, 2]).sum() == 35
    assert list(values[~np.isnan(values[:, 2])][0]) == [81.0, 100.0, -14.0]


def test_smoothing_window_pools_the_last_returns_of_all_seeds(tmp_path):
    returns = [[-5, -1, -2, -3], [-5, -4, -5, -6], [-5, -7, -8, -9], [-5, -10, -20, -900], [-5, -30, -40, -1000]]
    frame = {
        f"run_{seed}": {"v_lr": 0.001, "q_lr": 0.05, "episodic_returns": values}
        for seed, values in zip(launcher.SEEDS, returns)
    }
    with open(tmp_path / "frame.json", "w") as f:
        json.dump(frame, f)

    # the inter-quartile mean drops a quarter of the pooled returns, rounded down, at either end:
    # 1 of 5 for a window of 1, 3 of 15 for a window of 3
    for window, expected in ((1, (-6 - 9 - 900) / 3), (3, -(4 + 5 + 6 + 7 + 8 + 9 + 10 + 20 + 30) / 9)):
        result = run_module(
            "koopmanrl_utils.process_sakc_ablations",
            [
                f"--root_dir={tmp_path}",
                "--data_frame=frame",
                f"--output_dir={tmp_path}",
                "--output_name=table.csv",
                f"--smoothing_window={window}",
            ],
        )
        assert result.returncode == 0, result.stderr
        _, rows = read_table(tmp_path / "table.csv")
        (value,) = [float(z) for x, y, z in rows if (float(x), float(y)) == (0.001, 0.05)]
        assert value == expected
