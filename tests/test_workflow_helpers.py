"""Tests of workflow/helpers.py: the checks of the settings of the Snakemake workflows, the folder of run folders
and the check of a data frame. The module is loaded by its path, since workflow/ is not a package, and without
Snakemake, which is not in the project environment: a check that fails raises `ValueError` here."""

import importlib.util
import json
import os
import re
import sys

import pytest

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_helpers():
    spec = importlib.util.spec_from_file_location(
        "workflow_helpers", os.path.join(REPOSITORY, "workflow", "helpers.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_helpers()
Refused = helpers.WorkflowError

ALGORITHMS = ["lqr", "sac_q", "sac_v", "skvi", "sakc"]
BENCHMARKS = {"LinearSystem-v0": "linear_system", "Lorenz-v0": "lorenz"}


def settings(name):
    with open(os.path.join(REPOSITORY, "configurations", f"{name}.json")) as f:
        return json.load(f)[name]


def rules(name):
    with open(os.path.join(REPOSITORY, "workflow", "rules", f"{name}.smk")) as f:
        return f.read()


def test_the_module_is_loaded_without_snakemake():
    # a check that fails raises the error of Snakemake where there is one, and ValueError in the project environment
    assert (Refused is ValueError) == (importlib.util.find_spec("snakemake") is None)


def test_alternatives_match_the_names_only():
    constraint = helpers.alternatives(["0.0001", "sac_q"])
    assert re.fullmatch(constraint, "0.0001") and re.fullmatch(constraint, "sac_q")
    assert not re.fullmatch(constraint, "0x0001") and not re.fullmatch(constraint, "sac_q2")


@pytest.mark.parametrize("value", [None, "null", "NULL", "none", "None"])
def test_the_words_null_and_none_are_null(value):
    assert helpers.is_null(value)
    assert helpers.selection(value, ALGORITHMS, "only_algorithms") == ALGORITHMS
    assert helpers.whole_number(value, "total_timesteps", 1, or_null=True) is None
    assert helpers.choice(value, ["orders"], "sweep", or_null=True) is None


@pytest.mark.parametrize("value", ["", "nul", 0, False, [], "nothing"])
def test_other_values_are_not_null(value):
    assert not helpers.is_null(value)


def test_unknown_keys_are_refused_with_the_keys_that_exist():
    # the verifiers: `lorenz` in place of `Lorenz-v0` under `seeds` gave the 25 seeds of the paper without a word,
    # as did a misspelled key of a workflow and a flag under `only_values` that no grid has
    given = {"LinearSystem-v0": [1]}
    assert helpers.known_keys(given, BENCHMARKS, "episodic_returns: seeds") is given
    with pytest.raises(Refused, match=r"episodic_returns: seeds does not have: lorenz\. It takes .*Lorenz-v0"):
        helpers.known_keys({"lorenz": [1]}, BENCHMARKS, "episodic_returns: seeds")
    with pytest.raises(Refused, match=r"ablations does not have: total_timestep, seed\."):
        helpers.known_keys({"total_timestep": 400, "seed": [1], "seeds": [1]}, settings("ablations"), "ablations")
    with pytest.raises(Refused, match=r"ablations: only_values does not have: vlr\."):
        helpers.known_keys({"vlr": ["0.0001"]}, ["num_actions", "v_lr"], "ablations: only_values")
    with pytest.raises(Refused, match=r"only_values takes a mapping with keys from: v_lr, q_lr\. Got: \['0.0001'\]"):
        helpers.known_keys(["0.0001"], ["v_lr", "q_lr"], "only_values")


@pytest.mark.parametrize("name", ["episodic_returns", "ablations", "tsne"])
def test_the_settings_of_the_json_files_pass_their_own_checks(name):
    defaults = settings(name)
    assert helpers.known_keys(defaults, defaults, name) is defaults
    if name == "tsne":
        assert helpers.script_options(defaults["identification"], name, known=defaults["identification"]) == (
            "--sweep inferred_layout"
        )
        return
    assert helpers.whole_number(defaults["total_timesteps"], "total_timesteps", 1, or_null=True) is None
    assert defaults["table_format"] == "csv"
    for key in ("only_algorithms", "only_benchmarks"):
        assert helpers.selection(defaults[key], ALGORITHMS, key) == ALGORITHMS
    seeds = defaults["seeds"]
    for listed in seeds.values() if isinstance(seeds, dict) else [seeds]:
        assert helpers.whole_numbers(listed, "seeds") == listed


def test_a_selection_keeps_the_order_of_the_known_names():
    assert helpers.selection(["sakc", "lqr"], ALGORITHMS, "only_algorithms") == ["lqr", "sakc"]
    assert helpers.selection(["Lorenz-v0"], BENCHMARKS, "only_benchmarks") == ["Lorenz-v0"]
    # the values of a grid: numbers of a file, strings of the command line
    assert helpers.selection([0.0005, "0.0001"], ["0.0001", "0.0005", "0.001"], "v_lr") == ["0.0001", "0.0005"]
    assert helpers.selection([71], ["71", "81"], "num_actions") == ["71"]


@pytest.mark.parametrize(
    "only",
    [
        [],  # selects nothing: the target was done at once, with no table
        ["lqr", "lqr"],  # twice
        ["LQR"],  # not a name
        ["lqr", "ppo"],
        "lqr",  # not a list
        {"lqr": None},
        "",
    ],
)
def test_a_selection_that_is_empty_repeats_a_name_or_names_nothing_known_is_refused(only):
    with pytest.raises(Refused, match=r"only_algorithms takes null or a list of names from: lqr, .*each of them once"):
        helpers.selection(only, ALGORITHMS, "only_algorithms")


@pytest.mark.parametrize("only", [["0.0001", 0.0001], ["5e-4"], ["0.00050"], []])
def test_values_of_a_grid_are_selected_in_the_spelling_of_the_grid_and_once(only):
    with pytest.raises(Refused, match="only_values: v_lr takes"):
        helpers.selection(only, ["0.0001", "0.0005"], "only_values: v_lr")


@pytest.mark.parametrize("value, number", [(400, 400), ("400", 400), ("0400", 400), (1, 1), ("50000", 50000)])
def test_whole_numbers_are_read_from_numbers_and_from_strings(value, number):
    assert helpers.whole_number(value, "total_timesteps", 1) == number
    assert helpers.whole_number(value, "total_timesteps", 1, or_null=True) == number


@pytest.mark.parametrize("value", [0, "0", -5, "-5", 1.5, "1.5", 400.0, "4e2", "abc", "", True, [400], None, "null"])
def test_total_timesteps_below_one_or_not_whole_are_refused(value):
    with pytest.raises(Refused, match=r"total_timesteps takes a whole number from 1\. Got: "):
        helpers.whole_number(value, "total_timesteps", 1)
    if not helpers.is_null(value):
        with pytest.raises(Refused, match=r"total_timesteps takes null or a whole number from 1\. Got: "):
            helpers.whole_number(value, "total_timesteps", 1, or_null=True)


def test_memory_given_on_the_command_line_is_a_number():
    # the verifiers: `mem_mb: 6000` given with --config reached the resources as a string
    assert helpers.whole_number("6000", "mem_mb", 1) == 6000
    with pytest.raises(Refused, match="mem_mb takes a whole number from 1"):
        helpers.whole_number("6G", "mem_mb", 1)


def test_seeds_are_numbers_with_one_spelling():
    assert helpers.whole_numbers([4430, "2738", "007", 0], "seeds") == [4430, 2738, 7, 0]
    assert helpers.whole_numbers(["1", "3", "5"], "smoothing_windows", 1) == [1, 3, 5]


@pytest.mark.parametrize(
    "values",
    [
        [],  # no seed: the data frame job had no run to read
        [1, 21, 1],  # twice: two jobs for one run directory
        [1, "01"],  # twice as numbers
        ["1", 1],
        [-1],
        [1.5],
        ["a"],
        [None],
        [[1]],
        7,  # not a list
        "7",
        None,
        "null",
    ],
)
def test_seed_lists_that_are_empty_repeat_a_seed_or_hold_no_whole_number_are_refused(values):
    with pytest.raises(Refused, match=r"seeds takes a list of whole numbers from 0, each of them once\. Got: "):
        helpers.whole_numbers(values, "seeds")


@pytest.mark.parametrize("values", [[0], [3, 0], [3, "3"], [3, "03"], [], 3])
def test_smoothing_windows_start_at_one_and_are_listed_once(values):
    with pytest.raises(Refused, match="smoothing_windows takes a list of whole numbers from 1, each of them once"):
        helpers.whole_numbers(values, "smoothing_windows", 1)


def test_a_choice_is_one_of_the_known_names():
    assert helpers.choice("dat", ("csv", "dat"), "table_format") == "dat"
    assert helpers.choice("orders", ("inferred_layout", "orders"), "sweep", or_null=True) == "orders"
    for value in ("tsv", "CSV", "", None, "null", ["csv"], 1):
        with pytest.raises(Refused, match=r"table_format takes one of: csv, dat\. Got: "):
            helpers.choice(value, ("csv", "dat"), "table_format")
    # the verifiers: a misspelled sweep failed inside the job, which had removed the tensors by then
    with pytest.raises(Refused, match=r"sweep takes null or one of: inferred_layout, orders\. Got: 'order'"):
        helpers.choice("order", ("inferred_layout", "orders"), "sweep", or_null=True)


def test_the_sweeps_that_the_workflow_accepts_are_those_of_the_script():
    listed = re.search(r"TS_SWEEPS = \((.*?)\)", rules("tsne"), re.S).group(1)
    with open(os.path.join(REPOSITORY, "koopmanrl_utils", "tsne_koopman_tensor.py")) as f:
        script = f.read()
    grids = script[script.index("def sweep_grids(") :].split("\ndef ")[0]
    assert re.findall(r'"(\w+)"', listed) == re.findall(r'args\.sweep == "(\w+)"', grids) != []


def test_the_drivers_that_the_workflow_accepts_are_those_of_the_script():
    listed = re.search(r"TS_DRIVERS = \((.*?)\)", rules("tsne"), re.S).group(1)
    with open(os.path.join(REPOSITORY, "koopmanrl_utils", "tsne_koopman_tensor.py")) as f:
        drivers = re.search(r"LSTSQ_DRIVERS = \((.*?)\)", f.read(), re.S).group(1)
    assert re.findall(r'"(\w+)"', listed) == re.findall(r'"(\w+)"', drivers) != []


def test_entries_that_the_scripts_read_from_the_file_are_not_settings():
    listed = settings("ablations")["algorithms"]
    assert helpers.unchanged(json.loads(json.dumps(listed)), listed, "ablations: algorithms") == listed
    # the final review: another grid listed runs that the launcher refuses, and an algorithm more gave a KeyError
    for change in (
        lambda given: given["sakc"]["grid"].update(v_lr=[0.1]),
        lambda given: given["sakc"].update(foo=1),
        lambda given: given.update(ppo={}),
        lambda given: given.pop("skvi"),
    ):
        given = json.loads(json.dumps(listed))
        change(given)
        with pytest.raises(Refused, match="ablations: algorithms is not a setting"):
            helpers.unchanged(given, listed, "ablations: algorithms")
    with pytest.raises(Refused, match="is not a setting"):
        helpers.unchanged(None, listed, "ablations: algorithms")

    listed = settings("episodic_returns")["algorithms"]
    given = json.loads(json.dumps(listed))
    given["skvi"]["mem_mb"] = "6000"  # the memory of an algorithm is a setting
    assert helpers.unchanged(given, listed, "episodic_returns: algorithms", settable=("mem_mb",)) is given
    given["skvi"]["runs_dir"] = "runs"
    with pytest.raises(Refused, match="episodic_returns: algorithms is not a setting.*apart from mem_mb of an entry"):
        helpers.unchanged(given, listed, "episodic_returns: algorithms", settable=("mem_mb",))
    benchmarks = settings("episodic_returns")["benchmarks"]
    with pytest.raises(Refused, match="episodic_returns: benchmarks is not a setting"):
        helpers.unchanged({**benchmarks, "Pendulum-v1": "pendulum"}, benchmarks, "episodic_returns: benchmarks")


def test_the_formats_of_the_tables_are_those_that_the_processing_scripts_tell_apart():
    formats = re.search(r"TABLE_FORMATS = \((.*?)\)", rules("common"), re.S).group(1)
    assert re.findall(r'"(\w+)"', formats) == ["csv", "dat"]
    for script in ("process_episodic_returns", "process_skvi_ablations", "process_sakc_ablations"):
        with open(os.path.join(REPOSITORY, "koopmanrl_utils", f"{script}.py")) as f:
            assert 'args.output_name.endswith(".csv")' in f.read()
        # the table jobs take the script they call as an input, and so do the data frame jobs
        assert f"koopmanrl_utils.{script}" in rules("episodic_returns") + json.dumps(settings("ablations"))
    assert rules("episodic_returns").count("script=") == rules("ablations").count("script=") == 2
    assert os.path.isfile(os.path.join(REPOSITORY, "koopmanrl_utils", "dataframe_creator.py"))


def test_script_options_write_the_arguments_in_the_order_of_their_names():
    options = {
        "units": "box",
        "perplexity": "50",
        "subtract_persistence": "true",
        "common_basis": False,
        "double_well_coordinates": [1, 2],
        "metric": None,
        "scaling": "null",
        "neighbours": [],
        "tsne_init": "it's",
    }
    assert helpers.script_options(options, "tsne: embedding") == (
        "--double_well_coordinates 1 2 --perplexity 50 --subtract_persistence --tsne_init 'it'\"'\"'s' --units box"
    )
    assert helpers.script_options({}, "tsne: embedding") == ""


def test_script_options_refuse_what_the_step_does_not_read():
    identification = settings("tsne")["identification"]
    assert {"sweep", "seeds"} <= set(identification) and "perplexity" not in identification
    assert helpers.script_options({"sweep": "orders", "seeds": 4}, "identification", known=identification) == (
        "--seeds 4 --sweep orders"
    )
    # the verifiers: `perplexity` under `identification` was passed on and identified every tensor again
    with pytest.raises(Refused, match=r"identification does not take: perplexity\. It takes: sweep, state_orders, "):
        helpers.script_options({"sweep": "orders", "perplexity": 50}, "identification", known=identification)
    with pytest.raises(Refused, match=r"embedding does not take: sweep, output_dir\.$"):
        helpers.script_options(
            {"sweep": "orders", "output_dir": "x", "units": "box"}, "embedding", ["output_dir", "sweep"]
        )
    with pytest.raises(Refused, match="embedding does not take: Units, tsne-seed"):
        helpers.script_options({"Units": "box", "tsne-seed": 1}, "embedding")
    with pytest.raises(Refused, match=r"embedding takes a mapping of arguments to values\. Got: \['units'\]"):
        helpers.script_options(["units"], "embedding")


def make_run(root, run_dir, name, runs_dir="runs"):
    folder = os.path.join(root, run_dir, runs_dir, name)
    os.makedirs(folder)
    with open(os.path.join(folder, "events"), "w") as f:
        f.write(name)
    return os.path.join(root, run_dir)


def test_run_folders_link_the_run_folder_of_every_run_directory(tmp_path):
    root = str(tmp_path / "sp ace")
    first = make_run(
        root, "runs/sakc/lorenz/1", "Lorenz-v0__soft_actor_koopman_critic__1__0.001__0.0005__17", "runs/SAKC"
    )
    second = make_run(
        root, "runs/sakc/lorenz/21", "Lorenz-v0__soft_actor_koopman_critic__21__0.001__0.0005__18", "runs/SAKC"
    )
    with helpers.run_folders([first, second], "runs/SAKC") as (folder, run_dir_of):
        assert sorted(os.listdir(folder)) == sorted(run_dir_of)
        assert list(run_dir_of.values()) == [first, second]
        for name in run_dir_of:
            with open(os.path.join(folder, name, "events")) as f:
                assert f.read() == name
    assert not os.path.exists(folder) and os.path.isdir(first)


def test_run_folders_refuse_a_run_directory_without_exactly_one_run_folder(tmp_path):
    root = str(tmp_path)
    one = make_run(root, "7", "LinearSystem-v0__linear_quadratic_regulator__7__100")
    make_run(root, "7", "LinearSystem-v0__linear_quadratic_regulator__7__200")
    with pytest.raises(Refused, match=rf"Expected one run folder in {re.escape(one)}/runs, found 2\."):
        with helpers.run_folders([one], "runs"):
            pass
    os.makedirs(os.path.join(root, "8", "runs"))
    os.makedirs(os.path.join(root, "9"))
    for empty in ("8", "9"):
        with pytest.raises(Refused, match=r"Expected one run folder in .*, found 0\."):
            with helpers.run_folders([os.path.join(root, empty)], "runs"):
                pass


def test_run_folders_refuse_one_run_folder_in_two_run_directories(tmp_path):
    # a run folder copied into the directory of another seed: the second link had no free name
    root = str(tmp_path)
    name = "LinearSystem-v0__linear_quadratic_regulator__7__100"
    dirs = [make_run(root, "7", name), make_run(root, "8", name)]
    with pytest.raises(Refused, match=rf"The run folder {name} lies in two run directories: {dirs[0]} and {dirs[1]}\."):
        with helpers.run_folders(dirs, "runs"):
            pass


ER_RUN = os.path.join("results", "episodic_returns", "runs", "{algorithm}", "{benchmark}", "{seed}")
AB_RUN = os.path.join("results", "ablations", "runs", "{algorithm}", "{benchmark}", "{first}__{second}", "{seed}")


def test_the_wildcards_of_a_run_directory_are_read_from_the_end_of_its_path():
    assert helpers.path_wildcards(ER_RUN, "results/episodic_returns/runs/sac_q/linear_system/4430") == {
        "algorithm": "sac_q",
        "benchmark": "linear_system",
        "seed": "4430",
    }
    expected = {"algorithm": "sakc", "benchmark": "double_well", "first": "0.0001", "second": "0.05", "seed": "21"}
    for path in (
        "results/ablations/runs/sakc/double_well/0.0001__0.05/21",
        "/scratch/sp ace/results/ablations/runs/sakc/double_well/0.0001__0.05/21/",
        "./results//ablations/runs/sakc/double_well/0.0001__0.05/21",
    ):
        assert helpers.path_wildcards(AB_RUN, path) == expected
    with pytest.raises(Refused, match="is not of the form"):
        helpers.path_wildcards(AB_RUN, "results/ablations/runs/sakc/double_well/0.0001_0.05/21")


@pytest.mark.parametrize("a, b", [(1, "1"), (4430, "4430"), (0.0005, "0.0005"), (71, "71"), (1.0, 1), ("x", "x")])
def test_fields_are_compared_as_numbers(a, b):
    assert helpers.same_number(a, b) and helpers.same_number(b, a)


@pytest.mark.parametrize("a, b", [(1, "21"), (0.0005, "0.005"), (9999, "7"), ("x", "y"), (None, "1"), ("x", 1)])
def test_different_fields_are_told_apart(a, b):
    assert not helpers.same_number(a, b) and not helpers.same_number(b, a)


STEPS = [199, 399, 599]


def sakc_frame(points, seeds=(1, 21), steps=STEPS):
    """Data frame and runs of SAKC on the Lorenz system as the ablation workflow has them."""
    frame, runs = {}, {}
    for v_lr, q_lr in points:
        for seed in seeds:
            name = f"Lorenz-v0__soft_actor_koopman_critic__{seed}__{float(v_lr)}__{float(q_lr)}__17900{len(frame)}"
            frame[name] = {"seed": seed, "v_lr": float(v_lr), "q_lr": float(q_lr), "steps": list(steps)}
            run_dir = f"res/ablations/runs/sakc/lorenz/{v_lr}__{q_lr}/{seed}"
            runs[name] = (run_dir, {"seed": str(seed), "v_lr": v_lr, "q_lr": q_lr})
    return frame, runs


def test_a_data_frame_of_the_runs_of_its_jobs_has_no_problem():
    frame, runs = sakc_frame([("0.0001", "0.0001"), ("0.0005", "0.0001"), ("0.05", "0.01")])
    assert helpers.frame_problems(frame, runs) == []
    assert helpers.frame_problems({}, {}) == []


def test_a_run_folder_in_the_directory_of_another_grid_point_is_named():
    # the verifiers: a run of (0.05, 0.05) in the directory of (0.0005, 0.0001) entered the table of the wrong point
    frame, runs = sakc_frame([("0.0001", "0.0001"), ("0.0005", "0.0001")])
    name = list(frame)[-1]
    frame[name].update(v_lr=0.05, q_lr=0.05)
    problems = helpers.frame_problems(frame, runs)
    assert len(problems) == 2 and all(
        p.startswith("res/ablations/runs/sakc/lorenz/0.0005__0.0001/21: ") for p in problems
    )
    assert f"its run folder {name} is a run with v_lr 0.05, and the directory is that of v_lr 0.0005." in problems[0]
    assert "is a run with q_lr 0.05, and the directory is that of q_lr 0.0001." in problems[1]


def test_a_run_folder_in_the_directory_of_another_seed_is_named():
    # the verifiers: a run with seed 41 in the directory of seed 1, and a run of the tuned configuration with seed
    # 8801 and values off the grid in the directory of seed 21, both passed
    frame, runs = sakc_frame([("0.0001", "0.0001"), ("0.0005", "0.0001")])
    names = list(frame)
    frame[names[2]]["seed"] = 41
    frame[names[3]].update(seed=8801, v_lr=0.00047001054701930456, q_lr=0.001802061953715088)
    problems = helpers.frame_problems(frame, runs)
    assert [p.split(": ")[0] for p in problems] == [runs[names[2]][0]] + [runs[names[3]][0]] * 3
    assert "is a run with seed 41, and the directory is that of seed 1." in problems[0]
    assert "is a run with seed 8801, and the directory is that of seed 21." in problems[1]
    assert "v_lr 0.00047001054701930456" in problems[2] and "q_lr 0.001802061953715088" in problems[3]


def test_a_run_directory_without_an_entry_and_an_entry_without_a_run_directory_are_named():
    # the verifiers: a run of the double well in a directory of the Lorenz system is not read into the data frame
    frame, runs = sakc_frame([("0.0001", "0.0001")])
    name, other = list(frame)
    del frame[name]
    frame["Lorenz-v0__soft_actor_koopman_critic__61__0.0001__0.0001__1"] = dict(frame[other])
    problems = helpers.frame_problems(frame, runs)
    assert problems == [
        f"{runs[name][0]}: its run folder {name} is not in the data frame, so it is not a run of the benchmark and"
        " the algorithm of this data frame.",
        "The entry Lorenz-v0__soft_actor_koopman_critic__61__0.0001__0.0001__1 of the data frame is not the run"
        " folder of any run directory.",
    ]


def test_the_runs_that_logged_other_steps_than_most_are_named():
    frame, runs = sakc_frame([("0.0001", "0.0001"), ("0.0005", "0.0001"), ("0.001", "0.0001")])
    names = list(frame)
    frame[names[1]]["steps"] = STEPS[:1]  # interrupted
    frame[names[4]]["steps"] = [199, 399, 600]  # the same number of returns, at other steps
    frame[names[5]]["steps"] = []
    assert helpers.frame_problems(frame, runs) == [
        f"{runs[names[1]][0]}: logged 1 returns, the last at step 199; the most common list of steps (3 of 6 runs)"
        " has 3 returns, the last at step 599.",
        f"{runs[names[4]][0]}: logged 3 returns, the last at step 600; the most common list of steps (3 of 6 runs)"
        " has 3 returns, the last at step 599.",
        f"{runs[names[5]][0]}: logged no return.",
    ]
    frame[names[4]]["steps"] = [199, 400, 599]
    assert "logged 3 returns, the last at step 599 at other steps; the most" in helpers.frame_problems(frame, runs)[1]


def test_of_two_equally_common_lists_of_steps_the_longer_one_is_the_common_one():
    # one of two runs was interrupted: the short run is the one that deviates
    frame, runs = sakc_frame([("0.0001", "0.0001")])
    short, full = list(frame)
    frame[short]["steps"] = STEPS[:2]
    (problem,) = helpers.frame_problems(frame, runs)
    assert problem.startswith(f"{runs[short][0]}: logged 2 returns") and "(1 of 2 runs) has 3 returns" in problem


def test_runs_without_a_return_are_all_named():
    frame, runs = sakc_frame([("0.0001", "0.0001")], steps=[])
    assert helpers.frame_problems(frame, runs) == [f"{run_dir}: logged no return." for run_dir, _ in runs.values()]


def write_frame(tmp_path, frame):
    path = tmp_path / "frames" / "sakc_lorenz.json"
    path.parent.mkdir()
    path.write_text(json.dumps(frame))
    log = tmp_path / "sakc_lorenz.log"
    log.write_text("output of the script\n")
    return str(path), str(log)


def test_check_frame_compares_the_entries_with_the_wildcards_of_their_run_directories(tmp_path):
    frame, runs = sakc_frame([("0.0001", "0.0001"), ("0.0005", "0.05")])
    run_dir_of = {name: run_dir for name, (run_dir, _) in runs.items()}
    fields = {"seed": "seed", "v_lr": "first", "q_lr": "second"}
    path, log = write_frame(tmp_path, frame)
    assert helpers.check_frame(path, run_dir_of, AB_RUN, fields, log) is None
    with open(log) as f:
        assert f.read() == "output of the script\n"

    # the same frame is wrong for the directories of another seed, which is the mapping of the episodic returns
    name = list(frame)[0]
    run_dir_of[name] = "res/ablations/runs/sakc/lorenz/0.0001__0.0001/41"
    with pytest.raises(Refused) as error:
        helpers.check_frame(path, run_dir_of, AB_RUN, {"seed": "seed"}, log)
    message = str(error.value)
    assert message.startswith(f"{path}: the data frame does not hold the 4 runs of its jobs as the tables need them.\n")
    assert f"res/ablations/runs/sakc/lorenz/0.0001__0.0001/41: its run folder {name} is a run with seed 1" in message
    with open(log) as f:
        assert f.read() == "output of the script\n" + message + "\n"


def test_check_frame_writes_every_problem_into_the_log_and_the_first_ones_into_the_error(tmp_path):
    points = [(v_lr, q_lr) for v_lr in ("0.0001", "0.0005", "0.001") for q_lr in ("0.0001", "0.0005")]
    frame, runs = sakc_frame(points, seeds=(1, 21, 41, 61, 81))
    for entry in list(frame.values())[:12]:
        entry["steps"] = STEPS[:1]
    path, log = write_frame(tmp_path, frame)
    run_dir_of = {name: run_dir for name, (run_dir, _) in runs.items()}
    with pytest.raises(Refused) as error:
        helpers.check_frame(path, run_dir_of, AB_RUN, {"seed": "seed", "v_lr": "first", "q_lr": "second"}, log)
    lines = str(error.value).split("\n")
    assert len(lines) == 13 and lines[-2] == f"... and 2 more, all listed in {log}"
    # the last line says what to do about the tables of an earlier data frame, which Snakemake leaves in place
    assert lines[-1].startswith("The data frame is removed.") and "name the data frame as a target" in lines[-1]
    with open(log) as f:
        logged = f.read().split("\n")
    assert logged[1:12] == lines[:11] and len(logged) == 16 and logged[13].startswith(list(run_dir_of.values())[11])
    assert logged[14] == lines[-1]
