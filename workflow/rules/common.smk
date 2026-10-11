# Shared by all workflows: where things are, how the code of the project is called, and what needs
# `config`, `shell` or `workflow`. The helpers that need none of them are in workflow/helpers.py:
# the checks of the settings, the selections, run folders and the check of a data frame. See
# workflow/README.md.

import json
import os
import shlex
import sys

# helpers.py lies next to the Snakefile.
sys.path.insert(0, workflow.basedir)
from helpers import (
    WorkflowError,
    alternatives,
    check_frame,
    choice,
    known_keys,
    unchanged,
    run_folders,
    script_options,
    selection,
    whole_number,
    whole_numbers,
)

# Root of the repository, from whatever directory Snakemake is started.
REPO_ROOT = os.path.dirname(workflow.basedir)

# Folder with the JSON file of each workflow: configurations/<workflow>.json, read with `configfile:`.
CONFIG_DIR = os.path.join(REPO_ROOT, "configurations")

# Each workflow writes below RESULTS/<workflow>/. A relative path is taken from the working directory
# of Snakemake.
RESULTS = str(config.get("results_dir", "results"))

# Command that starts the Python of the project environment. Snakemake runs outside of that
# environment, since the project is pinned to Python 3.10 and Snakemake needs 3.11 or newer.
# `--no-sync` uses the environment as it is, so it has to exist (`uv sync`).
PYTHON = config.get(
    "python", f"uv run --project {shlex.quote(REPO_ROOT)} --no-sync python"
)

# The script that builds a data frame. A job that builds a data frame or a table takes the script
# it calls as an input, so that it is made again when the script changes.
FRAME_SCRIPT = os.path.join(REPO_ROOT, "koopmanrl_utils", "dataframe_creator.py")

# Formats of a table, which are the endings of its file: comma-separated with named columns, or the
# layout of the tables that the figure sources of the paper read.
TABLE_FORMATS = ("csv", "dat")


# The wildcards that are constrained for all workflows: a seed is a whole number in one spelling,
# and {ext} the format of a table. A workflow constrains its other wildcards in its own rules, with
# its own names.
wildcard_constraints:
    seed=r"0|[1-9][0-9]*",
    ext=alternatives(TABLE_FORMATS),


def no_target(wildcards):
    """Stops a call without a target with the list of the targets; the input of the default target."""
    raise WorkflowError(
        "Name a target: episodic_returns, ablations, tsne, or all for the three of them."
        " Add -n to list the jobs of a target without running them; all is 1,935 training runs."
    )


def defaults(name):
    """The settings of a workflow as configurations/<name>.json lists them: the keys it has."""
    with open(os.path.join(CONFIG_DIR, f"{name}.json")) as f:
        return json.load(f)[name]


def registry():
    """The algorithms and the benchmarks of configurations/episodic_returns.json, which all workflows use.

    They are not settings, apart from the memory of an algorithm: the launchers read them from the
    file. Call this after `configfile:` has loaded the file.
    """
    listed = defaults("episodic_returns")
    given = config["episodic_returns"]
    unchanged(
        given.get("benchmarks"),
        listed["benchmarks"],
        "episodic_returns: benchmarks",
    )
    unchanged(
        given.get("algorithms"),
        listed["algorithms"],
        "episodic_returns: algorithms",
        settable=("mem_mb",),
    )
    return given["algorithms"], given["benchmarks"]


def script_file(module):
    """Path of the file of a module of the repository, as an input of the jobs that call it."""
    return os.path.join(REPO_ROOT, *module.split(".")) + ".py"


def data_frame(run_dirs, pattern, fields, runs_dir, options, frame, log):
    """Write the data frame of the runs made in `run_dirs` with koopmanrl_utils.dataframe_creator.

    `options` are its --mode, --system and --rl_algo. `pattern` is the path pattern of a run
    directory, and `fields` maps the fields of an entry of the data frame to the wildcards of
    the pattern they are compared with. Fails unless the data frame holds one entry per run
    directory, with the fields of the job of that directory, and all runs logged the same
    steps, which the scripts that process a data frame assume. What is wrong is appended to
    `log`, with the run directories it concerns.
    """
    storage_dir, output_file = os.path.split(frame)
    try:
        with run_folders(run_dirs, runs_dir) as (target_dir, run_dir_of):
            shell(
                "{PYTHON} -m koopmanrl_utils.dataframe_creator {options} --target_dir {target_dir:q}"
                " --storage_dir {storage_dir:q} --output_file {output_file:q} > {log:q} 2>&1"
            )
    except WorkflowError as error:
        # the run directories are not as a run leaves them, and the script was not called
        with open(log, "w") as f:
            f.write(f"{error}\n")
        raise
    check_frame(frame, run_dir_of, pattern, fields, log)
