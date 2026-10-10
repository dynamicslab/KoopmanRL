"""Helpers of the Snakemake workflows that need nothing of Snakemake: the checks of the settings, the command line
of a mapping of arguments, the folder of run folders that a data frame is built from, and the check of a data frame.

`rules/common.smk` imports this module; `tests/test_workflow_helpers.py` loads it in the project environment, which
has no Snakemake. A check that fails raises `WorkflowError` (`ValueError` where Snakemake is not installed), with a
message that names the setting, what it takes and what was given.
"""

import collections
import contextlib
import json
import os
import re
import shlex
import tempfile

try:
    from snakemake.exceptions import WorkflowError
except ImportError:  # the project environment: Python 3.10, where Snakemake cannot be installed
    WorkflowError = ValueError


def alternatives(names):
    """Wildcard constraint that matches exactly the given names."""
    return "|".join(re.escape(name) for name in names)


def is_null(value):
    """Whether a setting is null. A value given with --config arrives as a string, so the words null and none
    are read as null."""
    return value is None or (isinstance(value, str) and value.lower() in ("null", "none"))


def refuse(what, takes, given):
    raise WorkflowError(f"{what} takes {takes}. Got: {given!r}.")


def known_keys(given, known, what):
    """The mapping `given`, after a check that all of its keys are in `known`."""
    takes = f"a mapping with keys from: {', '.join(known)}"
    if not isinstance(given, dict):
        refuse(what, takes, given)
    unknown = [str(key) for key in given if key not in known]
    if unknown:
        raise WorkflowError(f"{what} does not have: {', '.join(unknown)}. It takes {takes}.")
    return given


def unchanged(given, listed, what, settable=()):
    """The mapping `given`, after a check that it is the mapping `listed` of the JSON file.

    For the entries of a JSON file that are not settings: the scripts of koopmanrl_utils/ read them from the file,
    so another value given with --config would reach the workflow and not the scripts. `settable` are the keys of
    the entries of the mapping that may differ from the file.
    """

    def fixed(mapping):
        if not isinstance(mapping, dict):
            return mapping
        return {
            name: {key: value for key, value in entry.items() if key not in settable}
            if isinstance(entry, dict)
            else entry
            for name, entry in mapping.items()
        }

    if fixed(given) != fixed(listed):
        but = f", apart from {', '.join(settable)} of an entry" if settable else ""
        raise WorkflowError(
            f"{what} is not a setting: the scripts of koopmanrl_utils/ read it from the JSON file, so it has to be"
            f" as the file lists it{but}. Got: {given!r}."
        )
    return given


def choice(value, known, what, or_null=False):
    """The name `value`, after a check that it is one of `known`; null stays None where it is allowed."""
    if or_null and is_null(value):
        return None
    if not isinstance(value, str) or value not in known:
        refuse(what, ("null or " if or_null else "") + f"one of: {', '.join(known)}", value)
    return value


def selection(only, known, what):
    """The names of `known` that are listed in `only`, in the order of `known`; all of them if `only` is null.

    `only` is a list that names something of `known` at least once and nothing twice. Its entries are compared as
    strings, since a number given with --config arrives as a string and a number of a file as a number.
    """
    if is_null(only):
        return list(known)
    takes = f"null or a list of names from: {', '.join(known)}, each of them once"
    if not isinstance(only, list) or not only:
        refuse(what, takes, only)
    names = [str(name) for name in only]
    if len(set(names)) != len(names) or any(name not in known for name in names):
        refuse(what, takes, only)
    return [name for name in known if name in names]


def whole_number(value, what, minimum=0, or_null=False):
    """The whole number `value` as an int; null stays None where it is allowed.

    Takes an int or a string of digits, which is how a number given with --config arrives. Refuses everything else
    (1.5, -5, true, a word) and a number below `minimum`.
    """
    if or_null and is_null(value):
        return None
    text = str(value)
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", text) or int(text) < minimum:
        refuse(what, ("null or " if or_null else "") + f"a whole number from {minimum}", value)
    return int(text)


def whole_numbers(values, what, minimum=0):
    """The whole numbers of the list `values` as ints: at least one, and none of them twice.

    They are compared as numbers, so 1 and 01 are the same number, and returned as numbers, so that a number has
    one spelling in a path.
    """
    takes = f"a list of whole numbers from {minimum}, each of them once"
    if not isinstance(values, list) or not values:
        refuse(what, takes, values)
    try:
        numbers = [whole_number(value, what, minimum) for value in values]
    except WorkflowError:
        refuse(what, takes, values)
    if len(set(numbers)) != len(numbers):
        refuse(what, takes, values)
    return numbers


def script_options(options, what, refused=(), known=None):
    """Command line of a mapping of the arguments of a script to their values.

    The arguments are written in the order of their names: a list as `--name a b`, true as
    `--name`, and nothing for false, null and an empty list, which leave the default of the
    script. A value given with --config arrives as a string, so the words true, false and null
    are read as these values. `refused` are the names that may not be given in this mapping;
    with `known`, every other name is refused as well.
    """
    if not isinstance(options, dict):
        refuse(what, "a mapping of arguments to values", options)
    unusable = [
        str(name)
        for name in options
        if name in refused
        or (known is not None and name not in known)
        or not re.fullmatch(r"[a-z][a-z0-9_]*", str(name))
    ]
    if unusable:
        takes = "" if known is None else f" It takes: {', '.join(known)}."
        raise WorkflowError(f"{what} does not take: {', '.join(unusable)}.{takes}")
    words = []
    for name in sorted(options):
        value = options[name]
        values = [str(v) for v in (value if isinstance(value, list) else [value])]
        if values == [] or values[0].lower() in ("false", "null", "none"):
            continue
        words.append(f"--{name}")
        if values[0].lower() != "true":
            words += [shlex.quote(v) for v in values]
    return " ".join(words)


@contextlib.contextmanager
def run_folders(run_dirs, runs_dir):
    """Temporary folder with a link to the run folder of each run directory, and whose folder each link is.

    Every run is made in a directory of its own, in which the algorithm writes
    `<runs_dir>/<run name>/`. The scripts that read TensorBoard logs take a folder that
    holds run folders, which is what this gives, together with a mapping of the name of
    each run folder to its run directory.
    """
    with tempfile.TemporaryDirectory() as folder:
        run_dir_of = {}
        for run_dir in run_dirs:
            parent = os.path.join(os.path.abspath(run_dir), runs_dir)
            names = [entry.name for entry in os.scandir(parent) if entry.is_dir()] if os.path.isdir(parent) else []
            if len(names) != 1:
                raise WorkflowError(f"Expected one run folder in {parent}, found {len(names)}.")
            if names[0] in run_dir_of:
                raise WorkflowError(
                    f"The run folder {names[0]} lies in two run directories: {run_dir_of[names[0]]} and {run_dir}."
                )
            os.symlink(os.path.join(parent, names[0]), os.path.join(folder, names[0]))
            run_dir_of[names[0]] = run_dir
        yield folder, run_dir_of


def path_wildcards(pattern, path):
    """The values of the wildcards of the path pattern `pattern` in `path`, which was made from it.

    Only the components from the first one with a wildcard on are compared, from the end of the path, so the
    spelling of the directories above them does not matter.
    """
    components = pattern.split(os.sep)
    first = next(index for index, component in enumerate(components) if "{" in component)
    values = {}
    for component, part in zip(components[first:], os.path.normpath(path).split(os.sep)[first - len(components) :]):
        pieces = re.split(r"\{(\w+)\}", component)
        regex = "".join(f"(?P<{p}>.+?)" if index % 2 else re.escape(p) for index, p in enumerate(pieces))
        match = re.fullmatch(regex, part)
        if match is None:
            raise WorkflowError(f"The path {path} is not of the form {pattern}.")
        values.update(match.groupdict())
    return values


def same_number(a, b):
    """Whether two values are the same number (1, 1.0 and "1"; 0.0005 and "0.0005"), or the same string."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def logged(steps):
    return f"{len(steps)} returns, the last at step {steps[-1]}" if steps else "no return"


def frame_problems(frame, runs):
    """What is wrong with a data frame: a list of sentences that name the run directories, empty if nothing is.

    `frame` is the data frame: the name of a run folder -> the fields that
    koopmanrl_utils/dataframe_creator.py read from that name, and the `steps` at which the run logged a return.
    `runs` is what the jobs made: the name of a run folder -> (its run directory, {field: the value of the job
    that made the directory}). The scripts that process a data frame assume what is checked here:

    - every run directory has its entry, and every entry its run directory;
    - the fields of an entry are those of the job of its directory (the seed, and for the ablations the two
      swept values), compared as numbers: a run folder that lies in the directory of another seed or grid point
      would enter the table as a run of that seed or grid point;
    - every run logged a return, and all of them at the same steps. The runs that deviate from the most common
      list of steps are named; among equally common lists the longest counts as the common one.
    """
    problems = []
    for name, (run_dir, _) in runs.items():
        if name not in frame:
            problems.append(
                f"{run_dir}: its run folder {name} is not in the data frame, so it is not a run of the benchmark"
                " and the algorithm of this data frame."
            )
    for name in frame:
        if name not in runs:
            problems.append(f"The entry {name} of the data frame is not the run folder of any run directory.")
    present = {name: frame[name] for name in runs if name in frame}
    for name, entry in present.items():
        run_dir, fields = runs[name]
        for field, value in fields.items():
            if field not in entry or not same_number(entry[field], value):
                problems.append(
                    f"{run_dir}: its run folder {name} is a run with {field} {entry.get(field)}, and the directory"
                    f" is that of {field} {value}."
                )
    steps = {name: tuple(entry["steps"]) for name, entry in present.items()}
    counts = collections.Counter(listed for listed in steps.values() if listed)
    common = max(counts, key=lambda listed: (counts[listed], len(listed))) if counts else ()
    for name, listed in steps.items():
        if not listed:
            problems.append(f"{runs[name][0]}: logged no return.")
        elif listed != common:
            other = " at other steps" if logged(listed) == logged(common) else ""
            problems.append(
                f"{runs[name][0]}: logged {logged(listed)}{other}; the most common list of steps"
                f" ({counts[common]} of {len(steps)} runs) has {logged(common)}."
            )
    return problems


def check_frame(frame, run_dir_of, pattern, fields, log, shown=10):
    """Fail unless the data frame in the file `frame` holds the runs of the jobs it was built from.

    `run_dir_of` is the mapping of `run_folders`, `pattern` the path pattern of a run directory and `fields` a
    mapping of the fields of an entry to the wildcards of the pattern they are compared with. What is wrong is
    appended to the log file `log` in full, and the error names the first `shown` of the problems.
    """
    with open(frame) as f:
        entries = json.load(f)
    runs = {}
    for name, run_dir in run_dir_of.items():
        wildcards = path_wildcards(pattern, run_dir)
        runs[name] = (run_dir, {field: wildcards[wildcard] for field, wildcard in fields.items()})
    problems = frame_problems(entries, runs)
    if not problems:
        return
    head = f"{frame}: the data frame does not hold the {len(runs)} runs of its jobs as the tables need them."
    # Snakemake removes the data frame of the failed job and keeps the tables of an earlier one, which it takes
    # to be up to date as long as the data frame is missing
    hint = (
        "The data frame is removed. Tables made from an earlier data frame stay and are not made again by"
        " themselves: once the runs are corrected, name the data frame as a target together with its tables."
    )
    with open(log, "a") as f:
        f.write("\n".join([head, *problems, hint]) + "\n")
    more = [f"... and {len(problems) - shown} more, all listed in {log}"] if len(problems) > shown else []
    raise WorkflowError("\n".join([head, *problems[:shown], *more, hint]))
