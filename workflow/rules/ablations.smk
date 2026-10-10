# Ablations: the tables behind the two ablation figures of the paper.
#
#     one run per algorithm, benchmark, grid point and seed    (ablations_run)
#  -> one data frame per algorithm and benchmark               (ablations_frame)
#  -> one table per algorithm, benchmark and smoothing window  (ablations_table)
#
# The steps are those of koopmanrl_utils/ABLATIONS.md. The grids and the seeds are listed in
# configurations/ablations.json, which koopmanrl_utils/run_ablations.py reads too. Its algorithms are
# named as in configurations/episodic_returns.json, where their modules, the folders of their logs,
# their memory and the benchmarks are listed; that file is loaded here as well, so this rule file
# does not depend on the one of the episodic returns.


configfile: os.path.join(CONFIG_DIR, "episodic_returns.json")
configfile: os.path.join(CONFIG_DIR, "ablations.json")


# A key that the JSON file does not have is refused: it would change nothing.
AB_DEFAULTS = defaults("ablations")
AB = known_keys(config["ablations"], AB_DEFAULTS, "ablations")
# The grids, and how the data frames and tables of an algorithm are made, are not settings:
# koopmanrl_utils/run_ablations.py reads them from the file.
unchanged(AB["algorithms"], AB_DEFAULTS["algorithms"], "ablations: algorithms")
AB_ALL_ALGORITHMS, AB_BENCHMARKS = registry()

# Wildcards: {algorithm} is a key of AB["algorithms"] (skvi, sakc), {benchmark} the name of a benchmark
# in file names (linear_system, ...), {table_name} the name of an algorithm in the tables (SKVI, SAKC),
# {first} and {second} the values of the two swept flags of the algorithm, {window} the number of
# returns that a table takes from the end of every run, {ext} the format of a table (csv, dat).
AB_REGISTRY = {name: AB_ALL_ALGORITHMS[name] for name in AB["algorithms"]}
AB_ENV_ID = {benchmark: env_id for env_id, benchmark in AB_BENCHMARKS.items()}
AB_ALGORITHM = {
    algorithm["table_name"]: name for name, algorithm in AB_REGISTRY.items()
}

# algorithm: {wildcard: (swept flag, its values)}, each value as str() of the number in the JSON file.
# This one string stands in the path of a run, is passed to the launcher, and is what the launcher
# passes on and the algorithm prints into the name of its run, since both print the number with
# str() again.
AB_GRID = {
    name: {
        axis: (flag, [str(value) for value in values])
        for axis, (flag, values) in zip(("first", "second"), algorithm["grid"].items())
    }
    for name, algorithm in AB["algorithms"].items()
}

# algorithm: {field of an entry of its data frames: the wildcard of a run directory it is compared
# with}. The data frame job fails when a run folder lies in the directory of another seed or grid point.
AB_FIELDS = {
    name: {"seed": "seed", **{flag: axis for axis, (flag, _) in grid.items()}}
    for name, grid in AB_GRID.items()
}

# (algorithm, first, second): the two swept flags with these values, for the command line of the
# launcher. Only the points of the grid of an algorithm have an entry, so a path that joins an
# algorithm with a value of the other grid is refused when the jobs are worked out.
AB_POINT = {
    (name, first, second): "--{} {} --{} {}".format(
        grid["first"][0], first, grid["second"][0], second
    )
    for name, grid in AB_GRID.items()
    for first in grid["first"][1]
    for second in grid["second"][1]
}

# {first} and {second} match the values of the grids in the spelling of AB_GRID only, and {window} a
# whole number from one without leading zeros.
AB_CONSTRAINTS = {
    "algorithm": alternatives(AB["algorithms"]),
    "benchmark": alternatives(AB_ENV_ID),
    "table_name": alternatives(AB_ALGORITHM),
    "first": alternatives(
        {value: None for grid in AB_GRID.values() for value in grid["first"][1]}
    ),
    "second": alternatives(
        {value: None for grid in AB_GRID.values() for value in grid["second"][1]}
    ),
    "window": r"[1-9][0-9]*",
}

AB_DIR = os.path.join(RESULTS, "ablations")
AB_RUN = os.path.join(
    AB_DIR, "runs", "{algorithm}", "{benchmark}", "{first}__{second}", "{seed}"
)
AB_FRAME = os.path.join(AB_DIR, "frames", "{algorithm}_{benchmark}.json")
AB_TABLE = os.path.join(
    AB_DIR,
    "tables",
    "{table_name}",
    "window_{window}",
    "{benchmark}_ablation.{ext}",
)
AB_LOGS = os.path.join(AB_DIR, "logs")

# Settings. A value given with --config arrives as a string and a value of a file given with
# --configfile with its type; the helpers take both, and compare the values of a grid as strings.
AB_STEPS = whole_number(
    AB["total_timesteps"], "ablations: total_timesteps", 1, or_null=True
)
AB_FORMAT = choice(AB["table_format"], TABLE_FORMATS, "ablations: table_format")
AB_SEEDS = whole_numbers(AB["seeds"], "ablations: seeds")
AB_WINDOWS = whole_numbers(AB["smoothing_windows"], "ablations: smoothing_windows", 1)

# algorithm: {wildcard: the values of its swept flag that the data frames are built from}
AB_ONLY = known_keys(
    AB["only_values"],
    [flag for grid in AB_GRID.values() for flag, _ in grid.values()],
    "ablations: only_values",
)
AB_SELECTED = {
    name: {
        axis: selection(AB_ONLY.get(flag), values, f"ablations: only_values: {flag}")
        for axis, (flag, values) in grid.items()
    }
    for name, grid in AB_GRID.items()
}

# algorithm: the memory that a run declares, which is listed with the episodic returns
AB_MEMORY = {
    name: whole_number(
        algorithm["mem_mb"], f"episodic_returns: algorithms: {name}: mem_mb", 1
    )
    for name, algorithm in AB_REGISTRY.items()
}


rule ablations:
    input:
        expand(
            AB_TABLE,
            table_name=[
                AB_REGISTRY[name]["table_name"]
                for name in selection(
                    AB["only_algorithms"],
                    AB["algorithms"],
                    "ablations: only_algorithms",
                )
            ],
            benchmark=[
                AB_BENCHMARKS[env_id]
                for env_id in selection(
                    AB["only_benchmarks"],
                    AB_BENCHMARKS,
                    "ablations: only_benchmarks",
                )
            ],
            window=AB_WINDOWS,
            ext=AB_FORMAT,
        ),


# One run, in a directory of its own, made by the launcher: the command line of a run is built there.
# The launcher sends the console output of the run to a file in the run directory, which Snakemake
# removes when the run fails, so that output is appended to the log of the job first.
rule ablations_run:
    output:
        directory(AB_RUN),
    log:
        os.path.join(
            AB_LOGS,
            "run",
            "{algorithm}",
            "{benchmark}",
            "{first}__{second}",
            "{seed}.log",
        ),
    wildcard_constraints:
        **AB_CONSTRAINTS,
    params:
        env_id=lambda wildcards: AB_ENV_ID[wildcards.benchmark],
        point=lambda wildcards: AB_POINT[
            wildcards.algorithm, wildcards.first, wildcards.second
        ],
        total_timesteps=("" if AB_STEPS is None else f"--total_timesteps {AB_STEPS}"),
    threads: 1
    resources:
        mem_mb=lambda wildcards: AB_MEMORY[wildcards.algorithm],
    shell:
        "{PYTHON} -m koopmanrl_utils.run_ablations"
        " --algorithms {wildcards.algorithm} --environments {params.env_id} --seeds {wildcards.seed}"
        " {params.point} {params.total_timesteps} --output_dir {output:q} > {log:q} 2>&1"
        " || {{ cat {output:q}/logs/{wildcards.algorithm}__{params.env_id}__{wildcards.first}__{wildcards.second}__{wildcards.seed}.log >> {log:q}; exit 1; }}"


# The script is an input, so that a change of it makes the data frames again; the same holds for
# the tables below. The algorithms and the launcher are not inputs of the runs: a change of them
# does not make the runs again.
rule ablations_frame:
    input:
        runs=lambda wildcards: expand(
            AB_RUN,
            **AB_SELECTED[wildcards.algorithm],
            seed=AB_SEEDS,
            allow_missing=True,
        ),
        script=FRAME_SCRIPT,
    output:
        AB_FRAME,
    log:
        os.path.join(AB_LOGS, "frame", "{algorithm}_{benchmark}.log"),
    wildcard_constraints:
        **AB_CONSTRAINTS,
    params:
        runs_dir=lambda wildcards: AB_REGISTRY[wildcards.algorithm]["runs_dir"],
        options=lambda wildcards: "--mode {} --system {}".format(
            AB["algorithms"][wildcards.algorithm]["frame_mode"],
            AB_ENV_ID[wildcards.benchmark],
        ),
    threads: 1
    resources:
        mem_mb=1000,
    run:
        data_frame(
            input.runs,
            AB_RUN,
            AB_FIELDS[wildcards.algorithm],
            params.runs_dir,
            params.options,
            output[0],
            log[0],
        )


# The window is part of the path of a table and enters this job only, so another list of windows
# makes other tables from the same data frames.
rule ablations_table:
    input:
        frame=lambda wildcards: AB_FRAME.format(
            algorithm=AB_ALGORITHM[wildcards.table_name],
            benchmark=wildcards.benchmark,
        ),
        script=lambda wildcards: script_file(
            AB["algorithms"][AB_ALGORITHM[wildcards.table_name]]["table_module"]
        ),
    output:
        AB_TABLE,
    log:
        os.path.join(
            AB_LOGS, "table", "{table_name}_{benchmark}_window_{window}.{ext}.log"
        ),
    wildcard_constraints:
        **AB_CONSTRAINTS,
    params:
        module=lambda wildcards: AB["algorithms"][AB_ALGORITHM[wildcards.table_name]][
            "table_module"
        ],
        root_dir=lambda wildcards, input: os.path.dirname(input.frame),
        data_frame=lambda wildcards, input: os.path.splitext(
            os.path.basename(input.frame)
        )[0],
        output_dir=lambda wildcards, output: os.path.dirname(output[0]),
        output_name=lambda wildcards, output: os.path.basename(output[0]),
    threads: 1
    resources:
        mem_mb=1000,
    shell:
        "{PYTHON} -m {params.module} --root_dir {params.root_dir:q}"
        " --data_frame {params.data_frame:q} --output_dir {params.output_dir:q}"
        " --output_name {params.output_name:q} --smoothing_window {wildcards.window} > {log:q} 2>&1"
