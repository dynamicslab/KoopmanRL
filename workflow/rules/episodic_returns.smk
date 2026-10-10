# Episodic returns: the tables behind the episodic-return figures of the paper.
#
#     one run per algorithm, benchmark and seed  (episodic_returns_run)
#  -> one data frame per algorithm and benchmark (episodic_returns_frame)
#  -> one table per algorithm and benchmark      (episodic_returns_table)
#
# The steps are those of koopmanrl_utils/EPISODIC_RETURNS.md. What is run is listed in
# configurations/episodic_returns.json, which koopmanrl_utils/run_optimized_experiments.py reads too.


configfile: os.path.join(CONFIG_DIR, "episodic_returns.json")


# A key that the JSON file does not have is refused, here and below: it would change nothing.
ER_DEFAULTS = defaults("episodic_returns")
ER = known_keys(config["episodic_returns"], ER_DEFAULTS, "episodic_returns")

# Wildcards: {algorithm} is a key of ER["algorithms"] (lqr, ...), {benchmark} the name of a benchmark
# in file names (linear_system, ...), {table_name} the name of an algorithm in the tables (LQR, ...),
# {ext} the format of a table (csv, dat).
ER_ENV_ID = {benchmark: env_id for env_id, benchmark in ER["benchmarks"].items()}
ER_ALGORITHM = {
    algorithm["table_name"]: name
    for name, algorithm in known_keys(
        ER["algorithms"], ER_DEFAULTS["algorithms"], "episodic_returns: algorithms"
    ).items()
}
ER_CONSTRAINTS = {
    "algorithm": alternatives(ER["algorithms"]),
    "benchmark": alternatives(ER_ENV_ID),
    "table_name": alternatives(ER_ALGORITHM),
}

ER_DIR = os.path.join(RESULTS, "episodic_returns")
ER_RUN = os.path.join(ER_DIR, "runs", "{algorithm}", "{benchmark}", "{seed}")
ER_FRAME = os.path.join(ER_DIR, "frames", "{algorithm}_{benchmark}.json")
ER_TABLE = os.path.join(
    ER_DIR, "tables", "{benchmark}", "{table_name}_{benchmark}.{ext}"
)
ER_LOGS = os.path.join(ER_DIR, "logs")

# Settings. A value given with --config arrives as a string and a value of a file given with
# --configfile with its type; the helpers take both.
ER_STEPS = whole_number(
    ER["total_timesteps"], "episodic_returns: total_timesteps", 1, or_null=True
)
ER_FORMAT = choice(ER["table_format"], TABLE_FORMATS, "episodic_returns: table_format")

# benchmark: its seeds, as numbers, so that a seed has one spelling in the path of its run
ER_SEEDS = {
    env_id: whole_numbers(seeds, f"episodic_returns: seeds: {env_id}")
    for env_id, seeds in known_keys(
        ER["seeds"], ER["benchmarks"], "episodic_returns: seeds"
    ).items()
}

# algorithm: the memory that a run declares
ER_MEMORY = {
    name: whole_number(
        known_keys(
            algorithm,
            ER_DEFAULTS["algorithms"][name],
            f"episodic_returns: algorithms: {name}",
        )["mem_mb"],
        f"episodic_returns: algorithms: {name}: mem_mb",
        1,
    )
    for name, algorithm in ER["algorithms"].items()
}

# The algorithms and the benchmarks are not settings, apart from the memory of an algorithm.
registry()


rule episodic_returns:
    input:
        expand(
            ER_TABLE,
            table_name=[
                ER["algorithms"][name]["table_name"]
                for name in selection(
                    ER["only_algorithms"],
                    ER["algorithms"],
                    "episodic_returns: only_algorithms",
                )
            ],
            benchmark=[
                ER["benchmarks"][env_id]
                for env_id in selection(
                    ER["only_benchmarks"],
                    ER["benchmarks"],
                    "episodic_returns: only_benchmarks",
                )
            ],
            ext=ER_FORMAT,
        ),


# One run, in a directory of its own, made by the launcher: the command line of a run is built there.
# The launcher sends the console output of the run to a file in the run directory, which Snakemake
# removes when the run fails, so that output is appended to the log of the job first.
rule episodic_returns_run:
    output:
        directory(ER_RUN),
    log:
        os.path.join(ER_LOGS, "run", "{algorithm}", "{benchmark}", "{seed}.log"),
    wildcard_constraints:
        **ER_CONSTRAINTS,
    params:
        env_id=lambda wildcards: ER_ENV_ID[wildcards.benchmark],
        total_timesteps=("" if ER_STEPS is None else f"--total_timesteps {ER_STEPS}"),
    threads: 1
    resources:
        mem_mb=lambda wildcards: ER_MEMORY[wildcards.algorithm],
    shell:
        "{PYTHON} -m koopmanrl_utils.run_optimized_experiments"
        " --algorithms {wildcards.algorithm} --environments {params.env_id} --seeds {wildcards.seed}"
        " {params.total_timesteps} --output_dir {output:q} > {log:q} 2>&1"
        " || {{ cat {output:q}/logs/{wildcards.algorithm}__{params.env_id}__{wildcards.seed}.log >> {log:q}; exit 1; }}"


# The script is an input, so that a change of it makes the data frames again; the same holds for
# the tables below. The algorithms and the launcher are not inputs of the runs: a change of them
# does not make the runs again.
rule episodic_returns_frame:
    input:
        runs=lambda wildcards: expand(
            ER_RUN,
            seed=ER_SEEDS[ER_ENV_ID[wildcards.benchmark]],
            allow_missing=True,
        ),
        script=FRAME_SCRIPT,
    output:
        ER_FRAME,
    log:
        os.path.join(ER_LOGS, "frame", "{algorithm}_{benchmark}.log"),
    wildcard_constraints:
        **ER_CONSTRAINTS,
    params:
        runs_dir=lambda wildcards: ER["algorithms"][wildcards.algorithm]["runs_dir"],
        options=lambda wildcards: "--mode Episodic_Returns --system {} --rl_algo {}".format(
            ER_ENV_ID[wildcards.benchmark],
            ER["algorithms"][wildcards.algorithm]["module"].rsplit(".", 1)[1],
        ),
    threads: 1
    resources:
        mem_mb=1000,
    run:
        data_frame(
            input.runs,
            ER_RUN,
            {"seed": "seed"},
            params.runs_dir,
            params.options,
            output[0],
            log[0],
        )


rule episodic_returns_table:
    input:
        frame=lambda wildcards: ER_FRAME.format(
            algorithm=ER_ALGORITHM[wildcards.table_name],
            benchmark=wildcards.benchmark,
        ),
        script=script_file("koopmanrl_utils.process_episodic_returns"),
    output:
        ER_TABLE,
    log:
        os.path.join(ER_LOGS, "table", "{table_name}_{benchmark}.{ext}.log"),
    wildcard_constraints:
        **ER_CONSTRAINTS,
    params:
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
        "{PYTHON} -m koopmanrl_utils.process_episodic_returns --root_dir {params.root_dir:q}"
        " --data_frame {params.data_frame:q} --output_dir {params.output_dir:q}"
        " --output_name {params.output_name:q} --deterministic_bootstrap > {log:q} 2>&1"
