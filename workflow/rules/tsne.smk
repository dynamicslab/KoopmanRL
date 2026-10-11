# t-SNE: the tables behind the t-SNE figure of the Koopman tensors of the paper.
#
#     the tensors of one benchmark, identified and stored    (tsne_identify)
#  -> one embedding of the stored tensors of all benchmarks  (tsne_embed)
#
# Both steps are calls of koopmanrl_utils/tsne_koopman_tensor.py, which koopmanrl_utils/TSNE.md
# describes: one with --identify_only per benchmark, one with --embed_only. They are two steps so
# that another embedding is made from the stored tensors, and the tensors of one benchmark without
# those of the others. The benchmarks are those of configurations/episodic_returns.json, which is
# loaded here as well; the sweeps are defined in the script, and configurations/tsne.json holds the
# arguments that are passed to it. Both jobs take the script as an input, so a changed script makes
# tensors and tables again.


configfile: os.path.join(CONFIG_DIR, "episodic_returns.json")
configfile: os.path.join(CONFIG_DIR, "tsne.json")


# A key that the JSON file does not have is refused, here and under `identification`: it would
# change nothing, or reach a step that does not read it.
TS_DEFAULTS = defaults("tsne")
TS = known_keys(config["tsne"], TS_DEFAULTS, "tsne")

# Wildcard: {benchmark} is the name of a benchmark in file names (linear_system, ...), which is also
# its name on the command line of the script.
_, TS_BENCHMARKS = registry()
TS_CONSTRAINTS = {"benchmark": alternatives(TS_BENCHMARKS.values())}
TS_SELECTED = [
    TS_BENCHMARKS[env_id]
    for env_id in selection(TS["only_benchmarks"], TS_BENCHMARKS, "tsne: only_benchmarks")
]

# The tensors have a folder of their own, since several embeddings are made from them. tables/ is the
# output directory of the script as TSNE.md describes it, with a copy of the tensors it was made from.
TS_DIR = os.path.join(RESULTS, "tsne")
TS_TENSORS = os.path.join(TS_DIR, "tensors", "tensors_{benchmark}.npz")
TS_TABLES = os.path.join(TS_DIR, "tables")
TS_TABLE = os.path.join(TS_TABLES, "{benchmark}_tsne.csv")
TS_EMBEDDED = os.path.join(TS_TABLES, "tensors_{benchmark}.npz")
TS_OTHER = [
    os.path.join(TS_TABLES, name)
    for name in (
        "tsne.csv",
        "separation.csv",
        "settings.json",
        "t_sne_figure.tex",
        "tsne_preview.pdf",
        "tsne_preview.png",
    )
]
TS_LOGS = os.path.join(TS_DIR, "logs")

# What an earlier embedding with other benchmarks may have left in tables/: these files would not
# belong to the coordinates of the files next to them, so the embedding removes them.
TS_NOT_SELECTED = expand(
    [TS_TABLE, TS_EMBEDDED],
    benchmark=[name for name in TS_BENCHMARKS.values() if name not in TS_SELECTED],
)

# Settings: arguments of the script, by the step that reads them. Those of the identification are
# the ones that the JSON file lists, and are passed to the embedding as well, where the script does
# not read them but writes them into settings.json. An argument of the identification under
# `embedding` would reach the embedding only and change nothing, so it is refused, as are the
# arguments that the rules set, and an argument of the embedding under `identification`, which
# would identify the tensors again for nothing.
TS_SET_BY_RULES = (
    "environments",
    "output_dir",
    "identify_only",
    "embed_only",
    "resume",
)
TS_IDENTIFICATION = script_options(
    TS["identification"],
    "tsne: identification",
    TS_SET_BY_RULES,
    known=TS_DEFAULTS["identification"],
)
TS_EMBEDDING = script_options(
    TS["embedding"],
    "tsne: embedding",
    (*TS_SET_BY_RULES, *TS_DEFAULTS["identification"]),
)

# The sweeps of the script. The sweep is checked here, before any job: an identification job that
# fails on a sweep the script does not have has lost the tensors it was to replace, since Snakemake
# removes the outputs of a job that fails.
TS_SWEEPS = ("inferred_layout", "orders")
choice(
    TS["identification"]["sweep"],
    TS_SWEEPS,
    "tsne: identification: sweep",
    or_null=True,
)
# The drivers of the least-squares solver that the script takes, checked here for the same reason.
TS_DRIVERS = ("gelsd", "gelss", "gelsy")
choice(
    TS["identification"]["lstsq_driver"],
    TS_DRIVERS,
    "tsne: identification: lstsq_driver",
    or_null=True,
)
TS_MEMORY = whole_number(TS["mem_mb"], "tsne: mem_mb", 1)
TS_SCRIPT = script_file("koopmanrl_utils.tsne_koopman_tensor")


rule tsne:
    input:
        expand(TS_TABLE, benchmark=TS_SELECTED),
        TS_OTHER,
        expand(TS_EMBEDDED, benchmark=TS_SELECTED),


# The tensors of one benchmark. They do not depend on the other benchmarks: the script seeds the
# data of each benchmark by itself.
rule tsne_identify:
    input:
        script=TS_SCRIPT,
    output:
        TS_TENSORS,
    log:
        os.path.join(TS_LOGS, "identify", "{benchmark}.log"),
    wildcard_constraints:
        **TS_CONSTRAINTS,
    params:
        identification=TS_IDENTIFICATION,
        output_dir=lambda wildcards, output: os.path.dirname(output[0]),
    threads: 1
    resources:
        mem_mb=TS_MEMORY,
    shell:
        "{PYTHON} -m koopmanrl_utils.tsne_koopman_tensor --identify_only"
        " --environments {wildcards.benchmark} {params.identification}"
        " --output_dir {params.output_dir:q} > {log:q} 2>&1"


# The script reads the tensors from its output directory, so they are copied there first. The
# benchmarks are passed in the order of the configuration file, which is the order of the rows of
# tsne.csv and of the legend of the figure.
rule tsne_embed:
    input:
        tensors=expand(TS_TENSORS, benchmark=TS_SELECTED),
        script=TS_SCRIPT,
    output:
        expand(TS_TABLE, benchmark=TS_SELECTED),
        TS_OTHER,
        expand(TS_EMBEDDED, benchmark=TS_SELECTED),
    log:
        os.path.join(TS_LOGS, "embed", "embed.log"),
    params:
        environments=" ".join(TS_SELECTED),
        identification=TS_IDENTIFICATION,
        embedding=TS_EMBEDDING,
        output_dir=lambda wildcards, output: os.path.dirname(output[0]),
        not_selected=TS_NOT_SELECTED,
    threads: 1
    resources:
        mem_mb=TS_MEMORY,
    shell:
        "(rm -f {params.not_selected:q} && cp {input.tensors:q} {params.output_dir:q} &&"
        " {PYTHON} -m koopmanrl_utils.tsne_koopman_tensor --embed_only"
        " --environments {params.environments} {params.identification} {params.embedding}"
        " --output_dir {params.output_dir:q}) > {log:q} 2>&1"
