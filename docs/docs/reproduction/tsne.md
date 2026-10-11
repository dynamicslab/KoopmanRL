---
id: tsne
sidebar_position: 4
title: t-SNE of the Koopman Tensors
---

# t-SNE of the Koopman Tensors

The supplementary material of the paper embeds Koopman tensors of the four benchmarks together in two dimensions with t-SNE. One script identifies the tensors, writes them in a common basis and embeds them. The guide [`koopmanrl_utils/TSNE.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/TSNE.md) also records what is known about how the published figure was made, what the embedding shows, and the reproducibility checks.

## One call

```bash
uv run -m koopmanrl_utils.tsne_koopman_tensor
```

The three steps of a call:

1. **Identification.** For each benchmark, Koopman tensors are identified by ordinary least squares from random-agent trajectories (collected as in `koopmanrl.soft_koopman_value_iteration.generate_koopman_tensor`) with monomial dictionaries, and stored in `tensors_<benchmark>.npz`.
2. **Common basis.** Every tensor is written in the dictionaries of the largest state and action orders of the sweep (`--common_basis largest`) and flattened.
3. **Embedding.** All tensors of all benchmarks are embedded together with `sklearn.manifold.TSNE` (perplexity 30, PCA initialisation, Barnes-Hut, `random_state=42` by default).

Everything is written into `tsne_koopman_tensor_results/` (`--output_dir`), which git ignores.

## Sweeps

| `--sweep` | Tensors per benchmark |
|-----------|-----------------------|
| `inferred_layout` (default) | 161: state and action order 2 on an 11 × 11 and a 5 × 5 grid of identification budgets, plus 15 combinations of state and action orders 1–4. This is the layout that the published coordinates point to. |
| `orders` | 128: state orders 1–4 × action orders 1–4 × 8 data seeds at the tuned SKVI budget. This follows the text of the paper. |

```bash
uv run -m koopmanrl_utils.tsne_koopman_tensor --sweep orders \
    --output_dir tsne_koopman_tensor_results/orders
```

`--state_orders`, `--action_orders`, `--num_paths`, `--num_steps_per_path`, `--seeds` and `--seed0` set the grid of the `orders` sweep.

## Splitting identification and embedding

| Flag | Effect |
|------|--------|
| `--identify_only` | Identify and store the tensors of `--environments`, without embedding them. |
| `--embed_only` | Repeat steps 2 and 3 from the tensors stored in `--output_dir`. The identification arguments must match those the tensors were made with; otherwise the call is refused and nothing is written. |
| `--resume` | Reuse the stored tensors of benchmarks that an interrupted run of the same sweep and driver has finished. |

```bash
uv run -m koopmanrl_utils.tsne_koopman_tensor --embed_only --units box
```

The embedding options (`--units`, `--scaling`, `--metric`, `--perplexity`, `--tsne_init`, `--tsne_method`, `--subtract_persistence`, `--shared_coordinates_only`, ...) are listed in the [guide's argument table](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/TSNE.md#arguments).

## Least-squares driver

`--lstsq_driver` sets the LAPACK driver of `torch.linalg.lstsq` used for the identification:

| Driver | Result |
|--------|--------|
| `gelsd` (default) | The same tensor in every call. |
| `gelss` | The same tensor in every call; equal to `gelsd` up to rounding. |
| `gelsy` | The driver used when none is named; the tensor differs in its last digits from call to call. |

With `gelsd`, on one machine and with one number of numerical threads, a run reproduces the tensors bit for bit and the tables byte for byte. See also [Least-squares driver](../koopman-tensor/least-squares-driver.md).

## Outputs

| File | Content |
|------|---------|
| `<benchmark>_tsne.csv` | One row per tensor; the first columns `index,x-val,y-val` are those the figure source of the paper reads. |
| `tsne.csv` | All rows, with `benchmark` and `class` columns. |
| `separation.csv` | Silhouette and nearest-neighbour purity, per benchmark and overall, in the embedding and on the flattened tensors. |
| `t_sne_figure.tex` | Standalone pgfplots figure reading the four tables. |
| `tsne_preview.pdf`, `tsne_preview.png` | Matplotlib previews. |
| `settings.json` | Arguments of the run, checked by later `--embed_only` calls. |

## With Snakemake

```bash
uvx --python 3.12 snakemake -n tsne
uvx --python 3.12 snakemake --cores 1 tsne
```

The workflow runs one `--identify_only` job per benchmark and one `--embed_only` job. See [Snakemake workflows](./snakemake.md).
