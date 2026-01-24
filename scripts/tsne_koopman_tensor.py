# /// script
# requires-python = "== 3.10"
# dependencies = [
#   "koopmanrl",
#   "matplotlib",
#   "sklearn",
#   "typed-argument-parser",
# ]
# ///

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tap import Tap

from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor
from koopmanrl.koopman_tensor.utils import load_tensor


class ArgumentParser(Tap):
    env_id: str = "LinearSystem-v0"  # the id of the environment (default: LinearSystem-v0)
    transpose: bool = False  # if the matrix M should be transposed before passing through tSNE (default: False)
    perplexity: int = 9  # t-SNE perplexity (default: 9)


def main():
    args = ArgumentParser().parse_args()
    koopman_tensor: KoopmanTensor = load_tensor(args.env_id, "path_based_tensor")
    if args.env_id == "DoubleWell-v0" and not args.transpose:
        args.perplexity = 5

    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42)
    tsne_input = koopman_tensor.M.T if args.transpose else koopman_tensor.M
    tsne_M = tsne.fit_transform(tsne_input)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_M[:, 0], tsne_M[:, 1])
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.title(f"{args.env_id} t-SNE visualization {'(transposed matrix)' if args.transpose else ''}")
    plt.show()


if __name__ == "__main__":
    main()
