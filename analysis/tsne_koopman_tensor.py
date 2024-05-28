"""
Example usage:

python -m analysis.tsne_koopman_tensor --env-id=DoubleWell-v0
python -m analysis.tsne_koopman_tensor --env-id=DoubleWell-v0 --transpose

First command passes the M matrix to t-SNE algo as is
Second command passes the transpose of the M matrix to t-SNE algo
"""

import argparse
import matplotlib.pyplot as plt

from koopman_tensor.torch_tensor import KoopmanTensor
from koopman_tensor.utils import load_tensor
from sklearn.manifold import TSNE


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="LinearSystem-v0",
        help="the id of the environment (default: LinearSystem-v0)")
parser.add_argument("--transpose", type=str2bool, nargs='?', const=True, default=False,
        help="if the matrix M should be transposed before passing through tSNE (default: False)")
parser.add_argument("--perplexity", type=int, default=9,
        help="t-SNE perplexity (default: 9)")
args = parser.parse_args()


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