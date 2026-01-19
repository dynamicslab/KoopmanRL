import os

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.style.use("ggplot")


def tabulate_events(dpath, _scalar_name):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in sorted(os.listdir(dpath))]

    for summary_iterator in summary_iterators:
        folder_name = summary_iterator.path.split("\\")[-1]
        hyperparams = str(summary_iterator.Tensors("hyperparameters/text_summary")[0]).split("|")
        env_id = hyperparams[hyperparams.index("env_id") + 1]

        steps = [e.step for e in summary_iterator.Scalars(_scalar_name)]
        data = [e.value for e in summary_iterator.Scalars(_scalar_name)]

        df = pd.DataFrame()
        df["steps"] = steps
        df["data"] = data
        df.to_csv(f"./analysis/{folder_name}_episodic_returns_data.csv")

    data = {"LinearSystem-v0": {}, "FluidFlow-v0": {}, "Lorenz-v0": {}, "DoubleWell-v0": {}}

    for scalar_name in ("charts/episodic_return", "losses/qf_loss", "losses/actor_loss", "losses/vf_loss"):
        for summary_iterator in summary_iterators:
            if scalar_name not in summary_iterator.Tags()["scalars"]:
                continue

            folder_name = summary_iterator.path.split("\\")[-1]
            hyperparams = str(summary_iterator.Tensors("hyperparameters/text_summary")[0]).split("|")
            env_id = hyperparams[hyperparams.index("env_id") + 1]

            if folder_name not in data[env_id]:
                data[env_id][folder_name] = {}
            if scalar_name not in data[env_id][folder_name]:
                data[env_id][folder_name][scalar_name] = {}

            data[env_id][folder_name][scalar_name]["steps"] = [e.step for e in summary_iterator.Scalars(scalar_name)]
            data[env_id][folder_name][scalar_name]["data"] = [e.value for e in summary_iterator.Scalars(scalar_name)]
    return data


def main():
    path = "./final_tensorboard_runs"
    scalar_name = "charts/episodic_return"
    data = tabulate_events(path, _scalar_name=scalar_name)
    env_ids = ("LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0")
    fig = plt.figure(figsize=(8, 4.5))
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    for env_id in env_ids:
        ax = fig.add_subplot(111)
        plot_data = list(data[env_id].values())
        for i in range(len(plot_data)):
            ax.plot(
                plot_data[i]["charts/episodic_return"]["steps"],
                plot_data[i]["charts/episodic_return"]["data"],
                color=colors[i],
            )
        ax.set_title(env_id)
        ax.set_xlabel("Steps In Environment")
        ax.set_ylabel("Episodic Return")
        ax.legend(["Value Iteration", "LQR", "SAC (Q)", "SAC (V)", "SAKC (V)"])
        plt.savefig(f"./analysis/{env_id}.png")
        fig.clf()


if __name__ == "__main__":
    main()
