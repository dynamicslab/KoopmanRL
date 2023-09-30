import matplotlib.pyplot as plt
plt.style.use('ggplot')

import os

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    data = {
        'LinearSystem-v0': {},
        'FluidFlow-v0': {},
        'Lorenz-v0': {},
        'DoubleWell-v0': {}
    }

    for scalar_name in ('charts/episodic_return', 'losses/qf_loss', 'losses/actor_loss', 'losses/vf_loss'):
        for summary_iterator in summary_iterators:
            if scalar_name not in summary_iterator.Tags()['scalars']:
                continue

            folder_name = summary_iterator.path.split('\\')[-1]
            hyperparams = str(summary_iterator.Tensors('hyperparameters/text_summary')[0]).split('|')
            env_id = hyperparams[hyperparams.index('env_id')+1]

            if folder_name not in data[env_id]:
                data[env_id][folder_name] = {}
            if scalar_name not in data[env_id][folder_name]:
                data[env_id][folder_name][scalar_name] = {}

            data[env_id][folder_name][scalar_name]['steps'] = [e.step for e in summary_iterator.Scalars(scalar_name)]
            data[env_id][folder_name][scalar_name]['data'] = [e.value for e in summary_iterator.Scalars(scalar_name)]

    return data

if __name__ == '__main__':
    path = "./sac-q-vs-sac-v-vs-sakc-v"
    data = tabulate_events(path)
    scalar_name = 'charts/episodic_return'
    env_ids = ("LinearSystem-v0", "FluidFlow-v0", "Lorenz-v0", "DoubleWell-v0")
    for env_id in env_ids:
        plot_data = list(data[env_id].values())
        for i in range(len(plot_data)):
            plt.plot(plot_data[i]['charts/episodic_return']['steps'], plot_data[i]['charts/episodic_return']['data'])
        plt.title(f"SAC (Q) vs SAC (V) vs SAKC (V): {env_id}")
        plt.xlabel("Steps In Environment")
        plt.ylabel("Episodic Return")
        plt.legend(['SAC (Q)', 'SAC (V)', 'SAKC (V)'])
        plt.tight_layout()
        plt.savefig(f'./analysis/{env_id}.png')
        plt.clf()