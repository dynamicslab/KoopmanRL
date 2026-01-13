"""
Example usage:
python -m movies.hundred_episode_cost_average
"""

import matplotlib.pyplot as plt
import numpy as np

from movies.env_enum import EnvEnum

# Set the style for a more modern look
plt.style.use('seaborn')

# Define a professional color palette
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f']

timestamp_map = {
    EnvEnum.LinearSystem: {
        "SAKC": 1738508701,
        "SAC (V)": 1738509084,
        "SAC (Q)": 1738510753,
        "SKVI": 1738512109,
        "LQR": 1738512185,
    },
    EnvEnum.FluidFlow: {
        "SAKC": 1738514025,
        "SAC (V)": 1738513905,
        "SAC (Q)": 1738513785,
        "SKVI": 1738513636,
        "LQR": 1738513072,
    },
    EnvEnum.Lorenz: {
        "SAKC": 1738514131,
        "SAC (V)": 1738514243,
        "SAC (Q)": 1738514346,
        "SKVI": 1738514754,
        "LQR": 1738514827,
    },
    EnvEnum.DoubleWell: {
        "SAKC": 1738515701,
        "SAC (V)": 1738515765,
        "SAC (Q)": 1738515835,
        "SKVI": 1738515613,
        "LQR": 1738515229,
    },
}

# Set figure size for better visibility
plt.figure(figsize=(10, 6))

for env, policy_timestamps in timestamp_map.items():
    print(env.value)
    policies = list(policy_timestamps.keys())
    average_episodic_costs = []
    std_episodic_costs = []

    for policy, timestamp in policy_timestamps.items():
        folder_name = f"./video_frames/{env.value}_{timestamp}"

        try:
            policy_costs = np.load(f"{folder_name}/main_policy_costs.npy")
            print(f"\t{policy} policy costs shape: {policy_costs.shape}")

            average_episodic_cost = policy_costs.sum(axis=1).mean()
            std_episodic_cost = policy_costs.sum(axis=1).std()
            print(f"\tAverage episodic cost from {policy} in {folder_name}: {average_episodic_cost}")
            print(f"\tStandard deviation of episodic cost from {policy} in {folder_name}: {std_episodic_cost}\n")

            average_episodic_costs.append(average_episodic_cost)
            std_episodic_costs.append(std_episodic_cost)
        except:
            print(f"\t{policy} policy costs not found in {folder_name}\n")

    # Create bar plot with enhanced styling
    bars = plt.bar(
        policies,
        average_episodic_costs,
        yerr=std_episodic_costs,
        capsize=4,  # Reduced capsize
        color=colors,
        alpha=0.8,
        width=0.6,
        edgecolor='black',
        linewidth=1,
        error_kw={
            'elinewidth': 1,     # Thinner error bars
            'capthick': 1,       # Thinner caps
            'alpha': 0.5         # More transparent error bars
        }
    )

    # Customize the plot
    plt.title(f"Average Episodic Cost for {env.value}", fontsize=14, pad=20)
    plt.xlabel("Policy", fontsize=12, labelpad=10)
    plt.ylabel("Average Episodic Cost", fontsize=12, labelpad=10)

    # Add value labels on top of each bar with adjusted position
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Calculate label position considering error bars
        label_height = height + std_episodic_costs[i] + (max(average_episodic_costs) * 0.02)  # Add 2% of max height as padding
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            label_height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Customize grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Customize spines
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.5)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust y-axis limits to accommodate labels
    ymax = plt.gca().get_ylim()[1]
    plt.ylim(0, ymax * 1.1)  # Add 10% padding at the top

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()
