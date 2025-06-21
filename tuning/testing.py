from opt_wrappers import ppo_tuning_wrapper

def evaluate():
    _experiment = ppo_tuning_wrapper()
    return _experiment


if __name__ == "__main__":

    # Run the experiment
    experiment = evaluate()

    # Extract the values
    raw_metric_values = [
        scalar_event[0] for scalar_event in experiment["charts/episodic_return"][-50:]
    ]

    # Apply the sliding window to raw metric values

    print(raw_metric_values)
    print(len(raw_metric_values))