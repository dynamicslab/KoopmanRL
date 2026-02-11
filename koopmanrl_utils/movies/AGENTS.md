# Movies Guides

The movies subdirectory contains the requisite logic to generate movies, or GIFs of the applied control for illustration purposes.

## Running Scripts

All scripts need to be run from the root of the repository, and can be run as

- `uv run koopmanrl_utils/movies/<name of scripts>`

## Directory Structure

```
movies/
├── fluid_flow/                      # Subdirectory for experiments on the fluid flow environment
├── __init__.py                      # Initialization file
├── abstract_policy.py               # Abstract method for the policy
├── AGENTS.md                        # Agents.md file for the subdirectory
├── algo_policies.py                 # Algorithmic inspection of the Koopman policies
├── default_policies.py              # Generate uncontrolled policies on the environments
├── env_enum.py                      # Enumeration of the reinforcement learning environments
├── generate_csvs.ipynb              # Stores the policies into a CSV
├── generate_gifs.py                 # Generates GIF illustrations of the applied control policy
├── generator.py                     # Generates controlled or uncontrolled trajectories
├── hundred_episode_cost_average.py  # Averages across 100 episodes
└── plotting_trajectories.ipynb      # Plot the control trajectories
```

## Working Checklist

1. Review the relevant AGENTS guide(s) and existing tests/examples for the script you touch.
2. Prototype changes in single files or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (tests/test_*.py) alongside code changes.
4. Run the scoped pytest command (uv run test -m ...) before submitting.
5. Keep documentation edits minimal and aligned.
