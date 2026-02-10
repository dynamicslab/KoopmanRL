import subprocess
from typing import List


def run_command(command: List[str]) -> None:
    """Run a shell command and raise on failure."""
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def run_linear_system() -> None:
    seeds = [
        4430,
        2738,
        9700,
        3478,
        8578,
        3602,
        1228,
        1749,
        6687,
        7659,
        3362,
        521,
        435,
        8229,
        762,
        7154,
        1045,
        4754,
        2936,
        868,
        2574,
        3169,
        4605,
        7,
    ]

    for seed in seeds:
        # SKVI
        run_command(
            [
                "python",
                "cleanrl/soft_koopman_value_iteration.py",
                "--env-id=LinearSystem-v0",
                f"--seed={seed}",
                "--lr=0.0010866250292703465",
                "--num-training-epochs=125",
                "--num-paths=75",
                "--num-steps-per-path=250",
                "--state-order=2",
                "--action-order=3",
            ]
        )

        # SAKC
        run_command(
            [
                "python",
                "cleanrl/soft_actor_koopman_critic.py",
                "--env-id=LinearSystem-v0",
                f"--seed={seed}",
                "--v-lr=0.00047001054701930456",
                "--q-lr=0.001802061953715088",
                "--num-paths=150",
                "--num-steps-per-path=175",
                "--state-order=2",
                "--action-order=3",
            ]
        )


def run_fluid_flow() -> None:
    seeds = [
        5412,
        3839,
        5062,
        3776,
        9127,
        3910,
        9604,
        5458,
        1745,
        5575,
        7601,
        2447,
        1584,
        3289,
        8699,
        5437,
        3771,
        1065,
        4787,
        9253,
        7844,
        7922,
        4050,
        6517,
        6597,
    ]

    for seed in seeds:
        # SKVI
        run_command(
            [
                "python",
                "cleanrl/soft_koopman_value_iteration.py",
                "--env-id=FluidFlow-v0",
                f"--seed={seed}",
                "--lr=0.00031904756404241047",
                "--num-training-epochs=125",
                "--num-paths=200",
                "--num-steps-per-path=225",
                "--state-order=4",
                "--action-order=2",
            ]
        )

        # SAKC
        run_command(
            [
                "python",
                "cleanrl/soft_actor_koopman_critic.py",
                "--env-id=FluidFlow-v0",
                f"--seed={seed}",
                "--v-lr=0.009423359172870875",
                "--q-lr=0.0017865746944645956",
                "--num-paths=50",
                "--num-steps-per-path=175",
                "--state-order=3",
                "--action-order=3",
            ]
        )


def run_lorenz() -> None:
    seeds = [
        8801,
        8207,
        7115,
        9370,
        6503,
        5442,
        1053,
        7904,
        5611,
        1635,
        2064,
        41,
        7644,
        1427,
        8573,
        1779,
        9355,
        169,
        3786,
        6957,
        4788,
        5900,
        3158,
        8953,
        6504,
    ]

    for seed in seeds:
        # SKVI
        run_command(
            [
                "python",
                "cleanrl/soft_koopman_value_iteration.py",
                "--env-id=Lorenz-v0",
                f"--seed={seed}",
                "--lr=0.0005076064158494174",
                "--num-training-epochs=125",
                "--num-paths=150",
                "--num-steps-per-path=250",
                "--state-order=3",
                "--action-order=1",
            ]
        )

        # SAKC
        run_command(
            [
                "python",
                "cleanrl/soft_actor_koopman_critic.py",
                "--env-id=Lorenz-v0",
                f"--seed={seed}",
                "--v-lr=0.05156797026843538",
                "--q-lr=0.023617484550332347",
                "--num-paths=200",
                "--num-steps-per-path=150",
                "--state-order=2",
                "--action-order=1",
            ]
        )


def run_double_well() -> None:
    seeds = [
        5991,
        5243,
        5581,
        726,
        2549,
        6408,
        6146,
        159,
        6382,
        1078,
        500,
        4000,
        9761,
        8178,
        8623,
        8866,
        9135,
        3642,
        9150,
        1106,
        5549,
        5202,
        6617,
        8294,
        469,
    ]

    for seed in seeds:
        # SKVI
        run_command(
            [
                "python",
                "cleanrl/soft_koopman_value_iteration.py",
                "--env-id=DoubleWell-v0",
                f"--seed={seed}",
                "--lr=0.0016556726497130062",
                "--num-training-epochs=175",
                "--num-paths=175",
                "--num-steps-per-path=100",
                "--state-order=2",
                "--action-order=4",
            ]
        )

        # SAKC
        run_command(
            [
                "python",
                "cleanrl/soft_actor_koopman_critic.py",
                "--env-id=DoubleWell-v0",
                f"--seed={seed}",
                "--v-lr=0.0003310304069101045",
                "--q-lr=0.00039795751924458065",
                "--num-paths=150",
                "--num-steps-per-path=300",
                "--state-order=4",
                "--action-order=4",
            ]
        )


def main() -> None:
    """Run all optimized variants with 25 seeds for final results.

    It assumes it is run from the repository root so that the `cleanrl/...` paths
    resolve correctly and that the correct virtual environment is already active.
    """
    run_linear_system()
    run_fluid_flow()
    run_lorenz()
    run_double_well()


if __name__ == "__main__":
    main()
