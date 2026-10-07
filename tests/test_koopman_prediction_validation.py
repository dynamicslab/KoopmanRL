"""koopmanrl_utils.koopman_prediction_validation uses the tuned settings and measures what it says it measures."""

import contextlib
import io
import json
import os

import numpy as np

import koopmanrl_utils.koopman_prediction_validation as validation

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configurations")
CONFIG_FILES = {
    "LinearSystem-v0": "skvi_linear_system_hparams.json",
    "Lorenz-v0": "skvi_lorenz_hparams.json",
    "FluidFlow-v0": "skvi_fluid_flow_hparams.json",
    "DoubleWell-v0": "skvi_double_well_hparams.json",
}


def small_run(monkeypatch, env_id, **overrides):
    """One seed of `run_env` with a short horizon, few rollouts and one subsample size."""
    monkeypatch.setattr(validation, "HORIZON", 5)
    monkeypatch.setattr(validation, "N_ROLLOUT_PATHS", 4)
    monkeypatch.setattr(validation, "N_MC", 8)
    monkeypatch.setattr(validation, "N_ID_GRID", [250])
    config = {**validation.CONFIGS[env_id], "num_paths": 10, "steps": 50, **overrides}
    with contextlib.redirect_stdout(io.StringIO()):
        return validation.run_env(env_id, config, seed=0)


def test_settings_match_the_tuned_configurations():
    for env_id, settings in validation.CONFIGS.items():
        with open(os.path.join(CONFIG_DIR, CONFIG_FILES[env_id])) as f:
            tuned = json.load(f)
        assert settings["num_paths"] == tuned["num-paths"]
        assert settings["steps"] == tuned["num-steps-per-path"]
        assert settings["so"] == tuned["state-order"]
        assert settings["ao"] == tuned["action-order"]


def test_nrmse_is_relative_to_the_reference():
    reference = np.array([[3.0, 0.0], [4.0, 5.0]])
    assert validation.nrmse(reference, reference) == 0.0
    assert np.isclose(validation.nrmse(np.zeros_like(reference), reference), 1.0)
    assert np.isclose(validation.nrmse(1.1 * reference, reference), 0.1)


def test_lifted_metrics_scale_each_function_and_drop_constants():
    rng = np.random.RandomState(0)
    reference = np.vstack([np.ones(200), rng.randn(200), 1e3 * rng.randn(200)])
    base = reference + np.vstack([np.zeros(200), 0.2 * reference[1], 0.2 * reference[2]])
    prediction = reference + np.vstack([5.0 * np.ones(200), 0.1 * reference[1], 0.1 * reference[2]])
    metrics = validation.lifted_metrics(prediction, reference, base)
    # the error of the constant function is ignored, and the large function does not dominate
    assert np.isclose(metrics["err"], 0.1)
    assert np.isclose(metrics["persistence"], 0.2)
    assert np.isclose(metrics["increment"], 0.5)
    assert metrics["noise_floor"] == 0.0


def test_linear_system_is_predicted_to_rounding(monkeypatch):
    onestep, multistep, sweep, _ = small_run(monkeypatch, "LinearSystem-v0")
    assert onestep["phi_heldout"] < 1e-10
    assert onestep["heldout"] < 1e-10
    assert onestep["phi_persistence"] > 1e-2
    assert multistep.shape == (5, 3) and np.all(multistep[:, :2] < 1e-8)
    assert sweep.shape == (1, 2)


def test_first_rollout_step_is_the_same_for_both_rollouts(monkeypatch):
    # iterating K^u in dictionary space and re-evaluating the dictionary differ only from the second step on
    _, multistep, _, dt = small_run(monkeypatch, "Lorenz-v0")
    assert np.isclose(multistep[0, 0], multistep[0, 1], rtol=1e-9)
    assert dt == 0.01


def test_double_well_reports_a_noise_floor(monkeypatch):
    # also runs the check of the vectorised drift against the environment's
    onestep, multistep, _, _ = small_run(monkeypatch, "DoubleWell-v0")
    assert onestep["phi_noise_floor"] > 0.0 and onestep["noise_floor"] > 0.0
    assert np.all(np.isfinite(multistep))
