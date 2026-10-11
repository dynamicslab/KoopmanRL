# Third-party licenses

KoopmanRL itself is licensed under the MIT License in `../LICENSE.md`. Parts of it are adapted from other projects and remain under their licenses, whose texts are in this directory:

* `koopmanrl/sac_continuous_action.py`, `koopmanrl/value_based_sac_continuous_action.py`,
  `koopmanrl/soft_actor_koopman_critic.py` and `koopmanrl/soft_koopman_value_iteration.py`
  are adapted from CleanRL (https://github.com/vwxyzjn/cleanrl), MIT License,
  Copyright (c) 2019 CleanRL Developers: `MIT-CleanRL.txt`.
* `koopmanrl/koopman_observables.py`, `koopmanrl/koopman_tensor/observables/numpy_observables.py`
  and `koopmanrl/koopman_tensor/observables/torch_observables.py` are adapted from d3s
  (https://github.com/sklus/d3s) and are licensed under the GNU Lesser General Public
  License v3.0 only: `LGPL-3.0-only.txt`, which supplements the GNU General
  Public License v3.0 in `GPL-3.0-only.txt`.

The package as a whole is therefore distributed under `MIT AND LGPL-3.0-only`.
