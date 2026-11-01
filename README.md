# EnSet: Boosting MARL at High Replay Ratios

Code for the NeurIPS 2026 paper "EnSet". EnSet trains multiagent reinforcement learning (MARL) at high replay ratios with two components on the centralized global Q-network:

- **Global Q-network ensemble reset**: an ensemble of $H$ global Q-networks trained with the averaged loss, plus periodic Shrink & Perturb on the global Q-networks and agent networks.
- **Multiagent translation augmentation**: a random translation of all unit positions in the global state.

## Structure

```
smac/     EnSet-QMIX in SMAC (built on PyMARL)
smacv2/   EnSet-QMIX in SMACv2 (built on pymarl2)
mpe/      EnSet-MADDPG and EnSet-FACMAC in MPE predator-prey (built on FACMAC)
```

Main EnSet files:

| | SMAC | SMACv2 | MPE |
|---|---|---|---|
| Learner | `smac/src/learners/enset_q_learner.py` | `smacv2/src/learners/enset_nq_learner.py` | `mpe/src/learners/enset_maddpg_learner.py`, `mpe/src/learners/enset_facmac_learner.py` |
| Global Q-network ensemble | `smac/src/modules/mixers/eqmix.py` | `smacv2/src/modules/mixers/enmix.py` | `mpe/src/modules/critics/emaddpg.py`, `mpe/src/modules/mixers/eqmix.py`, `mpe/src/modules/critics/efacmac.py` |
| Config | `smac/src/config/algs/enset_qmix.yaml` | `smacv2/src/config/algs/enset_qmix.yaml` | `mpe/src/config/algs/enset_maddpg_pp.yaml`, `mpe/src/config/algs/enset_facmac_pp.yaml` |

## Installation

SMAC follows PyMARL and MPE follows FACMAC. See `smac/README.md` and `mpe/README.md` for the Docker setup. In short:

```
cd smac            # or mpe
cd docker && bash build.sh && cd ..
bash install_sc2.sh  # SMAC only
```

SMACv2 follows pymarl2. See `smacv2/install.txt` for the conda setup, which installs SMACv2 from `git+https://github.com/oxwhirl/smacv2.git@ranges`. We use StarCraft II version 4.6.2.69232.

## Running

### SMAC

```
cd smac
python src/main.py --config=enset_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=1020000 seed=1
```

Use `t_max=520000` for 2s3z, 3s5z, and 10m_vs_11m, and `t_max=1020000` for the other maps. The baseline QMIX uses `--config=qmix` (replay ratio 1).

### SMACv2

```
cd smacv2
python src/main.py --config=enset_qmix --env-config=sc2_gen_zerg with env_args.capability_config.n_units=10 env_args.capability_config.start_positions.n_enemies=10 seed=1
```

Use `--env-config=sc2_gen_protoss`, `sc2_gen_terran`, or `sc2_gen_zerg` for the three races. The default replay ratio is 15. Add `replay_ratio=10` for a replay ratio of 10. The baseline QMIX uses `--config=qmix`.

### MPE

```
cd mpe
python src/main.py --config=enset_maddpg_pp --env-config=particle with env_args.scenario_name=continuous_pred_prey_3a seed=1
python src/main.py --config=enset_facmac_pp --env-config=particle with env_args.scenario_name=continuous_pred_prey_3a seed=1
```

Scenarios: `continuous_pred_prey_3a`, `continuous_pred_prey_6a`, `continuous_pred_prey_9a`. Add `t_max=270000` for pp3a (0.25M steps). The default `t_max=520000` is for pp6a and pp9a (0.5M steps). Baselines use `--config=maddpg_pp` and `--config=facmac_pp` (replay ratio 1).

## Hyperparameters

All EnSet hyperparameters are shared across algorithms, tasks, and environments.

| Hyperparameter | Config key | Value |
|---|---|---|
| Ensemble size $H$ | `mixer_ensemble_N` | 5 |
| Replay ratio $N_{RR}$ | `replay_ratio` | 10 (SMAC), 15 or 10 (SMACv2), 25 (MPE) |
| Reset interval $T_R$ | `reset_interval` | 2000 |
| Shrink & Perturb interpolation $\alpha_r$ | `reset_mixer_alpha`, `reset_agent_alpha` | 0.8 |
| Translation range $\alpha_z$ | `perturb_alpha` | 0.2 |
| Observation scaling range | (in the learner) | [0.8, 1.2] |

Results are written by Sacred to `results/`.

## Acknowledgements

This code is built on [PyMARL](https://github.com/oxwhirl/pymarl), [pymarl2](https://github.com/hijkzzz/pymarl2), [SMAC](https://github.com/oxwhirl/smac), [SMACv2](https://github.com/oxwhirl/smacv2), and [FACMAC](https://github.com/oxwhirl/facmac). We thank the authors for releasing their code.

## License

This code is released under the Apache License 2.0, the same as PyMARL and pymarl2. The `mpe/` folder is based on FACMAC, which is released without a license file.
