# Tokamak `\tau_E` baseline project

This repository benchmarks three model families for tokamak energy-confinement scaling on the ITPA H-mode database:

- `OLS` power-law baseline
- `MLP` nonlinear baseline
- `KAN` interpretable nonlinear baseline

The fixed feature set is:

- `BT`
- `IP`
- `NEL`
- `PL`
- `RGEO`
- `EPSILON`
- `KAPPA`
- `DELTA`
- `MEFF`

The target is `TAUTH`, and all models operate in log space.

## Evaluation layout

This repo now uses three distinct evaluation views:

1. `extrap_jet` is the primary performance test.
   It sorts the cleaned dataset by `TAUTH`, takes the global top 20%, and uses only the `1505` high-`TAUTH` `JET` samples as test data.
2. `group` is a supplementary stress test.
   It evaluates cross-device generalization under `TOK` group splits.
3. `random` is kept for interpretability analysis.
   It is the source of the local-exponent tables and KAN symbolic analysis.

For the current cleaned dataset, the `extrap_jet` split is fixed at:

- `train = 5286`
- `val = 755`
- `test = 1505`

## Environment

The project was run in:

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe
```

Install dependencies with:

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe -m pip install -r requirements.txt
```

`pykan` is also required:

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe -m pip install pykan
```

## Recommended runs

### Primary baseline suite

This runs the main `extrap_jet + group` workflow:

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\run_all_final.py --config configs\final_baseline.yaml --python C:\Users\12610\anaconda3\envs\KAN\python.exe --split-suite primary
```

If you want the simpler non-overnight wrapper:

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\run_baseline_suite.py --config configs\base.yaml --split-type primary
```

### Single split runs

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\run_ols.py --config configs\final_baseline.yaml --split-type extrap_jet
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\tune_mlp.py --config configs\final_baseline.yaml --split-type extrap_jet
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\tune_kan.py --config configs\final_baseline.yaml --split-type extrap_jet
```

Supplementary `group` test:

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\run_ols.py --config configs\final_baseline.yaml --split-type group
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\tune_mlp.py --config configs\final_baseline.yaml --split-type group
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\tune_kan.py --config configs\final_baseline.yaml --split-type group
```

## Interpretability workflow

`random` remains the interpretability split. The interpretability suite now analyzes all 9 features by default.

```powershell
C:\Users\12610\anaconda3\envs\KAN\python.exe scripts\run_interpretability_suite.py `
  --mlp-run-dir outputs_final\mlp\20260423_205839_random `
  --kan-run-dir outputs_final\kan\20260423_210939_random `
  --config configs\final_baseline.yaml `
  --split-type random `
  --device cpu `
  --attempt-symbolic
```

Key outputs:

- `interpretability_mlp\local_exponent_summary.csv`
- `interpretability\local_exponent_summary.csv`
- `interpretability*\analysis_meta.json`
- `kan_specific\kan_specific_summary.json`
- `kan_specific\symbolic_formula.txt`
- `kan_specific\symbolic_test_audit.csv`

## Metric semantics

This distinction is important:

- `best_params.json -> rmse_log` is the search-stage validation metric.
- `metrics.json -> rmse_log` is the final held-out test metric.

Do not compare these two as if they were measured on the same split.

## Current headline results

### Primary `extrap_jet` test

| Model | RMSE(log) | MAE(log) | R²(log) |
|---|---:|---:|---:|
| OLS | 0.2611 | 0.2259 | 0.3386 |
| MLP | 0.2579 | 0.2265 | 0.3549 |
| KAN | 0.3654 | 0.3199 | -0.2955 |

Interpretation:

- OLS and MLP are close on the high-`TAUTH` `JET` tail.
- The current KAN configuration does not show extrapolation advantage here.

### Supplementary `group` stress test

`group` is kept as a stress test for cross-device transfer. In the project’s documented supplementary group run, KAN showed a collapse mode with `R²(log) = -1.5049`, associated with weak regularization and a deeper `hidden_dims=[16,8]` configuration. A natural follow-up is:

- `KAN group + stronger regularization`
- start with `lamb=1e-3`
- start with `lamb_entropy=2`

## OLS global exponent vs MLP/KAN local exponents

The table below uses the `random` interpretability outputs. `OLS coef` is the global log-log coefficient; the MLP and KAN columns are local-exponent distribution quantiles on the random test set.

| Feature | OLS coef | MLP p10 | MLP p50 | MLP p90 | KAN p10 | KAN p50 | KAN p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BT` | 0.1245 | -0.7327 | -0.0966 | 0.5800 | -2.3252 | 0.1560 | 1.6886 |
| `IP` | 1.0471 | 0.5973 | 1.4069 | 1.9271 | -0.1680 | 1.2631 | 2.8247 |
| `NEL` | 0.1198 | -0.3279 | -0.0307 | 0.5460 | -0.7924 | 0.1809 | 0.9668 |
| `PL` | -0.6517 | -1.0543 | -0.7685 | -0.4984 | -1.3790 | -0.7666 | -0.1708 |
| `RGEO` | 1.4770 | 0.5242 | 1.0661 | 2.7215 | -0.4727 | 2.6169 | 6.5450 |
| `EPSILON` | 0.0537 | -0.9767 | -0.1818 | 0.7341 | -3.0431 | 0.5443 | 4.4324 |
| `KAPPA` | 0.1866 | -3.4159 | 0.9829 | 2.4316 | -7.1669 | 1.0891 | 6.1914 |
| `DELTA` | 0.2189 | -0.4827 | 1.0301 | 2.6056 | -5.0583 | 1.8257 | 8.1928 |
| `MEFF` | 0.2973 | -0.4261 | 0.6459 | 1.3281 | -2.1725 | 0.2141 | 2.3194 |

The strongest evidence for near-power-law behavior is on:

- `IP`
- `PL`

The clearest state-dependent variables are:

- `BT`
- `RGEO`
- `EPSILON`

## KAN symbolic note

The current `random` KAN symbolic extraction now records test-set consistency numbers in `kan_specific_summary.json`.

Current values:

- numerical KAN test RMSE(log): `0.1301`
- symbolic KAN test RMSE(log): `1.2003`
- symbolic-vs-numerical RMSE(log): `1.2005`

This means the current auto-symbolic expression is still a qualitative structure hint, not a numerically faithful replacement for the trained KAN.

## Output structure

Typical final runs write to:

```text
outputs_final/<model>/<timestamp>_<split_type>/
```

Common files:

- `metrics.json`
- `predictions.csv`
- `best_params.json`
- `trial_results.csv`
- `ols_coefficients.csv`
- `model.pt`
- `parity_original.png`
- `parity_log.png`

## Notes

- OLS is trained in log-log space without standardization, so its coefficients are directly interpretable as global scaling exponents.
- MLP and KAN are trained on z-scored log features and z-scored log target.
- `TOK` is never used as a model feature.
- `SHOT` is not used as a model feature.
- The v3 report is in `KAN_time_group_report_final_v3.md`.
