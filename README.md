# Tokamak `\tau_E` baseline project

This project implements the **first-layer baseline study** for the ITPA H-mode confinement database:

- **OLS power-law baseline**
- **MLP baseline**
- **KAN baseline**

The study uses the fixed engineering-variable set

- `BT`
- `IP`
- `NEL`
- `PL`
- `RGEO`
- `EPSILON`
- `KAPPA`
- `DELTA`
- `MEFF`

and predicts **`log(TAUTH)`**.

## What this project does

1. Loads `processed.csv`
2. Drops rows with missing or non-positive values in the chosen inputs/target
3. Builds either a **random split** or a **group split by `TOK`**
4. Fits:
   - OLS in log-log space
   - MLP on standardized log features/target
   - KAN on standardized log features/target
5. Saves:
   - metrics
   - best hyperparameters
   - test predictions
   - OLS coefficients
   - trained model weights where applicable

## Directory layout

```text
tokamak_tauE_baselines/
├── configs/
│   └── base.yaml
├── data/
│   └── processed.csv
├── outputs/
├── scripts/
│   ├── run_baseline_suite.py
│   ├── run_ols.py
│   ├── tune_kan.py
│   └── tune_mlp.py
├── src/
│   └── tokamak_tauE_baselines/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── data.py
│       ├── io_utils.py
│       ├── metrics.py
│       ├── search.py
│       ├── seed.py
│       ├── splits.py
│       └── models/
│           ├── __init__.py
│           ├── kan_wrapper.py
│           ├── mlp.py
│           └── ols.py
└── requirements.txt
```

## Installation

Create a clean environment first.

```bash
pip install -r requirements.txt
```

For KAN you also need:

```bash
pip install pykan
```

## Recommended run order

### 1) OLS
```bash
python scripts/run_ols.py --config configs/base.yaml --split-type random
python scripts/run_ols.py --config configs/base.yaml --split-type group
```

### 2) MLP search + final refit
```bash
python scripts/tune_mlp.py --config configs/base.yaml --split-type random
python scripts/tune_mlp.py --config configs/base.yaml --split-type group
```

### 3) KAN search + final refit
```bash
python scripts/tune_kan.py --config configs/base.yaml --split-type random
python scripts/tune_kan.py --config configs/base.yaml --split-type group
```

### 4) One-click run
```bash
python scripts/run_baseline_suite.py --config configs/base.yaml --split-type both
```

## Output structure

Each run writes to:

```text
outputs/<model>/<timestamp>_<split_type>/
```

Typical files:

- `metrics.json`
- `predictions.csv`
- `best_params.json`
- `trial_results.csv`
- `ols_coefficients.csv`
- `model.pt`

## Notes

- OLS is trained in **log-log space without standardization**, so coefficients are directly interpretable as scaling exponents.
- MLP and KAN are trained on **z-scored log features** and **z-scored log target**.
- `TOK` is used only for grouped splits and downstream diagnostics, not as a model feature.
- `SHOT` is not used as an input.
