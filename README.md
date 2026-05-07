# Drift Observability Synthetic Artifact

This repository contains a deterministic synthetic sanity check for drift observability on simplex-valued operational signals. It compares distance-to-reference monitors with a policy-aligned log-ratio balance monitor and includes a split/merge churn check with lineage-aware aggregation.

No real operational data is used. All trajectories are synthetic and generated from a fixed random seed.

## Contents

- `sanity_check.py`: executable Python script that generates all results.
- `sanity_outputs/summary.csv`: machine-readable summary table.
- `sanity_outputs/summary.md`: human-readable summary and suggested compact paper text.
- `sanity_outputs/example_trajectories.pdf`: example benign and risky trajectories.
- `requirements.txt`: minimal Python dependencies.

## Reproduce

Use Python 3. Install dependencies and run the script:

```bash
python -m pip install -r requirements.txt
python sanity_check.py
```

The script writes outputs under `sanity_outputs/` and prints the Markdown summary to stdout.

## Synthetic Setup

The script uses:

- seed: `20260505`
- trajectories/windows: `N=300`, `T=100`
- reference composition `(F,R,O)`: `(0.32, 0.32, 0.36)`
- policy boundary: `F/R > 1.5`
- stationary noise in ilr space: `(0.025, 0.025)`
- benign redistribution: drift primarily in the `z2` balance
- risky ratio drift: drift primarily in the `z1 = log(F/R)/sqrt(2)` balance
- churn regime: split/merge of `F` into `F1,F2`, then canonical aggregation back to `F`

## Interpretation

This is a synthetic mechanism check, not empirical evidence from operational systems. It tests whether a policy-aligned balance monitor distinguishes benign redistribution from drift toward a ratio boundary, and whether canonical lineage-aware aggregation preserves the signal under split/merge churn.

## License

This artifact is released under the MIT License. See `LICENSE`.
