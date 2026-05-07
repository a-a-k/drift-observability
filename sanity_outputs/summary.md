# Synthetic sanity check summary

## Parameters
- seed: `20260505`
- trajectories/windows: `N=300`, `T=100`
- reference composition `(F,R,O)`: `(0.32, 0.32, 0.36)`
- policy boundary: `F/R > 1.5`; `z1_boundary=0.286707`
- stationary sigma in balance space: `(0.025, 0.025)`
- benign drift: `z2 += 0.55`, `z1_sigma=0.012`, `z2_sigma=0.025`
- risky drift: `z1 += 0.43`, `sigma=(0.018, 0.022)`
- boundary projection: horizon `8`, smooth window `8`
- calibrated thresholds: Euclidean `0.0320248`, Aitchison `0.0952964`

## Summary table
| regime | monitor | false_alarm_rate | detection_rate | median_delay | median_lead | attribution | churn_error | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stationary | euclidean | 0.05 |  |  |  |  |  | 0.0320248 |
| stationary | aitchison | 0.05 |  |  |  |  |  | 0.0952964 |
| stationary | boundary_balance | 0 |  |  |  |  |  | 0.286707 |
| benign_redistribution | euclidean | 1 |  |  |  | 1 |  | 0.0320248 |
| benign_redistribution | aitchison | 1 |  |  |  | 1 |  | 0.0952964 |
| benign_redistribution | boundary_balance | 0 |  |  |  | 1 |  | 0.286707 |
| risky_ratio_drift | euclidean |  | 1 | -44 | 44 | 1 |  | 0.0320248 |
| risky_ratio_drift | aitchison |  | 1 | -45 | 45 | 1 |  | 0.0952964 |
| risky_ratio_drift | boundary_balance |  | 1 | -13 | 13 | 1 |  | 0.286707 |
| churn_lineage_aware | euclidean |  | 1 | -44 | 44 | 1 | 0 | 0.0320248 |
| churn_lineage_aware | aitchison |  | 1 | -45 | 45 | 1 | 0 | 0.0952964 |
| churn_lineage_aware | boundary_balance |  | 1 | -13 | 13 | 1 | 0 | 0.286707 |

## Interpretation
- Euclidean distance alerted on benign redistribution at rate 1, while the boundary-aware monitor alerted at 0.
- The boundary-aware monitor detected risky ratio drift before or at boundary crossing with detection rate 1 and median lead time 13 windows.
- Attribution fidelity matched the injected direction: benign redistribution was attributed to z2 at 1, and risky drift to z1 at 1.
- Lineage-aware aggregation made the churned canonical signal numerically identical to the matched no-churn signal at median max error 0.

## Suggested compact paper insertion
In a deterministic synthetic sanity check over 300 trajectories and 100 windows, the boundary-aware balance monitor detected risky ratio drift with median lead time 13 windows while producing 0 false alarms under benign redistribution; lineage-aware churn aggregation had median max canonical error 0.
