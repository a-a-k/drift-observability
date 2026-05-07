from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260505
N = 300
T = 100

X_REF = np.array([0.32, 0.32, 0.36], dtype=float)
PART_NAMES = ("F", "R", "O")
POLICY_RATIO = 1.5
Z1_BOUNDARY = math.log(POLICY_RATIO) / math.sqrt(2.0)

STATIONARY_SIGMA = np.array([0.025, 0.025], dtype=float)
BENIGN_Z2_DRIFT = 0.55
BENIGN_Z1_SIGMA = 0.012
BENIGN_Z2_SIGMA = 0.025
BENIGN_Z1_CLIP = 0.07

RISKY_Z1_DRIFT = 0.43
RISKY_SIGMA = np.array([0.018, 0.022], dtype=float)

BOUNDARY_HORIZON = 8
BOUNDARY_SMOOTH_WINDOW = 8
MIN_PROJECTED_Z1_PROGRESS = 0.12
MIN_PROJECTED_SLOPE = 0.0015

SPLIT_START = T // 2
MERGE_START = 3 * T // 4

OUTPUT_DIR = Path("sanity_outputs")
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
EXAMPLE_PDF = OUTPUT_DIR / "example_trajectories.pdf"


def closure(x: np.ndarray | Iterable[float]) -> np.ndarray:
    """Normalize a positive vector or array of vectors along the last axis."""
    arr = np.asarray(x, dtype=float)
    if arr.shape[-1] < 1:
        raise ValueError("closure requires a non-empty vector")
    if np.any(arr <= 0.0):
        raise ValueError("closure requires strictly positive entries")
    total = np.sum(arr, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("closure requires a positive total")
    return arr / total


def ilr3(x: np.ndarray | Iterable[float]) -> np.ndarray:
    """Map a 3-part composition (F, R, O) to the specified two balances."""
    comp = closure(x)
    if comp.shape[-1] != 3:
        raise ValueError("ilr3 requires 3-part compositions")
    f = comp[..., 0]
    r = comp[..., 1]
    o = comp[..., 2]
    z1 = (1.0 / math.sqrt(2.0)) * np.log(f / r)
    z2 = math.sqrt(2.0 / 3.0) * np.log(np.sqrt(f * r) / o)
    return np.stack((z1, z2), axis=-1)


def ilr3_inv(z: np.ndarray | Iterable[float]) -> np.ndarray:
    """Inverse map for the ilr3 basis."""
    z_arr = np.asarray(z, dtype=float)
    if z_arr.shape[-1] != 2:
        raise ValueError("ilr3_inv requires 2 balance coordinates")
    z1 = z_arr[..., 0]
    z2 = z_arr[..., 1]

    clr_f = z1 / math.sqrt(2.0) + z2 / math.sqrt(6.0)
    clr_r = -z1 / math.sqrt(2.0) + z2 / math.sqrt(6.0)
    clr_o = -2.0 * z2 / math.sqrt(6.0)
    clr = np.stack((clr_f, clr_r, clr_o), axis=-1)
    clr = clr - np.max(clr, axis=-1, keepdims=True)
    exp_clr = np.exp(clr)
    return exp_clr / np.sum(exp_clr, axis=-1, keepdims=True)


def euclidean_distance(x: np.ndarray, x_ref: np.ndarray = X_REF) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=float) - x_ref, axis=-1)


def aitchison_distance(x: np.ndarray, x_ref: np.ndarray = X_REF) -> np.ndarray:
    return np.linalg.norm(ilr3(x) - ilr3(x_ref), axis=-1)


def policy_violation(x: np.ndarray | Iterable[float]) -> np.ndarray | bool:
    arr = np.asarray(x, dtype=float)
    violations = arr[..., 0] / arr[..., 1] > POLICY_RATIO
    if violations.shape == ():
        return bool(violations)
    return violations


def simulate_stationary(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z_ref = ilr3(X_REF)
    noise = rng.normal(0.0, STATIONARY_SIGMA, size=(N, T, 2))
    z = z_ref + noise
    return ilr3_inv(z), z


def simulate_benign(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z_ref = ilr3(X_REF)
    time = np.linspace(0.0, 1.0, T)
    z1_noise = rng.normal(0.0, BENIGN_Z1_SIGMA, size=(N, T))
    z1_noise = np.clip(z1_noise, -BENIGN_Z1_CLIP, BENIGN_Z1_CLIP)
    z2_noise = rng.normal(0.0, BENIGN_Z2_SIGMA, size=(N, T))

    z = np.empty((N, T, 2), dtype=float)
    z[..., 0] = z_ref[0] + z1_noise
    z[..., 1] = z_ref[1] + time * BENIGN_Z2_DRIFT + z2_noise
    return ilr3_inv(z), z


def simulate_risky(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z_ref = ilr3(X_REF)
    time = np.linspace(0.0, 1.0, T)
    noise = rng.normal(0.0, RISKY_SIGMA, size=(N, T, 2))
    drift = np.zeros((T, 2), dtype=float)
    drift[:, 0] = time * RISKY_Z1_DRIFT
    z = z_ref + drift[None, :, :] + noise
    return ilr3_inv(z), z


def simulate_churn(risky_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    leaves = np.zeros((N, T, 4), dtype=float)
    f = risky_x[..., 0]
    r = risky_x[..., 1]
    o = risky_x[..., 2]

    split_fraction = 0.58 + 0.08 * np.sin(np.arange(N, dtype=float))
    split_fraction = split_fraction[:, None]

    leaves[..., 0] = f
    leaves[..., 2] = r
    leaves[..., 3] = o

    split_slice = slice(SPLIT_START, MERGE_START)
    leaves[:, split_slice, 0] = f[:, split_slice] * split_fraction
    leaves[:, split_slice, 1] = f[:, split_slice] * (1.0 - split_fraction)

    canonical = np.stack(
        (
            leaves[..., 0] + leaves[..., 1],
            leaves[..., 2],
            leaves[..., 3],
        ),
        axis=-1,
    )
    return canonical, ilr3(canonical)


def first_true(mask: np.ndarray) -> np.ndarray:
    first = np.argmax(mask, axis=1).astype(int)
    first[~np.any(mask, axis=1)] = -1
    return first


def boundary_balance_alerts(z: np.ndarray) -> np.ndarray:
    z1 = z[..., 0]
    alerts = z1 >= Z1_BOUNDARY

    xs = np.arange(BOUNDARY_SMOOTH_WINDOW, dtype=float)
    xs = xs - xs.mean()
    denom = np.sum(xs * xs)

    for t in range(BOUNDARY_SMOOTH_WINDOW - 1, T):
        window = z1[:, t - BOUNDARY_SMOOTH_WINDOW + 1 : t + 1]
        centered = window - window.mean(axis=1, keepdims=True)
        slope = centered @ xs / denom
        remaining = Z1_BOUNDARY - z1[:, t]
        steps_to_boundary = np.divide(
            remaining,
            slope,
            out=np.full_like(remaining, np.inf),
            where=slope > 0.0,
        )
        projected = (
            (slope >= MIN_PROJECTED_SLOPE)
            & ((z1[:, t] - ilr3(X_REF)[0]) >= MIN_PROJECTED_Z1_PROGRESS)
            & (steps_to_boundary >= 0.0)
            & (steps_to_boundary <= BOUNDARY_HORIZON)
        )
        alerts[:, t] = alerts[:, t] | projected

    return alerts


def attribution_fidelity(z: np.ndarray, expected_component: int) -> float:
    z_ref = ilr3(X_REF)
    energy = np.sum((z - z_ref) ** 2, axis=1)
    largest = np.argmax(energy, axis=1)
    return float(np.mean(largest == expected_component))


def finite_closed_positive(name: str, x: np.ndarray) -> None:
    if not np.all(np.isfinite(x)):
        raise AssertionError(f"{name} contains non-finite values")
    if np.any(x <= 0.0):
        raise AssertionError(f"{name} contains non-positive canonical masses")
    if not np.allclose(np.sum(x, axis=-1), 1.0, atol=1e-12):
        raise AssertionError(f"{name} compositions are not closed")


def calibrate_thresholds(stationary_x: np.ndarray) -> tuple[float, float]:
    euclidean = euclidean_distance(stationary_x)
    aitchison = aitchison_distance(stationary_x)
    return (
        float(np.quantile(np.max(euclidean, axis=1), 0.95)),
        float(np.quantile(np.max(aitchison, axis=1), 0.95)),
    )


def monitor_alerts(
    x: np.ndarray, z: np.ndarray, euclidean_threshold: float, aitchison_threshold: float
) -> dict[str, np.ndarray]:
    return {
        "euclidean": euclidean_distance(x) > euclidean_threshold,
        "aitchison": aitchison_distance(x) > aitchison_threshold,
        "boundary_balance": boundary_balance_alerts(z),
    }


def blank_or_float(value: float | None) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if abs(value) < 5e-13:
        value = 0.0
    return f"{value:.6g}"


def row_for_monitor(
    regime: str,
    monitor: str,
    alerts: np.ndarray,
    x: np.ndarray,
    threshold: float,
    attribution: float | None = None,
    churn_error: float | None = None,
) -> dict[str, float | str | None]:
    alert_times = first_true(alerts)
    crossing_times = first_true(policy_violation(x))

    row: dict[str, float | str | None] = {
        "regime": regime,
        "monitor": monitor,
        "false_alarm_rate": None,
        "detection_rate": None,
        "median_detection_delay": None,
        "median_lead_time": None,
        "attribution_fidelity": attribution,
        "churn_continuity_error": churn_error,
        "threshold": threshold,
    }

    if regime in {"stationary", "benign_redistribution"}:
        row["false_alarm_rate"] = float(np.mean(alert_times >= 0))
        return row

    detected = (alert_times >= 0) & (crossing_times >= 0) & (alert_times <= crossing_times)
    row["detection_rate"] = float(np.mean(detected))
    if np.any(detected):
        delays = alert_times[detected] - crossing_times[detected]
        median_delay = float(np.median(delays))
        row["median_detection_delay"] = median_delay
        row["median_lead_time"] = -median_delay
    return row


def build_rows(
    regimes: dict[str, tuple[np.ndarray, np.ndarray]],
    euclidean_threshold: float,
    aitchison_threshold: float,
    churn_error: float,
) -> list[dict[str, float | str | None]]:
    monitor_thresholds = {
        "euclidean": euclidean_threshold,
        "aitchison": aitchison_threshold,
        "boundary_balance": Z1_BOUNDARY,
    }
    attribution_by_regime: dict[str, float | None] = {
        "stationary": None,
        "benign_redistribution": attribution_fidelity(regimes["benign_redistribution"][1], 1),
        "risky_ratio_drift": attribution_fidelity(regimes["risky_ratio_drift"][1], 0),
        "churn_lineage_aware": attribution_fidelity(regimes["churn_lineage_aware"][1], 0),
    }
    churn_by_regime: dict[str, float | None] = {
        "stationary": None,
        "benign_redistribution": None,
        "risky_ratio_drift": None,
        "churn_lineage_aware": churn_error,
    }

    rows: list[dict[str, float | str | None]] = []
    for regime, (x, z) in regimes.items():
        alerts_by_monitor = monitor_alerts(x, z, euclidean_threshold, aitchison_threshold)
        for monitor, alerts in alerts_by_monitor.items():
            rows.append(
                row_for_monitor(
                    regime=regime,
                    monitor=monitor,
                    alerts=alerts,
                    x=x,
                    threshold=monitor_thresholds[monitor],
                    attribution=attribution_by_regime[regime],
                    churn_error=churn_by_regime[regime],
                )
            )
    return rows


def write_summary_csv(rows: list[dict[str, float | str | None]]) -> None:
    fields = [
        "regime",
        "monitor",
        "false_alarm_rate",
        "detection_rate",
        "median_detection_delay",
        "median_lead_time",
        "attribution_fidelity",
        "churn_continuity_error",
        "threshold",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: blank_or_float(row[field]) for field in fields})


def markdown_table(rows: list[dict[str, float | str | None]]) -> str:
    headers = [
        "regime",
        "monitor",
        "false_alarm_rate",
        "detection_rate",
        "median_delay",
        "median_lead",
        "attribution",
        "churn_error",
        "threshold",
    ]
    fields = [
        "regime",
        "monitor",
        "false_alarm_rate",
        "detection_rate",
        "median_detection_delay",
        "median_lead_time",
        "attribution_fidelity",
        "churn_continuity_error",
        "threshold",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row[field]) if field in {"regime", "monitor"} else blank_or_float(row[field]) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parameter_block(euclidean_threshold: float, aitchison_threshold: float) -> str:
    x_ref = tuple(float(v) for v in X_REF)
    stationary_sigma = tuple(float(v) for v in STATIONARY_SIGMA)
    risky_sigma = tuple(float(v) for v in RISKY_SIGMA)
    return "\n".join(
        [
            f"- seed: `{SEED}`",
            f"- trajectories/windows: `N={N}`, `T={T}`",
            f"- reference composition `(F,R,O)`: `{x_ref}`",
            f"- policy boundary: `F/R > {POLICY_RATIO}`; `z1_boundary={Z1_BOUNDARY:.6g}`",
            f"- stationary sigma in balance space: `{stationary_sigma}`",
            f"- benign drift: `z2 += {BENIGN_Z2_DRIFT}`, `z1_sigma={BENIGN_Z1_SIGMA}`, `z2_sigma={BENIGN_Z2_SIGMA}`",
            f"- risky drift: `z1 += {RISKY_Z1_DRIFT}`, `sigma={risky_sigma}`",
            f"- boundary projection: horizon `{BOUNDARY_HORIZON}`, smooth window `{BOUNDARY_SMOOTH_WINDOW}`",
            f"- calibrated thresholds: Euclidean `{euclidean_threshold:.6g}`, Aitchison `{aitchison_threshold:.6g}`",
        ]
    )


def find_row(
    rows: list[dict[str, float | str | None]], regime: str, monitor: str
) -> dict[str, float | str | None]:
    for row in rows:
        if row["regime"] == regime and row["monitor"] == monitor:
            return row
    raise KeyError((regime, monitor))


def build_markdown(
    rows: list[dict[str, float | str | None]],
    euclidean_threshold: float,
    aitchison_threshold: float,
) -> str:
    benign_euclidean = find_row(rows, "benign_redistribution", "euclidean")
    benign_boundary = find_row(rows, "benign_redistribution", "boundary_balance")
    risky_boundary = find_row(rows, "risky_ratio_drift", "boundary_balance")
    risky_attr = find_row(rows, "risky_ratio_drift", "euclidean")
    benign_attr = find_row(rows, "benign_redistribution", "euclidean")
    churn_row = find_row(rows, "churn_lineage_aware", "boundary_balance")

    suggested = (
        "In a deterministic synthetic sanity check over "
        f"{N} trajectories and {T} windows, the boundary-aware balance monitor "
        f"detected risky ratio drift with median lead time "
        f"{blank_or_float(risky_boundary['median_lead_time'])} windows while producing "
        f"{blank_or_float(benign_boundary['false_alarm_rate'])} false alarms under benign redistribution; "
        f"lineage-aware churn aggregation had median max canonical error "
        f"{blank_or_float(churn_row['churn_continuity_error'])}."
    )

    bullets = [
        (
            "Euclidean distance alerted on benign redistribution at rate "
            f"{blank_or_float(benign_euclidean['false_alarm_rate'])}, while the boundary-aware monitor alerted at "
            f"{blank_or_float(benign_boundary['false_alarm_rate'])}."
        ),
        (
            "The boundary-aware monitor detected risky ratio drift before or at boundary crossing with detection rate "
            f"{blank_or_float(risky_boundary['detection_rate'])} and median lead time "
            f"{blank_or_float(risky_boundary['median_lead_time'])} windows."
        ),
        (
            "Attribution fidelity matched the injected direction: benign redistribution was attributed to z2 at "
            f"{blank_or_float(benign_attr['attribution_fidelity'])}, and risky drift to z1 at "
            f"{blank_or_float(risky_attr['attribution_fidelity'])}."
        ),
        (
            "Lineage-aware aggregation made the churned canonical signal numerically identical to the matched no-churn signal "
            f"at median max error {blank_or_float(churn_row['churn_continuity_error'])}."
        ),
    ]

    return "\n\n".join(
        [
            "# Synthetic sanity check summary",
            "## Parameters\n" + parameter_block(euclidean_threshold, aitchison_threshold),
            "## Summary table\n" + markdown_table(rows),
            "## Interpretation\n" + "\n".join(f"- {bullet}" for bullet in bullets),
            "## Suggested compact paper insertion\n" + suggested,
        ]
    )


def write_summary_md(markdown: str) -> None:
    SUMMARY_MD.write_text(markdown + "\n", encoding="utf-8")


def representative_index(x: np.ndarray, prefer_crossing: bool) -> int:
    crossing_times = first_true(policy_violation(x))
    if prefer_crossing:
        candidates = np.where(crossing_times >= 0)[0]
        target = int(np.median(crossing_times[candidates]))
        return int(candidates[np.argmin(np.abs(crossing_times[candidates] - target))])
    return int(np.argmin(np.max(np.abs(x[..., 0] / x[..., 1] - 1.0), axis=1)))


def plot_examples(
    benign_x: np.ndarray,
    risky_x: np.ndarray,
    euclidean_threshold: float,
    aitchison_threshold: float,
) -> None:
    examples = [
        ("benign redistribution", benign_x, representative_index(benign_x, prefer_crossing=False)),
        ("risky ratio drift", risky_x, representative_index(risky_x, prefer_crossing=True)),
    ]
    time = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), constrained_layout=True)

    for ax, (title, x, idx) in zip(axes, examples):
        trajectory = x[idx]
        ratio = trajectory[:, 0] / trajectory[:, 1]
        euclidean = euclidean_distance(trajectory)
        aitchison = aitchison_distance(trajectory)

        ax.plot(time, ratio, color="#2f6f73", label="F/R")
        ax.axhline(POLICY_RATIO, color="#9b2f2f", linestyle="--", linewidth=1.0, label="policy boundary")
        ax.set_title(title)
        ax.set_xlabel("window")
        ax.set_ylabel("F/R")
        ax.set_ylim(bottom=0.0)

        ax2 = ax.twinx()
        ax2.plot(time, euclidean, color="#6b5ca5", linewidth=1.2, label="Euclidean distance")
        ax2.plot(time, aitchison, color="#c47f2c", linewidth=1.2, label="Aitchison distance")
        ax2.axhline(euclidean_threshold, color="#6b5ca5", linestyle=":", linewidth=0.9)
        ax2.axhline(aitchison_threshold, color="#c47f2c", linestyle=":", linewidth=0.9)
        ax2.set_ylabel("distance")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=7)

    fig.savefig(EXAMPLE_PDF)
    plt.close(fig)


def run_checks(
    regimes: dict[str, tuple[np.ndarray, np.ndarray]],
    risky_x: np.ndarray,
    churn_x: np.ndarray,
    churn_error: float,
    rows: list[dict[str, float | str | None]],
) -> None:
    reconstructed = ilr3_inv(ilr3(X_REF))
    if not np.allclose(reconstructed, X_REF, atol=1e-12):
        raise AssertionError("ilr3_inv(ilr3(X_REF)) failed to reconstruct X_REF")

    for name, (x, _z) in regimes.items():
        finite_closed_positive(name, x)

    if np.any(policy_violation(regimes["benign_redistribution"][0])):
        raise AssertionError("benign redistribution produced a policy violation")

    max_churn_error = float(np.max(np.abs(churn_x - risky_x)))
    if max_churn_error > 1e-12 or churn_error > 1e-12:
        raise AssertionError("lineage-aware churn aggregation is not continuous")

    benign_euclidean = find_row(rows, "benign_redistribution", "euclidean")
    benign_boundary = find_row(rows, "benign_redistribution", "boundary_balance")
    risky_boundary = find_row(rows, "risky_ratio_drift", "boundary_balance")
    benign_attr = find_row(rows, "benign_redistribution", "euclidean")
    risky_attr = find_row(rows, "risky_ratio_drift", "euclidean")

    if not (benign_euclidean["false_alarm_rate"] > benign_boundary["false_alarm_rate"]):
        raise AssertionError("Euclidean benign false alarms did not exceed boundary-aware false alarms")
    if not (risky_boundary["detection_rate"] >= 0.9):
        raise AssertionError("boundary-aware risky detection rate is below acceptance target")
    if not (risky_boundary["median_detection_delay"] is not None and risky_boundary["median_detection_delay"] <= 0):
        raise AssertionError("boundary-aware monitor did not detect before or at crossing")
    if not (benign_attr["attribution_fidelity"] >= 0.9 and risky_attr["attribution_fidelity"] >= 0.9):
        raise AssertionError("attribution fidelity is below acceptance target")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    stationary_x, stationary_z = simulate_stationary(rng)
    benign_x, benign_z = simulate_benign(rng)
    risky_x, risky_z = simulate_risky(rng)
    churn_x, churn_z = simulate_churn(risky_x)

    churn_per_trajectory = np.max(np.abs(churn_x - risky_x), axis=(1, 2))
    churn_error = float(np.median(churn_per_trajectory))

    regimes = {
        "stationary": (stationary_x, stationary_z),
        "benign_redistribution": (benign_x, benign_z),
        "risky_ratio_drift": (risky_x, risky_z),
        "churn_lineage_aware": (churn_x, churn_z),
    }

    euclidean_threshold, aitchison_threshold = calibrate_thresholds(stationary_x)
    rows = build_rows(regimes, euclidean_threshold, aitchison_threshold, churn_error)
    run_checks(regimes, risky_x, churn_x, churn_error, rows)

    write_summary_csv(rows)
    markdown = build_markdown(rows, euclidean_threshold, aitchison_threshold)
    write_summary_md(markdown)
    plot_examples(benign_x, risky_x, euclidean_threshold, aitchison_threshold)

    print(markdown)


if __name__ == "__main__":
    main()
