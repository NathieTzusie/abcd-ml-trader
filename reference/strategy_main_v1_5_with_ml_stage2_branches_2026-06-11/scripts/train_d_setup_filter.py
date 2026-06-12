from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "bc_ratio",
    "tp1_r",
    "tp2_r",
    "d_search_bars",
    "d_confirm_bars",
    "d_overshoot_atr",
    "atr_percent",
    "volume_zscore",
    "trend_slope",
    "d_position_range",
]
CATEGORICAL_FEATURES = ["timeframe", "pattern_type"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ML filter for confirmed AB=CD D setups")
    parser.add_argument("--events", default="reports/abcd_d_confirmation_events.csv", help="D 点事件表")
    parser.add_argument("--output-dir", default="reports/ml_d_filter", help="输出目录")
    parser.add_argument("--train-months", type=int, default=12, help="滚动训练窗口月数")
    parser.add_argument("--test-months", type=int, default=3, help="滚动验证窗口月数")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="学习率")
    parser.add_argument("--epochs", type=int, default=4000, help="训练轮数")
    parser.add_argument("--l2", type=float, default=0.02, help="L2 正则强度")
    parser.add_argument("--target", choices=("reached_tp1", "reached_tp2"), default="reached_tp1", help="训练目标")
    return parser.parse_args()


def load_events(path: Path, target: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows["d_confirm_time"] = pd.to_datetime(rows["d_confirm_time"])
    rows = rows.sort_values("d_confirm_time").reset_index(drop=True)
    for feature in NUMERIC_FEATURES:
        rows[feature] = pd.to_numeric(rows[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows[target] = rows[target].astype(int)
    return rows


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_numeric = train[NUMERIC_FEATURES].astype(float)
    test_numeric = test[NUMERIC_FEATURES].astype(float)
    mean = train_numeric.mean()
    std = train_numeric.std(ddof=0).replace(0.0, 1.0)
    train_scaled = (train_numeric - mean) / std
    test_scaled = (test_numeric - mean) / std

    combined_categories = pd.concat([train[CATEGORICAL_FEATURES], test[CATEGORICAL_FEATURES]], axis=0)
    category_columns = pd.get_dummies(combined_categories, columns=CATEGORICAL_FEATURES, dtype=float).columns
    train_categories = pd.get_dummies(train[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, dtype=float).reindex(columns=category_columns, fill_value=0.0)
    test_categories = pd.get_dummies(test[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, dtype=float).reindex(columns=category_columns, fill_value=0.0)

    x_train = np.column_stack([np.ones(len(train_scaled)), train_scaled.to_numpy(), train_categories.to_numpy()])
    x_test = np.column_stack([np.ones(len(test_scaled)), test_scaled.to_numpy(), test_categories.to_numpy()])
    feature_names = ["intercept"] + NUMERIC_FEATURES + list(category_columns)
    return x_train, x_test, train[target].to_numpy(), feature_names


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic_regression(x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int, l2: float) -> np.ndarray:
    weights = np.zeros(x.shape[1])
    for _ in range(epochs):
        predictions = sigmoid(x @ weights)
        gradient = x.T @ (predictions - y) / len(y)
        regularization = l2 * weights
        regularization[0] = 0.0
        weights -= learning_rate * (gradient + regularization)
    return weights


def evaluate(rows: pd.DataFrame, probabilities: np.ndarray, threshold: float, target: str) -> dict[str, float | int]:
    selected = rows[probabilities >= threshold]
    selected_results = selected["result_r"].astype(float)
    baseline_results = rows["result_r"].astype(float)
    return {
        "threshold": threshold,
        "test_rows": int(len(rows)),
        "selected_rows": int(len(selected)),
        "selection_rate": float(len(selected) / len(rows)) if len(rows) else 0.0,
        "baseline_target_prob": float(rows[target].mean()) if len(rows) else 0.0,
        "selected_target_prob": float(selected[target].mean()) if len(selected) else 0.0,
        "baseline_tp1_prob": float(rows["reached_tp1"].mean()) if len(rows) else 0.0,
        "selected_tp1_prob": float(selected["reached_tp1"].mean()) if len(selected) else 0.0,
        "baseline_tp2_prob": float(rows["reached_tp2"].mean()) if len(rows) else 0.0,
        "selected_tp2_prob": float(selected["reached_tp2"].mean()) if len(selected) else 0.0,
        "baseline_avg_r": float(baseline_results.mean()) if len(rows) else 0.0,
        "selected_avg_r": float(selected_results.mean()) if len(selected) else 0.0,
        "baseline_sl_prob": float((rows["outcome"] == "sl_before_tp1").mean()) if len(rows) else 0.0,
        "selected_sl_prob": float((selected["outcome"] == "sl_before_tp1").mean()) if len(selected) else 0.0,
    }


def month_add(timestamp: pd.Timestamp, months: int) -> pd.Timestamp:
    return timestamp + pd.DateOffset(months=months)


def run_walk_forward(rows: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    reports: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    current = rows["d_confirm_time"].min().normalize()
    final_time = rows["d_confirm_time"].max()
    while True:
        train_start = current
        train_end = month_add(train_start, args.train_months)
        test_end = month_add(train_end, args.test_months)
        if train_end >= final_time:
            break
        train = rows[(rows["d_confirm_time"] >= train_start) & (rows["d_confirm_time"] < train_end)].copy()
        test = rows[(rows["d_confirm_time"] >= train_end) & (rows["d_confirm_time"] < test_end)].copy()
        if len(train) < 300 or len(test) < 50 or train[args.target].nunique() < 2:
            current = month_add(current, args.test_months)
            continue
        x_train, x_test, y_train, feature_names = prepare_features(train, test, args.target)
        weights = train_logistic_regression(x_train, y_train, args.learning_rate, args.epochs, args.l2)
        probabilities = sigmoid(x_test @ weights)
        for row_index, probability in zip(test.index, probabilities):
            prediction_rows.append(
                {
                    "row_index": int(row_index),
                    "d_confirm_time": rows.loc[row_index, "d_confirm_time"],
                    "timeframe": rows.loc[row_index, "timeframe"],
                    "pattern_type": rows.loc[row_index, "pattern_type"],
                    "outcome": rows.loc[row_index, "outcome"],
                    "result_r": float(rows.loc[row_index, "result_r"]),
                    "reached_tp1": int(rows.loc[row_index, "reached_tp1"]),
                    "reached_tp2": int(rows.loc[row_index, "reached_tp2"]),
                    "probability": float(probability),
                    "target": args.target,
                    "test_start": train_end.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                }
            )
        for threshold in thresholds:
            report = evaluate(test, probabilities, threshold, args.target)
            report.update(
                {
                    "target": args.target,
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": train_end.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "train_rows": len(train),
                }
            )
            reports.append(report)
        for feature, weight in zip(feature_names, weights):
            coefficient_rows.append(
                {
                    "test_start": train_end.strftime("%Y-%m-%d"),
                    "feature": feature,
                    "coefficient": float(weight),
                }
            )
        current = month_add(current, args.test_months)
    return reports, coefficient_rows, prediction_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_events(Path(args.events), args.target)
    reports, coefficients, predictions = run_walk_forward(rows, args)
    write_csv(output_dir / "d_setup_filter_walk_forward.csv", reports)
    write_csv(output_dir / "d_setup_filter_coefficients.csv", coefficients)
    write_csv(output_dir / "d_setup_filter_predictions.csv", predictions)


if __name__ == "__main__":
    main()
