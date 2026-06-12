from __future__ import annotations

import argparse
import bisect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from abcd_backtester import Candle, calculate_atr, read_candles
from abcd_d_confirmation_event_study import bar_event, partial_exit_cost_r


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MSS-delayed entries for Candidate v1 AB=CD setups")
    parser.add_argument("--events", default="reports/wf_fib0382_risk125_partial_rerun/abcd_d_confirmation_events.csv")
    parser.add_argument("--predictions", default="reports/wf_fib0382_risk125_partial_rerun/ml_tp2/d_setup_filter_predictions.csv")
    parser.add_argument("--ltf-data", default="data/raw/BTCUSDT_5m.csv")
    parser.add_argument("--output-dir", default="reports/wf_fib0382_risk125_partial_rerun/mss_v2")
    parser.add_argument("--tp2-score-min", type=float, default=0.25)
    parser.add_argument("--atr-percent-min", type=float, default=0.185411)
    parser.add_argument("--atr-percent-max", type=float, default=0.873754)
    parser.add_argument("--lookback-bars", type=int, default=12)
    parser.add_argument("--max-mss-bars", type=int, default=96)
    parser.add_argument("--entry-max-wait-bars", type=int, default=24)
    parser.add_argument("--risk-atr-multiple", type=float, default=1.25)
    parser.add_argument("--risk-source", choices=("ltf_atr", "setup_v1"), default="setup_v1")
    parser.add_argument("--tp2-r", type=float, default=2.0)
    parser.add_argument("--maker-bps", type=float, default=2.0)
    parser.add_argument("--taker-bps", type=float, default=4.0)
    parser.add_argument("--entry-mode", choices=("mss_midpoint", "fvg_ce", "de_fib_0382"), default="mss_midpoint")
    parser.add_argument("--break-type", choices=("close", "wick"), default="close")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = load_candidate(args)
    ltf_candles = read_candles(Path(args.ltf_data))
    ltf_times = [pd.Timestamp(candle.timestamp) for candle in ltf_candles]
    ltf_atr = calculate_atr(ltf_candles, 14)

    rows: list[dict[str, Any]] = []
    for _, setup in candidate.iterrows():
        row = test_setup(setup, ltf_candles, ltf_times, ltf_atr, args)
        rows.append(row)

    result = pd.DataFrame(rows)
    result_path = output_dir / f"mss_delayed_entry_{args.entry_mode}_{args.break_type}.csv"
    summary_path = output_dir / f"mss_delayed_entry_{args.entry_mode}_{args.break_type}_summary.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    summarize(result).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")
    print(summarize(result).to_string(index=False))


def load_candidate(args: argparse.Namespace) -> pd.DataFrame:
    events = pd.read_csv(args.events).sort_values("d_confirm_time").reset_index(drop=True)
    events = events.reset_index().rename(columns={"index": "row_index"})
    predictions = pd.read_csv(args.predictions).rename(columns={"probability": "tp2_score"})
    rows = predictions.merge(events, on=["row_index", "d_confirm_time", "timeframe", "pattern_type", "outcome", "result_r", "reached_tp1", "reached_tp2"], how="left")
    rows = rows[
        (rows["timeframe"] != "4h")
        & (rows["tp2_score"] >= args.tp2_score_min)
        & (rows["atr_percent"] >= args.atr_percent_min)
        & (rows["atr_percent"] <= args.atr_percent_max)
    ].copy()
    return rows


def test_setup(
    setup: pd.Series,
    candles: list[Candle],
    times: list[pd.Timestamp],
    atr_values: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    d_index = first_index_at_or_after(times, pd.Timestamp(setup["d_extreme_time"]))
    if d_index is None:
        return skipped_row(setup, "no_ltf_data")
    mss = find_mss(setup, candles, times, d_index, args)
    if not mss["mss_confirmed"]:
        return skipped_row(setup, "no_mss", mss)
    entry = find_entry(setup, candles, mss, args)
    if not entry["filled"]:
        return skipped_row(setup, "no_retest", mss | entry)
    levels = build_levels(setup, candles, atr_values, mss, entry, args)
    if levels["risk"] <= 0:
        return skipped_row(setup, "bad_risk", mss | entry | levels)
    outcome = simulate_trade(candles, int(entry["entry_index"]), levels, str(setup["pattern_type"]), args)
    return base_row(setup) | mss | entry | levels | outcome | {
        "v2_reached_tp1": 1 if outcome["v2_outcome"] in {"tp1_then_be", "tp2"} else 0,
        "v2_reached_tp2": 1 if outcome["v2_outcome"] == "tp2" else 0,
    }


def first_index_at_or_after(times: list[pd.Timestamp], timestamp: pd.Timestamp) -> int | None:
    index = bisect.bisect_left(times, timestamp)
    return index if index < len(times) else None


def find_mss(setup: pd.Series, candles: list[Candle], times: list[pd.Timestamp], d_index: int, args: argparse.Namespace) -> dict[str, Any]:
    start = max(0, d_index - args.lookback_bars)
    if d_index <= start:
        return {"mss_confirmed": 0}
    previous = candles[start:d_index]
    is_bullish = setup["pattern_type"] == "bullish"
    break_level = max(c.high for c in previous) if is_bullish else min(c.low for c in previous)
    end = min(len(candles), d_index + args.max_mss_bars + 1)
    for index in range(d_index, end):
        candle = candles[index]
        value = candle.close if args.break_type == "close" else (candle.high if is_bullish else candle.low)
        if is_bullish and value > break_level:
            return mss_row(candles, index, d_index, break_level, setup)
        if not is_bullish and value < break_level:
            return mss_row(candles, index, d_index, break_level, setup)
    return {"mss_confirmed": 0, "mss_break_level": break_level}


def mss_row(candles: list[Candle], index: int, d_index: int, break_level: float, setup: pd.Series) -> dict[str, Any]:
    candle = candles[index]
    is_bullish = setup["pattern_type"] == "bullish"
    displacement_extreme = candle.high if is_bullish else candle.low
    midpoint = (break_level + displacement_extreme) / 2.0
    fvg = find_fvg(candles, d_index, index, is_bullish)
    return {
        "mss_confirmed": 1,
        "mss_index": index,
        "mss_time": candle.timestamp,
        "mss_bars": index - d_index,
        "mss_break_level": break_level,
        "mss_displacement_extreme": displacement_extreme,
        "mss_midpoint": midpoint,
        **fvg,
    }


def find_fvg(candles: list[Candle], start: int, end: int, is_bullish: bool) -> dict[str, Any]:
    for index in range(max(start + 2, 2), end + 1):
        left = candles[index - 2]
        right = candles[index]
        if is_bullish and right.low > left.high:
            return {"fvg_found": 1, "fvg_ce": (right.low + left.high) / 2.0}
        if not is_bullish and right.high < left.low:
            return {"fvg_found": 1, "fvg_ce": (right.high + left.low) / 2.0}
    return {"fvg_found": 0, "fvg_ce": 0.0}


def find_entry(setup: pd.Series, candles: list[Candle], mss: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.entry_mode == "mss_midpoint":
        level = float(mss["mss_midpoint"])
    elif args.entry_mode == "fvg_ce":
        if not mss.get("fvg_found"):
            return {"filled": 0, "entry_index": -1, "entry_price": 0.0, "entry_time": "", "entry_wait_bars": 0}
        level = float(mss["fvg_ce"])
    else:
        level = float(setup["d_price"]) + (float(setup["e_price"]) - float(setup["d_price"])) * 0.382
    start = int(mss["mss_index"]) + 1
    end = min(len(candles), start + args.entry_max_wait_bars)
    for index in range(start, end):
        candle = candles[index]
        if setup["pattern_type"] == "bullish" and candle.low <= level:
            return {"filled": 1, "entry_index": index, "entry_price": level, "entry_time": candle.timestamp, "entry_wait_bars": index - int(mss["mss_index"])}
        if setup["pattern_type"] == "bearish" and candle.high >= level:
            return {"filled": 1, "entry_index": index, "entry_price": level, "entry_time": candle.timestamp, "entry_wait_bars": index - int(mss["mss_index"])}
    return {"filled": 0, "entry_index": -1, "entry_price": level, "entry_time": "", "entry_wait_bars": 0}


def build_levels(
    setup: pd.Series,
    candles: list[Candle],
    atr_values: list[float],
    mss: dict[str, Any],
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    entry_price = float(entry["entry_price"])
    atr = atr_values[int(mss["mss_index"])]
    if args.risk_source == "setup_v1":
        risk = abs(float(setup["entry_price"]) - float(setup["stop"]))
    else:
        risk = atr * args.risk_atr_multiple
    if setup["pattern_type"] == "bullish":
        stop = entry_price - risk
        tp1 = float(setup["c_price"])
        tp2 = min(float(setup["a_price"]), entry_price + risk * args.tp2_r)
    else:
        stop = entry_price + risk
        tp1 = float(setup["c_price"])
        tp2 = max(float(setup["a_price"]), entry_price - risk * args.tp2_r)
    return {
        "entry": entry_price,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk,
        "tp1_r": abs(tp1 - entry_price) / risk if risk > 0 else 0.0,
        "tp2_r": abs(tp2 - entry_price) / risk if risk > 0 else 0.0,
    }


def simulate_trade(
    candles: list[Candle], entry_index: int, levels: dict[str, float], pattern_type: str, args: argparse.Namespace
) -> dict[str, Any]:
    fee_args = SimpleNamespace(cost_model="maker_taker", maker_bps=args.maker_bps, taker_bps=args.taker_bps, commission_bps=0.0, slippage_bps=0.0)
    tp1_hit = False
    end = min(len(candles), entry_index + 300)
    for index in range(entry_index, end):
        event = bar_event(candles[index], levels, pattern_type, tp1_hit)
        if event == "sl":
            result_r = -1.0 - exit_cost_r(levels, fee_args, "sl")
            return {"v2_outcome": "sl_before_tp1", "v2_result_r": result_r, "v2_exit_time": candles[index].timestamp}
        if event == "tp1":
            tp1_hit = True
            continue
        if event == "be":
            result_r = 0.5 * levels["tp1_r"] - partial_exit_cost_r(levels, fee_args, "be")
            return {"v2_outcome": "tp1_then_be", "v2_result_r": result_r, "v2_exit_time": candles[index].timestamp}
        if event == "tp2":
            result_r = 0.5 * levels["tp1_r"] + 0.5 * levels["tp2_r"] - partial_exit_cost_r(levels, fee_args, "tp2")
            return {"v2_outcome": "tp2", "v2_result_r": result_r, "v2_exit_time": candles[index].timestamp}
    return {"v2_outcome": "timeout", "v2_result_r": 0.0, "v2_exit_time": candles[end - 1].timestamp if end > entry_index else ""}


def exit_cost_r(levels: dict[str, float], args: Any, exit_type: str) -> float:
    if levels["risk"] <= 0:
        return 0.0
    entry_cost = levels["entry"] * args.maker_bps / 10000.0
    if exit_type == "tp2":
        exit_price = levels["tp2"]
        exit_bps = args.maker_bps
    elif exit_type == "be":
        exit_price = levels["entry"]
        exit_bps = args.taker_bps
    else:
        exit_price = levels["stop"]
        exit_bps = args.taker_bps
    return (entry_cost + exit_price * exit_bps / 10000.0) / levels["risk"]


def skipped_row(setup: pd.Series, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return base_row(setup) | (extra or {}) | {
        "skip_reason": reason,
        "filled": 0,
        "v2_outcome": "skip",
        "v2_result_r": 0.0,
        "v2_reached_tp1": 0,
        "v2_reached_tp2": 0,
    }


def base_row(setup: pd.Series) -> dict[str, Any]:
    return {
        "row_index": int(setup["row_index"]),
        "timeframe": setup["timeframe"],
        "pattern_type": setup["pattern_type"],
        "d_extreme_time": setup["d_extreme_time"],
        "d_confirm_time": setup["d_confirm_time"],
        "v1_outcome": setup["outcome"],
        "v1_result_r": setup["result_r"],
        "tp2_score": setup["tp2_score"],
        "atr_percent": setup["atr_percent"],
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    executed = rows[rows["v2_outcome"] != "skip"].copy()
    groups = [("ALL", executed)]
    groups.extend((str(name), group) for name, group in executed.groupby("timeframe"))
    groups.extend((str(name), group) for name, group in executed.groupby("pattern_type"))
    summary = []
    for name, group in groups:
        count = len(group)
        sl = int((group["v2_outcome"] == "sl_before_tp1").sum())
        tp2 = int((group["v2_outcome"] == "tp2").sum())
        summary.append(
            {
                "group": name,
                "rows": count,
                "sl": sl,
                "tp2": tp2,
                "sl_rate": sl / count if count else 0.0,
                "tp2_rate": tp2 / count if count else 0.0,
                "avg_r": group["v2_result_r"].mean() if count else 0.0,
                "total_r": group["v2_result_r"].sum() if count else 0.0,
            }
        )
    return pd.DataFrame(summary)


if __name__ == "__main__":
    main()
