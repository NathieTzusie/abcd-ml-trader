from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from abcd_backtester import Candle, calculate_atr, read_candles


TIMEFRAMES = ("15m", "30m", "1h", "4h")


@dataclass(frozen=True)
class ConfirmedSwing:
    index: int
    confirm_index: int
    timestamp: str
    confirm_time: str
    price: float
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward AB=CD D-confirmation event study")
    parser.add_argument("--data-dir", default="data/raw", help="原始 OHLCV CSV 目录")
    parser.add_argument("--output-dir", default="reports", help="输出目录")
    parser.add_argument("--symbol", default="BTCUSDT", help="品种，例如 BTCUSDT")
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES), help="逗号分隔周期")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR 周期")
    parser.add_argument("--atr-multiple", type=float, default=1.5, help="D 点反向确认 ATR 倍数")
    parser.add_argument("--atr-buffer", type=float, default=0.25, help="SL 放在 D 点外侧的 ATR 缓冲")
    parser.add_argument("--max-d-search-bars", type=int, default=300, help="C 点确认后最多等待多少根 K 线触达 D 区")
    parser.add_argument("--max-confirm-bars", type=int, default=120, help="触达 D 区后最多等待多少根 K 线确认反转")
    parser.add_argument("--max-hold-bars", type=int, default=300, help="确认 D 后最多观察多少根 K 线")
    parser.add_argument("--tp2-r", type=float, default=2.0, help="TP2 最大按多少 R 计算")
    parser.add_argument("--entry-mode", choices=("immediate", "fib_retrace"), default="immediate", help="D 确认后的入场模式")
    parser.add_argument("--entry-fib", type=float, default=0.5, help="D 到 E 位移的回踩比例")
    parser.add_argument("--entry-max-wait-bars", type=int, default=24, help="确认 D 后最多等待多少根 K 线回踩入场")
    parser.add_argument("--risk-mode", choices=("structure", "fixed_atr"), default="structure", help="R 的计算模式")
    parser.add_argument("--risk-atr-multiple", type=float, default=1.75, help="固定 ATR 风险倍数")
    parser.add_argument("--commission-bps", type=float, default=0.0, help="单边手续费，单位 bps")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="单边滑点，单位 bps")
    parser.add_argument("--cost-model", choices=("uniform", "maker_taker"), default="uniform", help="成本模型")
    parser.add_argument("--maker-bps", type=float, default=2.0, help="maker 单边手续费，单位 bps")
    parser.add_argument("--taker-bps", type=float, default=4.0, help="taker 单边手续费，单位 bps")
    parser.add_argument("--tp1-action", choices=("partial", "be_only"), default="partial", help="TP1 到达后的处理")
    return parser.parse_args()


def detect_confirmed_swings(candles: list[Candle], atr_values: list[float], atr_multiple: float) -> list[ConfirmedSwing]:
    """按时间向前确认 swing，记录极点时间和确认时间。"""
    if len(candles) < 10:
        return []
    direction = initial_direction(candles, atr_values, atr_multiple)
    if direction == 0:
        return []
    initial = initial_swing(candles, direction)
    swings = [initial]
    extreme_index = initial.index
    extreme_price = initial.price
    for index in range(extreme_index + 1, len(candles)):
        candle = candles[index]
        reversal = reversal_amount(atr_values, index, candle.close, atr_multiple)
        if direction == 1:
            if candle.high > extreme_price:
                extreme_index = index
                extreme_price = candle.high
            elif candle.low <= extreme_price - reversal:
                swings.append(
                    ConfirmedSwing(
                        extreme_index,
                        index,
                        candles[extreme_index].timestamp,
                        candle.timestamp,
                        extreme_price,
                        "high",
                    )
                )
                direction = -1
                extreme_index = index
                extreme_price = candle.low
        else:
            if candle.low < extreme_price:
                extreme_index = index
                extreme_price = candle.low
            elif candle.high >= extreme_price + reversal:
                swings.append(
                    ConfirmedSwing(
                        extreme_index,
                        index,
                        candles[extreme_index].timestamp,
                        candle.timestamp,
                        extreme_price,
                        "low",
                    )
                )
                direction = 1
                extreme_index = index
                extreme_price = candle.high
    return dedupe_swings(swings)


def initial_direction(candles: list[Candle], atr_values: list[float], atr_multiple: float) -> int:
    base_high = candles[0].high
    base_low = candles[0].low
    for index, candle in enumerate(candles[1:], start=1):
        reversal = reversal_amount(atr_values, index, candle.close, atr_multiple)
        if candle.high >= base_low + reversal:
            return 1
        if candle.low <= base_high - reversal:
            return -1
        base_high = max(base_high, candle.high)
        base_low = min(base_low, candle.low)
    return 0


def initial_swing(candles: list[Candle], direction: int) -> ConfirmedSwing:
    prices = [candle.low if direction == 1 else candle.high for candle in candles[:10]]
    index = prices.index(min(prices) if direction == 1 else max(prices))
    kind = "low" if direction == 1 else "high"
    return ConfirmedSwing(index, index, candles[index].timestamp, candles[index].timestamp, prices[index], kind)


def reversal_amount(atr_values: list[float], index: int, price: float, atr_multiple: float) -> float:
    atr = atr_values[index] if index < len(atr_values) else 0.0
    return atr * atr_multiple if atr > 0 else price * 0.015


def dedupe_swings(swings: list[ConfirmedSwing]) -> list[ConfirmedSwing]:
    cleaned: list[ConfirmedSwing] = []
    for swing in sorted(swings, key=lambda item: (item.confirm_index, item.index)):
        if cleaned and cleaned[-1].kind == swing.kind:
            cleaned[-1] = more_extreme(cleaned[-1], swing)
        else:
            cleaned.append(swing)
    return cleaned


def more_extreme(left: ConfirmedSwing, right: ConfirmedSwing) -> ConfirmedSwing:
    if left.kind == "high":
        return left if left.price >= right.price else right
    return left if left.price <= right.price else right


def run_timeframe(path: Path, timeframe: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    candles = read_candles(path)
    atr_values = calculate_atr(candles, args.atr_period)
    swings = detect_confirmed_swings(candles, atr_values, args.atr_multiple)
    rows: list[dict[str, Any]] = []
    for index in range(len(swings) - 2):
        a, b, c = swings[index : index + 3]
        pattern_type = abc_pattern_type(a, b, c)
        if not pattern_type:
            continue
        ab = abs(b.price - a.price)
        bc = abs(c.price - b.price)
        if ab <= 0:
            continue
        bc_ratio = bc / ab
        if not 0.382 <= bc_ratio <= 0.886:
            continue
        event = study_abc_event(candles, atr_values, a, b, c, pattern_type, args)
        if event:
            event["symbol"] = args.symbol
            event["timeframe"] = timeframe
            event["bc_ratio"] = bc_ratio
            rows.append(event)
    return rows


def abc_pattern_type(a: ConfirmedSwing, b: ConfirmedSwing, c: ConfirmedSwing) -> str | None:
    kinds = (a.kind, b.kind, c.kind)
    if kinds == ("high", "low", "high"):
        return "bullish"
    if kinds == ("low", "high", "low"):
        return "bearish"
    return None


def study_abc_event(
    candles: list[Candle],
    atr_values: list[float],
    a: ConfirmedSwing,
    b: ConfirmedSwing,
    c: ConfirmedSwing,
    pattern_type: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    ab = abs(b.price - a.price)
    zone_low, zone_high = d_projection_zone(c.price, ab, pattern_type)
    reach = find_d_zone_reach(candles, c.confirm_index + 1, zone_low, zone_high, pattern_type, args.max_d_search_bars)
    if not reach:
        return None
    confirmation = confirm_d_reversal(candles, atr_values, reach["index"], pattern_type, args)
    if not confirmation:
        return None
    entry_plan = resolve_entry(candles, confirmation, pattern_type, args)
    levels = trade_levels_from_candles(candles, a, c, confirmation, atr_values, pattern_type, args, entry_plan)
    if levels["risk"] <= 0:
        return None
    if not entry_plan["filled"]:
        outcome = {"outcome": "no_fill", "result_r": 0.0, "exit_time": "", "bars_after_confirm": 0}
    else:
        outcome = simulate_outcome(candles, int(entry_plan["entry_index"]), levels, pattern_type, args)
    confirm_index = confirmation["confirm_index"]
    d_index = confirmation["d_index"]
    d_atr = atr_values[d_index] if d_index < len(atr_values) else 0.0
    return {
        "pattern_type": pattern_type,
        "a_time": a.timestamp,
        "a_price": a.price,
        "b_time": b.timestamp,
        "b_price": b.price,
        "c_time": c.timestamp,
        "c_confirm_time": c.confirm_time,
        "c_price": c.price,
        "d_zone_low": zone_low,
        "d_zone_high": zone_high,
        "d_reach_time": candles[reach["index"]].timestamp,
        "d_extreme_time": candles[confirmation["d_index"]].timestamp,
        "d_confirm_time": candles[confirmation["confirm_index"]].timestamp,
        "d_price": confirmation["d_price"],
        "d_search_bars": reach["index"] - c.confirm_index,
        "d_confirm_bars": confirm_index - reach["index"],
        "d_overshoot_atr": d_overshoot_atr(confirmation["d_price"], zone_low, zone_high, pattern_type, d_atr),
        "atr_percent": atr_values[confirm_index] / candles[confirm_index].close * 100.0 if candles[confirm_index].close else 0.0,
        "volume_zscore": volume_zscore(candles, confirm_index, 20),
        "trend_slope": trend_slope(candles, reach["index"], 20),
        "d_position_range": range_position(candles, d_index, 50, confirmation["d_price"]),
        "reached_tp1": 1 if outcome["outcome"] in {"tp1_then_be", "tp2"} else 0,
        "reached_tp2": 1 if outcome["outcome"] == "tp2" else 0,
        "entry_mode": args.entry_mode,
        "entry_fib": args.entry_fib,
        "risk_mode": args.risk_mode,
        "risk_atr_multiple": args.risk_atr_multiple,
        "e_price": entry_plan["e_price"],
        "planned_entry": entry_plan["entry_price"],
        "filled": 1 if entry_plan["filled"] else 0,
        "entry_wait_bars": entry_plan["entry_wait_bars"],
        "entry_time": entry_plan["entry_time"],
        "entry_price": levels["entry"],
        "stop": levels["stop"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "tp1_r": levels["tp1_r"],
        "tp2_r": levels["tp2_r"],
        "outcome": outcome["outcome"],
        "result_r": outcome["result_r"],
        "exit_time": outcome["exit_time"],
        "bars_after_confirm": outcome["bars_after_confirm"],
    }


def resolve_entry(
    candles: list[Candle], confirmation: dict[str, Any], pattern_type: str, args: argparse.Namespace
) -> dict[str, Any]:
    confirm_index = confirmation["confirm_index"]
    if confirm_index + 1 >= len(candles):
        return empty_entry(0.0, 0.0)
    if pattern_type == "bullish":
        e_price = candles[confirm_index].high
    else:
        e_price = candles[confirm_index].low
    entry_price = confirmation["d_price"] + (e_price - confirmation["d_price"]) * args.entry_fib
    if args.entry_mode == "immediate":
        entry_index = confirm_index + 1
        return {
            "filled": True,
            "entry_index": entry_index,
            "entry_time": candles[entry_index].timestamp,
            "entry_price": candles[entry_index].open,
            "entry_wait_bars": 1,
            "e_price": e_price,
            "planned_entry": candles[entry_index].open,
        }
    end = min(len(candles), confirm_index + args.entry_max_wait_bars + 1)
    for index in range(confirm_index + 1, end):
        candle = candles[index]
        if pattern_type == "bullish" and candle.low <= entry_price:
            return filled_entry(candle, index, confirm_index, entry_price, e_price)
        if pattern_type == "bearish" and candle.high >= entry_price:
            return filled_entry(candle, index, confirm_index, entry_price, e_price)
    return empty_entry(entry_price, e_price)


def filled_entry(candle: Candle, index: int, confirm_index: int, entry_price: float, e_price: float) -> dict[str, Any]:
    return {
        "filled": True,
        "entry_index": index,
        "entry_time": candle.timestamp,
        "entry_price": entry_price,
        "entry_wait_bars": index - confirm_index,
        "e_price": e_price,
        "planned_entry": entry_price,
    }


def empty_entry(entry_price: float, e_price: float) -> dict[str, Any]:
    return {
        "filled": False,
        "entry_index": -1,
        "entry_time": "",
        "entry_price": entry_price,
        "entry_wait_bars": 0,
        "e_price": e_price,
        "planned_entry": entry_price,
    }


def d_overshoot_atr(d_price: float, zone_low: float, zone_high: float, pattern_type: str, atr: float) -> float:
    if atr <= 0:
        return 0.0
    if pattern_type == "bullish":
        return max(0.0, zone_low - d_price) / atr
    return max(0.0, d_price - zone_high) / atr


def volume_zscore(candles: list[Candle], index: int, window: int) -> float:
    start = max(0, index + 1 - window)
    volumes = [candle.volume for candle in candles[start : index + 1]]
    if len(volumes) < 2:
        return 0.0
    mean = sum(volumes) / len(volumes)
    variance = sum((value - mean) ** 2 for value in volumes) / (len(volumes) - 1)
    return (volumes[-1] - mean) / variance ** 0.5 if variance > 0 else 0.0


def trend_slope(candles: list[Candle], index: int, window: int) -> float:
    start = max(0, index - window)
    base = candles[start].close
    return (candles[index].close - base) / base * 100.0 if base else 0.0


def range_position(candles: list[Candle], index: int, window: int, price: float) -> float:
    start = max(0, index + 1 - window)
    high = max(candle.high for candle in candles[start : index + 1])
    low = min(candle.low for candle in candles[start : index + 1])
    return (price - low) / (high - low) if high > low else 0.5


def d_projection_zone(c_price: float, ab: float, pattern_type: str) -> tuple[float, float]:
    if pattern_type == "bullish":
        low = c_price - 1.1 * ab
        high = c_price - 0.9 * ab
    else:
        low = c_price + 0.9 * ab
        high = c_price + 1.1 * ab
    return min(low, high), max(low, high)


def find_d_zone_reach(
    candles: list[Candle], start: int, zone_low: float, zone_high: float, pattern_type: str, max_bars: int
) -> dict[str, int] | None:
    end = min(len(candles), start + max_bars)
    for index in range(start, end):
        candle = candles[index]
        if pattern_type == "bullish" and candle.low <= zone_high:
            return {"index": index}
        if pattern_type == "bearish" and candle.high >= zone_low:
            return {"index": index}
    return None


def confirm_d_reversal(
    candles: list[Candle], atr_values: list[float], reach_index: int, pattern_type: str, args: argparse.Namespace
) -> dict[str, Any] | None:
    extreme_index = reach_index
    extreme_price = candles[reach_index].low if pattern_type == "bullish" else candles[reach_index].high
    end = min(len(candles), reach_index + args.max_confirm_bars + 1)
    for index in range(reach_index + 1, end):
        candle = candles[index]
        reversal = reversal_amount(atr_values, index, candle.close, args.atr_multiple)
        if pattern_type == "bullish":
            if candle.low < extreme_price:
                extreme_price = candle.low
                extreme_index = index
            elif candle.high >= extreme_price + reversal:
                return {"d_index": extreme_index, "confirm_index": index, "d_price": extreme_price}
        else:
            if candle.high > extreme_price:
                extreme_price = candle.high
                extreme_index = index
            elif candle.low <= extreme_price - reversal:
                return {"d_index": extreme_index, "confirm_index": index, "d_price": extreme_price}
    return None


def trade_levels_from_candles(
    candles: list[Candle],
    a: ConfirmedSwing,
    c: ConfirmedSwing,
    confirmation: dict[str, Any],
    atr_values: list[float],
    pattern_type: str,
    args: argparse.Namespace,
    entry_plan: dict[str, Any],
) -> dict[str, float]:
    if not entry_plan["filled"]:
        return {"risk": 1.0, "entry": entry_plan["entry_price"], "stop": 0.0, "tp1": 0.0, "tp2": 0.0, "tp1_r": 0.0, "tp2_r": 0.0}
    entry_index = int(entry_plan["entry_index"])
    if entry_index >= len(candles):
        return {"risk": 0.0, "entry": 0.0, "stop": 0.0, "tp1": 0.0, "tp2": 0.0, "tp1_r": 0.0, "tp2_r": 0.0}
    entry = float(entry_plan["entry_price"])
    atr = atr_values[confirmation["d_index"]]
    fixed_risk = atr * args.risk_atr_multiple
    buffer = atr * args.atr_buffer
    if pattern_type == "bullish":
        stop = entry - fixed_risk if args.risk_mode == "fixed_atr" else confirmation["d_price"] - buffer
        risk = entry - stop
        tp1 = c.price
        tp2 = min(a.price, entry + risk * args.tp2_r)
    else:
        stop = entry + fixed_risk if args.risk_mode == "fixed_atr" else confirmation["d_price"] + buffer
        risk = stop - entry
        tp1 = c.price
        tp2 = max(a.price, entry - risk * args.tp2_r)
    return {
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk,
        "tp1_r": abs(tp1 - entry) / risk if risk > 0 else 0.0,
        "tp2_r": abs(tp2 - entry) / risk if risk > 0 else 0.0,
    }


def simulate_outcome(
    candles: list[Candle], entry_index: int, levels: dict[str, float], pattern_type: str, args: argparse.Namespace
) -> dict[str, Any]:
    if entry_index >= len(candles):
        return {"outcome": "no_entry", "result_r": 0.0, "exit_time": "", "bars_after_confirm": 0}
    tp1_hit = False
    end = min(len(candles), entry_index + args.max_hold_bars)
    uniform_cost_r = round_trip_cost_r(levels, args.commission_bps, args.slippage_bps)
    for index in range(entry_index, end):
        candle = candles[index]
        event = bar_event(candle, levels, pattern_type, tp1_hit)
        if event == "sl":
            result_r = -1.0 - exit_cost_r(levels, args, "sl")
            return {"outcome": "sl_before_tp1", "result_r": result_r, "exit_time": candle.timestamp, "bars_after_confirm": index - entry_index + 1}
        if event == "tp1":
            tp1_hit = True
            continue
        if event == "be":
            if args.tp1_action == "be_only":
                result_r = 0.0 - exit_cost_r(levels, args, "be")
            else:
                result_r = 0.5 * levels["tp1_r"] - partial_exit_cost_r(levels, args, "be")
            return {"outcome": "tp1_then_be", "result_r": result_r, "exit_time": candle.timestamp, "bars_after_confirm": index - entry_index + 1}
        if event == "tp2":
            if args.tp1_action == "be_only":
                result_r = levels["tp2_r"] - exit_cost_r(levels, args, "tp2")
            else:
                result_r = 0.5 * levels["tp1_r"] + 0.5 * levels["tp2_r"] - partial_exit_cost_r(levels, args, "tp2")
            return {
                "outcome": "tp2",
                "result_r": result_r,
                "exit_time": candle.timestamp,
                "bars_after_confirm": index - entry_index + 1,
            }
    return {"outcome": "timeout", "result_r": 0.0, "exit_time": candles[end - 1].timestamp, "bars_after_confirm": end - entry_index}


def round_trip_cost_r(levels: dict[str, float], commission_bps: float, slippage_bps: float) -> float:
    if levels["risk"] <= 0:
        return 0.0
    total_bps = (commission_bps + slippage_bps) * 2.0
    cost_price = levels["entry"] * total_bps / 10000.0
    return cost_price / levels["risk"]


def exit_cost_r(levels: dict[str, float], args: argparse.Namespace, exit_type: str) -> float:
    if levels["risk"] <= 0:
        return 0.0
    if args.cost_model == "uniform":
        return round_trip_cost_r(levels, args.commission_bps, args.slippage_bps)
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
    exit_cost = exit_price * exit_bps / 10000.0
    return (entry_cost + exit_cost) / levels["risk"]


def partial_exit_cost_r(levels: dict[str, float], args: argparse.Namespace, final_exit_type: str) -> float:
    if args.cost_model == "uniform":
        return round_trip_cost_r(levels, args.commission_bps, args.slippage_bps)
    if levels["risk"] <= 0:
        return 0.0
    entry_cost = levels["entry"] * args.maker_bps / 10000.0
    tp1_cost = 0.5 * levels["tp1"] * args.maker_bps / 10000.0
    if final_exit_type == "tp2":
        final_exit_cost = 0.5 * levels["tp2"] * args.maker_bps / 10000.0
    else:
        final_exit_cost = 0.5 * levels["entry"] * args.taker_bps / 10000.0
    return (entry_cost + tp1_cost + final_exit_cost) / levels["risk"]


def bar_event(candle: Candle, levels: dict[str, float], pattern_type: str, tp1_hit: bool) -> str | None:
    if pattern_type == "bullish":
        if not tp1_hit and candle.low <= levels["stop"]:
            return "sl"
        if tp1_hit and candle.low <= levels["entry"]:
            return "be"
        if candle.high >= levels["tp2"]:
            return "tp2"
        if not tp1_hit and candle.high >= levels["tp1"]:
            return "tp1"
    else:
        if not tp1_hit and candle.high >= levels["stop"]:
            return "sl"
        if tp1_hit and candle.high >= levels["entry"]:
            return "be"
        if candle.low <= levels["tp2"]:
            return "tp2"
        if not tp1_hit and candle.low <= levels["tp1"]:
            return "tp1"
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for timeframe in sorted({row["timeframe"] for row in rows}):
        subset = [row for row in rows if row["timeframe"] == timeframe]
        result.append(summary_row(timeframe, subset))
    result.append(summary_row("all", rows))
    return result


def summary_row(timeframe: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = ["no_fill", "sl_before_tp1", "tp1_then_be", "tp2", "timeout", "no_entry"]
    total = len(rows)
    results = [float(row["result_r"]) for row in rows]
    row: dict[str, Any] = {
        "timeframe": timeframe,
        "events": total,
        "avg_result_r": sum(results) / total if total else 0.0,
        "median_result_r": median(results) if results else 0.0,
    }
    for outcome in outcomes:
        count = sum(1 for item in rows if item["outcome"] == outcome)
        row[f"{outcome}_count"] = count
        row[f"{outcome}_prob"] = count / total if total else 0.0
    tp1_related = sum(1 for item in rows if item["outcome"] in {"tp1_then_be", "tp2"})
    row["tp1_reached_count"] = tp1_related
    row["tp1_reached_prob"] = tp1_related / total if total else 0.0
    return row


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    for timeframe in [value.strip() for value in args.timeframes.split(",") if value.strip()]:
        path = data_dir / f"{args.symbol}_{timeframe}.csv"
        if not path.exists():
            continue
        rows = run_timeframe(path, timeframe, args)
        all_rows.extend(rows)
    write_csv(output_dir / "abcd_d_confirmation_events.csv", all_rows)
    write_csv(output_dir / "abcd_d_confirmation_summary.csv", summarize(all_rows))


if __name__ == "__main__":
    main()
