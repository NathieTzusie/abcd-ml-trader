from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


TIMEFRAMES = ("15m", "30m", "1h", "4h")
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class SwingPoint:
    index: int
    timestamp: str
    price: float
    kind: str


@dataclass(frozen=True)
class Pattern:
    pattern_type: str
    a: SwingPoint
    b: SwingPoint
    c: SwingPoint
    d: SwingPoint
    bc_ratio: float
    cd_ratio: float
    time_symmetry: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AB=CD harmonic reversal backtester")
    parser.add_argument("--data-dir", default="data/raw", help="原始 OHLCV CSV 目录")
    parser.add_argument("--output-dir", default="reports", help="结果输出目录")
    parser.add_argument("--symbol", default=None, help="只回测指定品种，例如 BTCUSDT")
    parser.add_argument("--zigzag-mode", choices=("atr", "percent"), default="atr", help="ZigZag 极点确认模式")
    parser.add_argument("--zigzag-pct", type=float, default=1.5, help="ZigZag 最小反转百分比")
    parser.add_argument("--zigzag-atr-multiple", type=float, default=1.0, help="ZigZag ATR 反转倍数")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR 周期")
    parser.add_argument("--atr-buffer", type=float, default=0.25, help="止损 ATR 缓冲倍数")
    parser.add_argument("--max-hold-bars", type=int, default=200, help="单笔最多持有 K 线数")
    parser.add_argument("--min-tp1-r", type=float, default=1.0, help="TP1 最小 R 值")
    parser.add_argument("--tp2-r", type=float, default=2.0, help="TP2 最大按多少 R 计算")
    parser.add_argument("--commission-bps", type=float, default=0.0, help="单边手续费，单位 bps")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="单边滑点，单位 bps")
    parser.add_argument("--require-mss", action="store_true", help="要求 D 点后出现 MSS 才入场")
    parser.add_argument("--mss-lookback-bars", type=int, default=5, help="MSS 参考的 D 点前局部结构 K 线数")
    parser.add_argument("--mss-max-confirm-bars", type=int, default=12, help="D 点后最多等待多少根 K 线确认 MSS")
    parser.add_argument("--mss-confirm-close", action="store_true", help="使用收盘价突破确认 MSS，默认使用 high/low 突破")
    return parser.parse_args()


def read_candles(path: Path) -> list[Candle]:
    """读取 OHLCV CSV，并显式校验字段，避免用错数据源。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} 缺少字段: {', '.join(missing)}")
        return [_row_to_candle(row) for row in reader]


def _row_to_candle(row: dict[str, str]) -> Candle:
    return Candle(
        timestamp=row["timestamp"],
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )


def detect_swings(
    candles: list[Candle],
    zigzag_mode: str,
    zigzag_pct: float,
    atr_values: list[float],
    atr_multiple: float,
) -> list[SwingPoint]:
    """用 ZigZag 逻辑把连续 K 线压缩成 high/low 波段点。"""
    if len(candles) < 10:
        return []
    direction = _initial_direction(candles, zigzag_mode, zigzag_pct, atr_values, atr_multiple)
    if direction == 0:
        return []
    swings = [_initial_swing(candles, direction)]
    extreme_index = swings[0].index
    extreme_price = swings[0].price
    return _walk_swings(candles, zigzag_mode, zigzag_pct, atr_values, atr_multiple, direction, extreme_index, extreme_price, swings)


def _initial_direction(
    candles: list[Candle], zigzag_mode: str, zigzag_pct: float, atr_values: list[float], atr_multiple: float
) -> int:
    base_high = candles[0].high
    base_low = candles[0].low
    for index, candle in enumerate(candles[1:], start=1):
        reversal = _reversal_amount(base_low, index, zigzag_mode, zigzag_pct, atr_values, atr_multiple)
        if candle.high >= base_low + reversal:
            return 1
        reversal = _reversal_amount(base_high, index, zigzag_mode, zigzag_pct, atr_values, atr_multiple)
        if candle.low <= base_high - reversal:
            return -1
        base_high = max(base_high, candle.high)
        base_low = min(base_low, candle.low)
    return 0


def _initial_swing(candles: list[Candle], direction: int) -> SwingPoint:
    prices = [candle.low if direction == 1 else candle.high for candle in candles[:10]]
    index = prices.index(min(prices) if direction == 1 else max(prices))
    kind = "low" if direction == 1 else "high"
    return SwingPoint(index, candles[index].timestamp, prices[index], kind)


def _walk_swings(
    candles: list[Candle],
    zigzag_mode: str,
    zigzag_pct: float,
    atr_values: list[float],
    atr_multiple: float,
    direction: int,
    extreme_index: int,
    extreme_price: float,
    swings: list[SwingPoint],
) -> list[SwingPoint]:
    for index in range(extreme_index + 1, len(candles)):
        candle = candles[index]
        if direction == 1:
            extreme_index, extreme_price, direction = _handle_uptrend(
                candle, index, zigzag_mode, zigzag_pct, atr_values, atr_multiple, extreme_index, extreme_price, direction, candles, swings
            )
        else:
            extreme_index, extreme_price, direction = _handle_downtrend(
                candle, index, zigzag_mode, zigzag_pct, atr_values, atr_multiple, extreme_index, extreme_price, direction, candles, swings
            )
    return _dedupe_swings(swings)


def _handle_uptrend(
    candle: Candle,
    index: int,
    zigzag_mode: str,
    zigzag_pct: float,
    atr_values: list[float],
    atr_multiple: float,
    extreme_index: int,
    extreme_price: float,
    direction: int,
    candles: list[Candle],
    swings: list[SwingPoint],
) -> tuple[int, float, int]:
    if candle.high > extreme_price:
        return index, candle.high, direction
    reversal = _reversal_amount(extreme_price, index, zigzag_mode, zigzag_pct, atr_values, atr_multiple)
    if candle.low <= extreme_price - reversal:
        swings.append(SwingPoint(extreme_index, candles[extreme_index].timestamp, extreme_price, "high"))
        return index, candle.low, -1
    return extreme_index, extreme_price, direction


def _handle_downtrend(
    candle: Candle,
    index: int,
    zigzag_mode: str,
    zigzag_pct: float,
    atr_values: list[float],
    atr_multiple: float,
    extreme_index: int,
    extreme_price: float,
    direction: int,
    candles: list[Candle],
    swings: list[SwingPoint],
) -> tuple[int, float, int]:
    if candle.low < extreme_price:
        return index, candle.low, direction
    reversal = _reversal_amount(extreme_price, index, zigzag_mode, zigzag_pct, atr_values, atr_multiple)
    if candle.high >= extreme_price + reversal:
        swings.append(SwingPoint(extreme_index, candles[extreme_index].timestamp, extreme_price, "low"))
        return index, candle.high, 1
    return extreme_index, extreme_price, direction


def _reversal_amount(
    price: float, index: int, zigzag_mode: str, zigzag_pct: float, atr_values: list[float], atr_multiple: float
) -> float:
    if zigzag_mode == "atr":
        atr = atr_values[index] if index < len(atr_values) else 0.0
        return atr * atr_multiple if atr > 0 else price * zigzag_pct / 100.0
    return price * zigzag_pct / 100.0


def _dedupe_swings(swings: list[SwingPoint]) -> list[SwingPoint]:
    cleaned: list[SwingPoint] = []
    for swing in sorted(swings, key=lambda item: item.index):
        if cleaned and cleaned[-1].index == swing.index:
            continue
        if cleaned and cleaned[-1].kind == swing.kind:
            cleaned[-1] = _more_extreme(cleaned[-1], swing)
            continue
        cleaned.append(swing)
    return cleaned


def _more_extreme(left: SwingPoint, right: SwingPoint) -> SwingPoint:
    if left.kind == "high":
        return left if left.price >= right.price else right
    return left if left.price <= right.price else right


def detect_abcd_patterns(swings: list[SwingPoint]) -> list[Pattern]:
    """从连续四个 swing points 中识别 AB=CD 候选形态。"""
    patterns: list[Pattern] = []
    for index in range(len(swings) - 3):
        a, b, c, d = swings[index : index + 4]
        pattern_type = _pattern_type(a, b, c, d)
        if not pattern_type:
            continue
        pattern = _build_pattern(pattern_type, a, b, c, d)
        if pattern:
            patterns.append(pattern)
    return patterns


def _pattern_type(a: SwingPoint, b: SwingPoint, c: SwingPoint, d: SwingPoint) -> str | None:
    kinds = (a.kind, b.kind, c.kind, d.kind)
    if kinds == ("high", "low", "high", "low"):
        return "bullish"
    if kinds == ("low", "high", "low", "high"):
        return "bearish"
    return None


def _build_pattern(
    pattern_type: str, a: SwingPoint, b: SwingPoint, c: SwingPoint, d: SwingPoint
) -> Pattern | None:
    ab = abs(b.price - a.price)
    bc = abs(c.price - b.price)
    cd = abs(d.price - c.price)
    if ab == 0:
        return None
    bc_ratio = bc / ab
    cd_ratio = cd / ab
    ab_bars = max(1, b.index - a.index)
    cd_bars = max(1, d.index - c.index)
    if 0.382 <= bc_ratio <= 0.886 and 0.90 <= cd_ratio <= 1.10:
        return Pattern(pattern_type, a, b, c, d, bc_ratio, cd_ratio, cd_bars / ab_bars)
    return None


def calculate_atr(candles: list[Candle], period: int) -> list[float]:
    """计算 ATR，用于给 D 点外侧止损留结构缓冲。"""
    values = [0.0] * len(candles)
    true_ranges: list[float] = []
    for index, candle in enumerate(candles):
        previous_close = candles[index - 1].close if index > 0 else candle.close
        true_range = max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close))
        true_ranges.append(true_range)
        if index + 1 >= period:
            values[index] = sum(true_ranges[index + 1 - period : index + 1]) / period
    return values


def backtest_pattern(
    candles: list[Candle],
    atr_values: list[float],
    pattern: Pattern,
    args: argparse.Namespace,
) -> dict[str, object]:
    atr = atr_values[pattern.d.index] or candles[pattern.d.index].high - candles[pattern.d.index].low
    mss = find_mss_confirmation(candles, pattern, atr, args)
    features = build_features(candles, pattern, atr_values, mss)
    if args.require_mss and not mss["mss_confirmed"]:
        return _pattern_row(pattern, "skip", 0.0, "no_mss_confirmation", 0.0, 0.0, features)
    entry_index = int(mss["mss_index"]) + 1 if args.require_mss else pattern.d.index + 1
    if entry_index >= len(candles):
        return _pattern_row(pattern, "skip", 0.0, "no_entry_candle", 0.0, 0.0, features)
    entry = candles[entry_index].open
    levels = _trade_levels(pattern, entry, atr, args.atr_buffer, args.tp2_r)
    if levels["risk"] <= 0 or levels["tp1_r"] < args.min_tp1_r:
        return _pattern_row(pattern, "skip", 0.0, "bad_reward_risk", levels["tp1_r"], levels["tp2_r"], features)
    result = _simulate_trade(
        candles,
        entry_index,
        levels,
        pattern.pattern_type,
        args.max_hold_bars,
        args.commission_bps,
        args.slippage_bps,
    )
    return _pattern_row(pattern, result["label"], result["result_r"], result["exit_reason"], levels["tp1_r"], levels["tp2_r"], features)


def find_mss_confirmation(
    candles: list[Candle], pattern: Pattern, atr: float, args: argparse.Namespace
) -> dict[str, object]:
    """用 D 点附近局部结构突破近似 MSS，避免直接在 D 点盲目入场。"""
    start = max(0, pattern.d.index - args.mss_lookback_bars)
    previous = candles[start : pattern.d.index]
    if not previous:
        return _empty_mss()
    if pattern.pattern_type == "bullish":
        break_level = max(candle.high for candle in previous)
    else:
        break_level = min(candle.low for candle in previous)
    end = min(len(candles), pattern.d.index + args.mss_max_confirm_bars + 1)
    for index in range(pattern.d.index + 1, end):
        candle = candles[index]
        if _mss_breaks_level(candle, pattern.pattern_type, break_level, args.mss_confirm_close):
            distance = abs(break_level - pattern.d.price) / atr if atr > 0 else 0.0
            return {
                "mss_confirmed": 1,
                "mss_index": index,
                "mss_time": candle.timestamp,
                "mss_bars_to_confirm": index - pattern.d.index,
                "mss_break_level": break_level,
                "mss_break_distance_atr": distance,
            }
    result = _empty_mss()
    result["mss_break_level"] = break_level
    return result


def _empty_mss() -> dict[str, object]:
    return {
        "mss_confirmed": 0,
        "mss_index": -1,
        "mss_time": "",
        "mss_bars_to_confirm": 0,
        "mss_break_level": 0.0,
        "mss_break_distance_atr": 0.0,
    }


def _mss_breaks_level(candle: Candle, pattern_type: str, break_level: float, confirm_close: bool) -> bool:
    if pattern_type == "bullish":
        return candle.close > break_level if confirm_close else candle.high > break_level
    return candle.close < break_level if confirm_close else candle.low < break_level


def _trade_levels(
    pattern: Pattern, entry: float, atr: float, atr_buffer: float, tp2_r: float
) -> dict[str, float]:
    buffer = atr * atr_buffer
    if pattern.pattern_type == "bullish":
        stop = pattern.d.price - buffer
        risk = entry - stop
        tp1 = pattern.c.price
        tp2 = min(pattern.a.price, entry + risk * tp2_r)
    else:
        stop = pattern.d.price + buffer
        risk = stop - entry
        tp1 = pattern.c.price
        tp2 = max(pattern.a.price, entry - risk * tp2_r)
    return {"entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2, "risk": risk, "tp1_r": abs(tp1 - entry) / risk, "tp2_r": abs(tp2 - entry) / risk}


def _simulate_trade(
    candles: list[Candle],
    entry_index: int,
    levels: dict[str, float],
    pattern_type: str,
    max_hold_bars: int,
    commission_bps: float,
    slippage_bps: float,
) -> dict[str, object]:
    tp1_hit = False
    end_index = min(len(candles), entry_index + max_hold_bars)
    cost_r = _round_trip_cost_r(levels, commission_bps, slippage_bps)
    for candle in candles[entry_index:end_index]:
        event = _bar_event(candle, levels, pattern_type, tp1_hit)
        if event == "stop":
            return {"label": "loss", "result_r": -1.0 - cost_r, "exit_reason": "stop_loss"}
        if event == "breakeven":
            result_r = 0.5 * levels["tp1_r"] - cost_r
            return {"label": _label_after_cost(result_r, "partial"), "result_r": result_r, "exit_reason": "breakeven_after_tp1"}
        if event == "tp1":
            tp1_hit = True
        if event == "tp2":
            result_r = 0.5 * levels["tp1_r"] + 0.5 * levels["tp2_r"] - cost_r
            return {"label": _label_after_cost(result_r, "win"), "result_r": result_r, "exit_reason": "tp2"}
    return _timeout_result(candles[end_index - 1], levels, pattern_type, tp1_hit, cost_r)


def _round_trip_cost_r(levels: dict[str, float], commission_bps: float, slippage_bps: float) -> float:
    total_bps = (commission_bps + slippage_bps) * 2.0
    cost_price = levels["entry"] * total_bps / 10000.0
    return cost_price / levels["risk"] if levels["risk"] > 0 else 0.0


def _label_after_cost(result_r: float, original_label: str) -> str:
    if result_r <= -0.5:
        return "loss"
    if result_r <= 0:
        return "breakeven"
    return original_label


def _bar_event(candle: Candle, levels: dict[str, float], pattern_type: str, tp1_hit: bool) -> str | None:
    if pattern_type == "bullish":
        if not tp1_hit and candle.low <= levels["stop"]:
            return "stop"
        if tp1_hit and candle.low <= levels["entry"]:
            return "breakeven"
        if candle.high >= levels["tp2"]:
            return "tp2"
        if not tp1_hit and candle.high >= levels["tp1"]:
            return "tp1"
    else:
        if not tp1_hit and candle.high >= levels["stop"]:
            return "stop"
        if tp1_hit and candle.high >= levels["entry"]:
            return "breakeven"
        if candle.low <= levels["tp2"]:
            return "tp2"
        if not tp1_hit and candle.low <= levels["tp1"]:
            return "tp1"
    return None


def _timeout_result(
    candle: Candle, levels: dict[str, float], pattern_type: str, tp1_hit: bool, cost_r: float
) -> dict[str, object]:
    direction = 1.0 if pattern_type == "bullish" else -1.0
    open_r = direction * (candle.close - levels["entry"]) / levels["risk"]
    if tp1_hit:
        result_r = 0.5 * levels["tp1_r"] + 0.5 * max(0.0, open_r) - cost_r
        return {"label": _label_after_cost(result_r, "partial"), "result_r": result_r, "exit_reason": "timeout_after_tp1"}
    result_r = max(-1.0, open_r) - cost_r
    return {"label": _label_after_cost(result_r, "timeout"), "result_r": result_r, "exit_reason": "timeout"}


def build_features(
    candles: list[Candle], pattern: Pattern, atr_values: list[float], mss: dict[str, object]
) -> dict[str, object]:
    """构建机器学习特征，加入基础 ICT 结构字段。"""
    index = pattern.d.index
    candle = candles[index]
    atr_percent = atr_values[index] / candle.close * 100.0 if candle.close else 0.0
    liquidity_sweep = _liquidity_sweep(candles, pattern, 20)
    pd_position = _range_position(candles, index, 50, pattern.d.price)
    return {
        "atr_percent": atr_percent,
        "volume_zscore": _volume_zscore(candles, index, 20),
        "trend_slope": _trend_slope(candles, index, 20),
        "d_position_range": pd_position,
        "pd_location": _pd_location(pd_position),
        "liquidity_sweep": liquidity_sweep,
        "has_directional_sweep": _has_directional_sweep(pattern.pattern_type, liquidity_sweep),
        "distance_to_range_extreme": _distance_to_range_extreme(pattern.pattern_type, pd_position),
        "mss_confirmed": mss["mss_confirmed"],
        "mss_bars_to_confirm": mss["mss_bars_to_confirm"],
        "mss_break_distance_atr": mss["mss_break_distance_atr"],
        "mss_time": mss["mss_time"],
        "mss_break_level": mss["mss_break_level"],
    }


def _volume_zscore(candles: list[Candle], index: int, window: int) -> float:
    start = max(0, index + 1 - window)
    volumes = [candle.volume for candle in candles[start : index + 1]]
    if len(volumes) < 2:
        return 0.0
    mean = sum(volumes) / len(volumes)
    variance = sum((value - mean) ** 2 for value in volumes) / (len(volumes) - 1)
    return (volumes[-1] - mean) / math.sqrt(variance) if variance > 0 else 0.0


def _trend_slope(candles: list[Candle], index: int, window: int) -> float:
    start = max(0, index - window)
    base = candles[start].close
    return (candles[index].close - base) / base * 100.0 if base else 0.0


def _range_position(candles: list[Candle], index: int, window: int, price: float) -> float:
    start = max(0, index + 1 - window)
    high = max(candle.high for candle in candles[start : index + 1])
    low = min(candle.low for candle in candles[start : index + 1])
    return (price - low) / (high - low) if high > low else 0.5


def _pd_location(position: float) -> str:
    if position <= 0.33:
        return "discount"
    if position >= 0.67:
        return "premium"
    return "equilibrium"


def _liquidity_sweep(candles: list[Candle], pattern: Pattern, window: int) -> str:
    start = max(0, pattern.d.index - window)
    previous = candles[start : pattern.d.index]
    if not previous:
        return "none"
    previous_low = min(candle.low for candle in previous)
    previous_high = max(candle.high for candle in previous)
    if pattern.d.price < previous_low:
        return "SSL"
    if pattern.d.price > previous_high:
        return "BSL"
    return "none"


def _has_directional_sweep(pattern_type: str, liquidity_sweep: str) -> int:
    if pattern_type == "bullish" and liquidity_sweep == "SSL":
        return 1
    if pattern_type == "bearish" and liquidity_sweep == "BSL":
        return 1
    return 0


def _distance_to_range_extreme(pattern_type: str, position: float) -> float:
    if pattern_type == "bullish":
        return position
    return 1.0 - position


def _pattern_row(
    pattern: Pattern,
    label: str,
    result_r: float,
    exit_reason: str,
    tp1_r: float,
    tp2_r: float,
    features: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "pattern_type": pattern.pattern_type,
        "a_time": pattern.a.timestamp,
        "a_price": pattern.a.price,
        "b_time": pattern.b.timestamp,
        "b_price": pattern.b.price,
        "c_time": pattern.c.timestamp,
        "c_price": pattern.c.price,
        "d_time": pattern.d.timestamp,
        "d_price": pattern.d.price,
        "bc_ratio": pattern.bc_ratio,
        "cd_ratio": pattern.cd_ratio,
        "time_symmetry": pattern.time_symmetry,
        "tp1_r": tp1_r,
        "tp2_r": tp2_r,
        "result_r": result_r,
        "label": label,
        "exit_reason": exit_reason,
    }
    row.update(features)
    return row


def find_input_files(data_dir: Path, timeframe: str, symbol: str | None) -> list[Path]:
    if symbol:
        exact = data_dir / f"{symbol}_{timeframe}.csv"
        return [exact] if exact.exists() else []
    return sorted(data_dir.glob(f"*_{timeframe}.csv"))


def symbol_from_path(path: Path, timeframe: str) -> str:
    suffix = f"_{timeframe}"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def run_file(path: Path, timeframe: str, args: argparse.Namespace) -> list[dict[str, object]]:
    candles = read_candles(path)
    atr_values = calculate_atr(candles, args.atr_period)
    swings = detect_swings(candles, args.zigzag_mode, args.zigzag_pct, atr_values, args.zigzag_atr_multiple)
    patterns = detect_abcd_patterns(swings)
    symbol = symbol_from_path(path, timeframe)
    rows = [backtest_pattern(candles, atr_values, pattern, args) for pattern in patterns]
    for row in rows:
        row["symbol"] = symbol
        row["timeframe"] = timeframe
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    all_keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    leading = [key for key in ("symbol", "timeframe") if key in all_keys]
    fieldnames = leading + [key for key in all_keys if key not in leading]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(timeframe: str, rows: list[dict[str, object]]) -> dict[str, object]:
    tradable = [row for row in rows if row["label"] != "skip"]
    results = [float(row["result_r"]) for row in tradable]
    return {
        "timeframe": timeframe,
        "total_setups": len(rows),
        "tradable_setups": len(tradable),
        "win_rate": _rate(tradable, "win"),
        "partial_rate": _rate(tradable, "partial"),
        "loss_rate": _rate(tradable, "loss"),
        "avg_result_r": sum(results) / len(results) if results else 0.0,
        "median_result_r": median(results) if results else 0.0,
        "max_drawdown_r": _max_drawdown(results),
        "profit_factor": _profit_factor(results),
        "best_bc_ratio_zone": _best_ratio_zone(tradable, "bc_ratio"),
        "best_cd_ratio_zone": _best_ratio_zone(tradable, "cd_ratio"),
    }


def _rate(rows: list[dict[str, object]], label: str) -> float:
    return sum(1 for row in rows if row["label"] == label) / len(rows) if rows else 0.0


def _max_drawdown(results: Iterable[float]) -> float:
    peak = 0.0
    equity = 0.0
    max_drawdown = 0.0
    for result in results:
        equity += result
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def _profit_factor(results: list[float]) -> float:
    gross_profit = sum(result for result in results if result > 0)
    gross_loss = abs(sum(result for result in results if result < 0))
    return gross_profit / gross_loss if gross_loss else 0.0


def _best_ratio_zone(rows: list[dict[str, object]], key: str) -> str:
    zones: dict[str, list[float]] = {}
    for row in rows:
        value = float(row[key])
        zone = f"{math.floor(value * 10) / 10:.1f}-{math.floor(value * 10) / 10 + 0.1:.1f}"
        zones.setdefault(zone, []).append(float(row["result_r"]))
    if not zones:
        return "n/a"
    return max(zones.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for timeframe in TIMEFRAMES:
        rows = []
        for path in find_input_files(data_dir, timeframe, args.symbol):
            rows.extend(run_file(path, timeframe, args))
        write_csv(output_dir / f"abcd_setups_{timeframe}.csv", rows)
        summary_rows.append(summarize(timeframe, rows))
    write_csv(output_dir / "abcd_summary_by_timeframe.csv", summary_rows)


if __name__ == "__main__":
    main()
