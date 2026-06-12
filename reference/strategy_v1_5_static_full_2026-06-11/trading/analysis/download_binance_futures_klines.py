from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
TIMEFRAMES = ("5m", "15m", "30m", "1h", "4h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance USDT futures OHLCV data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance futures symbol")
    parser.add_argument("--start", default="2024-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-06-10", help="结束日期 YYYY-MM-DD，包含当天")
    parser.add_argument("--output-dir", default="data/raw", help="输出目录")
    parser.add_argument("--intervals", default=",".join(TIMEFRAMES), help="逗号分隔周期，例如 5m,15m,1h")
    return parser.parse_args()


def date_to_ms(value: str) -> int:
    """把日期转成 UTC 毫秒时间戳，确保不同电脑时区下结果一致。"""
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list[object]]:
    """分页下载 Binance Futures K 线，单次最多 1500 根。"""
    rows: list[list[object]] = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1500,
        }
        url = f"{BASE_URL}?{urlencode(params)}"
        with urlopen(url, timeout=30) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        rows.extend(batch)
        next_open_time = int(batch[-1][0]) + 1
        if next_open_time <= current:
            break
        current = next_open_time
        time.sleep(0.15)
    return rows


def write_ohlcv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in rows:
            timestamp = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, row[1], row[2], row[3], row[4], row[5]])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    start_ms = date_to_ms(args.start)
    end_ms = date_to_ms(args.end) + 24 * 60 * 60 * 1000 - 1
    for timeframe in [value.strip() for value in args.intervals.split(",") if value.strip()]:
        rows = fetch_klines(args.symbol, timeframe, start_ms, end_ms)
        write_ohlcv(output_dir / f"{args.symbol}_{timeframe}.csv", rows)
        print(f"{args.symbol}_{timeframe}: {len(rows)} rows")


if __name__ == "__main__":
    main()
