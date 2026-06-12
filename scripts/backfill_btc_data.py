"""
回填 Binance BTC 数据 → 完整覆盖 2024-01-01 到 2026-05-25

BTC 30m/1h/4h 只有 2025-01-01 起的数据。
回填 2024 年 + 补全 2025 年 → 合并为完整数据集。
"""
import sys
import os
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time

SOURCE_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data"
SYMBOL = "BTC/USDT"
TIMEFRAMES = ["30m", "1h", "4h"]
FETCH_START = "2024-01-01"
FETCH_END = "2026-05-25"
BATCH_DAYS = 30


def fetch_batch(exchange, symbol, timeframe, since_ms, limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def fetch_range(exchange, symbol, tf, start_str, end_str):
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    current = start
    all_data = []

    while current < end:
        batch_end = min(current + timedelta(days=BATCH_DAYS), end)
        since_ms = int(current.timestamp() * 1000)
        print(f"    抓取 {current.date()} → {batch_end.date()}...")

        try:
            df = fetch_batch(exchange, symbol, tf, since_ms)
            if df is not None and len(df) > 0:
                df = df[(df.index >= current) & (df.index <= batch_end)]
                all_data.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"    错误: {e}, 5s 后重试...")
            time.sleep(5)
            try:
                df = fetch_batch(exchange, symbol, tf, since_ms)
                if df is not None and len(df) > 0:
                    df = df[(df.index >= current) & (df.index <= batch_end)]
                    all_data.append(df)
            except Exception as e2:
                print(f"    重试失败: {e2}")

        current = batch_end

    if all_data:
        result = pd.concat(all_data)
        result = result[~result.index.duplicated()]
        result.sort_index(inplace=True)
        return result
    return None


def main():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    for tf in TIMEFRAMES:
        print(f"\n{'='*50}")
        print(f"处理 {tf}...")
        print(f"{'='*50}")

        data = fetch_range(exchange, SYMBOL, tf, FETCH_START, FETCH_END)
        if data is not None:
            output_path = os.path.join(SOURCE_DIR, f"binance_BTCUSDT_{tf}.parquet")
            data.to_parquet(output_path)
            print(f"  ✅ 已保存: {len(data)} 行, {data.index.min()} → {data.index.max()}")
        else:
            print(f"  ❌ 无数据")


if __name__ == "__main__":
    main()
