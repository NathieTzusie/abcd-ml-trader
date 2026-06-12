"""
Phase A — Step 1: 生成 V4 训练数据集

用 V4_swing3_vol 引擎跑 16 标的，收集所有交易结果。
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

from backtest.walk_forward_v4 import WalkForwardEngineV4

DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data/data_um"
OUTPUT_PATH = "outputs/v4_training_dataset.csv"
DATE_START = "2024-01-01"
DATE_END = "2026-05-01"

V4_PARAMS = dict(
    atr_mult=0.7,
    min_quality_score=50,
    sl_buffer_atr=0.2,
    min_tp1_pct=0.003,
    bc_ab_min=0.382,
    bc_ab_max=0.886,
    cd_ab_ratio=1.0,
    d_zone_tolerance=0.005,
    timeout_mult=2.0,
    tp1_pct=0.5,
    min_atr_mult=0.5,
    use_confirmation=True,
    confirm_bars=3,
    confirm_mode="swing",
    vol_adaptive=True,
    btc_atr_pct_ref=0.003,
)


def load_data(path):
    """加载 parquet 数据，按时间筛选，只保留必要列"""
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= DATE_START) & (df.index < DATE_END)][["high", "low", "close", "open", "volume"]]


def run_one(symbol, df):
    """跑 V4 引擎"""
    t0 = time.time()
    engine = WalkForwardEngineV4(**V4_PARAMS)
    result = engine.run(df, symbol, "15m")
    elapsed = time.time() - t0

    if result is None or len(result) == 0:
        return None, {"trades": 0, "wr": 0, "avg_r": 0, "total_r": 0, "sl_pct": 0, "elapsed": elapsed}

    trades = len(result)
    wr = result["label"].mean() * 100
    avg_r = result["final_r"].mean()
    total_r = result["final_r"].sum()
    med_r = result["final_r"].median()

    # SL 比例
    sl_count = sum(1 for r in result["exit_reason"] if "SL" in str(r))
    sl_pct = sl_count / trades * 100 if trades > 0 else 0

    return result, {
        "symbol": symbol, "trades": trades,
        "wr": round(wr, 1), "avg_r": round(avg_r, 3),
        "total_r": round(total_r, 1), "med_r": round(med_r, 3),
        "sl_pct": round(sl_pct, 1), "elapsed": round(elapsed, 1),
    }


def main():
    os.makedirs("outputs", exist_ok=True)

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_15m_um.parquet")])
    symbols = [f.split("_")[0] for f in files]
    print(f"标的 ({len(symbols)}): {', '.join(symbols)}", flush=True)
    print(f"引擎: V4_swing3_vol", flush=True)
    print(f"范围: {DATE_START} ~ {DATE_END}", flush=True)
    print(f"{'='*80}", flush=True)

    all_trades = []
    stats_rows = []

    for fname, sym in zip(files, symbols):
        path = os.path.join(DATA_DIR, fname)
        print(f"\n>>> {sym}", flush=True)
        df = load_data(path)
        print(f"    数据: {len(df)} bars", flush=True)

        trades_df, stats = run_one(sym, df)
        if trades_df is not None and len(trades_df) > 0:
            trades_df["symbol"] = sym
            trades_df["timeframe"] = "15m"
            all_trades.append(trades_df)
            print(f"    ✅ {stats['trades']} trades  WR={stats['wr']:.1f}%  "
                  f"avgR={stats['avg_r']:.3f}  totalR={stats['total_r']:.1f}  "
                  f"SL={stats['sl_pct']:.1f}%  {stats['elapsed']:.1f}s", flush=True)
        else:
            print(f"    ⚠️ 0 trades", flush=True)

        stats_rows.append(stats)

        # 释放内存
        del df, trades_df
        gc.collect()

    # ================================================================
    # 合并 & 保存
    # ================================================================
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"\n{'='*80}", flush=True)
        print(f"✅ 总计: {len(combined)} trades 已保存到 {OUTPUT_PATH}", flush=True)
    else:
        print("\n⚠️ 没有产生任何交易数据", flush=True)
        return

    # ================================================================
    # 每标统计
    # ================================================================
    stats_df = pd.DataFrame(stats_rows)
    total_trades = stats_df["trades"].sum()
    if total_trades > 0:
        avg_wr = (stats_df["wr"] * stats_df["trades"]).sum() / total_trades
        avg_sl = (stats_df["sl_pct"] * stats_df["trades"]).sum() / total_trades
        sum_tr = stats_df["total_r"].sum()
    else:
        avg_wr = avg_sl = sum_tr = 0

    print(f"\n{'='*80}", flush=True)
    print(f"📊 每标统计")
    print(f"{'='*80}", flush=True)
    print(f"{'Symbol':>8s} | {'Trades':>7s} | {'WR%':>6s} | {'avgR':>7s} | "
          f"{'totalR':>8s} | {'SL%':>6s} | {'Time':>7s}", flush=True)
    print("-" * 60, flush=True)
    for _, row in stats_df.iterrows():
        print(f"{row['symbol']:>8s} | {row['trades']:>7d} | {row['wr']:>5.1f}% | "
              f"{row['avg_r']:>7.3f} | {row['total_r']:>8.1f} | {row['sl_pct']:>5.1f}% | "
              f"{row['elapsed']:>6.1f}s", flush=True)
    print("-" * 60, flush=True)
    print(f"{'合计':>8s} | {total_trades:>7d} | {avg_wr:>5.1f}% | "
          f"{'--':>7s} | {sum_tr:>8.1f} | {avg_sl:>5.1f}% | "
          f"{stats_df['elapsed'].sum():>6.1f}s", flush=True)
    print(f"\n✅ Step 1 完成: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
