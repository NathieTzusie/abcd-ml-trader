"""
V1 vs V4 对比测试

A/B 测试:
  A: V1 基线 (无确认, 固定参数)
  B: V4 swing 确认 (confirm_bars=3)
  C: V4 swing 确认 + vol_adaptive
  D: V4 candle 确认 + vol_adaptive

在所有 16 个标的上运行，输出对比表。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

from backtest.walk_forward import WalkForwardEngine as V1
from backtest.walk_forward_v4 import WalkForwardEngineV4 as V4

DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data/data_um"
OUTPUT_DIR = "outputs/cross_market"
DATE_START = "2024-01-01"
DATE_END = "2026-05-01"

BASE_PARAMS = dict(
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
)

# 测试配置
CONFIGS = {
    "V1_baseline": {
        "engine": "V1",
        "params": {**BASE_PARAMS},
    },
    "V4_swing3": {
        "engine": "V4",
        "params": {**BASE_PARAMS, "use_confirmation": True, "confirm_bars": 3, "confirm_mode": "swing"},
    },
    "V4_swing3_vol": {
        "engine": "V4",
        "params": {**BASE_PARAMS, "use_confirmation": True, "confirm_bars": 3,
                   "confirm_mode": "swing", "vol_adaptive": True, "btc_atr_pct_ref": 0.003},
    },
    "V4_candle3_vol": {
        "engine": "V4",
        "params": {**BASE_PARAMS, "use_confirmation": True, "confirm_bars": 3,
                   "confirm_mode": "candle", "vol_adaptive": True, "btc_atr_pct_ref": 0.003},
    },
    "V4_both3_vol": {
        "engine": "V4",
        "params": {**BASE_PARAMS, "use_confirmation": True, "confirm_bars": 3,
                   "confirm_mode": "both", "vol_adaptive": True, "btc_atr_pct_ref": 0.003},
    },
}


def load_data(path):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= DATE_START) & (df.index < DATE_END)][["high", "low", "close", "open", "volume"]]


def run_one(engine_class, params, df, symbol, tf):
    t0 = time.time()
    engine = engine_class(**params)
    result = engine.run(df, symbol, tf)
    elapsed = time.time() - t0

    if result is None or len(result) == 0:
        return {
            "symbol": symbol, "tf": tf, "trades": 0,
            "wr": 0, "avg_r": 0, "total_r": 0, "elapsed": round(elapsed, 1)
        }

    trades = len(result)
    wr = result["label"].mean() * 100
    avg_r = result["final_r"].mean()
    total_r = result["final_r"].sum()
    med_r = result["final_r"].median()

    # 退出分布
    sl_trades = 0
    if "exit_reason" in result.columns:
        for r in result["exit_reason"]:
            if "SL" in str(r):
                sl_trades += 1

    return {
        "symbol": symbol, "tf": tf, "trades": trades,
        "wr": round(wr, 1), "avg_r": round(avg_r, 3),
        "total_r": round(total_r, 1), "med_r": round(med_r, 3),
        "sl_pct": round(sl_trades / trades * 100, 1),
        "elapsed": round(elapsed, 1),
    }


def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
    symbols = [f.split("_")[0] for f in files]

    # 结果矩阵: rows=symbols, cols=configs
    all_rows = []

    print(f"{'='*80}")
    print(f"V1 vs V4 对比测试")
    print(f"配置: {', '.join(CONFIGS.keys())}")
    print(f"范围: {DATE_START} ~ {DATE_END} | 标的: {len(symbols)} 个")
    print(f"{'='*80}\n")

    for f in files:
        sym = f.split("_")[0]
        path = os.path.join(DATA_DIR, f)
        df = load_data(path)

        row = {"symbol": sym}

        for cfg_name, cfg in CONFIGS.items():
            is_v1 = cfg["engine"] == "V1"
            engine_cls = V1 if is_v1 else V4
            stats = run_one(engine_cls, cfg["params"], df, sym, "15m")

            row[f"{cfg_name}_trades"] = stats["trades"]
            row[f"{cfg_name}_wr"] = stats["wr"]
            row[f"{cfg_name}_avgR"] = stats["avg_r"]
            row[f"{cfg_name}_totalR"] = stats["total_r"]
            row[f"{cfg_name}_sl"] = stats.get("sl_pct", 0)

            # 简要输出
            if stats["trades"] > 0:
                print(f"  {sym:6s} {cfg_name:18s}: {stats['trades']:>5d} trades  "
                      f"WR={stats['wr']:>5.1f}%  avgR={stats['avg_r']:>7.3f}  "
                      f"totalR={stats['total_r']:>8.1f}  SL={stats['sl_pct']:>5.1f}%  "
                      f"{stats['elapsed']:.1f}s")
            else:
                print(f"  {sym:6s} {cfg_name:18s}: 0 trades | {stats['elapsed']:.1f}s")

        all_rows.append(row)

    # ================================================================
    # 汇总报告
    # ================================================================
    results = pd.DataFrame(all_rows)
    results.to_csv(f"{OUTPUT_DIR}/v1_vs_v4_comparison.csv", index=False)

    print(f"\n{'='*120}")
    print(f"📊 V1 vs V4 汇总")
    print(f"{'='*120}")

    # 表头
    header = f"{'Symbol':<8s}"
    for cfg in CONFIGS:
        header += f" | {cfg:>18s}"
    print(header)
    print("-" * len(header))

    # 胜率对比
    print("\n🔹 胜率 (WR%) 对比:")
    wr_header = f"{'Symbol':<8s}"
    for cfg in CONFIGS:
        wr_header += f" | {cfg:>18s}"
    print(wr_header)
    print("-" * len(wr_header))
    for _, row in results.iterrows():
        sym = row["symbol"]
        line = f"{sym:<8s}"
        best_wr = 0
        best_cfg = ""
        for cfg in CONFIGS:
            wr_val = row.get(f"{cfg}_wr", 0)
            line += f" | {wr_val:>17.1f}%"
            if wr_val > best_wr:
                best_wr = wr_val
                best_cfg = cfg
        line += f"  ← best: {best_cfg}"
        print(line)

    # totalR 对比
    print(f"\n🔹 totalR 对比:")
    tr_header = f"{'Symbol':<8s}"
    for cfg in CONFIGS:
        tr_header += f" | {cfg:>18s}"
    print(tr_header)
    print("-" * len(tr_header))
    for _, row in results.iterrows():
        sym = row["symbol"]
        line = f"{sym:<8s}"
        best_tr = -9999
        best_cfg = ""
        for cfg in CONFIGS:
            tr_val = row.get(f"{cfg}_totalR", 0)
            line += f" | {tr_val:>18.1f}"
            if tr_val > best_tr:
                best_tr = tr_val
                best_cfg = cfg
        line += f"  ← best: {best_cfg}"
        print(line)

    # SL 比例对比
    print(f"\n🔹 SL 比例 (%) 对比:")
    sl_header = f"{'Symbol':<8s}"
    for cfg in CONFIGS:
        sl_header += f" | {cfg:>18s}"
    print(sl_header)
    print("-" * len(sl_header))
    for _, row in results.iterrows():
        sym = row["symbol"]
        line = f"{sym:<8s}"
        for cfg in CONFIGS:
            sl_val = row.get(f"{cfg}_sl", 0)
            line += f" | {sl_val:>17.1f}%"
        print(line)

    # 全体平均
    print(f"\n🔹 全体加权平均:")
    for cfg in CONFIGS:
        t_col = f"{cfg}_trades"
        wr_col = f"{cfg}_wr"
        tr_col = f"{cfg}_totalR"
        sl_col = f"{cfg}_sl"
        total_t = results[t_col].sum()
        if total_t > 0:
            avg_wr = (results[wr_col] * results[t_col]).sum() / total_t
            avg_sl = (results[sl_col] * results[t_col]).sum() / total_t
            sum_tr = results[tr_col].sum()
            print(f"  {cfg:20s}: {total_t:>5.0f} trades  WR={avg_wr:.1f}%  "
                  f"totalR={sum_tr:.1f}  SL={avg_sl:.1f}%")

    print(f"\n✅ 详细结果: {OUTPUT_DIR}/v1_vs_v4_comparison.csv")


if __name__ == "__main__":
    main()
