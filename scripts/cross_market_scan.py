"""
跨市场 Walk-Forward 回测

用 V1 最优参数对所有 15m 标的做回测，比较：
- 交易数量、胜率、avgR、totalR、PF
- 形态类型分布
- 识别低胜率原因
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

from backtest.walk_forward import WalkForwardEngine

# ================================================================
# 配置
# ================================================================
DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data/data_um"
OUTPUT_DIR = "outputs/cross_market"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V1 最优参数（BTC 30m 优化得到）
ENGINE_PARAMS = dict(
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

# 回测时间范围
DATE_START = "2024-01-01"
DATE_END = "2026-05-01"


def load_and_prep(path, symbol, tf):
    """加载 parquet，设置时间索引"""
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()
    # 只取回测范围
    df = df[(df.index >= DATE_START) & (df.index < DATE_END)]
    # walk_forward 需要这些列
    required = ["high", "low", "close", "open", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} 缺少列: {col}")
    return df[required]


def run_one(engine, path, symbol, tf):
    """运行单个标的，返回 (results_df, stats_dict)"""
    t0 = time.time()
    df = load_and_prep(path, symbol, tf)
    
    if len(df) < 500:
        elapsed = time.time() - t0
        return None, {"symbol": symbol, "tf": tf, "trades": 0, "error": f"数据不足 ({len(df)} bars)", "elapsed": elapsed}
    
    result = engine.run(df, symbol, tf)
    elapsed = time.time() - t0
    
    if result is None or len(result) == 0:
        return None, {"symbol": symbol, "tf": tf, "trades": 0, "elapsed": round(elapsed, 1)}
    
    # 统计
    trades = len(result)
    wr = result["label"].mean()
    avg_r = result["final_r"].mean()
    med_r = result["final_r"].median()
    total_r = result["final_r"].sum()
    pos_r = result[result["final_r"] > 0]["final_r"].mean() if (result["final_r"] > 0).sum() > 0 else 0
    neg_r = result[result["final_r"] < 0]["final_r"].mean() if (result["final_r"] < 0).sum() > 0 else 0
    pf = abs(pos_r * (result["final_r"] > 0).sum() / max(neg_r * (result["final_r"] < 0).sum(), 1e-10))
    
    # 退出原因分布
    exit_counts = result["exit_reason"].value_counts().to_dict() if "exit_reason" in result.columns else {}
    
    # 各形态统计
    pattern_stats = {}
    if "shape_id" in result.columns:
        for _, row in result.iterrows():
            sid = str(row["shape_id"])
            # 从 shape_id 提取形态名: SYM_TF_PATTERN_DIR_N
            parts = sid.split("_")
            if len(parts) >= 3:
                pattern = parts[-3]  # 倒数第三位是形态
            else:
                pattern = "UNKNOWN"
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {"trades": 0, "wr_sum": 0, "r_sum": 0.0}
            pattern_stats[pattern]["trades"] += 1
            pattern_stats[pattern]["wr_sum"] += result.loc[_, "label"]
            pattern_stats[pattern]["r_sum"] += result.loc[_, "final_r"]
    
    for p in pattern_stats:
        n = pattern_stats[p]["trades"]
        pattern_stats[p]["wr"] = round(pattern_stats[p]["wr_sum"] / n * 100, 1)
        pattern_stats[p]["avg_r"] = round(pattern_stats[p]["r_sum"] / n, 3)
    
    # 方向统计
    dir_stats = {}
    if "direction" in result.columns:
        for d in result["direction"].unique():
            sub = result[result["direction"] == d]
            dir_stats[d] = {
                "trades": len(sub),
                "wr": round(sub["label"].mean() * 100, 1),
                "avg_r": round(sub["final_r"].mean(), 3),
            }
    
    stats = {
        "symbol": symbol,
        "tf": tf,
        "trades": trades,
        "wr": round(wr * 100, 1),
        "avg_r": round(avg_r, 3),
        "med_r": round(med_r, 3),
        "total_r": round(total_r, 1),
        "pf": round(pf, 2),
        "avg_wr_per_trade": round(avg_r, 3),
        "pos_r": round(pos_r, 3),
        "neg_r": round(neg_r, 3),
        "exit_reasons": exit_counts,
        "directions": dir_stats,
        "patterns": pattern_stats,
        "elapsed": round(elapsed, 1),
    }
    
    return result, stats


def main():
    engine = WalkForwardEngine(**ENGINE_PARAMS)
    
    all_stats = []
    top_totalr = []
    
    files = sorted(os.listdir(DATA_DIR))
    parquet_files = [f for f in files if f.endswith(".parquet")]
    
    print(f"{'='*70}")
    print(f"跨市场 Walk-Forward 回测")
    print(f"参数: atr_m=0.7, minQ=50, sl_buf=0.2, minTP1=0.3%")
    print(f"范围: {DATE_START} ~ {DATE_END} | 标的: {len(parquet_files)} 个")
    print(f"{'='*70}\n")
    
    for f in parquet_files:
        # 解析符号名: BTCUSDT_15m_um.parquet → BTCUSDT
        parts = f.replace(".parquet", "").split("_")
        symbol = parts[0]
        tf = "15m"
        
        path = os.path.join(DATA_DIR, f)
        result, stats = run_one(engine, path, symbol, tf)
        all_stats.append(stats)
        
        if stats["trades"] > 0:
            top_totalr.append((stats["total_r"], stats))
        
        # 输出
        if stats["trades"] == 0:
            err = stats.get("error", "0笔")
            print(f"  {symbol:6s} {tf:4s}: ⚠️ {err} | {stats['elapsed']:.1f}s")
        else:
            wr_str = f"{stats['wr']}%"
            print(f"  {symbol:6s} {tf:4s}: {stats['trades']:>5d} trades | "
                  f"WR={wr_str:>6s} avgR={stats['avg_r']:>7.3f} "
                  f"totalR={stats['total_r']:>8.1f} PF={stats['pf']:>5.2f} | "
                  f"{stats['elapsed']:.1f}s")
    
    # ================================================================
    # 汇总报告
    # ================================================================
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(f"{OUTPUT_DIR}/cross_market_summary.csv", index=False)
    
    # 有交易记录的排名
    traded = stats_df[stats_df["trades"] > 0].sort_values("total_r", ascending=False)
    
    print(f"\n{'='*70}")
    print(f"📊 按 totalR 排名 (Top 10)")
    print(f"{'='*70}")
    col_headers = f"{'Rank':<5} {'Symbol':<8} {'Trades':>6} {'WR':>7} {'avgR':>8} {'totalR':>9} {'PF':>6}"
    print(col_headers)
    print("-" * len(col_headers))
    for rank, (_, row) in enumerate(traded.head(16).iterrows(), 1):
        print(f"{rank:<5} {row['symbol']:<8} {int(row['trades']):>6} "
              f"{row['wr']:>6.1f}% {row['avg_r']:>8.3f} {row['total_r']:>9.1f} {row['pf']:>6.2f}")
    
    print(f"\n按胜率排名 (Top 10):")
    by_wr = traded.sort_values("wr", ascending=False)
    for rank, (_, row) in enumerate(by_wr.head(10).iterrows(), 1):
        print(f"  {rank}. {row['symbol']:<8s} WR={row['wr']:.1f}%  trades={int(row['trades'])}  avgR={row['avg_r']:.3f}  totalR={row['total_r']:.1f}")
    
    # 全体汇总
    t_trades = traded["trades"].sum()
    t_wr = (traded["wr"] * traded["trades"]).sum() / t_trades if t_trades > 0 else 0
    t_totalr = traded["total_r"].sum()
    print(f"\n全体汇总: {t_trades}笔  WR={t_wr:.1f}%  totalR={t_totalr:.1f}")
    
    # 低胜率诊断
    print(f"\n{'='*70}")
    print(f"🔍 低胜率诊断 (WR<45%)")
    print(f"{'='*70}")
    low_wr = traded[traded["wr"] < 45].sort_values("wr")
    for _, row in low_wr.iterrows():
        patterns = row.get("patterns", {})
        pattern_detail = ", ".join(f"{k}({v['trades']}/{v['wr']}%)" for k, v in sorted(patterns.items(), key=lambda x: -x[1]['trades'])[:3])
        directions = row.get("directions", {})
        dir_detail = ", ".join(f"{k}({v['trades']}/{v['wr']}%)" for k, v in directions.items())
        print(f"  {row['symbol']:<8s} WR={row['wr']:.1f}%  trades={int(row['trades'])}  "
              f"avgR={row['avg_r']:.3f}  SL比例≈{row.get('exit_reasons', {}).get('SL', 0) + row.get('exit_reasons', {}).get('TP1+SL', 0)}/{int(row['trades'])}")
        print(f"         方向: {dir_detail}")
        print(f"         形态: {pattern_detail}")
    
    print(f"\n✅ 结果保存: {OUTPUT_DIR}/cross_market_summary.csv")
    return all_stats


if __name__ == "__main__":
    main()
