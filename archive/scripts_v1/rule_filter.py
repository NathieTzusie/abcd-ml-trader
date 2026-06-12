"""
规则过滤器 — 基于 walk-forward 回测结果 + ML 特征重要性

在形态入场前应用规则门槛，过滤低质量交易。
过滤器设计依据 SHAP 分析：
  1. ab_distance_pct  — AB 摆动幅度（15% 重要性）
  2. d_zone_momentum  — D_zone 价格动量方向（11%）
  3. atr_pct          — 波动率水平（10%）
  4. price_vs_ema20   — 趋势位置（8%）
  5. d_zone_wick_ratio — 影线/反转形态（5%）
"""
import pandas as pd
import numpy as np
from itertools import product


# 默认阈值（基于 SHAP 分布热区）
DEFAULT_THRESHOLDS = {
    "ab_distance_pct_min": 0.002,     # AB ≥ 0.2% 价格幅度
    "d_zone_momentum_align": True,     # 看涨时 D_zone 动量为正，看跌时为负
    "atr_pct_max": 0.05,               # ATR/价格 ≤ 5%
    "price_vs_ema20_align": True,      # 看涨时价格 < EMA20（回调买入），看跌时 > EMA20
    "d_zone_wick_ratio_min": 0.3,     # 影线比 ≥ 30%（有反转迹象）
}


def apply_filter(
    df: pd.DataFrame,
    thresholds: dict = None,
) -> pd.DataFrame:
    """
    对交易数据集应用规则过滤器

    Parameters
    ----------
    df : pd.DataFrame
        walk-forward 回测结果（每行=一笔交易）
    thresholds : dict
        过滤阈值，为空则不过滤

    Returns
    -------
    (filtered_df, stats_dict)
    """
    if thresholds is None:
        thresholds = {}

    mask = pd.Series(True, index=df.index)

    # 1. AB 最小幅度
    if "ab_distance_pct_min" in thresholds:
        t = thresholds["ab_distance_pct_min"]
        mask &= df["ab_distance_pct"] >= t

    # 2. D_zone 动量方向对齐
    if thresholds.get("d_zone_momentum_align", False):
        # 看涨：入场时价格应该在上行（动量 > 0）
        # 看跌：入场时价格应该在下行（动量 < 0）
        bull_ok = (df["direction"] == "bullish") & (df["d_zone_momentum"] > 0)
        bear_ok = (df["direction"] == "bearish") & (df["d_zone_momentum"] < 0)
        mask &= bull_ok | bear_ok

    # 3. ATR 上限（过高波动率不过滤——高波动也有高回报）
    if "atr_pct_max" in thresholds:
        t = thresholds["atr_pct_max"]
        mask &= df["atr_pct"] <= t

    # 4. 趋势位置对齐
    if thresholds.get("price_vs_ema20_align", False):
        # 看涨：价格低于 EMA20（回调买）
        # 看跌：价格高于 EMA20（反弹卖）
        bull_ok = (df["direction"] == "bullish") & (df["price_vs_ema20"] < 0)
        bear_ok = (df["direction"] == "bearish") & (df["price_vs_ema20"] > 0)
        mask &= bull_ok | bear_ok

    # 5. 影线比（反转确认）
    if "d_zone_wick_ratio_min" in thresholds:
        t = thresholds["d_zone_wick_ratio_min"]
        mask &= df["d_zone_wick_ratio"] >= t

    return df[mask].copy()


def compute_stats(df: pd.DataFrame, label: str = "") -> dict:
    """计算交易统计"""
    if len(df) == 0:
        return {"label": label, "trades": 0, "wr": 0, "avg_r": 0, "med_r": 0, "total_r": 0}
    return {
        "label": label,
        "trades": len(df),
        "wr": round(df["label"].mean() * 100, 1),
        "avg_r": round(df["final_r"].mean(), 3),
        "med_r": round(df["final_r"].median(), 3),
        "total_r": round(df["final_r"].sum(), 1),
    }


def grid_search(df: pd.DataFrame, tf_filter: str = None) -> pd.DataFrame:
    """
    网格搜索最佳过滤阈值

    Parameters
    ----------
    df : pd.DataFrame
        完整数据集
    tf_filter : str or None
        如果指定，只搜索该时间框架（如 '15m'）
    """
    if tf_filter:
        df = df[df["timeframe"] == tf_filter]

    param_grid = {
        "ab_distance_pct_min": [0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01],
        "d_zone_momentum_align": [False, True],
        "price_vs_ema20_align": [False, True],
        "d_zone_wick_ratio_min": [0, 0.2, 0.3, 0.4, 0.5],
    }

    results = []
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        t = dict(zip(keys, values))
        filtered = apply_filter(df, t)
        stats = compute_stats(filtered, str(t))
        stats["retention"] = round(len(filtered) / len(df) * 100, 1)
        stats["thresholds"] = t
        results.append(stats)

    return pd.DataFrame(results).sort_values("avg_r", ascending=False)


def print_filter_report(df_full: pd.DataFrame, thresholds: dict):
    """打印过滤器效果报告"""
    print(f"\n{'='*60}")
    print("规则过滤器效果报告")
    print(f"{'='*60}")

    total = len(df_full)
    base = compute_stats(df_full, "无过滤")

    print(f"\n{'过滤器':<40} {'交易':>6} {'保留%':>7} {'WR%':>6} {'avgR':>7} {'medR':>7} {'totalR':>8}")
    print("-" * 95)
    print(f"{'无过滤':<40} {base['trades']:>6} {100:>6.1f}% {base['wr']:>5.1f}% {base['avg_r']:>7.3f} {base['med_r']:>7.3f} {base['total_r']:>8.1f}")

    # 逐个过滤器效果
    single_filters = {
        "AB幅度 ≥ 0.2%": {"ab_distance_pct_min": 0.002},
        "D_zone动量对齐": {"d_zone_momentum_align": True},
        "趋势位置对齐": {"price_vs_ema20_align": True},
        "影线比 ≥ 0.3": {"d_zone_wick_ratio_min": 0.3},
    }

    for name, t in single_filters.items():
        filtered = apply_filter(df_full, t)
        s = compute_stats(filtered, name)
        ret = round(len(filtered) / total * 100, 1)
        print(f"{name:<40} {s['trades']:>6} {ret:>6.1f}% {s['wr']:>5.1f}% {s['avg_r']:>7.3f} {s['med_r']:>7.3f} {s['total_r']:>8.1f}")

    # 组合过滤器
    filtered = apply_filter(df_full, thresholds)
    s = compute_stats(filtered, "组合过滤")
    ret = round(len(filtered) / total * 100, 1)
    print(f"{'组合过滤':<40} {s['trades']:>6} {ret:>6.1f}% {s['wr']:>5.1f}% {s['avg_r']:>7.3f} {s['med_r']:>7.3f} {s['total_r']:>8.1f}")

    # 按 TF 分组
    print(f"\n{'='*60}")
    print("分组效果（组合过滤）")
    print(f"{'='*60}")
    print(f"{'Symbol':<10} {'TF':<5} {'无过滤':>8} {'过滤后':>8} {'保留%':>7} {'WR变化':>8} {'avgR变化':>9}")
    print("-" * 65)
    for (sym, tf), group in df_full.groupby(["symbol", "timeframe"]):
        base_s = compute_stats(group)
        if tf == "4h":
            continue
        filt_group = apply_filter(group, thresholds)
        filt_s = compute_stats(filt_group)
        ret = round(len(filt_group) / len(group) * 100, 1)
        wr_chg = f"+{filt_s['wr']-base_s['wr']:.1f}%" if filt_s['wr'] >= base_s['wr'] else f"{filt_s['wr']-base_s['wr']:.1f}%"
        ar_chg = f"+{filt_s['avg_r']-base_s['avg_r']:.3f}" if filt_s['avg_r'] >= base_s['avg_r'] else f"{filt_s['avg_r']-base_s['avg_r']:.3f}"
        print(f"{sym:<10} {tf:<5} {base_s['trades']:>8} {filt_s['trades']:>8} {ret:>6.1f}% {wr_chg:>8} {ar_chg:>9}")


if __name__ == "__main__":
    df = pd.read_csv("outputs/wf_training_dataset.csv")
    # 放弃 4h
    df = df[df["timeframe"] != "4h"]
    print(f"数据集: {len(df)} 笔交易 (不含 4h)", flush=True)

    # 默认过滤器效果
    print_filter_report(df, DEFAULT_THRESHOLDS)

    # 网格搜索最佳参数
    print(f"\n{'='*60}")
    print("网格搜索 Top 10 (by avgR)")
    print(f"{'='*60}")
    gs = grid_search(df)
    for _, row in gs.head(10).iterrows():
        t = row["thresholds"]
        print(f"  avgR={row['avg_r']:.3f} WR={row['wr']:.1f}% trades={row['trades']} ret={row['retention']:.1f}% | "
              f"ab={t.get('ab_distance_pct_min',0):.4f} mom={t.get('d_zone_momentum_align',0)} "
              f"ema={t.get('price_vs_ema20_align',0)} wick={t.get('d_zone_wick_ratio_min',0):.1f}", flush=True)

    # 保存最佳参数结果
    best = gs.iloc[0]
    best_t = best["thresholds"]
    print(f"\n🏆 最佳阈值: {best_t}", flush=True)
    print(f"   效果: WR={best['wr']:.1f}% avgR={best['avg_r']:.3f} trades={best['trades']} ({best['retention']:.1f}% 保留)", flush=True)
