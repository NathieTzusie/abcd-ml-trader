"""
Phase A — Step 3: ML 软过滤实验

扫描不同阈值以找到最优过滤。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "outputs/v4_training_dataset.csv"
MODEL_PATH = "outputs/xgboost_v4_model.json"
TRAIN_RATIO = 0.8

FEATURE_COLS = [
    "ab_distance_pct", "bc_ab_ratio", "quality_score",
    "ab_bars", "bc_bars", "d_zone_bars",
    "d_zone_volume_ratio", "d_zone_momentum", "d_zone_wick_ratio",
    "atr_pct", "atr_percentile_20",
    "ema20_50_direction", "price_vs_ema20",
    "volume_ratio", "volume_trend",
    "hour_of_day", "day_of_week", "vol_mult",
]

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]

# === 加载模型 ===
print("加载模型...", flush=True)
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print(f"✅ 模型已加载: {MODEL_PATH}", flush=True)

# === 加载数据 ===
df = pd.read_csv(DATA_PATH)
df["entry_time"] = pd.to_datetime(df["entry_time"])
df = df.sort_values("entry_time").reset_index(drop=True)

split_idx = int(len(df) * TRAIN_RATIO)
test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
print(f"测试集: {len(test_df):,} trades", flush=True)

# === 预测概率 ===
X_test = test_df[FEATURE_COLS].values
probas = model.predict_proba(X_test)[:, 1]
test_df["ml_score"] = probas

# === 基准 ===
total_all = len(test_df)
win_all = test_df["label"].sum()
wr_all = win_all / total_all * 100
avgR_all = test_df["final_r"].mean()
totalR_all = test_df["final_r"].sum()
medR_all = test_df["final_r"].median()

print(f"\n{'='*80}")
print("🔍 ML 软过滤实验: 按 predict_proba 阈值过滤")
print(f"{'='*80}")
print(f"\n{'-'*80}")
print(f"{'✅ 基准 (全测试集)':<60s}")
print(f"{'-'*80}")
print(f"  Trades = {total_all:,} | WR = {wr_all:.1f}% | avgR = {avgR_all:.3f} | "
      f"totalR = {totalR_all:.0f} | medR = {medR_all:.3f}", flush=True)

# === 按阈值筛选 ===
results = []
for th in THRESHOLDS:
    mask = test_df["ml_score"] > th
    filtered = test_df[mask]

    n_trades = len(filtered)
    n_win = filtered["label"].sum()
    wr = n_win / n_trades * 100 if n_trades > 0 else 0
    avg_r = filtered["final_r"].mean() if n_trades > 0 else 0
    total_r = filtered["final_r"].sum() if n_trades > 0 else 0
    med_r = filtered["final_r"].median() if n_trades > 0 else 0

    retention = n_trades / total_all * 100
    wr_delta = wr - wr_all
    avgR_delta = avg_r - avgR_all

    # SL比率
    sl_count = sum(1 for r in filtered["exit_reason"] if "SL" in str(r)) if "exit_reason" in filtered.columns else 0
    sl_pct = sl_count / n_trades * 100 if n_trades > 0 else 0

    results.append({
        "threshold": th,
        "trades": n_trades,
        "retention_pct": round(retention, 1),
        "wr": round(wr, 1),
        "wr_delta": round(wr_delta, 1),
        "avg_r": round(avg_r, 3),
        "avgR_delta": round(avgR_delta, 3),
        "total_r": round(total_r, 1),
        "med_r": round(med_r, 3),
        "sl_pct": round(sl_pct, 1),
        "wins": n_win,
    })

    print(f"\n✅ 阈值 = {th:.2f}", flush=True)
    print(f"   {n_trades:>7,} trades ({retention:.1f}% 保留率)", flush=True)
    print(f"   WR = {wr:.1f}% ({wr_delta:+.1f}pp) | avgR = {avg_r:.3f} ({avgR_delta:+.3f}) | "
          f"totalR = {total_r:.0f} | medR = {med_r:.3f}", flush=True)
    print(f"   SL = {sl_pct:.1f}% | Wins = {n_win}", flush=True)

# === 最优阈值建议 ===
print(f"\n{'='*80}")
print("📊 最优阈值分析")
print(f"{'='*80}")

# avgR-based: prefer highest avgR
best_avgR = max(results, key=lambda x: x["avg_r"])
print(f"\n根据 avgR 最优: threshold = {best_avgR['threshold']:.2f} (avgR = {best_avgR['avg_r']:.3f})")

# Sharpe-like: avgR * WR
for r in results:
    r["score"] = r["avg_r"] * r["wr"] * np.sqrt(r["trades"] / total_all)
best_score = max(results, key=lambda x: x["score"])
print(f"综合评分最优: threshold = {best_score['threshold']:.2f} (score = {best_score['score']:.2f})")

# 保留率 vs 改进的平衡
print(f"\n{'='*80}")
print("推荐方案:")
print(f"{'='*80}")
print(f"  保守 (强过滤): threshold=0.65 — avgR={[r['avg_r'] for r in results if r['threshold']==0.65][0]:.3f}, "
      f"retention={[r['retention_pct'] for r in results if r['threshold']==0.65][0]:.1f}%")
print(f"  平衡:         threshold=0.60 — avgR={[r['avg_r'] for r in results if r['threshold']==0.60][0]:.3f}, "
      f"retention={[r['retention_pct'] for r in results if r['threshold']==0.60][0]:.1f}%")
print(f"  积极:         threshold=0.55 — avgR={[r['avg_r'] for r in results if r['threshold']==0.55][0]:.3f}, "
      f"retention={[r['retention_pct'] for r in results if r['threshold']==0.55][0]:.1f}%")

# === 保存结果 ===
result_df = pd.DataFrame(results)
result_df.to_csv("outputs/ml_filter_results.csv", index=False)
print(f"\n✅ 详细结果: outputs/ml_filter_results.csv")

# === 累积分布 ===
print(f"\n{'='*80}")
print("📊 ML Score 分布")
print(f"{'='*80}")
test_df["ml_score_bin"] = pd.cut(test_df["ml_score"], bins=10, precision=2)
binned = test_df.groupby("ml_score_bin", observed=False).agg(
    trades=("label", "count"),
    wins=("label", "sum"),
    avgR=("final_r", "mean"),
    totalR=("final_r", "sum"),
).reset_index()
for _, row in binned.iterrows():
    wr = row["wins"] / row["trades"] * 100
    bar = "█" * int(row["trades"] / max(binned["trades"]) * 30)
    bin_label = str(row['ml_score_bin'])
    print(f"  {bin_label:>18s}: {row['trades']:>7,} trades  WR={wr:>5.1f}%  avgR={row['avgR']:.3f}  {bar}")

print(f"\n✅ Step 3 完成", flush=True)
