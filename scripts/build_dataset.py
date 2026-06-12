"""完整训练数据集生成脚本（独立版）"""
import sys, time
sys.path.insert(0, '.')
import pandas as pd
from ml import FeatureLabelPipeline

SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15m', '30m', '1h', '4h']
DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data"
SHAPES_CSV = "outputs/all_shapes_2024_2026.csv"
OUTPUT = "outputs/training_dataset.csv"

pipeline = FeatureLabelPipeline(d_zone_tolerance=0.005, sl_buffer_atr=0.3)
shapes_all = pd.read_csv(SHAPES_CSV)
print(f"加载 {len(shapes_all)} 形态", flush=True)

all_rows = []
stats = []
total_t0 = time.time()

for sym in SYMBOLS:
    for tf in TIMEFRAMES:
        t0 = time.time()
        group = shapes_all[(shapes_all['symbol'] == sym) & (shapes_all['timeframe'] == tf)]
        if len(group) == 0:
            continue
        
        parquet_path = f"{DATA_DIR}/binance_{sym}_{tf}.parquet"
        df = pd.read_parquet(parquet_path)
        df = df[(df.index >= '2024-01-01') & (df.index < '2026-05-01')]
        
        result = pipeline.process_shapes(df, group)
        elapsed = time.time() - t0
        
        if len(result) > 0:
            all_rows.append(result)
            stats.append({
                "symbol": sym, "tf": tf,
                "shapes": len(group), "traded": len(result),
                "pct": round(len(result) / len(group) * 100, 1),
                "win_rate": round(result["label"].mean() * 100, 1),
                "avg_r": round(result["final_r"].mean(), 3),
            })
            print(f"  {sym:8s} {tf:4s}: {len(group):>5d} → {len(result):>5d} trades | "
                  f"WR={result['label'].mean()*100:.1f}% avgR={result['final_r'].mean():.3f} | "
                  f"{elapsed:.1f}s", flush=True)
        else:
            print(f"  {sym:8s} {tf:4s}: {len(group):>5d} → 0 trades | {elapsed:.1f}s", flush=True)

# 合并保存
dataset = pd.concat(all_rows, ignore_index=True)
dataset.to_csv(OUTPUT, index=False)

print(f"\n{'='*60}")
print(f"📊 训练数据集: {OUTPUT}")
print(f"总形态: {len(shapes_all)} → 有效交易: {len(dataset)} ({len(dataset)/len(shapes_all)*100:.1f}%)")
print(f"总胜率: {dataset['label'].mean()*100:.1f}% | 平均 R: {dataset['final_r'].mean():.3f}")
print(f"总耗时: {time.time()-total_t0:.1f}s")
print(f"\n分组:", flush=True)
for s in stats:
    print(f"  {s['symbol']:8s} {s['tf']:4s}: {s['shapes']:>5d} → {s['traded']:>5d} "
          f"({s['pct']:>5.1f}%) WR={s['win_rate']:>5.1f}% avgR={s['avg_r']:>7.3f}", flush=True)
