"""
Step 2: 完整形态扫描 — 8 组 × BTC/ETH × 15m/30m/1h/4h
输出：outputs/all_shapes_2024_2026.csv
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from abcd_detector import ABCDDetector

SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15m', '30m', '1h', '4h']
DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data"
OUTPUT = "outputs/all_shapes_2024_2026.csv"

detector = ABCDDetector(min_quality_score=40, min_atr_mult=0.5)
all_shapes = []
stats = []

for sym in SYMBOLS:
    for tf in TIMEFRAMES:
        path = f"{DATA_DIR}/binance_{sym}_{tf}.parquet"
        df = pd.read_parquet(path)
        df = df[(df.index >= '2024-01-01') & (df.index < '2026-05-01')]
        
        print(f"扫描 {sym} {tf} ({len(df)} bars)...")
        shapes = detector.detect(df, sym, tf)
        
        for s in shapes:
            all_shapes.append(s.to_dict())
        
        stats.append({
            'symbol': sym, 'tf': tf,
            'bars': len(df), 'shapes': len(shapes),
        })
        print(f"  → {len(shapes)} 形态")

# 保存
df_all = pd.DataFrame(all_shapes)
df_all.to_csv(OUTPUT, index=False)

print(f"\n{'='*50}")
print(f"✅ 总形态: {len(df_all)}")
print(f"✅ 已保存: {OUTPUT}")

# 分组统计
print(f"\n分组统计:")
for _, r in pd.DataFrame(stats).iterrows():
    print(f"  {r['symbol']:8s} {r['tf']:4s}: {r['shapes']:>5d} 形态  ({r['bars']:>6d} bars)")
