"""V1 vs V4 全量对比 — 精简版（3配置）"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from backtest.walk_forward import WalkForwardEngine as V1
from backtest.walk_forward_v4 import WalkForwardEngineV4 as V4

DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data/data_um"
OUTPUT_DIR = "outputs/cross_market"
DATE_START, DATE_END = "2024-01-01", "2026-05-01"

BASE = dict(atr_mult=0.7, min_quality_score=50, sl_buffer_atr=0.2, min_tp1_pct=0.003,
            bc_ab_min=0.382, bc_ab_max=0.886, cd_ab_ratio=1.0, d_zone_tolerance=0.005,
            timeout_mult=2.0, tp1_pct=0.5, min_atr_mult=0.5)

CONFIGS = [
    ("V1_baseline",     V1, BASE),
    ("V4_swing3",       V4, {**BASE, "use_confirmation": True, "confirm_bars": 3, "confirm_mode": "swing"}),
    ("V4_swing3_vol",   V4, {**BASE, "use_confirmation": True, "confirm_bars": 3, "confirm_mode": "swing",
                              "vol_adaptive": True, "btc_atr_pct_ref": 0.003}),
]

def load_data(path):
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("ts").sort_index()
    return df[(df.index >= DATE_START) & (df.index < DATE_END)][["high","low","close","open","volume"]]

def stats(engine_cls, params, df, sym, tf):
    t0 = time.time()
    e = engine_cls(**params)
    r = e.run(df, sym, tf)
    el = time.time()-t0
    if r is None or len(r) == 0:
        return {"trades":0,"wr":0,"avg_r":0,"total_r":0,"sl_pct":0,"elapsed":el}
    t = len(r)
    return {"trades":t, "wr":r["label"].mean()*100, "avg_r":r["final_r"].mean(),
            "total_r":r["final_r"].sum(), "sl_pct":sum(1 for x in r["exit_reason"] if "SL" in str(x))/t*100,
            "elapsed":el}

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])

print(f"{'='*100}")
print(f"V1 vs V4 全量对比 (16 标的 × 3 配置)")
print(f"{'='*100}")

all_rows = []
for f in files:
    sym = f.split("_")[0]
    df = load_data(os.path.join(DATA_DIR, f))
    row = {"symbol": sym}
    for name, cls, params in CONFIGS:
        s = stats(cls, params, df, sym, "15m")
        for k, v in s.items():
            row[f"{name}_{k}"] = v
        if s["trades"] > 0:
            print(f"  {sym:6s} {name:18s}: {s['trades']:>5d}t  WR={s['wr']:.1f}%  "
                  f"avgR={s['avg_r']:.3f}  totalR={s['total_r']:.1f}  SL={s['sl_pct']:.1f}%  {s['elapsed']:.1f}s", flush=True)
    all_rows.append(row)

res = pd.DataFrame(all_rows)
res.to_csv(f"{OUTPUT_DIR}/v1_vs_v4_comparison.csv", index=False)

print(f"\n{'='*100}")
print(f"📊 全体加权平均")
print(f"{'='*100}")
for name, _, _ in CONFIGS:
    tc = f"{name}_trades"; wc = f"{name}_wr"; trc = f"{name}_total_r"; sc = f"{name}_sl_pct"
    tt = res[tc].sum()
    if tt > 0:
        aw = (res[wc] * res[tc]).sum() / tt
        atr = res[trc].sum()
        asl = (res[sc] * res[tc]).sum() / tt
        print(f"  {name:20s}: {tt:>6.0f} trades  WR={aw:.1f}%  totalR={atr:.1f}  SL={asl:.1f}%")

print(f"\n{'='*100}")
print(f"📊 按 V4_swing3_vol vs V1 胜率提升排名")
print(f"{'='*100}")
improved = []
for _, row in res.iterrows():
    v1_wr = row["V1_baseline_wr"]
    v4_wr = row["V4_swing3_vol_wr"]
    if v1_wr > 0 and v4_wr > 0:
        improved.append((v4_wr - v1_wr, row["symbol"], v1_wr, v4_wr,
                         row["V1_baseline_total_r"], row["V4_swing3_vol_total_r"]))
for gain, sym, v1w, v4w, v1tr, v4tr in sorted(improved, reverse=True):
    print(f"  {sym:6s}: {v1w:.1f}% → {v4w:.1f}%  (+{gain:+.1f}pp)  "
          f"totalR: {v1tr:.0f} → {v4tr:.0f}")

print(f"\n✅ {OUTPUT_DIR}/v1_vs_v4_comparison.csv")
