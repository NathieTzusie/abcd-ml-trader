"""
跨TF测试: V4+ML 在 BTC/ETH × 15m/30m/1h/4h
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, xgboost as xgb
from backtest.walk_forward_v4 import WalkForwardEngineV4

DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data"
DATE_START, DATE_END = "2024-01-01", "2026-05-01"
MODEL_PATH = "outputs/xgboost_v4_model.json"

ML_FEATURES = [
    "ab_distance_pct", "bc_ab_ratio", "quality_score", "ab_bars", "bc_bars",
    "d_zone_bars", "d_zone_volume_ratio", "d_zone_momentum", "d_zone_wick_ratio",
    "atr_pct", "atr_percentile_20", "ema20_50_direction", "price_vs_ema20",
    "volume_ratio", "volume_trend", "hour_of_day", "day_of_week", "vol_mult",
]

BASE = dict(atr_mult=0.7, min_quality_score=50, sl_buffer_atr=0.2, min_tp1_pct=0.003,
            bc_ab_min=0.382, bc_ab_max=0.886, cd_ab_ratio=1.0, d_zone_tolerance=0.005,
            timeout_mult=2.0, tp1_pct=0.5, min_atr_mult=0.5,
            use_confirmation=True, confirm_bars=3, confirm_mode="swing",
            vol_adaptive=True, btc_atr_pct_ref=0.003)

# 不同TF的 btc_atr_pct_ref 需要调整
ATR_REF = {"15m": 0.003, "30m": 0.0045, "1h": 0.006, "4h": 0.012}

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]

print(f"{'='*100}")
print(f"跨TF测试: BTC/ETH × 15m/30m/1h/4h")
print(f"{'='*100}\n")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

all_rows = []

for sym in ["BTCUSDT", "ETHUSDT"]:
    for tf in ["15m", "30m", "1h", "4h"]:
        path = f"{DATA_DIR}/binance_{sym}_{tf}.parquet"
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"]); df = df.set_index("ts")
        df = df.sort_index()
        df = df[(df.index >= DATE_START) & (df.index < DATE_END)][["high","low","close","open","volume"]]

        t0 = time.time()
        engine = WalkForwardEngineV4(**{**BASE, "btc_atr_pct_ref": ATR_REF[tf]})
        result = engine.run(df, sym, tf)
        el = time.time() - t0

        if result is None or len(result) == 0:
            print(f"  {sym} {tf}: 0 trades", flush=True)
            all_rows.append({"symbol": sym, "tf": tf, "v4_trades": 0})
            continue

        # V4 baseline
        v4_t = len(result)
        v4_wr = result["label"].mean() * 100
        v4_tr = result["final_r"].sum()
        v4_avg = result["final_r"].mean()
        v4_sl = sum(1 for x in result["exit_reason"] if "SL" in str(x)) / v4_t * 100

        # ML 评分
        feats = result[ML_FEATURES].values.astype(np.float32)
        result["ml_score"] = model.predict_proba(feats)[:, 1]

        row = {"symbol": sym, "tf": tf, "v4_trades": v4_t, "v4_wr": round(v4_wr, 1),
               "v4_avgR": round(v4_avg, 3), "v4_totalR": round(v4_tr, 1), "v4_sl": round(v4_sl, 1)}

        line = f"  {sym} {tf}: V4={v4_t}t WR={v4_wr:.1f}% avgR={v4_avg:.3f} totalR={v4_tr:.0f} SL={v4_sl:.1f}% |"
        for th in THRESHOLDS:
            fg = result[result["ml_score"] >= th]
            if len(fg) > 0:
                t = len(fg); wr = fg["label"].mean() * 100; tr = fg["final_r"].sum()
                sl = sum(1 for x in fg["exit_reason"] if "SL" in str(x)) / t * 100
                ret = t / v4_t * 100
                row[f"th{th:.2f}_wr"] = round(wr, 1)
                row[f"th{th:.2f}_totalR"] = round(tr, 1)
                row[f"th{th:.2f}_sl"] = round(sl, 1)
                row[f"th{th:.2f}_ret"] = round(ret, 1)
                line += f" th{th:.2f}={wr:.1f}%/{tr:.0f}"
        line += f" | {el:.1f}s"
        print(line, flush=True)
        all_rows.append(row)

res = pd.DataFrame(all_rows)
res.to_csv("outputs/cross_market/cross_tf_summary.csv", index=False)

print(f"\n{'='*100}")
print(f"📊 跨TF对比 — V4 vs V4+ML(th=0.60)")
print(f"{'='*100}")
for _, r in res.iterrows():
    if r["v4_trades"] == 0: continue
    v4w = r["v4_wr"]; v4r = r["v4_totalR"]
    mlw = r.get("th0.60_wr", 0); mlr = r.get("th0.60_totalR", 0); mls = r.get("th0.60_sl", 0)
    ret = r.get("th0.60_ret", 0)
    print(f"  {r['symbol']:8s} {r['tf']:4s}: V4 WR={v4w:.1f}% totalR={v4r:.0f} → ML WR={mlw:.1f}% totalR={mlr:.0f} SL={mls:.1f}% ret={ret:.1f}%")

print(f"\n✅ outputs/cross_market/cross_tf_summary.csv")
