"""
V4 + ML 集成过滤器

等价于 V5 引擎，但用分离式：V4跑完 → XGBoost批量评分 → 过滤
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
import xgboost as xgb

from backtest.walk_forward_v4 import WalkForwardEngineV4

DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data/data_um"
OUTPUT_DIR = "outputs/cross_market"
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

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def load_data(path):
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("ts").sort_index()
    df = df[(df.index >= DATE_START) & (df.index < DATE_END)]
    return df[["high", "low", "close", "open", "volume"]]


def run_v4_with_ml(sym, df):
    """跑V4引擎 → 加ML评分 → 多阈值过滤"""
    engine = WalkForwardEngineV4(**BASE)
    result = engine.run(df, sym, "15m")
    if result is None or len(result) == 0:
        return None

    # 批量 ML 预测
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    feats = result[ML_FEATURES].values.astype(np.float32)
    result["ml_score"] = model.predict_proba(feats)[:, 1]

    stats = {}
    for th in THRESHOLDS:
        filtered = result[result["ml_score"] >= th]
        if len(filtered) == 0:
            stats[th] = {"trades": 0, "wr": 0, "avg_r": 0, "total_r": 0, "sl_pct": 0}
            continue
        t = len(filtered)
        wr = filtered["label"].mean() * 100
        avg_r = filtered["final_r"].mean()
        total_r = filtered["final_r"].sum()
        sl_pct = sum(1 for x in filtered["exit_reason"] if "SL" in str(x)) / t * 100
        stats[th] = {"trades": t, "wr": round(wr, 1), "avg_r": round(avg_r, 3),
                     "total_r": round(total_r, 1), "sl_pct": round(sl_pct, 1),
                     "retention": round(t / len(result) * 100, 1)}
    return stats


def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])

    print(f"{'='*100}")
    print(f"V4 + ML Filter 集成验证 (16标的 × 5阈值)")
    print(f"{'='*100}\n")

    # Baseline V4 (纯引擎, 无ML)
    print("Running V4 baseline...", flush=True)
    engine_v4 = WalkForwardEngineV4(**BASE)
    all_results = []

    for f in files:
        sym = f.split("_")[0]
        df = load_data(os.path.join(DATA_DIR, f))
        t0 = time.time()
        stats = run_v4_with_ml(sym, df)
        el = time.time() - t0

        if stats is None:
            print(f"  {sym:6s}: no trades", flush=True)
            continue

        # V4 baseline stats
        v4 = engine_v4.run(df, sym, "15m")
        v4_t = len(v4)
        v4_wr = v4["label"].mean() * 100
        v4_tr = v4["final_r"].sum()
        v4_sl = sum(1 for x in v4["exit_reason"] if "SL" in str(x)) / v4_t * 100

        print(f"  {sym:6s}: V4={v4_t}t WR={v4_wr:.1f}% totalR={v4_tr:.0f} |", end="", flush=True)
        for th in THRESHOLDS:
            s = stats.get(th, {})
            if s.get("trades", 0) > 0:
                print(f" th{th:.2f}={s['wr']:.1f}%/{s['trades']}t/{s['total_r']:.0f}", end="", flush=True)
        print(f" | {el:.1f}s", flush=True)

        all_results.append({"symbol": sym, "v4_trades": v4_t, "v4_wr": round(v4_wr, 1),
                           "v4_avgR": round(v4["final_r"].mean(), 3),
                           "v4_totalR": round(v4_tr, 1), "v4_sl": round(v4_sl, 1),
                           **{f"th{th:.2f}_wr": stats.get(th, {}).get("wr", 0) for th in THRESHOLDS},
                           **{f"th{th:.2f}_totalR": stats.get(th, {}).get("total_r", 0) for th in THRESHOLDS},
                           **{f"th{th:.2f}_trades": stats.get(th, {}).get("trades", 0) for th in THRESHOLDS},
                           **{f"th{th:.2f}_sl": stats.get(th, {}).get("sl_pct", 0) for th in THRESHOLDS},
                           })

    # 汇总
    res = pd.DataFrame(all_results)
    res.to_csv(f"{OUTPUT_DIR}/v4_ml_integrated.csv", index=False)

    print(f"\n{'='*100}")
    print(f"📊 V4 vs V4+ML 加权汇总")
    print(f"{'='*100}")

    # V4 全体
    t_v4 = res["v4_trades"].sum()
    w_v4 = (res["v4_wr"] * res["v4_trades"]).sum() / t_v4
    tr_v4 = res["v4_totalR"].sum()
    sl_v4 = (res["v4_sl"] * res["v4_trades"]).sum() / t_v4

    print(f"  {'V4 baseline':20s}: {t_v4:>6.0f}t  WR={w_v4:.1f}%  totalR={tr_v4:.1f}  SL={sl_v4:.1f}%")

    for th in THRESHOLDS:
        t_col = f"th{th:.2f}_trades"
        w_col = f"th{th:.2f}_wr"
        r_col = f"th{th:.2f}_totalR"
        s_col = f"th{th:.2f}_sl"
        tt = res[t_col].dropna().sum()
        if tt > 0:
            ww = (res[w_col] * res[t_col]).fillna(0).sum() / tt
            rr = res[r_col].sum()
            ss = (res[s_col] * res[t_col]).fillna(0).sum() / tt
            ret = tt / t_v4 * 100
            print(f"  {'V4+ML th='+str(th):20s}: {tt:>6.0f}t  WR={ww:.1f}%  totalR={rr:.1f}  SL={ss:.1f}%  retention={ret:.1f}%")

    print(f"\n✅ {OUTPUT_DIR}/v4_ml_integrated.csv")


if __name__ == "__main__":
    main()
