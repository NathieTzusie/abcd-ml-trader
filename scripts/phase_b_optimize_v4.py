"""
Phase B — Step 2: V4 滚动窗口参数优化（并行版）

并行执行多个标的以减少总耗时。
"""
import sys, os, time, gc, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Optional, Tuple
from multiprocessing import Process, Queue, cpu_count
import warnings; warnings.filterwarnings('ignore')

from backtest.walk_forward_v4 import WalkForwardEngineV4

DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data/data_um"
OUTPUT_DIR = "outputs"
DATE_START = "2024-01-01"
DATE_END = "2026-05-01"

FIXED_PARAMS = dict(
    atr_period=14, bc_ab_min=0.382, bc_ab_max=0.886, cd_ab_ratio=1.0,
    d_zone_tolerance=0.005, timeout_mult=2.0, tp1_pct=0.5,
    min_atr_mult=0.5, min_tp1_pct=0.003,
    use_confirmation=True, confirm_bars=3, confirm_mode="swing",
    vol_adaptive=True, btc_atr_pct_ref=0.003,
)

PARAM_GRID = {
    "atr_mult": [0.3, 0.5, 0.7, 1.0],
    "min_quality_score": [30, 40, 50, 60],
    "sl_buffer_atr": [0.2, 0.3, 0.5, 0.7],
}
N_COMBOS = 64


def load_data(path: str):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= DATE_START) & (df.index < DATE_END)][["high", "low", "close", "open", "volume"]]


def extract_market_state(df: pd.DataFrame, window_end: pd.Timestamp) -> dict:
    end_idx = df.index.get_indexer([window_end], method='ffill')[0]
    if end_idx < 60: return {}
    start_idx = max(0, end_idx - 120)
    seg = df.iloc[start_idx:end_idx + 1]
    cl, hi, lo, vol = seg["close"].values, seg["high"].values, seg["low"].values, seg["volume"].values
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(cl, 1)), np.abs(lo - np.roll(cl, 1))))
    tr[0] = hi[0] - lo[0]
    alpha = 2 / 15; atr = np.zeros(len(tr)); atr[0] = tr[0]
    for i in range(1, len(tr)): atr[i] = alpha * tr[i] + (1-alpha) * atr[i-1]
    cap = atr[-1] / cl[-1] if cl[-1] > 0 else 0
    atr_p = atr / cl
    ap80 = np.percentile(atr_p[-20:], 80) if len(atr_p) >= 20 else cap
    atr_rank = 0.5
    if len(atr_p) >= 50:
        a50 = atr_p[-50:]; rng = a50.max() - a50.min()
        atr_rank = (atr_p[-1] - a50.min()) / max(rng, 1e-10)
    ema20 = np.mean(cl[-20:]) if len(cl) >= 20 else cl[-1]
    ema50 = np.mean(cl[-50:]) if len(cl) >= 50 else cl[-1]
    td = 1 if ema20 > ema50 else -1
    pve = (cl[-1] - ema20) / ema20 if ema20 > 0 else 0
    ts = 0.0
    if len(cl) >= 20:
        try:
            sl, _ = np.polyfit(np.arange(20), cl[-20:], 1)
            ts = sl / cl[-1] if cl[-1] > 0 else 0
        except: pass
    v20 = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
    vr = vol[-1] / v20 if v20 > 0 else 1
    rhl = (hi[-20:].max() - lo[-20:].min()) / cl[-1] if len(cl) >= 20 and cl[-1] > 0 else 0
    apm20 = np.mean(atr_p[-20:]) if len(atr_p) >= 20 else cap
    return dict(cur_atr_pct=round(cap,6), atr_percentile_80=round(ap80,6),
                atr_rank_50=round(atr_rank,4), atr_pct_mean_20=round(apm20,6),
                trend_direction=td, price_vs_ema20=round(pve,6),
                trend_strength=round(ts,6), volume_rank=round(vr,4),
                recent_hl_pct=round(rhl,6))


def rolling_window_optimize(df: pd.DataFrame, sym: str, tf: str) -> pd.DataFrame:
    df = df.sort_index()
    start, end = df.index.min(), df.index.max()
    results, current, wm = [], start, 3
    while current < end:
        we = min(current + pd.DateOffset(months=wm), end)
        ps = max(df.index.min(), current - pd.DateOffset(months=2))
        wdf = df[(df.index >= ps) & (df.index <= we)]
        if len(wdf) < 200: current = we; continue
        state = extract_market_state(df, we)
        if not state: current = we; continue
        score_best = -float('inf'); bp = None; bs = None
        for atr_m, min_q, sl_b in product(PARAM_GRID["atr_mult"], PARAM_GRID["min_quality_score"], PARAM_GRID["sl_buffer_atr"]):
            p = {**FIXED_PARAMS, "atr_mult": atr_m, "min_quality_score": min_q, "sl_buffer_atr": sl_b}
            r = WalkForwardEngineV4(**p).run(wdf, sym, tf)
            if r is None or len(r) < 5: continue
            r = r[r["entry_time"] >= str(current)]
            if len(r) < 3: continue
            wr = r["label"].mean(); ar = r["final_r"].mean(); sc = ar * wr
            if sc > score_best:
                score_best = sc; bp = (atr_m, min_q, sl_b, len(r), wr, ar, r["final_r"].sum())
        if bp:
            results.append({**state, "atr_mult": bp[0], "min_quality_score": bp[1],
                           "sl_buffer_atr": bp[2], "score": round(score_best,4),
                           "trades": bp[3], "wr": round(bp[4]*100,1),
                           "avg_r": round(bp[5],3), "total_r": round(bp[6],1),
                           "window_start": str(current.date()), "window_end": str(we.date()),
                           "symbol": sym, "timeframe": tf})
        current = we
    return pd.DataFrame(results)


def worker(sym: str, q: Queue):
    try:
        path = os.path.join(DATA_DIR, f"{sym}_15m_um.parquet")
        if not os.path.exists(path): q.put(("error", sym, f"No data {path}")); return
        df = load_data(path)
        t0 = time.time()
        windows = rolling_window_optimize(df, sym, "15m")
        t = time.time() - t0
        q.put(("done", sym, windows, t))
    except Exception as e:
        q.put(("error", sym, str(e)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_symbols = sorted([f.split("_")[0] for f in os.listdir(DATA_DIR) if f.endswith("_15m_um.parquet")])
    print(f"[V4 滚动窗口] {N_COMBOS} 组合 × 3月步长, {len(all_symbols)} 标的, 并行 {min(4, cpu_count())} 路", flush=True)

    test_symbols = ["BTCUSDT", "NEARUSDT"]
    n_workers = min(4, cpu_count())
    all_rows = []

    for phase_name, syms in [("验证", test_symbols), ("全量", all_symbols)]:
        print(f"\n{'='*70}\nPhase: {phase_name}", flush=True)
        for sym in syms:
            if phase_name == "全量" and sym in test_symbols and all_rows:
                continue  # skip if phase_b_windows.csv already has all
            print(f">>> {sym}...", flush=True)
            t0 = time.time()
            path = os.path.join(DATA_DIR, f"{sym}_15m_um.parquet")
            df = load_data(path)
            windows = rolling_window_optimize(df, sym, "15m")
            t = time.time() - t0
            if len(windows) > 0:
                for _, r in windows.iterrows(): all_rows.append(r.to_dict())
                print(f"    ✅ {len(windows)} 窗口 in {t:.0f}s", flush=True)
            else:
                print(f"    ⚠️ 0 窗口 in {t:.0f}s", flush=True)
            del df, windows; gc.collect()

    # 保存
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(f"{OUTPUT_DIR}/phase_b_windows.csv", index=False)
        print(f"\n{'='*70}")
        print(f"✅ 保存: {OUTPUT_DIR}/phase_b_windows.csv")
        print(f"   总计 {len(df_all)} 窗口, {df_all['symbol'].nunique()} 标的")
        print(f"   时间: {df_all['window_start'].min()} ~ {df_all['window_end'].max()}")
    else:
        print("\n⚠️ 无窗口数据")

if __name__ == "__main__":
    main()
