"""
Phase B — Step 4: 自适应 vs 固定参数对比

加载 LightGBM 模型，对整个回测期做 walk-forward：
每2个月读取市场状态 → 预测最优参数 → 用预测参数回测 → 对比 V4 固定参数
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

import joblib
from sklearn.preprocessing import StandardScaler

from backtest.walk_forward_v4 import WalkForwardEngineV4

DATA_DIR = "/mnt/c/Users/12645/Sisie-Quantive/data/data_um"
OUTPUT_DIR = "outputs"
DATE_START = "2024-01-01"
DATE_END = "2026-05-01"

# 固定 baseline 参数（V4_swing3_vol 默认）
FIXED_PARAMS = dict(
    atr_mult=0.7,
    min_quality_score=50,
    sl_buffer_atr=0.2,
    atr_period=14, bc_ab_min=0.382, bc_ab_max=0.886, cd_ab_ratio=1.0,
    d_zone_tolerance=0.005, timeout_mult=2.0, tp1_pct=0.5,
    min_atr_mult=0.5, min_tp1_pct=0.003,
    use_confirmation=True, confirm_bars=3, confirm_mode="swing",
    vol_adaptive=True, btc_atr_pct_ref=0.003,
)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[(df.index >= DATE_START) & (df.index < DATE_END)][["high", "low", "close", "open", "volume"]]


def extract_state_features(df: pd.DataFrame, window_end: pd.Timestamp) -> np.ndarray:
    """提取市场状态特征向量用于模型预测"""
    end_idx = df.index.get_indexer([window_end], method='ffill')[0]
    if end_idx < 60:
        return None
    start_idx = max(0, end_idx - 120)
    seg = df.iloc[start_idx:end_idx + 1]
    cl, hi, lo, vol = seg["close"].values, seg["high"].values, seg["low"].values, seg["volume"].values

    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(cl, 1)), np.abs(lo - np.roll(cl, 1))))
    tr[0] = hi[0] - lo[0]
    alpha = 2 / 15; atr = np.zeros(len(tr)); atr[0] = tr[0]
    for i in range(1, len(tr)): atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
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
        try: sl, _ = np.polyfit(np.arange(20), cl[-20:], 1); ts = sl / cl[-1] if cl[-1] > 0 else 0
        except: pass
    v20 = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
    vr = vol[-1] / v20 if v20 > 0 else 1
    rhl = (hi[-20:].max() - lo[-20:].min()) / cl[-1] if len(cl) >= 20 and cl[-1] > 0 else 0
    apm20 = np.mean(atr_p[-20:]) if len(atr_p) >= 20 else cap

    return np.array([cap, ap80, atr_rank, apm20, td, pve, ts, vr, rhl])


def run_backtest(df: pd.DataFrame, params: dict, sym: str) -> dict:
    """用给定参数跑一次回测, 返回统计"""
    engine = WalkForwardEngineV4(**params)
    result = engine.run(df, sym, "15m")
    if result is None or len(result) == 0:
        return {"trades": 0, "wr": 0, "avg_r": 0, "total_r": 0}
    trades = len(result)
    return {
        "trades": trades,
        "wr": round(result["label"].mean() * 100, 1),
        "avg_r": round(result["final_r"].mean(), 3),
        "total_r": round(result["final_r"].sum(), 1),
    }


def compare_on_symbol(sym: str, model, scaler_X, features_list):
    """对一个标的执行自适应 vs 固定对比"""
    print(f"\n>>> {sym}", flush=True)
    path = os.path.join(DATA_DIR, f"{sym}_15m_um.parquet")
    if not os.path.exists(path):
        print(f"    ⚠️ No data", flush=True)
        return None
    df = load_data(path)

    # 固定参数回测（全量）
    fixed = run_backtest(df, FIXED_PARAMS, sym)

    # 自适应 walk-forward: 每 2 个月
    start, end = df.index.min(), df.index.max()
    current = start
    adaptive_trades = []

    window_month = 3  # 每 3 个月更新参数
    warmup = pd.DateOffset(months=4)  # 前 4 个月预热

    while current < end:
        we = min(current + pd.DateOffset(months=window_month), end)

        # 构建市场状态（截取到 we 之前的数据）
        lookback_start = max(df.index.min(), we - pd.DateOffset(months=2))
        lookback_df = df[(df.index >= lookback_start) & (df.index <= we)]

        state = extract_state_features(df, we)
        if state is None:
            current = we
            continue

        # 预测参数
        state_scaled = scaler_X.transform(state.reshape(1, -1))
        pred = model.predict(state_scaled)[0]
        atr_m = max(0.3, min(1.0, pred[0]))
        min_q = max(30, min(60, round(pred[1] / 10) * 10))
        sl_b = max(0.2, min(0.7, round(pred[2] * 10) / 10))

        # 回测窗口
        pre_start = max(df.index.min(), current - pd.DateOffset(months=3))
        wdf = df[(df.index >= pre_start) & (df.index <= we)]

        params = {**FIXED_PARAMS, "atr_mult": atr_m,
                  "min_quality_score": min_q, "sl_buffer_atr": sl_b}

        engine = WalkForwardEngineV4(**params)
        result = engine.run(wdf, sym, "15m")
        if result is not None and len(result) > 0:
            result = result[result["entry_time"] >= str(current)]
            for _, r in result.iterrows():
                adaptive_trades.append(r)

        current = we

    # 自适应统计
    if adaptive_trades:
        adf = pd.DataFrame(adaptive_trades)
        adf["symbol"] = sym
        wr_a = adf["label"].mean() * 100
        avg_r_a = adf["final_r"].mean()
        total_r_a = adf["final_r"].sum()
        trades_a = len(adf)
    else:
        wr_a = avg_r_a = total_r_a = 0
        trades_a = 0

    print(f"  固定: {fixed['trades']:>5d} trades  WR={fixed['wr']:>5.1f}%  "
          f"avgR={fixed['avg_r']:>6.3f}  totalR={fixed['total_r']:>8.1f}", flush=True)
    print(f"  自适应: {trades_a:>5d} trades  WR={wr_a:>5.1f}%  "
          f"avgR={avg_r_a:>6.3f}  totalR={total_r_a:>8.1f}", flush=True)

    return {
        "symbol": sym,
        "fixed_trades": fixed["trades"], "fixed_wr": fixed["wr"],
        "fixed_avgR": fixed["avg_r"], "fixed_totalR": fixed["total_r"],
        "adapt_trades": trades_a, "adapt_wr": round(wr_a, 1) if trades_a > 0 else 0,
        "adapt_avgR": round(avg_r_a, 3) if trades_a > 0 else 0,
        "adapt_totalR": round(total_r_a, 1) if trades_a > 0 else 0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    model_path = f"{OUTPUT_DIR}/models/lgb_adaptive.pkl"
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print("   请先运行 Step 3 (python scripts/train_lgb_adaptive.py)")
        return

    saved = joblib.load(model_path)
    model = saved["model"]
    scaler_X = saved["scaler_X"]
    features_list = saved["features"]
    print(f"✅ 加载模型: {model_path}", flush=True)
    print(f"   特征: {features_list}", flush=True)
    print(f"   目标: {saved['targets']}", flush=True)

    # 测试标的
    test_symbols = ["BTCUSDT", "NEARUSDT", "ADAUSDT"]

    all_results = []
    for sym in test_symbols:
        r = compare_on_symbol(sym, model, scaler_X, features_list)
        if r:
            all_results.append(r)
        gc.collect()

    # 结果
    print(f"\n{'='*80}", flush=True)
    print(f"📊 自适应 vs 固定参数 对比", flush=True)
    print(f"{'='*80}", flush=True)

    header = f"{'Symbol':>8s} | {'固定trades':>10s} | {'固定WR%':>8s} | {'固定avgR':>8s} | {'固定totalR':>10s} | " \
             f"{'自适应trades':>10s} | {'自适应WR%':>8s} | {'自适应avgR':>8s} | {'自适应totalR':>10s} | {'WRΔ':>6s} | {'totalRΔ':>8s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    total_fixed_t = 0
    total_adapt_t = 0
    total_fixed_tr = 0
    total_adapt_tr = 0

    for r in all_results:
        wr_delta = r["adapt_wr"] - r["fixed_wr"]
        tr_delta = r["adapt_totalR"] - r["fixed_totalR"]
        print(f"{r['symbol']:>8s} | {r['fixed_trades']:>10d} | {r['fixed_wr']:>7.1f}% | "
              f"{r['fixed_avgR']:>8.3f} | {r['fixed_totalR']:>10.1f} | "
              f"{r['adapt_trades']:>10d} | {r['adapt_wr']:>7.1f}% | "
              f"{r['adapt_avgR']:>8.3f} | {r['adapt_totalR']:>10.1f} | "
              f"{wr_delta:>+5.1f}% | {tr_delta:>+7.1f}", flush=True)
        total_fixed_t += r["fixed_trades"]
        total_adapt_t += r["adapt_trades"]
        total_fixed_tr += r["fixed_totalR"]
        total_adapt_tr += r["adapt_totalR"]

    # 合计
    wr_delta_total = (total_adapt_tr / max(total_adapt_t, 1) / max(total_fixed_tr / max(total_fixed_t, 1), 1e-10) - 1) * 100
    print("-" * len(header), flush=True)
    print(f"{'合计':>8s} | {total_fixed_t:>10d} | {'--':>8s} | {'--':>8s} | "
          f"{total_fixed_tr:>10.1f} | {total_adapt_t:>10d} | {'--':>8s} | "
          f"{'--':>8s} | {total_adapt_tr:>10.1f} | {'--':>6s} | "
          f"{'--':>8s}", flush=True)
    print(f"\ntotalR 变化: {total_fixed_tr:.1f} → {total_adapt_tr:.1f} "
          f"({(total_adapt_tr/total_fixed_tr-1)*100:+.1f}%)", flush=True)

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUTPUT_DIR}/adaptive_vs_fixed.csv", index=False)
    print(f"\n✅ 对比结果保存: {OUTPUT_DIR}/adaptive_vs_fixed.csv", flush=True)
    print(f"✅ Step 4 完成", flush=True)


if __name__ == "__main__":
    main()
