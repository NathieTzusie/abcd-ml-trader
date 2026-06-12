"""
Phase B: 参数自适应

流程：
  1. 滚动窗口（2个月）参数网格扫描
  2. 每窗口提取市场状态特征
  3. LightGBM 学习: market_state → optimal_params
  4. 自适应 vs 固定参数对比
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, Tuple
import warnings; warnings.filterwarnings('ignore')


# 搜索的参数空间
PARAM_GRID = {
    "atr_mult": [0.3, 0.5, 0.7, 1.0],
    "min_quality_score": [30, 40, 50, 60],
    "sl_buffer_atr": [0.2, 0.3, 0.5, 0.7],
}

# 固定参数（baseline）
FIXED_PARAMS = {"atr_mult": 0.5, "min_quality_score": 40, "sl_buffer_atr": 0.3}


def extract_market_state(df_ohlcv: pd.DataFrame, window_end: pd.Timestamp) -> dict:
    """
    提取窗口结束时的市场状态特征
    
    Returns dict with features like volatility regime, trend strength, etc.
    """
    end_idx = df_ohlcv.index.get_indexer([window_end], method='ffill')[0]
    if end_idx < 50:
        return {}
    
    start_idx = max(0, end_idx - 100)  # 最近100 bar
    segment = df_ohlcv.iloc[start_idx:end_idx + 1]
    close = segment["close"].values
    high = segment["high"].values
    low = segment["low"].values
    volume = segment["volume"].values
    
    # 波动率水平
    atr_arr = np.zeros(len(high))
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    tr[0] = high[0] - low[0]
    alpha = 2 / 15
    atr_arr[0] = tr[0]
    for i in range(1, len(tr)):
        atr_arr[i] = alpha * tr[i] + (1 - alpha) * atr_arr[i - 1]
    
    cur_atr_pct = atr_arr[-1] / close[-1] if close[-1] > 0 else 0
    atr_pct_arr = atr_arr / close
    atr_pct_20 = np.percentile(atr_pct_arr[-20:], 80) if len(atr_pct_arr) >= 20 else cur_atr_pct
    atr_rank = (atr_pct_arr[-1] - atr_pct_arr[-50:].min()) / max(atr_pct_arr[-50:].max() - atr_pct_arr[-50:].min(), 1e-10) if len(atr_pct_arr) >= 50 else 0.5
    
    # 趋势强度
    ema20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
    ema50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
    trend_dir = 1 if ema20 > ema50 else -1
    price_vs_ema20 = (close[-1] - ema20) / ema20 if ema20 > 0 else 0
    
    # 趋势斜率
    if len(close) >= 20:
        x = np.arange(20)
        slope, _ = np.polyfit(x, close[-20:], 1)
        trend_strength = slope / close[-1] if close[-1] > 0 else 0
    else:
        trend_strength = 0
    
    # 成交量
    vol_20 = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
    vol_rank = volume[-1] / vol_20 if vol_20 > 0 else 1
    
    # 价格波动率（近期高低差/价格）
    recent_hl_pct = (high[-20:].max() - low[-20:].min()) / close[-1] if len(close) >= 20 and close[-1] > 0 else 0
    
    return {
        "cur_atr_pct": round(cur_atr_pct, 6),
        "atr_percentile_80": round(atr_pct_20, 6),
        "atr_rank_50": round(atr_rank, 4),
        "trend_direction": trend_dir,
        "price_vs_ema20": round(price_vs_ema20, 6),
        "trend_strength": round(trend_strength, 6),
        "volume_rank": round(vol_rank, 4),
        "recent_hl_pct": round(recent_hl_pct, 6),
    }


def rolling_window_optimize(
    df: pd.DataFrame,
    sym: str,
    tf: str,
    window_months: int = 2,
) -> pd.DataFrame:
    """
    滚动窗口参数优化

    每 window_months 个月，网格搜索最优参数组合。
    返回: DataFrame(每行=一个窗口的最优参数 + 市场状态 + 性能)
    """
    from backtest.walk_forward import WalkForwardEngine
    
    df = df.sort_index()
    start = df.index.min()
    end = df.index.max()
    
    results = []
    current = start
    
    while current < end:
        window_end = min(current + pd.DateOffset(months=window_months), end)
        
        # 窗口数据（加前 2 个月预热）
        pre_start = max(df.index.min(), current - pd.DateOffset(months=2))
        window_df = df[(df.index >= pre_start) & (df.index <= window_end)]
        
        if len(window_df) < 200:
            current = window_end
            continue
        
        # 市场状态特征
        state = extract_market_state(df, window_end)
        if not state:
            current = window_end
            continue
        
        # 参数网格搜索
        best_avg_r = -float('inf')
        best_params = None
        best_stats = None
        
        for atr_m, min_q, sl_b in product(
            PARAM_GRID["atr_mult"],
            PARAM_GRID["min_quality_score"],
            PARAM_GRID["sl_buffer_atr"],
        ):
            engine = WalkForwardEngine(
                atr_mult=atr_m,
                min_quality_score=min_q,
                sl_buffer_atr=sl_b,
            )
            result = engine.run(window_df, sym, tf)
            
            if len(result) < 5:
                continue
            
            # 确保只看窗口期内的交易
            result = result[result["entry_time"] >= str(current)]
            
            if len(result) < 3:
                continue
            
            # 评分 = avgR × win_rate（平衡收益和稳定性）
            wr = result["label"].mean()
            avg_r = result["final_r"].mean()
            score = avg_r * wr
            
            if score > best_avg_r:
                best_avg_r = score
                best_params = {"atr_mult": atr_m, "min_quality_score": min_q, "sl_buffer_atr": sl_b}
                best_stats = {
                    "trades": len(result),
                    "wr": round(wr * 100, 1),
                    "avg_r": round(avg_r, 3),
                    "med_r": round(result["final_r"].median(), 3),
                    "total_r": round(result["final_r"].sum(), 1),
                }
        
        if best_params:
            results.append({
                **state,
                **best_params,
                **best_stats,
                "window_start": str(current.date()),
                "window_end": str(window_end.date()),
                "symbol": sym,
                "timeframe": tf,
            })
        
        current = window_end
    
    return pd.DataFrame(results)


def build_training_data(data_dir: str, output_path: str):
    """对所有 TF（不含 4h）× 2 币种做滚动窗口优化"""
    from backtest.walk_forward import WalkForwardEngine
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['15m', '30m', '1h']
    
    all_windows = []
    
    for sym in symbols:
        for tf in timeframes:
            path = f"{data_dir}/binance_{sym}_{tf}.parquet"
            df = pd.read_parquet(path)
            df = df[(df.index >= '2024-01-01') & (df.index < '2026-05-01')]
            
            print(f"扫描 {sym} {tf}...", flush=True)
            windows = rolling_window_optimize(df, sym, tf, window_months=3 if tf == '15m' else 4)
            all_windows.append(windows)
            print(f"  → {len(windows)} 窗口", flush=True)
    
    if all_windows:
        data = pd.concat(all_windows, ignore_index=True)
        data.to_csv(output_path, index=False)
        print(f"\n✅ 保存 {len(data)} 窗口 → {output_path}", flush=True)
    return pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()


if __name__ == "__main__":
    build_training_data(
        "/mnt/c/Users/12645/Sisie-Quantive/data",
        "outputs/phase_b_training.csv",
    )
