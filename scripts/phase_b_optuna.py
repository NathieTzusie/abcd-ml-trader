"""
Phase B Lite: Optuna 参数优化 + 规则型自适应

不需要完整滚动窗口网格搜索。
策略: 按波动率分三档 → 每档 Optuna 搜索最优参数 → 简单规则切换
"""
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
import optuna
from backtest.walk_forward import WalkForwardEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)


def classify_regime(atr_pct: float) -> str:
    """按 ATR/价格 分档"""
    if atr_pct < 0.008:     return "low_vol"    # < 0.8%
    elif atr_pct < 0.020:   return "mid_vol"    # 0.8-2%
    else:                   return "high_vol"   # > 2%


def optimize_regime(df, sym, tf, n_trials=50):
    """Optuna 搜索给定数据集的最优参数"""
    
    def objective(trial):
        atr_m = trial.suggest_float("atr_mult", 0.3, 1.2, step=0.1)
        min_q = trial.suggest_int("min_quality_score", 30, 60, step=5)
        sl_b = trial.suggest_float("sl_buffer_atr", 0.1, 0.8, step=0.1)
        
        engine = WalkForwardEngine(
            atr_mult=atr_m, min_quality_score=min_q, sl_buffer_atr=sl_b,
        )
        result = engine.run(df, sym, tf)
        
        if len(result) < 10:
            return -999
        
        # 得分 = avgR × win_rate × log(trades)  (鼓励更多交易)
        wr = max(result["label"].mean(), 0.01)
        avg_r = result["final_r"].mean()
        score = avg_r * wr * np.log(max(len(result), 10))
        return score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    data_dir = "/mnt/c/Users/12645/Sisie-Quantive/data"
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['15m', '30m', '1h']
    
    all_results = []
    
    for sym in symbols:
        for tf in timeframes:
            path = f"{data_dir}/binance_{sym}_{tf}.parquet"
            df = pd.read_parquet(path)
            df = df[(df.index >= '2024-01-01') & (df.index < '2026-05-01')]
            
            # 计算全局 ATR 分档阈值
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            atr = WalkForwardEngine.compute_atr(high, low, close, 14)
            atr_pct = atr / close
            
            # 分三档
            for regime_name, mask_fn in [
                ("low_vol", lambda: atr_pct < 0.008),
                ("mid_vol", lambda: (atr_pct >= 0.008) & (atr_pct < 0.020)),
                ("high_vol", lambda: atr_pct >= 0.020),
            ]:
                mask = mask_fn()
                # 取该 regime 的连续片段
                # 简化: 取价格在该 regime 时的前后数据
                regime_indices = np.where(mask)[0]
                if len(regime_indices) < 500:
                    continue
                
                # 取中间一段做优化（节省时间）
                mid = len(regime_indices) // 2
                sample_start = max(0, regime_indices[max(0, mid - 2000)])
                sample_end = min(len(df) - 1, regime_indices[min(len(regime_indices) - 1, mid + 2000)])
                
                sub_df = df.iloc[max(0, sample_start - 500):sample_end + 500]
                if len(sub_df) < 300:
                    continue
                
                print(f"  {sym} {tf} {regime_name}: {len(sub_df)} bars", flush=True)
                best_params, best_score = optimize_regime(sub_df, sym, tf, n_trials=30)
                
                all_results.append({
                    "symbol": sym, "timeframe": tf, "regime": regime_name,
                    **best_params, "score": round(best_score, 2),
                })
                print(f"    → {best_params} score={best_score:.1f}", flush=True)

    # 汇总
    df_r = pd.DataFrame(all_results)
    df_r.to_csv("outputs/phase_b_regime_params.csv", index=False)
    print(f"\n{'='*60}")
    print("Phase B 分档最优参数")
    print(f"{'='*60}")
    for _, row in df_r.iterrows():
        print(f"  {row['symbol']:8s} {row['timeframe']:4s} {row['regime']:10s}: "
              f"atr_mult={row['atr_mult']} minQ={row['min_quality_score']} "
              f"sl_buf={row['sl_buffer_atr']} score={row['score']}", flush=True)


if __name__ == "__main__":
    main()
