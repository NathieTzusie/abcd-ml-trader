"""
Phase B — Step 3: LightGBM 自适应模型训练

输入: outputs/phase_b_windows.csv
特征: 市场状态 (cur_atr_pct, atr_percentile_80, trend_direction, etc.)
标签: optimal_atr_mult, optimal_min_quality, optimal_sl_buffer (多输出回归)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT_DIR = "outputs"

# 特征列
FEATURE_COLS = [
    "cur_atr_pct", "atr_percentile_80", "atr_rank_50", "atr_pct_mean_20",
    "trend_direction", "price_vs_ema20", "trend_strength",
    "volume_rank", "recent_hl_pct",
]

# 标签列
TARGET_COLS = ["atr_mult", "min_quality_score", "sl_buffer_atr"]


def load_window_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"加载 {len(df)} 窗口, {df['symbol'].nunique()} 标的", flush=True)
    print(f"时间: {df['window_start'].min()} ~ {df['window_end'].max()}", flush=True)
    return df


def train_model(df: pd.DataFrame):
    """训练 LightGBM MultiOutputRegressor"""

    # 过滤缺失特征的行
    valid = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    print(f"有效样本: {len(valid)} / {len(df)}", flush=True)

    X = valid[FEATURE_COLS].values
    y = valid[TARGET_COLS].values

    # Time-based split (前80%时间训练, 后20%测试)
    valid = valid.sort_values("window_end")
    split_idx = int(len(valid) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n训练集: {len(X_train)} | 测试集: {len(X_test)}", flush=True)
    print(f"训练时间: {valid['window_end'].iloc[0]} ~ {valid['window_end'].iloc[split_idx-1]}", flush=True)
    print(f"测试时间: {valid['window_end'].iloc[split_idx]} ~ {valid['window_end'].iloc[-1]}", flush=True)

    # 标准化
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # LightGBM 基础参数
    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 16,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }

    # MultiOutputRegressor 包装
    base_lgb = lgb.LGBMRegressor(**lgb_params)
    model = MultiOutputRegressor(base_lgb)

    print("\n训练 LightGBM...", flush=True)
    model.fit(X_train_s, y_train)

    # 预测
    y_pred = model.predict(X_test_s)

    # 评估
    print(f"\n{'='*60}", flush=True)
    print("📊 测试集评估", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for i, col in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])

        # 分类准确率（四舍五入到最接近的离散值）
        if col == "atr_mult":
            true_disc = np.round(y_test[:, i] * 10) / 10  # 0.3, 0.5, 0.7, 1.0
            pred_disc = np.round(y_pred[:, i] * 10) / 10
        elif col == "min_quality_score":
            true_disc = np.round(y_test[:, i] / 10) * 10  # 30, 40, 50, 60
            pred_disc = np.round(y_pred[:, i] / 10) * 10
        else:  # sl_buffer_atr
            true_disc = np.round(y_test[:, i] * 10) / 10  # 0.2, 0.3, 0.5, 0.7
            pred_disc = np.round(y_pred[:, i] * 10) / 10

        acc = (true_disc == pred_disc).mean()

        print(f"  {col}:", flush=True)
        print(f"    MAE = {mae:.4f}, R² = {r2:.4f}", flush=True)
        print(f"    离散准确率 = {acc*100:.1f}%", flush=True)

        results[col] = {
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "discrete_accuracy": round(acc, 4),
        }

    # Feature importance (从每个输出取平均)
    print(f"\n📊 特征重要性 (平均)", flush=True)
    print(f"{'='*60}", flush=True)
    importances = []
    for est_idx, estimator in enumerate(model.estimators_):
        fi = estimator.feature_importances_
        importances.append(fi)
        print(f"  输出 {TARGET_COLS[est_idx]}:")
        for name, imp in sorted(zip(FEATURE_COLS, fi), key=lambda x: -x[1]):
            print(f"    {name:25s} = {imp:.2f}")

    # 平均特征重要性
    avg_imp = np.mean(importances, axis=0)
    print(f"\n  平均:", flush=True)
    for name, imp in sorted(zip(FEATURE_COLS, avg_imp), key=lambda x: -x[1]):
        print(f"    {name:25s} = {imp:.2f}", flush=True)

    # 保存模型
    import joblib
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    model_path = f"{OUTPUT_DIR}/models/lgb_adaptive.pkl"
    joblib.dump({"model": model, "scaler_X": scaler_X, "features": FEATURE_COLS,
                 "targets": TARGET_COLS, "results": results}, model_path)
    print(f"\n✅ 模型保存: {model_path}", flush=True)

    # 预测保存
    df_test = valid.iloc[split_idx:].copy()
    df_test["pred_atr_mult"] = y_pred[:, 0]
    df_test["pred_min_quality"] = y_pred[:, 1]
    df_test["pred_sl_buffer"] = y_pred[:, 2]
    pred_path = f"{OUTPUT_DIR}/adaptive_predictions.csv"
    df_test.to_csv(pred_path, index=False)
    print(f"✅ 预测保存: {pred_path}", flush=True)

    return model, scaler_X


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    window_path = f"{OUTPUT_DIR}/phase_b_windows.csv"
    test_path = f"{OUTPUT_DIR}/phase_b_windows_test.csv"

    # 优先用全量数据
    if os.path.exists(window_path):
        df = load_window_data(window_path)
    elif os.path.exists(test_path):
        print("⚠️ 全量数据不存在, 使用验证集", flush=True)
        df = load_window_data(test_path)
    else:
        print("❌ 无窗口数据", flush=True)
        return

    train_model(df)
    print(f"\n✅ Step 3 完成: LightGBM 自适应模型", flush=True)


if __name__ == "__main__":
    main()
