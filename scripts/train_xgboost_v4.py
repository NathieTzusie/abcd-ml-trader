"""
Phase A — Step 2: XGBoost 训练

用 V4 数据集训练 XGBoost 分类器。
Time-based split: 前80%训练，后20%测试。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "outputs/v4_training_dataset.csv"
MODEL_PATH = "outputs/xgboost_v4_model.json"
FEATURE_IMP_PATH = "outputs/feature_importance_v4.csv"
SHAP_PATH = "outputs/shap_v4_summary.png"
TRAIN_RATIO = 0.8

# === 特征列 ===
FEATURE_COLS = [
    "ab_distance_pct", "bc_ab_ratio", "quality_score",
    "ab_bars", "bc_bars", "d_zone_bars",
    "d_zone_volume_ratio", "d_zone_momentum", "d_zone_wick_ratio",
    "atr_pct", "atr_percentile_20",
    "ema20_50_direction", "price_vs_ema20",
    "volume_ratio", "volume_trend",
    "hour_of_day", "day_of_week", "vol_mult",
]

# === 加载 ===
print("加载数据...", flush=True)
df = pd.read_csv(DATA_PATH)
df["entry_time"] = pd.to_datetime(df["entry_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
print(f"总计: {len(df):,} trades | 正样本: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)", flush=True)

# === 特征准备 ===
X = df[FEATURE_COLS].values
y = df["label"].values

# === Time-based split ===
split_idx = int(len(df) * TRAIN_RATIO)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n训练集: {len(X_train):,} ({split_idx/len(df)*100:.0f}%)")
print(f"测试集: {len(X_test):,} ({(len(df)-split_idx)/len(df)*100:.0f}%)")

train_pos_rate = y_train.mean()
test_pos_rate = y_test.mean()
print(f"训练集正样本率: {train_pos_rate*100:.1f}%")
print(f"测试集正样本率: {test_pos_rate*100:.1f}%")

# 样本权重：BTC 3x
df["sample_weight"] = df["symbol"].map({"BTCUSDT": 3.0, "ETHUSDT": 1.5})
df["sample_weight"] = df["sample_weight"].fillna(1.0)
w_train = df["sample_weight"].values[:split_idx]

# scale_pos_weight
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# === XGBoost 训练 ===
print(f"\n{'='*60}")
print("XGBoost 训练中...")
print(f"{'='*60}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_lambda=1.0,
    reg_alpha=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=1,
    early_stopping_rounds=30,
    eval_metric=["logloss", "auc", "error"],
)

model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50,
)

# === 评估 ===
print(f"\n{'='*60}")
print("评估")
print(f"{'='*60}")

y_train_pred = model.predict_proba(X_train)[:, 1]
y_test_pred = model.predict_proba(X_test)[:, 1]
y_train_class = (y_train_pred >= 0.5).astype(int)
y_test_class = (y_test_pred >= 0.5).astype(int)

train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)
train_acc = accuracy_score(y_train, y_train_class)
test_acc = accuracy_score(y_test, y_test_class)
test_prec = precision_score(y_test, y_test_class)
test_rec = recall_score(y_test, y_test_class)
test_f1 = f1_score(y_test, y_test_class)

print(f"\n📊 AUC:")
print(f"  训练集: {train_auc:.4f}")
print(f"  测试集: {test_auc:.4f}")

print(f"\n📊 Accuracy:")
print(f"  训练集: {train_acc:.4f}")
print(f"  测试集: {test_acc:.4f}")

print(f"\n📊 测试集 (阈值=0.5):")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1:        {test_f1:.4f}")

print(f"\n📊 Classification Report (测试集):")
print(classification_report(y_test, y_test_class, target_names=["loss", "win"]))

cm = confusion_matrix(y_test, y_test_class)
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]} FP={cm[0,1]}")
print(f"  FN={cm[1,0]} TP={cm[1,1]}")

# === 特征重要性 ===
importance = model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importance,
    "gain": model.features_names_in_ if hasattr(model, "features_names_in_") else None,
})
feat_imp = feat_imp.sort_values("importance", ascending=False)

print(f"\n{'='*60}")
print("特征重要性")
print(f"{'='*60}")
for _, row in feat_imp.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:28s}: {row['importance']:.4f} {bar}")

feat_imp.to_csv(FEATURE_IMP_PATH, index=False)
print(f"\n✅ 特征重要性已保存: {FEATURE_IMP_PATH}")

# === 保存模型 ===
model.save_model(MODEL_PATH)
print(f"✅ 模型已保存: {MODEL_PATH}")

# === SHAP (如果可用) ===
try:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n计算 SHAP 值...", flush=True)
    # 用 subsets
    n_shap = min(2000, len(X_test))
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[:n_shap])

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_test[:n_shap],
                      feature_names=FEATURE_COLS, show=False, max_display=18)
    plt.tight_layout()
    plt.savefig(SHAP_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP 图已保存: {SHAP_PATH}")
except ImportError:
    print("⚠️  shap not available, skipping SHAP plot")
except Exception as e:
    print(f"⚠️  SHAP error: {e}")

print(f"\n{'='*60}")
print("✅ Step 2 完成")
print(f"{'='*60}")
