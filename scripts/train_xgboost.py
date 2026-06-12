"""
Phase A: XGBoost Baseline 训练 + SHAP 分析

标签：final_r >= 1.0 (至少 1R 利润)
BTC 样本权重：3x
Walk-forward 5-fold 交叉验证
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === 配置 ===
DATA_PATH = "outputs/training_dataset.csv"
BTC_WEIGHT = 3.0
LABEL_THRESHOLD = 1.0  # final_r >= 1.0
N_FOLDS = 5

# === 加载数据 ===
df = pd.read_csv(DATA_PATH)
print(f"加载: {len(df)} 行", flush=True)

# 标签
df['label'] = (df['final_r'] >= LABEL_THRESHOLD).astype(int)

# 特征列（排除非特征列）
exclude_cols = ['shape_id', 'symbol', 'timeframe', 'direction', 'entry_bar',
                'entry_price', 'entry_time', 'c_price', 'a_price', 'sl_price',
                'd_projected', 'exit_reason', 'final_r', 'label', 'total_bars',
                'tp1_hit_bar', 'tp1_hit_price', 'final_exit_bar', 'final_exit_price']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"特征: {len(feature_cols)} 列", flush=True)
print(f"标签分布: 1={df['label'].sum()} ({df['label'].mean()*100:.1f}%), 0={(~df['label'].astype(bool)).sum()}", flush=True)

# 样本权重（BTC=3, ETH=1）
df['sample_weight'] = df['symbol'].map({'BTCUSDT': BTC_WEIGHT, 'ETHUSDT': 1.0})

# === 按时间排序 ===
df = df.sort_values('entry_time').reset_index(drop=True)
X = df[feature_cols].values
y = df['label'].values
weights = df['sample_weight'].values

# === Walk-forward 交叉验证 ===
tscv = TimeSeriesSplit(n_splits=N_FOLDS)
fold_results = []
shap_values_all = []
y_test_all = []
y_pred_all = []

print(f"\n{'='*60}")
print(f"Walk-Forward (N={N_FOLDS}) XGBoost 训练")
print(f"{'='*60}")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train = weights[train_idx]

    # 计算 scale_pos_weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight * 1.5,  # 额外正样本权重
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred)
    precision = (y_pred_class[y_pred_class == 1] == y_test[y_pred_class == 1]).mean() if y_pred_class.sum() > 0 else 0
    recall = (y_test[y_pred_class == 1] == 1).mean() if (y_test == 1).sum() > 0 else 0

    fold_results.append({
        'fold': fold, 'train_n': len(train_idx), 'test_n': len(test_idx),
        'pos_rate': y_test.mean(), 'auc': auc,
        'precision': precision, 'recall': recall,
    })

    print(f"Fold {fold+1}: train={len(train_idx)} test={len(test_idx)} | "
          f"pos_rate={y_test.mean():.2f} | AUC={auc:.3f} | prec={precision:.3f} | rec={recall:.3f}", flush=True)

    shap_values_all.append(y_pred)
    y_test_all.extend(y_test)
    y_pred_all.extend(y_pred)

    # SHAP（仅最后一 fold）
    if fold == N_FOLDS - 1:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test[:min(2000, len(X_test))])
        shap.summary_plot(shap_vals, X_test[:min(2000, len(X_test))],
                          feature_names=feature_cols, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 SHAP 图已保存: outputs/shap_summary.png", flush=True)

# === 汇总 ===
print(f"\n{'='*60}")
print(f"汇总")
print(f"{'='*60}")
results_df = pd.DataFrame(fold_results)
print(f"平均 AUC: {results_df['auc'].mean():.3f} ± {results_df['auc'].std():.3f}")

y_test_arr = np.array(y_test_all)
y_pred_arr = np.array(y_pred_all)
total_auc = roc_auc_score(y_test_arr, y_pred_arr)
print(f"Overall AUC: {total_auc:.3f}")

# 特征重要性
importance = model.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False)
print(f"\nTop 15 特征重要性:")
for _, row in feat_imp.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# 保存特征重要性
feat_imp.to_csv("outputs/feature_importance.csv", index=False)
print(f"\n✅ 特征重要性已保存: outputs/feature_importance.csv")
print(f"✅ 训练完成")
