"""
Phase A: XGBoost Walk-Forward 训练 + SHAP 分析

使用 walk-forward 回测数据（无 lookahead bias）
标签：final_r >= 0
"""
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import shap, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

DATA = "outputs/wf_training_dataset.csv"
LABEL_THRESHOLD = 0.0
BTC_WEIGHT = 3.0
N_FOLDS = 5

df = pd.read_csv(DATA)
print(f"加载: {len(df)} 行", flush=True)

df['label'] = (df['final_r'] >= LABEL_THRESHOLD).astype(int)
print(f"标签: 1={df['label'].sum()} ({df['label'].mean()*100:.1f}%)", flush=True)

exclude = ['shape_id','symbol','timeframe','direction','entry_bar','entry_price',
           'entry_time','c_price','a_price','sl_price','d_projected',
           'exit_reason','final_r','label','total_bars','exit_bar','exit_price']
features = [c for c in df.columns if c not in exclude]
print(f"特征: {len(features)}", flush=True)

df['sample_weight'] = df['symbol'].map({'BTCUSDT': BTC_WEIGHT, 'ETHUSDT': 1.0})
df = df.sort_values('entry_time').reset_index(drop=True)

X = df[features].values
y = df['label'].values
w = df['sample_weight'].values

tscv = TimeSeriesSplit(n_splits=N_FOLDS)
folds = []
y_all, p_all = [], []

for fold, (tr, te) in enumerate(tscv.split(X)):
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    wtr = w[tr]

    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    yp = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, yp)
    folds.append({'fold':fold, 'auc':auc, 'n_train':len(tr), 'n_test':len(te), 'pos':yte.mean()})
    print(f"Fold {fold+1}: train={len(tr)} test={len(te)} pos={yte.mean():.2f} AUC={auc:.3f}", flush=True)
    y_all.extend(yte); p_all.extend(yp)

    if fold == N_FOLDS - 1:
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(Xte[:min(2000, len(Xte))])
        shap.summary_plot(sv, Xte[:min(2000, len(Xte))], feature_names=features,
                          show=False, max_display=15)
        plt.tight_layout()
        plt.savefig("outputs/shap_wf_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

print(f"\nAUC avg={np.mean([f['auc'] for f in folds]):.3f} ± {np.std([f['auc'] for f in folds]):.3f}", flush=True)
print(f"Overall AUC={roc_auc_score(y_all, p_all):.3f}", flush=True)

imp = pd.DataFrame({'feature':features, 'importance':m.feature_importances_})
imp.sort_values('importance', ascending=False, inplace=True)
print(f"\nTop 15 特征:", flush=True)
for _, r in imp.head(15).iterrows():
    print(f"  {r['feature']:30s}: {r['importance']:.4f}", flush=True)
imp.to_csv("outputs/feature_importance_wf.csv", index=False)
print(f"\nDone!", flush=True)
