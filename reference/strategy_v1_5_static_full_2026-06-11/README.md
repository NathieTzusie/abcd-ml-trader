# Strategy v1.5 Static Full Version

保存日期：2026-06-11

这是 Static Candidate v1.5 的完整可复现版本。

## 版本定位

Static Candidate v1.5 是当前主版本，不包含 ML Stage2 增强过滤，也不包含进攻型 5m execution 模型。

## 主规则

- 品种：BTCUSDT / BTCUSDT.P
- 数据区间：2024-01-01 到 2026-06-10
- Setup 时间级别：15m / 30m / 1h
- 4h：暂不使用
- D 点确认：价格进入 AB=CD D 投射区后，反向走出 1.5 ATR
- 入场：D 到 E 回踩 0.382 fib
- 风险：固定 1.25 ATR
- 成本：入场 maker 2 bps，TP maker 2 bps，SL/BE taker 4 bps
- TP1：到达 TP1 后 50% 减仓，剩余仓位移动到 BE
- Stage1 ML：TP2 Score >= 0.25
- ATR 过滤：0.185% <= atr_percent <= 0.874%
- MSS 过滤：5m close MSS，按 setup 时间级别分级等待

## MSS 分级窗口

| Setup 时间级别 | 5m MSS 最大等待 |
|---|---:|
| 15m | 72 根 |
| 30m | 120 根 |
| 1h | 160 根 |

## 主要结果

| 指标 | Static v1.5 |
|---|---:|
| 样本数 | 2073 |
| SL | 1114 |
| TP2 | 923 |
| SL Rate | 53.7% |
| TP2 Rate | 44.5% |
| Avg R | 0.477 |
| Total R | 988.9 |

## 目录说明

```text
strategy_v1_5_static_full_2026-06-11/
├── data/raw/
│   └── BTCUSDT 5m/15m/30m/1h/4h 原始数据
├── trading/analysis/
│   ├── abcd_d_confirmation_event_study.py
│   ├── train_d_setup_filter.py
│   ├── abcd_mss_delayed_entry_experiment.py
│   ├── abcd_backtester.py
│   └── download_binance_futures_klines.py
└── reports/wf_fib0382_risk125_partial_rerun/
    ├── abcd_d_confirmation_events.csv
    ├── candidate_v1_with_mss_features_160.csv
    ├── candidate_v1_5_stability.csv
    ├── mss_variable_window_15m72_30m120_1h160.csv
    └── ml_tp2/
```

## 复现步骤

在本目录作为工作目录时运行：

```powershell
python trading\analysis\abcd_d_confirmation_event_study.py --data-dir data\raw --output-dir reports\wf_fib0382_risk125_partial_rerun --symbol BTCUSDT --timeframes 15m,30m,1h,4h --atr-multiple 1.5 --entry-mode fib_retrace --entry-fib 0.382 --entry-max-wait-bars 24 --risk-mode fixed_atr --risk-atr-multiple 1.25 --cost-model maker_taker --maker-bps 2 --taker-bps 4 --tp1-action partial
```

然后训练 Stage1 TP2 模型：

```powershell
python trading\analysis\train_d_setup_filter.py --events reports\wf_fib0382_risk125_partial_rerun\abcd_d_confirmation_events.csv --output-dir reports\wf_fib0382_risk125_partial_rerun\ml_tp2 --target reached_tp2
```

MSS 特征表和 v1.5 过滤统计目前以 CSV 形式保存：

- `reports/wf_fib0382_risk125_partial_rerun/candidate_v1_with_mss_features_160.csv`
- `reports/wf_fib0382_risk125_partial_rerun/mss_variable_window_15m72_30m120_1h160.csv`
- `reports/wf_fib0382_risk125_partial_rerun/candidate_v1_5_stability.csv`

## 回退说明

如果后续实验偏离，可以回到这个目录中的 CSV 和脚本，恢复 Static Candidate v1.5 的完整状态。
