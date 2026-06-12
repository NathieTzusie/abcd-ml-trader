# Strategy Main v1.5 + ML Stage2 Branches

存档日期：2026-06-11

## 版本定位

本存档将 **静态 Candidate v1.5** 确认为当前主版本。

同时保留两个派生增强版本：

- ML Stage2 >= 0.25：偏向保留更多 TP2、提高 Total R
- ML Stage2 >= 0.30：偏向更接近“过滤约一半 SL，同时保留 75%+ TP2”的约束

## 主版本：Static Candidate v1.5

### 策略规则

- 品种：BTCUSDT / BTCUSDT.P
- 数据区间：2024-01-01 到 2026-06-10
- Setup 时间级别：15m / 30m / 1h，暂不使用 4h
- D 点确认：进入 AB=CD D 投射区后反向走出 1.5 ATR
- 入场：D 到 E 回踩 0.382 fib
- 风险：固定 1.25 ATR
- 成本：入场 maker 2 bps，TP maker 2 bps，SL/BE taker 4 bps
- TP1：到达 TP1 后 50% 减仓，剩余仓位移动到 BE
- Stage1 ML：TP2 Score >= 0.25
- ATR 过滤：0.185% <= atr_percent <= 0.874%
- MSS 过滤：5m close MSS，按 setup 时间级别分级等待

### MSS 分级窗口

| Setup 时间级别 | 5m MSS 最大等待 |
|---|---:|
| 15m | 72 根 |
| 30m | 120 根 |
| 1h | 160 根 |

### 主版本表现

| 指标 | Candidate v1 | Static v1.5 |
|---|---:|---:|
| 样本数 | 3321 | 2073 |
| SL 数 | 2138 | 1114 |
| TP2 数 | 1140 | 923 |
| SL Rate | 64.4% | 53.7% |
| TP2 Rate | 34.3% | 44.5% |
| SL 减少 | - | 47.9% |
| TP2 保留 | - | 81.0% |
| Avg R | 0.123 | 0.477 |
| Total R | 408.0 | 988.9 |

## 派生增强版本：ML Stage2

Stage2 使用 Candidate v1 样本作为输入，再用 MSS 特征做 walk-forward 二阶段过滤。

### OOS 测试表现

| 版本 | 样本 | SL | TP2 | SL减少 | TP2保留 | Avg R | Total R |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 测试集基准 | 1721 | 1066 | 628 | - | - | 0.216 | 372.5 |
| Static v1.5 | 1041 | 533 | 487 | 50.0% | 77.5% | 0.565 | 588.3 |
| ML Stage2 >= 0.25 | 1185 | 615 | 548 | 42.3% | 87.3% | 0.545 | 645.7 |
| ML Stage2 >= 0.30 | 1065 | 547 | 497 | 48.7% | 79.1% | 0.543 | 578.1 |

## 使用建议

- 主版本：Static Candidate v1.5
- 增强观察版本 1：ML Stage2 >= 0.25
- 增强观察版本 2：ML Stage2 >= 0.30

## 已知风险

- ML Stage2 的 OOS 测试窗口目前从 2025Q3 开始，样本窗口比全量 v1.5 更短。
- Static v1.5 更可解释，适合作为主版本。
- ML Stage2 适合作为增强过滤器继续观察，不应立即替代主版本。

## 存档内容

- `reports/candidate_v1_with_mss_features_160.csv`
- `reports/candidate_v1_5_stability.csv`
- `reports/mss_variable_window_15m72_30m120_1h160.csv`
- `reports/mss_stage2_ml/`
- `scripts/`
