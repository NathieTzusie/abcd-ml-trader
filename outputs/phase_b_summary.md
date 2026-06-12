# Phase B 总结 — V4引擎 × 滚动窗口参数自适应

## Step 1: V4 训练数据集
- **引擎**: V4_swing3_vol (反转确认 + 波动率自适应)
- **标的**: 16 个 (ADA, ATOM, AVAX, BNB, BTC, DOGE, DOT, ETC, ETH, FIL, LINK, LTC, NEAR, SOL, UNI, XRP)
- **时间范围**: 2024-01-01 ~ 2026-04-30
- **数据**: 181,953 trades 保存到 `outputs/v4_training_dataset.csv`
- **平均**: WR=63.8%, SL=36.2%, totalR=70,332.8

## Step 2: V4 滚动窗口参数优化

### 方法
- 滚动 3 个月步长窗口，每窗口 2 个月预热
- 64 种参数组合: atr_mult [0.3, 0.5, 0.7, 1.0] × min_quality [30, 40, 50, 60] × sl_buffer [0.2, 0.3, 0.5, 0.7]
- 评分 = avgR × win_rate（平衡收益和稳定性）
- 每窗口提取市场状态特征: cur_atr_pct, atr_percentile_80, trend_direction, volume_rank 等

### 验证结果（BTC + NEAR 实际运行）
| 指标 | BTCUSDT | NEARUSDT |
|------|---------|----------|
| 窗口数 | 10 | 10 |
| 最优 atr_mult 分布 | 0.3(5), 1.0(4), 0.7(1) | 0.3(6), 1.0(2), 0.7(2) |
| 最优 min_quality 分布 | 60(9), 50(1) | 60(5), 30(3), 50(1), 40(1) |
| 最优 sl_buffer | 0.2(10) | 0.2(10) |
| cur_atr_pct 范围 | 0.0014 ~ 0.0042 | 0.0034 ~ 0.0101 |

**关键发现**:
- sl_buffer=0.2 在所有窗口都是最优（低滑点容忍）
- 高波动期 (atr% 高) 倾向于 atr_mult=1.0（更积极入场）
- 低波动期 (atr% 低) 倾向于 atr_mult=0.3（更保守入场）
- BTC 大部分窗口最优 min_quality=60（仅优质形态），NEAR 在低波动窗口接受 min_quality=30

## Step 3: LightGBM 自适应模型

### 方法
- MultiOutputRegressor(LGBMRegressor) 同时预测三个参数
- Time-based split: 前 80% 训练，后 20% 测试

### 性能
| 目标 | MAE | R² | 离散准确率 |
|------|-----|-----|----------|
| atr_mult | 0.0045 | 0.999 | 100.0% |
| min_quality_score | 8.8410 | -0.367 | 40.6% |
| sl_buffer_atr | 0.0000 | 0.000 | 100.0% |

### 特征重要性（按参数）
- **atr_mult**: atr_pct_mean_20 > recent_hl_pct > volume_rank
- **min_quality_score**: atr_percentile_80 > cur_atr_pct > atr_rank_50
- **sl_buffer_atr**: 全部 0（sl_buffer 几乎不变化，模型无法学习）

### 模型保存
- `outputs/models/lgb_adaptive.pkl`

## Step 4: 自适应 vs 固定参数对比

| Symbol | 固定 WR | 固定 avgR | 固定 totalR | 自适应 WR | 自适应 avgR | 自适应 totalR | totalR Δ |
|--------|---------|-----------|-------------|-----------|-------------|---------------|----------|
| BTC | 65.7% | 0.451 | 5,344.2 | 65.8% | 0.474 | 5,128.5 | -215.7 |
| NEAR | 65.9% | 0.371 | 3,976.5 | 65.9% | 0.372 | 3,792.2 | -184.3 |
| ADA | 63.2% | 0.350 | 4,072.0 | 63.5% | 0.346 | 3,975.0 | -97.0 |

**合计**: totalR 13,392.7 → 12,895.7 (**-3.7%**)

## 结论与下一步

### 当前局限
1. **sl_buffer 不敏感** — 所有窗口最优 sl_buffer 都是 0.2，模型无法学习变异性
2. **min_quality 预测不准** — R² = -0.367，离散准确率仅 40.6%
3. **自适应收益不明显** — WR 持平但 totalR 下降 3.7%，因为参数切换有「冷启动」成本
4. **合成数据偏差** — 13 个标的的窗口数据通过 bootstrap 生成，需要实际跑完

### 改进方向
1. **扩大参数搜索范围** — 加入 0.1/0.15 的 sl_buffer, 20/70 的 min_quality
2. **增加低频特征** — 加入 ema_cross, adx, macd, support/resistance 距离
3. **减少滑动窗口步长** — 1 个月步长 vs 3 个月，更快适应市场切换
4. **参数平滑过渡** — 不硬切换，加权融合新旧参数
5. **实际跑完全部 16 标的** — 用脚本 `scripts/phase_b_optimize_v4.py` 跑全量数据
