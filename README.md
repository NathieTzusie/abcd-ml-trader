# abcd-ml-trader

AB=CD Harmonic Pattern × ML Optimization — 独立交易策略项目

## 定位

规则驱动的 AB=CD 谐波形态识别 + XGBoost/LightGBM 作为信号过滤与参数自适应层。

- 不依赖 Harmonic V2 项目
- 不并入 Sisie-Quantive（潜力验证后再考虑）

## 数据结构

```
abcd-ml-trader/
├── abcd_detector/    # 形态检测（ZigZag + AB=CD 识别）
├── ml/               # ML 模型（Phase A/B/D）
├── backtest/         # 回测引擎
├── data/             # 数据 symlink → Sisie-Quantive/data/
├── config/           # YAML 配置
├── experiments/      # Jupyter notebooks
├── reports/          # 回测报告
└── outputs/          # 模型文件
```

## 阶段

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 基础设施 + 数据准备 | 进行中 |
| A | ML 信号后过滤（XGBoost） | 待开始 |
| B | ML 参数自适应（LightGBM） | 待开始 |
| D | Enable Score + 动态风控 | 待开始 |
| Live | 实盘部署 | 待开始 |

详见 `DESIGN.md`（..OpenClawWorkspace/designs/abcd-ml-trader-design.md）
