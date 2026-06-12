"""
ATR-自适应 ZigZag 摆动点检测器 v2

核心逻辑：
  - 逐 bar 跟踪价格方向，当回撤超过 ATR 阈值时确认转折
  - 方向严格交替：高→低→高→低
"""
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """一个摆动点"""
    bar_index: int
    timestamp: pd.Timestamp
    price: float
    direction: str  # 'high' or 'low'


class ATRZigZag:
    """
    ATR 自适应 ZigZag 检测器

    Parameters
    ----------
    atr_period : int
        ATR 计算周期
    atr_mult : float
        回撤需超过 atr_mult × ATR 才能确认新摆动点
    """

    def __init__(self, atr_period: int = 14, atr_mult: float = 0.5):
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """EMA 平滑 ATR"""
        n = len(high)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1)),
            ),
        )
        tr[0] = high[0] - low[0]

        atr = np.zeros(n)
        atr[0] = tr[0]
        alpha = 2 / (self.atr_period + 1)
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        return atr

    def find_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        逐 bar 遍历检测摆动点

        算法：维护「当前极值」和「方向」，价格反向突破阈值时记录新摆点。
        """
        n = len(df)
        if n < self.atr_period + 5:
            return []

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        atr = self.compute_atr(high, low, close)

        swings: List[SwingPoint] = []
        start = self.atr_period + 2

        # 初始化：找第一个显著的摆点
        # 0 = 未确定，1 = 最近是高点（下一步找低点），-1 = 最近是低点（下一步找高点）
        direction = 0
        extreme_price = 0.0
        extreme_idx = 0

        for i in range(start, n):
            threshold = self.atr_mult * atr[i]

            if direction == 0:
                # 找第一个有效的高低点
                # 用简单的 local max/min
                lookback = 5
                if i < start + lookback:
                    continue

                is_high = high[i - lookback] == np.max(high[i - lookback * 2 : i])
                is_low = low[i - lookback] == np.min(low[i - lookback * 2 : i])

                if is_high:
                    swings.append(SwingPoint(
                        bar_index=i - lookback,
                        timestamp=df.index[i - lookback],
                        price=high[i - lookback],
                        direction="high",
                    ))
                    direction = -1  # 下一步找低点
                    extreme_price = low[i - lookback]
                    extreme_idx = i - lookback
                elif is_low:
                    swings.append(SwingPoint(
                        bar_index=i - lookback,
                        timestamp=df.index[i - lookback],
                        price=low[i - lookback],
                        direction="low",
                    ))
                    direction = 1  # 下一步找高点
                    extreme_price = high[i - lookback]
                    extreme_idx = i - lookback

            elif direction == 1:
                # 最近是低点，现在在找高点
                if high[i] > extreme_price:
                    extreme_price = high[i]
                    extreme_idx = i

                # 检查是否从高点回落超过阈值
                retreat = extreme_price - low[i]
                if retreat >= threshold and i > extreme_idx + 1:
                    swings.append(SwingPoint(
                        bar_index=extreme_idx,
                        timestamp=df.index[extreme_idx],
                        price=extreme_price,
                        direction="high",
                    ))
                    direction = -1  # 下一步找低点
                    extreme_price = low[i]
                    extreme_idx = i

            elif direction == -1:
                # 最近是高点，现在在找低点
                if low[i] < extreme_price:
                    extreme_price = low[i]
                    extreme_idx = i

                # 检查是否从低点反弹超过阈值
                rally = high[i] - extreme_price
                if rally >= threshold and i > extreme_idx + 1:
                    swings.append(SwingPoint(
                        bar_index=extreme_idx,
                        timestamp=df.index[extreme_idx],
                        price=extreme_price,
                        direction="low",
                    ))
                    direction = 1  # 下一步找高点
                    extreme_price = high[i]
                    extreme_idx = i

        return swings

    def to_dataframe(self, swings: List[SwingPoint]) -> pd.DataFrame:
        if not swings:
            return pd.DataFrame(columns=["bar_index", "timestamp", "price", "direction"])
        return pd.DataFrame([
            {"bar_index": s.bar_index, "timestamp": s.timestamp,
             "price": s.price, "direction": s.direction}
            for s in swings
        ])
