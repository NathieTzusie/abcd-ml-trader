"""
ABCD 形态检测器

基于 ATR-ZigZag 的摆动点，识别 AB=CD 反转谐波形态。
D 点由 C + AB 投影计算，不依赖额外 swing 确认。
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from .zigzag import ATRZigZag, SwingPoint


# 标准 Fibonacci 回撤水平
STANDARD_FIB_LEVELS = [0.382, 0.5, 0.618, 0.786, 0.886]


@dataclass
class ABCDShape:
    """一个检测到的 ABCD 形态"""

    # 基本信息
    shape_id: str
    symbol: str
    timeframe: str
    direction: str  # 'bullish' or 'bearish'

    # 摆动点索引 (在 df 中的 bar_index)
    a_idx: int
    b_idx: int
    c_idx: int

    # 价格
    a_price: float
    b_price: float
    c_price: float
    d_projected: float

    # Fibonacci 比率
    bc_ab_ratio: float
    cd_ab_ratio: float  # 预测值（投影用）

    # 形态质量 (0-100)
    quality_score: float

    # 维度分数
    fib_score: float
    time_symmetry_score: float
    price_symmetry_score: float
    leg_maturity_score: float

    # 元数据
    ab_bars: int
    bc_bars: int
    cd_bars_estimated: int  # 预测 CD 的 bar 数

    def to_dict(self) -> dict:
        return {
            "shape_id": self.shape_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "a_idx": self.a_idx,
            "b_idx": self.b_idx,
            "c_idx": self.c_idx,
            "a_price": self.a_price,
            "b_price": self.b_price,
            "c_price": self.c_price,
            "d_projected": self.d_projected,
            "bc_ab_ratio": self.bc_ab_ratio,
            "cd_ab_ratio": self.cd_ab_ratio,
            "quality_score": self.quality_score,
            "fib_score": self.fib_score,
            "time_symmetry_score": self.time_symmetry_score,
            "price_symmetry_score": self.price_symmetry_score,
            "leg_maturity_score": self.leg_maturity_score,
            "ab_bars": self.ab_bars,
            "bc_bars": self.bc_bars,
            "cd_bars_estimated": self.cd_bars_estimated,
        }


class ABCDDetector:
    """
    AB=CD 谐波形态检测器

    检测流程：
      1. ATR-ZigZag 识别摆动点
      2. 从最近 3 个摆动点提取 A, B, C
      3. 验证 BC/AB Fib 回撤比例
      4. 投影 D 点 = C + direction × |AB|
      5. 质量评分 (0-100)

    Parameters
    ----------
    bc_ab_min : float, default 0.382
        BC/AB 最小回撤比例
    bc_ab_max : float, default 0.886
        BC/AB 最大回撤比例
    cd_ab_ratio : float, default 1.0
        投影 D 所用的 CD/AB 比率（经典等长 = 1.0）
    d_zone_tolerance : float, default 0.005
        D 投影 ±0.5% 为 D_zone 进入区
    atr_period : int, default 14
        ATR 计算周期
    atr_mult : float, default 0.5
        ZigZag ATR 倍数
    min_quality_score : float, default 40
        最低质量分阈值
    min_atr_mult : float, default 0.5
        AB 最小幅度 = min_atr_mult × ATR（过滤微摆动噪声）
    """

    def __init__(
        self,
        bc_ab_min: float = 0.382,
        bc_ab_max: float = 0.886,
        cd_ab_ratio: float = 1.0,
        d_zone_tolerance: float = 0.005,
        atr_period: int = 14,
        atr_mult: float = 0.5,
        min_quality_score: float = 40,
        min_atr_mult: float = 0.5,
        fib_weights: Optional[Dict[str, float]] = None,
    ):
        self.bc_ab_min = bc_ab_min
        self.bc_ab_max = bc_ab_max
        self.cd_ab_ratio = cd_ab_ratio
        self.d_zone_tolerance = d_zone_tolerance
        self.min_quality_score = min_quality_score
        self.min_atr_mult = min_atr_mult

        self.zigzag = ATRZigZag(atr_period=atr_period, atr_mult=atr_mult)

        self.fib_weights = fib_weights or {
            "fib_precision": 45.0,
            "time_symmetry": 25.0,
            "price_symmetry": 20.0,
            "leg_maturity": 10.0,
        }

        self._shape_counter = 0

    def detect(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> List[ABCDShape]:
        """
        检测 ABCD 形态

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 数据，必须有 high/low/close 列，index 为 datetime
        symbol : str
            交易对（如 'BTCUSDT'）
        timeframe : str
            时间框架（如 '30m'）

        Returns
        -------
        List[ABCDShape]
        """
        swings = self.zigzag.find_swings(df)
        if len(swings) < 3:
            return []

        # 预计算 ATR（供 AB 最小幅度过滤用）
        atr = self.zigzag.compute_atr(df["high"].values, df["low"].values, df["close"].values)

        shapes = []
        for i in range(len(swings) - 2):
            c_swing = swings[i]
            b_swing = swings[i + 1]
            a_swing = swings[i + 2]

            if not self._valid_alternation(a_swing, b_swing, c_swing):
                continue

            # AB 最小幅度过滤 (min_atr_mult × ATR)
            ab_len = abs(a_swing.price - b_swing.price)
            b_atr_val = atr[min(b_swing.bar_index, len(atr) - 1)]
            if ab_len < self.min_atr_mult * b_atr_val:
                continue

            shape = self._try_form_shape(
                a_swing, b_swing, c_swing, df, symbol, timeframe
            )
            if shape is not None:
                shapes.append(shape)

        return shapes

    def _valid_alternation(
        self, a: SwingPoint, b: SwingPoint, c: SwingPoint
    ) -> bool:
        """检查 A→B→C 方向是否交替"""
        # 看涨：A=high, B=low, C=high (且 C < A)
        # 看跌：A=low, B=high, C=low (且 C > A)
        if a.direction == "high" and b.direction == "low" and c.direction == "high":
            return c.price < a.price  # 看涨
        if a.direction == "low" and b.direction == "high" and c.direction == "low":
            return c.price > a.price  # 看跌
        return False

    def _try_form_shape(
        self,
        a: SwingPoint,
        b: SwingPoint,
        c: SwingPoint,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Optional[ABCDShape]:
        """尝试从 A,B,C 三点构建 ABCD 形态"""

        ab_length = abs(a.price - b.price)
        bc_length = abs(b.price - c.price)

        if ab_length == 0:
            return None

        # 1. BC/AB 回撤验证
        bc_ab_ratio = bc_length / ab_length
        if not (self.bc_ab_min <= bc_ab_ratio <= self.bc_ab_max):
            return None

        # 2. 确定方向
        if a.direction == "high":
            direction = "bullish"
            direction_sign = -1  # D 在 C 之下
        else:
            direction = "bearish"
            direction_sign = 1  # D 在 C 之上

        # 3. D 点投影
        d_projected = c.price + direction_sign * ab_length * self.cd_ab_ratio

        # 4. 质量评分
        ab_bars = abs(b.bar_index - a.bar_index)
        bc_bars = abs(c.bar_index - b.bar_index)
        cd_bars_estimated = int(ab_bars * self.cd_ab_ratio)  # 假设时间对称

        scores = self._compute_quality(bc_ab_ratio, ab_bars, bc_bars)
        quality = sum(scores.values())

        if quality < self.min_quality_score:
            return None

        # 5. 创建形态对象
        self._shape_counter += 1
        shape_id = f"{symbol}_{timeframe}_{direction}_{self._shape_counter}"

        return ABCDShape(
            shape_id=shape_id,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            a_idx=a.bar_index,
            b_idx=b.bar_index,
            c_idx=c.bar_index,
            a_price=a.price,
            b_price=b.price,
            c_price=c.price,
            d_projected=d_projected,
            bc_ab_ratio=bc_ab_ratio,
            cd_ab_ratio=self.cd_ab_ratio,
            quality_score=quality,
            fib_score=scores["fib_precision"],
            time_symmetry_score=scores["time_symmetry"],
            price_symmetry_score=scores["price_symmetry"],
            leg_maturity_score=scores["leg_maturity"],
            ab_bars=ab_bars,
            bc_bars=bc_bars,
            cd_bars_estimated=cd_bars_estimated,
        )

    def _compute_quality(
        self, bc_ab_ratio: float, ab_bars: int, bc_bars: int
    ) -> Dict[str, float]:
        """4 因子质量评分"""

        # 1. Fibonacci 精度 (45 pts)
        distances = [abs(bc_ab_ratio - lvl) for lvl in STANDARD_FIB_LEVELS]
        min_dist = min(distances)
        if min_dist < 0.02:
            fib_score = self.fib_weights["fib_precision"]
        elif min_dist < 0.05:
            fib_score = self.fib_weights["fib_precision"] * 0.7
        elif min_dist < 0.10:
            fib_score = self.fib_weights["fib_precision"] * 0.4
        elif min_dist < 0.18:
            fib_score = self.fib_weights["fib_precision"] * 0.15
        else:
            fib_score = 0

        # 2. 时间对称 (25 pts)
        if ab_bars > 0 and bc_bars > 0:
            time_ratio = bc_bars / ab_bars
            time_sym = 1 - min(abs(time_ratio - 1), 1)
            time_sym_score = self.fib_weights["time_symmetry"] * time_sym
        else:
            time_sym_score = 0

        # 3. 价格对称 (20 pts) — BC 回撤越接近 0.618 越好
        price_sym = 1 - min(abs(bc_ab_ratio - 0.618) / 0.618, 1)
        price_sym_score = self.fib_weights["price_symmetry"] * price_sym

        # 4. 腿成熟度 (10 pts)
        if ab_bars >= 8:
            leg_score = self.fib_weights["leg_maturity"]
        elif ab_bars >= 5:
            leg_score = self.fib_weights["leg_maturity"] * 0.7
        elif ab_bars >= 3:
            leg_score = self.fib_weights["leg_maturity"] * 0.3
        else:
            leg_score = 0

        return {
            "fib_precision": float(fib_score),
            "time_symmetry": float(time_sym_score),
            "price_symmetry": float(price_sym_score),
            "leg_maturity": float(leg_score),
        }

    def to_dataframe(self, shapes: List[ABCDShape]) -> pd.DataFrame:
        """将形态列表转换为 DataFrame"""
        return pd.DataFrame([s.to_dict() for s in shapes])
