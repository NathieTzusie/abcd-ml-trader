"""
Walk-Forward 回测引擎

逐 bar 推进，实时确认摆动点 → 检测 ABCD 形态 → 模拟交易。
消除 lookahead bias：C 点确认后才投影 D，从当前 bar 起监测 D_zone。
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TradeResult:
    """一笔已完成的交易"""
    shape_id: str
    symbol: str
    timeframe: str
    direction: str
    # 形态信息
    a_price: float
    b_price: float
    c_price: float
    d_projected: float
    ab_distance_pct: float
    bc_ab_ratio: float
    quality_score: float
    ab_bars: int
    bc_bars: int
    # 入场
    entry_bar: int
    entry_price: float
    entry_time: pd.Timestamp
    # 退出
    exit_reason: str
    exit_bar: int
    exit_price: float
    final_r: float
    label: int  # 1 if final_r >= 0
    total_bars: int
    # D_zone 特征
    d_zone_bars: int = 0
    d_zone_volume_ratio: float = 1.0
    d_zone_momentum: float = 0.0
    d_zone_wick_ratio: float = 0.0
    # 市场状态
    atr_pct: float = 0.0
    atr_percentile_20: float = 0.0
    ema20_50_direction: int = 0
    price_vs_ema20: float = 0.0
    volume_ratio: float = 1.0
    volume_trend: float = 1.0
    # 时间
    hour_of_day: int = 0
    day_of_week: int = 0


@dataclass
@dataclass
class ActiveShape:
    """一个正在等待 D_zone 触发的活跃形态"""
    shape_id: str = ""
    direction: str = ""
    a_price: float = 0.0
    b_price: float = 0.0
    c_price: float = 0.0
    d_projected: float = 0.0
    bc_ab_ratio: float = 0.0
    cd_ab_ratio: float = 1.0
    quality_score: float = 0.0
    ab_bars: int = 0
    bc_bars: int = 0
    cd_bars_estimated: int = 0
    ab_distance_pct: float = 0.0
    a_idx: int = 0
    b_idx: int = 0
    c_idx: int = 0
    c_bar: int = 0
    d_lower: float = 0.0
    d_upper: float = 0.0
    first_in_zone_bar: Optional[int] = None
    trade_entered: bool = False


@dataclass
class ActiveTrade:
    """一个正在进行的交易"""
    shape: ActiveShape
    entry_bar: int
    entry_price: float
    tp1_hit: bool = False
    tp1_bar: Optional[int] = None
    sl_price: float = 0.0
    be_price: float = 0.0
    tp2_price: float = 0.0
    timeout_bar: int = 0


class WalkForwardEngine:
    """
    Walk-Forward 回测引擎

    逐 bar 推进，流式检测形态和模拟交易。
    消除所有 lookahead bias。

    Parameters
    ----------
    atr_period, atr_mult : ZigZag 参数
    bc_ab_min, bc_ab_max : ABCD Fib 验证范围
    cd_ab_ratio : D 点投影比率
    d_zone_tolerance : D_zone = D × (1 ± tolerance)
    sl_buffer_atr : SL = D_zone 边缘 + buffer × ATR
    timeout_mult : 超时 = |AD| bars × multiplier
    tp1_pct : TP1 平仓比例
    min_quality_score : 最低形态质量分
    min_atr_mult : AB 最小幅度 = mult × ATR(B)
    min_tp1_pct : TP1 最小距离 % (0.003=0.3%)
    max_tp1_pct : TP1 最大距离 % (0.05=5%)
    max_adx : ADX 上限（0=不限），谐波反转在趋势中易失败
    use_confluence : 是否启用 confluence 过滤
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 0.5,
        bc_ab_min: float = 0.382,
        bc_ab_max: float = 0.886,
        cd_ab_ratio: float = 1.0,
        d_zone_tolerance: float = 0.005,
        sl_buffer_atr: float = 0.3,
        timeout_mult: float = 2.0,
        tp1_pct: float = 0.5,
        min_quality_score: float = 40,
        min_atr_mult: float = 0.5,
        min_tp1_pct: float = 0.0,
        max_tp1_pct: float = 1.0,
        max_adx: float = 0.0,
        use_confluence: bool = False,
    ):
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.bc_ab_min = bc_ab_min
        self.bc_ab_max = bc_ab_max
        self.cd_ab_ratio = cd_ab_ratio
        self.d_zone_tolerance = d_zone_tolerance
        self.sl_buffer_atr = sl_buffer_atr
        self.timeout_mult = timeout_mult
        self.tp1_pct = tp1_pct
        self.min_quality_score = min_quality_score
        self.min_atr_mult = min_atr_mult
        self.min_tp1_pct = min_tp1_pct
        self.max_tp1_pct = max_tp1_pct
        self.max_adx = max_adx
        self.use_confluence = use_confluence

        self._shape_counter = 0

    # ================================================================
    # 预计算指标
    # ================================================================
    @staticmethod
    def _compute_adx(high, low, close, period=14):
        """Wilder's ADX"""
        n = len(high)
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        up = np.maximum(high - np.roll(high, 1), 0)
        down = np.maximum(np.roll(low, 1) - low, 0)
        up[0] = down[0] = 0
        atr_s = np.zeros(n); atr_s[:period] = np.mean(tr[:period])
        pdi = np.zeros(n); pdi[:period] = np.mean(up[:period])
        mdi = np.zeros(n); mdi[:period] = np.mean(down[:period])
        a = 1/period
        for i in range(period, n):
            atr_s[i] = atr_s[i-1]*(1-a) + tr[i]*a
            pdi[i] = pdi[i-1]*(1-a) + up[i]*a
            mdi[i] = mdi[i-1]*(1-a) + down[i]*a
        pdi_n = pdi / np.maximum(atr_s, 1e-10) * 100
        mdi_n = mdi / np.maximum(atr_s, 1e-10) * 100
        dx = np.abs(pdi_n - mdi_n) / np.maximum(pdi_n + mdi_n, 1e-10) * 100
        adx = np.zeros(n); adx[:period*2] = np.mean(dx[:period*2])
        for i in range(period*2, n):
            adx[i] = adx[i-1]*(1-a) + dx[i]*a
        return adx

    @staticmethod
    def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    period: int) -> np.ndarray:
        n = len(high)
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ))
        tr[0] = high[0] - low[0]
        atr = np.zeros(n)
        atr[0] = tr[0]
        alpha = 2 / (period + 1)
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        return atr

    @staticmethod
    def calc_ema(prices: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[:period] = np.mean(prices[:max(period, 1)])
        for i in range(max(period, 1), len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    # ================================================================
    # 核心回测
    # ================================================================
    def run(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        逐 bar walk-forward 回测

        Returns
        -------
        pd.DataFrame : 每行 = 一笔完成交易，含特征和标签
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        opens = df["open"].values
        volume = df["volume"].values
        n = len(df)

        # 预计算指标
        atr = self.compute_atr(high, low, close, self.atr_period)
        ema20 = self.calc_ema(close, 20)
        ema50 = self.calc_ema(close, 50)
        adx = self._compute_adx(high, low, close, 14) if self.max_adx > 0 else None

        # ZigZag 状态
        confirmed_swings: deque = deque(maxlen=10)  # (bar_idx, price, direction)
        all_swings: List[Tuple] = []  # 完整历史（供 confluence 用）
        direction = 0       # 0=未确定, 1=找高点, -1=找低点
        extreme_price = 0.0
        extreme_idx = 0

        # 活跃状态
        active_shapes: List[ActiveShape] = []
        active_trades: List[ActiveTrade] = []
        completed_trades: List[TradeResult] = []

        start_bar = self.atr_period + 10
        if start_bar >= n:
            return pd.DataFrame()

        # 初始化第一个摆动点
        direction, extreme_price, extreme_idx = self._init_first_swing(
            high, low, atr, start_bar, confirmed_swings, df
        )
        if direction == 0:
            return pd.DataFrame()
        # 第一个 swing 也加入 all_swings
        first = confirmed_swings[-1]
        all_swings.append((first[0], first[1], first[2]))

        # ================================================================
        # 逐 bar 主循环
        # ================================================================
        for i in range(start_bar, n):
            threshold = self.atr_mult * atr[i]
            adx_i = adx[i] if adx is not None else 0

            # ---- Step 1: ZigZag 状态更新 ----
            if direction == 1:  # 在找高点
                if high[i] > extreme_price:
                    extreme_price = high[i]
                    extreme_idx = i
                retreat = extreme_price - low[i]
                if retreat >= threshold and i > extreme_idx + 1:
                    # 确认高点
                    confirmed_swings.append((extreme_idx, extreme_price, "high"))
                    all_swings.append((extreme_idx, extreme_price, "high"))
                    direction = -1
                    extreme_price = low[i]
                    extreme_idx = i
                    # 新摆动点确认 → 尝试检测所有谐波形态
                    new_shape = self._try_detect_shape(confirmed_swings, i, atr,
                                                        symbol, timeframe)
                    if new_shape is not None and self._validate_shape(new_shape, adx_i):
                        active_shapes.append(new_shape)
                    multi_shapes = self._try_multi_patterns(confirmed_swings, i, atr,
                                                            symbol, timeframe)
                    for ms in multi_shapes:
                        if self._validate_shape(ms, adx_i):
                            active_shapes.append(ms)

            elif direction == -1:  # 在找低点
                if low[i] < extreme_price:
                    extreme_price = low[i]
                    extreme_idx = i
                rally = high[i] - extreme_price
                if rally >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "low"))
                    all_swings.append((extreme_idx, extreme_price, "low"))
                    direction = 1
                    extreme_price = high[i]
                    extreme_idx = i
                    new_shape = self._try_detect_shape(confirmed_swings, i, atr,
                                                        symbol, timeframe)
                    if new_shape is not None and self._validate_shape(new_shape, adx_i):
                        active_shapes.append(new_shape)
                    multi_shapes = self._try_multi_patterns(confirmed_swings, i, atr,
                                                            symbol, timeframe)
                    for ms in multi_shapes:
                        if self._validate_shape(ms, adx_i):
                            active_shapes.append(ms)

            # ---- Step 2: 监测 D_zone 入场 ----
            for shape in active_shapes:
                if shape.first_in_zone_bar is not None:
                    continue  # 已经进入 D_zone了
                if self._price_in_zone(high[i], low[i], shape):
                    shape.first_in_zone_bar = i

            # ---- Step 3: 活跃形态 → 入场 ----
            for shape in active_shapes:
                if shape.trade_entered:
                    continue
                if shape.first_in_zone_bar is not None:
                    if self.use_confluence:
                        if not self._check_confluence(
                            shape, all_swings, high, low, close, volume, i
                        ):
                            shape.trade_entered = True
                            continue
                    
                    trade = self._enter_trade(shape, i, atr, close, df)
                    if trade is not None:
                        active_trades.append(trade)
                        shape.trade_entered = True

            # 每 100 bar 清理已入场旧形状
            if i % 100 == 0:
                active_shapes = [s for s in active_shapes if not s.trade_entered]

            # ---- Step 4: 活跃交易 → 检查退出 ----
            still_active = []
            for trade in active_trades:
                result = self._check_exit(trade, i, high, low, close, df)
                if result is not None:
                    # 交易完成，提取特征
                    self._enrich_features(result, i, high, low, close, opens,
                                          volume, atr, ema20, ema50, df, df.index)
                    completed_trades.append(result)
                else:
                    still_active.append(trade)
            active_trades = still_active

        # ================================================================
        # 清理（bar 结束时仍未退出的交易 → Timeout）
        # ================================================================
        for trade in active_trades:
            final_bar = min(trade.timeout_bar, n - 1)
            result = self._force_close(trade, final_bar, close[final_bar], df)
            self._enrich_features(result, final_bar, high, low, close, opens,
                                  volume, atr, ema20, ema50, df, df.index)
            completed_trades.append(result)

        return pd.DataFrame([t.__dict__ for t in completed_trades])

    # ================================================================
    # ZigZag 初始化
    # ================================================================
    def _init_first_swing(self, high, low, atr, start_bar, swings, df):
        """找到第一个有效的摆动点"""
        lookback = 10
        for i in range(start_bar + lookback, min(start_bar + 200, len(high))):
            window_h = high[i - lookback:i]
            window_l = low[i - lookback:i]
            is_high = high[i - lookback] == np.max(window_h)
            is_low = low[i - lookback] == np.min(window_l)

            if is_high:
                swings.append((i - lookback, high[i - lookback], "high"))
                return -1, low[i - lookback], i - lookback
            elif is_low:
                swings.append((i - lookback, low[i - lookback], "low"))
                return 1, high[i - lookback], i - lookback
        return 0, 0, 0

    # ================================================================
    # ABCD 检测（在摆动点确认时触发）
    # ================================================================
    def _try_multi_patterns(self, swings, current_bar, atr, symbol, timeframe):
        """检测 Gartley / Butterfly / Bat（需要 X,A,B,C 四个摆动点）"""
        from abcd_detector.harmonic_patterns import detect_all_patterns
        
        if len(swings) < 4:
            return []
        
        # 复用 ABCD 检测的三点 (a, b, c) = swings[-3], swings[-2], swings[-1]
        a = swings[-3]
        b = swings[-2]
        c = swings[-1]
        
        # 验证方向
        if a[2] == "high" and b[2] == "low" and c[2] == "high" and c[1] < a[1]:
            direction = "bullish"
        elif a[2] == "low" and b[2] == "high" and c[2] == "low" and c[1] > a[1]:
            direction = "bearish"
        else:
            return []
        
        ab_len = abs(a[1] - b[1])
        bc_len = abs(b[1] - c[1])
        if ab_len == 0:
            return []
        
        bc_ab = bc_len / ab_len
        ab_bars = abs(b[0] - a[0])
        bc_bars = abs(c[0] - b[0])
        ab_dist_pct = ab_len / b[1]
        cd_ab = self.cd_ab_ratio
        
        if direction == "bullish":
            d_proj = c[1] - ab_len * cd_ab
        else:
            d_proj = c[1] + ab_len * cd_ab
        
        # 调用多形态检测
        detected = detect_all_patterns(swings, bc_ab, cd_ab, ab_bars, bc_bars,
                                       ab_dist_pct, d_proj, direction)
        
        shapes = []
        for hp in detected:
            if hp.quality_score < self.min_quality_score:
                continue
            b_atr = atr[min(b[0], len(atr) - 1)]
            if ab_len < self.min_atr_mult * b_atr:
                continue
            
            self._shape_counter += 1
            d_lower = hp.d_projected * (1 - self.d_zone_tolerance)
            d_upper = hp.d_projected * (1 + self.d_zone_tolerance)
            
            shape_id = f"{symbol}_{timeframe}_{hp.pattern}_{direction}_{self._shape_counter}"
            
            # 用 ActiveShape 包装（pattern 信息存在 shape_id 里，d_proj 用同一结构）
            from backtest.walk_forward import ActiveShape
            active = ActiveShape(
                shape_id=shape_id,
                direction=direction,
                a_price=a[1], b_price=b[1], c_price=c[1],
                d_projected=hp.d_projected,
                bc_ab_ratio=round(bc_ab, 4),
                quality_score=hp.quality_score,
                ab_bars=ab_bars, bc_bars=bc_bars,
                ab_distance_pct=round(ab_dist_pct, 6),
                c_bar=current_bar,
                d_lower=d_lower, d_upper=d_upper,
            )
            shapes.append(active)
        
        return shapes

    def _try_detect_shape(self, swings, current_bar, atr, symbol, timeframe):
        """检查最新 3 个摆动点是否构成有效 ABC → 投影 D"""
        if len(swings) < 3:
            return None

        c = swings[-1]
        b = swings[-2]
        a = swings[-3]

        if a[2] == "high":
            if not (b[2] == "low" and c[2] == "high" and c[1] < a[1]):
                return None
            direction = "bullish"
        else:
            if not (b[2] == "high" and c[2] == "low" and c[1] > a[1]):
                return None
            direction = "bearish"

        ab_len = abs(a[1] - b[1])
        bc_len = abs(b[1] - c[1])
        if ab_len == 0:
            return None

        bc_ab = bc_len / ab_len
        if not (self.bc_ab_min <= bc_ab <= self.bc_ab_max):
            return None

        b_atr = atr[min(b[0], len(atr) - 1)]
        if ab_len < self.min_atr_mult * b_atr:
            return None

        if direction == "bullish":
            d_proj = c[1] - ab_len * self.cd_ab_ratio
        else:
            d_proj = c[1] + ab_len * self.cd_ab_ratio

        ab_bars = abs(b[0] - a[0])
        bc_bars = abs(c[0] - b[0])
        quality = self._quality_score(bc_ab, ab_bars, bc_bars)
        if quality < self.min_quality_score:
            return None

        d_lower = d_proj * (1 - self.d_zone_tolerance)
        d_upper = d_proj * (1 + self.d_zone_tolerance)

        self._shape_counter += 1

        return ActiveShape(
            shape_id=f"{symbol}_{timeframe}_{direction}_{self._shape_counter}",
            direction=direction,
            a_price=a[1], b_price=b[1], c_price=c[1],
            d_projected=d_proj,
            bc_ab_ratio=round(bc_ab, 4),
            quality_score=round(quality, 1),
            ab_bars=ab_bars, bc_bars=bc_bars,
            cd_bars_estimated=int(ab_bars * self.cd_ab_ratio),
            a_idx=a[0], b_idx=b[0], c_idx=c[0],
            c_bar=current_bar,
            d_lower=d_lower, d_upper=d_upper,
        )


    def _validate_shape(self, shape, adx_at_entry: float = 0) -> bool:
        """额外的形态验证（TP1距离, ADX等）"""
        if self.min_tp1_pct > 0 or self.max_tp1_pct < 1.0:
            tp1_dist = abs(shape.c_price - shape.d_projected)
            tp1_pct = tp1_dist / shape.d_projected
            if tp1_pct < self.min_tp1_pct:
                return False
            if tp1_pct > self.max_tp1_pct:
                return False
        if self.max_adx > 0 and adx_at_entry > self.max_adx:
            return False
        return True

    @staticmethod
    def _check_confluence(shape, all_swings, high, low, close, volume, current_bar):
        """检查 D_zone 是否落在 confluence 区域"""
        from abcd_detector.confluence import ConfluenceDetector
        detector = ConfluenceDetector()
        
        result = detector.check(
            d_projected=shape.d_projected,
            direction=shape.direction,
            current_bar=current_bar,
            swings=all_swings,
            high=high, low=low, close=close, volume=volume,
        )
        return result["has_confluence"]

    @staticmethod
    def _quality_score(bc_ab: float, ab_bars: int, bc_bars: int) -> float:
        """4 因子质量评分"""
        std_levels = [0.382, 0.5, 0.618, 0.786, 0.886]
        min_dist = min(abs(bc_ab - l) for l in std_levels)
        if min_dist < 0.02: fib_s = 45
        elif min_dist < 0.05: fib_s = 31.5
        elif min_dist < 0.10: fib_s = 18
        elif min_dist < 0.18: fib_s = 6.75
        else: fib_s = 0

        time_sym = 1 - min(abs(bc_bars / max(ab_bars, 1) - 1), 1)
        time_s = 25 * time_sym

        price_sym = 1 - min(abs(bc_ab - 0.618) / 0.618, 1)
        price_s = 20 * price_sym

        if ab_bars >= 8: leg_s = 10
        elif ab_bars >= 5: leg_s = 7
        elif ab_bars >= 3: leg_s = 3
        else: leg_s = 0

        return fib_s + time_s + price_s + leg_s

    @staticmethod
    def _price_in_zone(high_i, low_i, shape) -> bool:
        if shape.direction == "bullish":
            return low_i <= shape.d_upper and high_i >= shape.d_lower
        else:
            return high_i >= shape.d_lower and low_i <= shape.d_upper

    # ================================================================
    # 入场
    # ================================================================
    def _enter_trade(self, shape, current_bar, atr, close, df):
        entry_price = shape.d_projected
        entry_bar = current_bar
        entry_atr = atr[min(entry_bar, len(atr) - 1)]

        if shape.direction == "bullish":
            sl_price = shape.d_lower - self.sl_buffer_atr * entry_atr
            tp2_price = shape.a_price
            be_price = entry_price
        else:
            sl_price = shape.d_upper + self.sl_buffer_atr * entry_atr
            tp2_price = shape.a_price
            be_price = entry_price

        ad_bars = shape.ab_bars + shape.bc_bars
        timeout_bar = entry_bar + int(ad_bars * self.timeout_mult)

        return ActiveTrade(
            shape=shape,
            entry_bar=entry_bar,
            entry_price=entry_price,
            sl_price=sl_price,
            be_price=be_price,
            tp2_price=tp2_price,
            timeout_bar=timeout_bar,
        )

    # ================================================================
    # 退出检查
    # ================================================================
    def _check_exit(self, trade, i, high, low, close, df):
        """检查是否触发退出。返回 TradeResult 或 None"""
        c_price = trade.shape.c_price
        direction = trade.shape.direction
        tp1_hit = trade.tp1_hit

        if direction == "bullish":
            tp1_ok = not tp1_hit and high[i] >= c_price
            tp2_ok = high[i] >= trade.tp2_price
            be_ok = tp1_hit and high[i] >= trade.be_price
            sl_ok = low[i] <= trade.sl_price
        else:
            tp1_ok = not tp1_hit and low[i] <= c_price
            tp2_ok = low[i] <= trade.tp2_price
            be_ok = tp1_hit and low[i] <= trade.be_price
            sl_ok = high[i] >= trade.sl_price

        exit_reason = None
        exit_price = 0.0

        # TP1 首次触发（不分平仓，只记录）
        if tp1_ok:
            trade.tp1_hit = True
            trade.tp1_bar = i

        # 退出判定（保守：同 bar SL 优先）
        if sl_ok:
            exit_reason = "TP1+SL" if trade.tp1_hit else "SL"
            exit_price = trade.sl_price
        elif be_ok and trade.tp1_hit:
            exit_reason = "TP1+BE"
            exit_price = trade.be_price
        elif tp2_ok:
            exit_reason = "TP1+TP2" if trade.tp1_hit else "TP2"
            exit_price = trade.tp2_price

        if exit_reason is None:
            return None

        # 计算 R
        risk = abs(trade.entry_price - trade.sl_price)
        if risk == 0:
            risk = 1e-8

        if direction == "bullish":
            tp1_r = 0
            if trade.tp1_hit:
                tp1_r = (c_price - trade.entry_price) / risk * self.tp1_pct
            remaining_pct = 1.0 - self.tp1_pct if trade.tp1_hit else 1.0
            r = tp1_r + (exit_price - trade.entry_price) / risk * remaining_pct
        else:
            tp1_r = 0
            if trade.tp1_hit:
                tp1_r = (trade.entry_price - c_price) / risk * self.tp1_pct
            remaining_pct = 1.0 - self.tp1_pct if trade.tp1_hit else 1.0
            r = tp1_r + (trade.entry_price - exit_price) / risk * remaining_pct

        return TradeResult(
            shape_id=trade.shape.shape_id,
            symbol="", timeframe="",
            direction=direction,
            a_price=trade.shape.a_price,
            b_price=trade.shape.b_price,
            c_price=c_price,
            d_projected=trade.shape.d_projected,
            ab_distance_pct=trade.shape.ab_distance_pct,
            bc_ab_ratio=trade.shape.bc_ab_ratio,
            quality_score=trade.shape.quality_score,
            ab_bars=trade.shape.ab_bars,
            bc_bars=trade.shape.bc_bars,
            entry_bar=trade.entry_bar,
            entry_price=trade.entry_price,
            entry_time=df.index[trade.entry_bar],
            exit_reason=exit_reason,
            exit_bar=i,
            exit_price=exit_price,
            final_r=round(r, 4),
            label=1 if r >= 0 else 0,
            total_bars=i - trade.entry_bar,
        )

    def _force_close(self, trade, final_bar, close_price, df):
        """超时强制平仓"""
        direction = trade.shape.direction
        risk = abs(trade.entry_price - trade.sl_price)
        if risk == 0: risk = 1e-8

        if direction == "bullish":
            tp1_r = (trade.shape.c_price - trade.entry_price) / risk * self.tp1_pct if trade.tp1_hit else 0
            remaining = 1.0 - self.tp1_pct if trade.tp1_hit else 1.0
            r = tp1_r + (close_price - trade.entry_price) / risk * remaining
        else:
            tp1_r = (trade.entry_price - trade.shape.c_price) / risk * self.tp1_pct if trade.tp1_hit else 0
            remaining = 1.0 - self.tp1_pct if trade.tp1_hit else 1.0
            r = tp1_r + (trade.entry_price - close_price) / risk * remaining

        exit_reason = "TP1+Timeout" if trade.tp1_hit else "Timeout"

        return TradeResult(
            shape_id=trade.shape.shape_id,
            symbol="", timeframe="",
            direction=direction,
            a_price=trade.shape.a_price, b_price=trade.shape.b_price,
            c_price=trade.shape.c_price,
            d_projected=trade.shape.d_projected,
            ab_distance_pct=trade.shape.ab_distance_pct,
            bc_ab_ratio=trade.shape.bc_ab_ratio,
            quality_score=trade.shape.quality_score,
            ab_bars=trade.shape.ab_bars, bc_bars=trade.shape.bc_bars,
            entry_bar=trade.entry_bar, entry_price=trade.entry_price,
            entry_time=df.index[trade.entry_bar],
            exit_reason=exit_reason, exit_bar=final_bar,
            exit_price=close_price,
            final_r=round(r, 4), label=1 if r >= 0 else 0,
            total_bars=final_bar - trade.entry_bar,
        )

    # ================================================================
    # 特征提取（交易完成后）
    # ================================================================
    @staticmethod
    def _enrich_features(result, bar_i, high, low, close, opens, volume,
                         atr, ema20, ema50, df, index_arr):
        """填充市场状态和 D_zone 特征"""
        n = len(high)
        entry_bar = result.entry_bar
        entry_price = result.entry_price

        # D_zone 内 bar 数
        d_lower = result.d_projected * (1 - 0.005)
        d_upper = result.d_projected * (1 + 0.005)
        zone_bars = 0
        c_bar = result.entry_bar - 1  # approximate
        for j in range(max(c_bar + 1, 0), entry_bar + 1):
            if j >= n: break
            if low[j] <= d_upper and high[j] >= d_lower:
                zone_bars += 1
        result.d_zone_bars = zone_bars

        # D_zone 成交量
        start_i = max(0, entry_bar - 3)
        end_i = min(n, entry_bar + 1)
        dzone_vol = volume[start_i:end_i]
        vol_ma20 = np.mean(volume[max(0, entry_bar - 20):entry_bar + 1]) if entry_bar >= 20 else np.mean(dzone_vol)
        result.d_zone_volume_ratio = round(float(np.mean(dzone_vol) / max(vol_ma20, 1e-10)), 4)

        # D_zone 动量
        if entry_bar >= 5:
            result.d_zone_momentum = round(float((close[entry_bar] - close[entry_bar - 5]) / max(close[entry_bar - 5], 1e-10)), 4)
        else:
            result.d_zone_momentum = 0.0

        # D_zone 影线比
        body = abs(close[entry_bar] - opens[entry_bar])
        total_range = high[entry_bar] - low[entry_bar]
        result.d_zone_wick_ratio = round(float((total_range - body) / max(total_range, 1e-10)), 4)

        # 波动率
        result.atr_pct = round(float(atr[min(entry_bar, len(atr) - 1)] / entry_price), 6)
        if entry_bar >= 20:
            atr_window = atr[entry_bar - 19:entry_bar + 1] / entry_price
            result.atr_percentile_20 = round(float(np.percentile(atr_window, 80)), 6)
        else:
            result.atr_percentile_20 = result.atr_pct

        # 趋势
        if entry_bar >= 14:
            result.ema20_50_direction = 1 if ema20[entry_bar] > ema50[entry_bar] else -1
            result.price_vs_ema20 = round(float(close[entry_bar] / max(ema20[entry_bar], 1e-10) - 1), 6)
        else:
            result.ema20_50_direction = 0
            result.price_vs_ema20 = 0.0

        # 成交量
        if entry_bar >= 20:
            vol_20 = np.mean(volume[entry_bar - 19:entry_bar + 1])
            vol_5 = np.mean(volume[max(0, entry_bar - 4):entry_bar + 1])
            result.volume_ratio = round(float(volume[entry_bar] / max(vol_20, 1e-10)), 4)
            result.volume_trend = round(float(vol_5 / max(vol_20, 1e-10)), 4)
        else:
            result.volume_ratio = 1.0
            result.volume_trend = 1.0

        # 时间
        result.hour_of_day = index_arr[entry_bar].hour
        result.day_of_week = index_arr[entry_bar].dayofweek


# ================================================================
# 一键运行
# ================================================================
def run_walk_forward_all(
    shapes_csv: str = None,  # 不再需要，引擎内部生成
    data_dir: str = "/mnt/c/Users/12645/Sisie-Quantive/data",
    output_path: str = "outputs/wf_training_dataset.csv",
    **engine_kwargs,
) -> pd.DataFrame:
    """
    全量 Walk-Forward 回测
    """
    import time
    engine = WalkForwardEngine(**engine_kwargs)
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['15m', '30m', '1h', '4h']

    all_results = []
    stats = []
    total_t0 = time.time()

    for sym in symbols:
        for tf in timeframes:
            t0 = time.time()
            path = f"{data_dir}/binance_{sym}_{tf}.parquet"
            df = pd.read_parquet(path)
            df = df[(df.index >= '2024-01-01') & (df.index < '2026-05-01')]

            result = engine.run(df, sym, tf)
            elapsed = time.time() - t0

            if len(result) > 0:
                result['symbol'] = sym
                result['timeframe'] = tf
                all_results.append(result)
                stats.append({
                    'symbol': sym, 'tf': tf,
                    'shapes': result['shape_id'].nunique() if 'shape_id' in result.columns else len(result),
                    'trades': len(result),
                    'win_rate': round(result['label'].mean() * 100, 1),
                    'avg_r': round(result['final_r'].mean(), 3),
                    'median_r': round(result['final_r'].median(), 3),
                })
                print(f"  {sym:8s} {tf:4s}: {len(result):>5d} trades | "
                      f"WR={result['label'].mean()*100:.1f}% avgR={result['final_r'].mean():.3f} "
                      f"medR={result['final_r'].median():.3f} | {elapsed:.1f}s", flush=True)
            else:
                print(f"  {sym:8s} {tf:4s}: 0 trades | {elapsed:.1f}s", flush=True)

    if all_results:
        dataset = pd.concat(all_results, ignore_index=True)
        dataset.to_csv(output_path, index=False)

        total_elapsed = time.time() - total_t0
        print(f"\n{'='*60}")
        print(f"📊 Walk-Forward 回测完成: {output_path}")
        print(f"总交易: {len(dataset)} | 胜率: {dataset['label'].mean()*100:.1f}%")
        print(f"平均 R: {dataset['final_r'].mean():.3f} | 中位 R: {dataset['final_r'].median():.3f}")
        print(f"总耗时: {total_elapsed:.1f}s")
        print(f"\n分组:")
        for s in stats:
            print(f"  {s['symbol']:8s} {s['tf']:4s}: {s['trades']:>5d} trades "
                  f"WR={s['win_rate']:>5.1f}% avgR={s['avg_r']:>7.3f} medR={s['median_r']:>7.3f}", flush=True)

        return dataset
    return pd.DataFrame()


if __name__ == "__main__":
    run_walk_forward_all()
