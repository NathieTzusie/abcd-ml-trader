"""
Walk-Forward V4 引擎

V1 基础 + 两项改进:
  1. D_zone 反转确认: 触及 D_zone 后等待反转K线才入场
  2. 波动率自适应: 以 BTC ATR% 为基准，动态调整 d_zone_tolerance / sl_buffer_atr

与 V1 对比测试用。
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TradeResult:
    shape_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    direction: str = ""
    a_price: float = 0.0
    b_price: float = 0.0
    c_price: float = 0.0
    d_projected: float = 0.0
    ab_distance_pct: float = 0.0
    bc_ab_ratio: float = 0.0
    quality_score: float = 0.0
    ab_bars: int = 0
    bc_bars: int = 0
    entry_bar: int = 0
    entry_price: float = 0.0
    entry_time: pd.Timestamp = None
    exit_reason: str = ""
    exit_bar: int = 0
    exit_price: float = 0.0
    final_r: float = 0.0
    label: int = 0
    total_bars: int = 0
    d_zone_bars: int = 0
    d_zone_volume_ratio: float = 1.0
    d_zone_momentum: float = 0.0
    d_zone_wick_ratio: float = 0.0
    atr_pct: float = 0.0
    atr_percentile_20: float = 0.0
    ema20_50_direction: int = 0
    price_vs_ema20: float = 0.0
    volume_ratio: float = 1.0
    volume_trend: float = 1.0
    hour_of_day: int = 0
    day_of_week: int = 0
    vol_mult: float = 1.0  # [V4] 波动率倍率


@dataclass
class ActiveShape:
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
    # [V4] 新增
    vol_mult: float = 1.0
    d_zone_extreme: float = 0.0  # D_zone 内极限价格（bullish=最低, bearish=最高）


@dataclass
class ActiveTrade:
    shape: ActiveShape
    entry_bar: int
    entry_price: float
    tp1_hit: bool = False
    tp1_bar: Optional[int] = None
    sl_price: float = 0.0
    be_price: float = 0.0
    tp2_price: float = 0.0
    timeout_bar: int = 0


class WalkForwardEngineV4:
    """Walk-Forward V4: V1 + 反转确认 + 波动率自适应"""

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
        # ── V4 新增参数 ──
        use_confirmation: bool = True,
        confirm_bars: int = 3,
        confirm_mode: str = "swing",    # "candle" | "swing" | "both"
        vol_adaptive: bool = False,
        btc_atr_pct_ref: float = 0.003,  # BTC 15m ATR% 中位
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
        self.use_confirmation = use_confirmation
        self.confirm_bars = confirm_bars
        self.confirm_mode = confirm_mode
        self.vol_adaptive = vol_adaptive
        self.btc_atr_pct_ref = btc_atr_pct_ref
        self._shape_counter = 0

    # ================================================================
    # 预计算指标
    # ================================================================
    @staticmethod
    def _compute_adx(high, low, close, period=14):
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
    def compute_atr(high, low, close, period):
        n = len(high)
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = np.zeros(n); atr[0] = tr[0]
        alpha = 2 / (period + 1)
        for i in range(1, n): atr[i] = alpha * tr[i] + (1-alpha) * atr[i-1]
        return atr

    @staticmethod
    def calc_ema(prices, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[:period] = np.mean(prices[:max(period, 1)])
        for i in range(max(period, 1), len(prices)):
            ema[i] = alpha * prices[i] + (1-alpha) * ema[i-1]
        return ema

    def _get_vol_mult(self, atr_val: float, price: float) -> float:
        """计算波动率倍率。atr_val/price vs BTC 基准"""
        if not self.vol_adaptive or price <= 0:
            return 1.0
        cur_atr_pct = atr_val / price
        ratio = cur_atr_pct / max(self.btc_atr_pct_ref, 1e-10)
        return max(1.0, ratio)  # 不低于 1.0（不缩小 BTC 自己的参数）

    # ================================================================
    # 核心回测
    # ================================================================
    def run(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        opens = df["open"].values
        volume = df["volume"].values
        n = len(df)

        atr = self.compute_atr(high, low, close, self.atr_period)
        ema20 = self.calc_ema(close, 20)
        ema50 = self.calc_ema(close, 50)
        adx = self._compute_adx(high, low, close, 14) if self.max_adx > 0 else None

        confirmed_swings: deque = deque(maxlen=10)
        all_swings: List[Tuple] = []
        direction = 0
        extreme_price = 0.0
        extreme_idx = 0

        active_shapes: List[ActiveShape] = []
        active_trades: List[ActiveTrade] = []
        completed_trades: List[TradeResult] = []

        start_bar = self.atr_period + 10
        if start_bar >= n:
            return pd.DataFrame()

        direction, extreme_price, extreme_idx = self._init_first_swing(
            high, low, atr, start_bar, confirmed_swings, df)
        if direction == 0:
            return pd.DataFrame()
        first = confirmed_swings[-1]
        all_swings.append((first[0], first[1], first[2]))

        # ================================================================
        # 逐 bar 主循环
        # ================================================================
        for i in range(start_bar, n):
            threshold = self.atr_mult * atr[i]
            adx_i = adx[i] if adx is not None else 0

            # ---- Step 1: ZigZag ----
            if direction == 1:
                if high[i] > extreme_price:
                    extreme_price = high[i]; extreme_idx = i
                retreat = extreme_price - low[i]
                if retreat >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "high"))
                    all_swings.append((extreme_idx, extreme_price, "high"))
                    direction = -1; extreme_price = low[i]; extreme_idx = i
                    new_shapes = self._detect_all(i, confirmed_swings, atr, close,
                                                   symbol, timeframe, adx_i)
                    for s in new_shapes: active_shapes.append(s)
            elif direction == -1:
                if low[i] < extreme_price:
                    extreme_price = low[i]; extreme_idx = i
                rally = high[i] - extreme_price
                if rally >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "low"))
                    all_swings.append((extreme_idx, extreme_price, "low"))
                    direction = 1; extreme_price = high[i]; extreme_idx = i
                    new_shapes = self._detect_all(i, confirmed_swings, atr, close,
                                                   symbol, timeframe, adx_i)
                    for s in new_shapes: active_shapes.append(s)

            # ---- Step 2: 监测 D_zone ----
            for shape in active_shapes:
                if shape.first_in_zone_bar is not None:
                    # 已在 D_zone → 更新极限价格
                    if shape.direction == "bullish":
                        shape.d_zone_extreme = min(shape.d_zone_extreme, low[i])
                    else:
                        shape.d_zone_extreme = max(shape.d_zone_extreme, high[i])
                    continue
                if self._price_in_zone(high[i], low[i], shape):
                    shape.first_in_zone_bar = i
                    shape.d_zone_extreme = low[i] if shape.direction == "bullish" else high[i]

            # ---- Step 3: 入场（含确认） ----
            for shape in active_shapes:
                if shape.trade_entered:
                    continue
                if shape.first_in_zone_bar is None:
                    continue

                if self.use_confirmation:
                    confirmed, entry_bar = self._check_confirmation(
                        shape, i, close, opens, low, high)
                    if not confirmed:
                        if entry_bar == -1:
                            shape.trade_entered = True  # 超时放弃
                        continue
                    trade = self._enter_trade(shape, entry_bar, atr, close, df)
                else:
                    trade = self._enter_trade(shape, i, atr, close, df)

                if trade is not None:
                    active_trades.append(trade)
                    shape.trade_entered = True

            if i % 100 == 0:
                active_shapes = [s for s in active_shapes if not s.trade_entered]

            # ---- Step 4: 检查退出 ----
            still_active = []
            for trade in active_trades:
                result = self._check_exit(trade, i, high, low, close, df)
                if result is not None:
                    result.vol_mult = trade.shape.vol_mult
                    self._enrich_features(result, i, high, low, close, opens,
                                          volume, atr, ema20, ema50, df, df.index)
                    completed_trades.append(result)
                else:
                    still_active.append(trade)
            active_trades = still_active

        # ---- 清理 ----
        for trade in active_trades:
            final_bar = min(trade.timeout_bar, n - 1)
            result = self._force_close(trade, final_bar, close[final_bar], df)
            result.vol_mult = trade.shape.vol_mult
            self._enrich_features(result, final_bar, high, low, close, opens,
                                  volume, atr, ema20, ema50, df, df.index)
            completed_trades.append(result)

        return pd.DataFrame([t.__dict__ for t in completed_trades])

    # ================================================================
    # [V4] 反转确认
    # ================================================================
    def _check_confirmation(self, shape, i, close, opens, low, high):
        """
        检查 D_zone 触及后是否出现反转确认信号。

        Returns
        -------
        (confirmed: bool, entry_bar: int)
          entry_bar = i (确认) / -1 (超时放弃) / -2 (继续等待)
        """
        bars_in_zone = i - shape.first_in_zone_bar
        if bars_in_zone > self.confirm_bars:
            return False, -1  # 超时

        mode = self.confirm_mode
        is_bull = shape.direction == "bullish"

        candle_ok = False
        swing_ok = False

        # 蜡烛确认
        if is_bull:
            candle_ok = close[i] > opens[i]
        else:
            candle_ok = close[i] < opens[i]

        # 摆动确认：价格离开 D_zone 极值方向
        # 需要至少 1 bar（不能在同一 bar 确认 swing，需要时间验证）
        if bars_in_zone >= 1:
            if is_bull:
                swing_ok = low[i] > shape.d_zone_extreme
            else:
                swing_ok = high[i] < shape.d_zone_extreme

        if mode == "candle":
            confirmed = candle_ok
        elif mode == "swing":
            confirmed = swing_ok
        else:  # "both"
            confirmed = candle_ok and swing_ok

        if confirmed:
            return True, i
        return False, -2  # 继续等待

    # ================================================================
    # 形态检测聚合
    # ================================================================
    def _detect_all(self, current_bar, swings, atr, close, symbol, tf, adx_i):
        shapes = []
        # ABCD
        s = self._try_detect_abcd(swings, current_bar, atr, close, symbol, tf)
        if s is not None and self._validate_shape(s, adx_i):
            shapes.append(s)
        # 多谐波
        multi = self._try_multi_patterns(swings, current_bar, atr, close, symbol, tf)
        for ms in multi:
            if self._validate_shape(ms, adx_i):
                shapes.append(ms)
        return shapes

    def _try_detect_abcd(self, swings, current_bar, atr, close, symbol, timeframe):
        if len(swings) < 3: return None
        c = swings[-1]; b = swings[-2]; a = swings[-3]

        if a[2] == "high":
            if not (b[2] == "low" and c[2] == "high" and c[1] < a[1]): return None
            direction = "bullish"
        else:
            if not (b[2] == "high" and c[2] == "low" and c[1] > a[1]): return None
            direction = "bearish"

        ab_len = abs(a[1] - b[1])
        bc_len = abs(b[1] - c[1])
        if ab_len == 0: return None
        bc_ab = bc_len / ab_len
        if not (self.bc_ab_min <= bc_ab <= self.bc_ab_max): return None

        b_idx = min(b[0], len(atr) - 1)
        b_atr = atr[b_idx]
        if ab_len < self.min_atr_mult * b_atr: return None

        if direction == "bullish":
            d_proj = c[1] - ab_len * self.cd_ab_ratio
        else:
            d_proj = c[1] + ab_len * self.cd_ab_ratio

        ab_bars = abs(b[0] - a[0])
        bc_bars = abs(c[0] - b[0])
        quality = self._quality_score(bc_ab, ab_bars, bc_bars)
        if quality < self.min_quality_score: return None

        # [V4] 波动率自适应倍率
        vol_mult = self._get_vol_mult(b_atr, b[1])
        d_tol = self.d_zone_tolerance * vol_mult
        d_lower = d_proj * (1 - d_tol)
        d_upper = d_proj * (1 + d_tol)

        self._shape_counter += 1
        return ActiveShape(
            shape_id=f"{symbol}_{timeframe}_{direction}_{self._shape_counter}",
            direction=direction,
            a_price=a[1], b_price=b[1], c_price=c[1],
            d_projected=d_proj, bc_ab_ratio=round(bc_ab, 4),
            quality_score=round(quality, 1),
            ab_bars=ab_bars, bc_bars=bc_bars,
            cd_bars_estimated=int(ab_bars * self.cd_ab_ratio),
            a_idx=a[0], b_idx=b[0], c_idx=c[0], c_bar=current_bar,
            d_lower=d_lower, d_upper=d_upper,
            vol_mult=round(vol_mult, 3),
        )

    def _try_multi_patterns(self, swings, current_bar, atr, close, symbol, timeframe):
        from abcd_detector.harmonic_patterns import detect_all_patterns
        if len(swings) < 4: return []
        a = swings[-3]; b = swings[-2]; c = swings[-1]

        if a[2] == "high" and b[2] == "low" and c[2] == "high" and c[1] < a[1]:
            direction = "bullish"
        elif a[2] == "low" and b[2] == "high" and c[2] == "low" and c[1] > a[1]:
            direction = "bearish"
        else:
            return []

        ab_len = abs(a[1] - b[1]); bc_len = abs(b[1] - c[1])
        if ab_len == 0: return []
        bc_ab = bc_len / ab_len
        ab_bars = abs(b[0] - a[0]); bc_bars = abs(c[0] - b[0])
        ab_dist_pct = ab_len / b[1]

        if direction == "bullish":
            d_proj = c[1] - ab_len * self.cd_ab_ratio
        else:
            d_proj = c[1] + ab_len * self.cd_ab_ratio

        detected = detect_all_patterns(swings, bc_ab, self.cd_ab_ratio,
                                       ab_bars, bc_bars, ab_dist_pct, d_proj, direction)
        shapes = []
        b_idx = min(b[0], len(atr) - 1)
        b_atr = atr[b_idx]
        vol_mult = self._get_vol_mult(b_atr, b[1])
        d_tol = self.d_zone_tolerance * vol_mult

        for hp in detected:
            if hp.quality_score < self.min_quality_score: continue
            if ab_len < self.min_atr_mult * b_atr: continue
            self._shape_counter += 1
            d_lower = hp.d_projected * (1 - d_tol)
            d_upper = hp.d_projected * (1 + d_tol)
            shapes.append(ActiveShape(
                shape_id=f"{symbol}_{timeframe}_{hp.pattern}_{direction}_{self._shape_counter}",
                direction=direction,
                a_price=a[1], b_price=b[1], c_price=c[1],
                d_projected=hp.d_projected, bc_ab_ratio=round(bc_ab, 4),
                quality_score=hp.quality_score,
                ab_bars=ab_bars, bc_bars=bc_bars,
                ab_distance_pct=round(ab_dist_pct, 6), c_bar=current_bar,
                d_lower=d_lower, d_upper=d_upper,
                vol_mult=round(vol_mult, 3),
            ))
        return shapes

    # ================================================================
    # 入场 / 退出 (沿用 V1 逻辑)
    # ================================================================
    def _enter_trade(self, shape, current_bar, atr, close, df):
        entry_price = shape.d_projected
        entry_bar = current_bar
        entry_atr = atr[min(entry_bar, len(atr) - 1)]

        # [V4] SL buffer 也乘以 vol_mult
        sl_buf = self.sl_buffer_atr * shape.vol_mult

        if shape.direction == "bullish":
            sl_price = shape.d_lower - sl_buf * entry_atr
            tp2_price = shape.a_price
            be_price = entry_price
        else:
            sl_price = shape.d_upper + sl_buf * entry_atr
            tp2_price = shape.a_price
            be_price = entry_price

        ad_bars = shape.ab_bars + shape.bc_bars
        timeout_bar = entry_bar + int(ad_bars * self.timeout_mult)

        return ActiveTrade(
            shape=shape, entry_bar=entry_bar, entry_price=entry_price,
            sl_price=sl_price, be_price=be_price,
            tp2_price=tp2_price, timeout_bar=timeout_bar,
        )

    def _check_exit(self, trade, i, high, low, close, df):
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

        exit_reason = None; exit_price = 0.0
        if tp1_ok:
            trade.tp1_hit = True; trade.tp1_bar = i

        if sl_ok:
            exit_reason = "TP1+SL" if trade.tp1_hit else "SL"
            exit_price = trade.sl_price
        elif be_ok and trade.tp1_hit:
            exit_reason = "TP1+BE"; exit_price = trade.be_price
        elif tp2_ok:
            exit_reason = "TP1+TP2" if trade.tp1_hit else "TP2"
            exit_price = trade.tp2_price

        if exit_reason is None: return None

        risk = abs(trade.entry_price - trade.sl_price)
        if risk == 0: risk = 1e-8

        if direction == "bullish":
            tp1_r = (c_price - trade.entry_price) / risk * self.tp1_pct if trade.tp1_hit else 0
            remaining = 1.0 - self.tp1_pct if trade.tp1_hit else 1.0
            r = tp1_r + (exit_price - trade.entry_price) / risk * remaining
        else:
            tp1_r = (trade.entry_price - c_price) / risk * self.tp1_pct if trade.tp1_hit else 0
            remaining = 1.0 - self.tp1_pct if trade.tp1_hit else 1.0
            r = tp1_r + (trade.entry_price - exit_price) / risk * remaining

        return TradeResult(
            shape_id=trade.shape.shape_id, symbol="", timeframe="",
            direction=direction, a_price=trade.shape.a_price,
            b_price=trade.shape.b_price, c_price=c_price,
            d_projected=trade.shape.d_projected,
            ab_distance_pct=trade.shape.ab_distance_pct,
            bc_ab_ratio=trade.shape.bc_ab_ratio,
            quality_score=trade.shape.quality_score,
            ab_bars=trade.shape.ab_bars, bc_bars=trade.shape.bc_bars,
            entry_bar=trade.entry_bar, entry_price=trade.entry_price,
            entry_time=df.index[trade.entry_bar],
            exit_reason=exit_reason, exit_bar=i, exit_price=exit_price,
            final_r=round(r, 4), label=1 if r >= 0 else 0,
            total_bars=i - trade.entry_bar,
        )

    def _force_close(self, trade, final_bar, close_price, df):
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
            shape_id=trade.shape.shape_id, symbol="", timeframe="",
            direction=direction, a_price=trade.shape.a_price,
            b_price=trade.shape.b_price, c_price=trade.shape.c_price,
            d_projected=trade.shape.d_projected,
            ab_distance_pct=trade.shape.ab_distance_pct,
            bc_ab_ratio=trade.shape.bc_ab_ratio,
            quality_score=trade.shape.quality_score,
            ab_bars=trade.shape.ab_bars, bc_bars=trade.shape.bc_bars,
            entry_bar=trade.entry_bar, entry_price=trade.entry_price,
            entry_time=df.index[trade.entry_bar],
            exit_reason=exit_reason, exit_bar=final_bar, exit_price=close_price,
            final_r=round(r, 4), label=1 if r >= 0 else 0,
            total_bars=final_bar - trade.entry_bar,
        )

    # ================================================================
    # ZigZag / 验证 / 辅助
    # ================================================================
    def _init_first_swing(self, high, low, atr, start_bar, swings, df):
        lookback = 10
        for i in range(start_bar + lookback, min(start_bar + 200, len(high))):
            window_h = high[i - lookback:i]; window_l = low[i - lookback:i]
            is_high = high[i - lookback] == np.max(window_h)
            is_low = low[i - lookback] == np.min(window_l)
            if is_high:
                swings.append((i - lookback, high[i - lookback], "high"))
                return -1, low[i - lookback], i - lookback
            elif is_low:
                swings.append((i - lookback, low[i - lookback], "low"))
                return 1, high[i - lookback], i - lookback
        return 0, 0, 0

    @staticmethod
    def _price_in_zone(high_i, low_i, shape):
        if shape.direction == "bullish":
            return low_i <= shape.d_upper and high_i >= shape.d_lower
        else:
            return high_i >= shape.d_lower and low_i <= shape.d_upper

    def _validate_shape(self, shape, adx_at_entry=0):
        if self.min_tp1_pct > 0 or self.max_tp1_pct < 1.0:
            tp1_dist = abs(shape.c_price - shape.d_projected)
            tp1_pct = tp1_dist / shape.d_projected
            if tp1_pct < self.min_tp1_pct: return False
            if tp1_pct > self.max_tp1_pct: return False
        if self.max_adx > 0 and adx_at_entry > self.max_adx: return False
        return True

    @staticmethod
    def _quality_score(bc_ab, ab_bars, bc_bars):
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

    # ================================================================
    # 特征提取
    # ================================================================
    @staticmethod
    def _enrich_features(result, bar_i, high, low, close, opens, volume,
                         atr, ema20, ema50, df, index_arr):
        n = len(high)
        entry_bar = result.entry_bar; entry_price = result.entry_price
        d_lower = result.d_projected * (1 - 0.005)
        d_upper = result.d_projected * (1 + 0.005)

        zone_bars = 0
        c_bar = result.entry_bar - 1
        for j in range(max(c_bar + 1, 0), entry_bar + 1):
            if j >= n: break
            if low[j] <= d_upper and high[j] >= d_lower:
                zone_bars += 1
        result.d_zone_bars = zone_bars

        start_i = max(0, entry_bar - 3); end_i = min(n, entry_bar + 1)
        dzone_vol = volume[start_i:end_i]
        vol_ma20 = np.mean(volume[max(0, entry_bar - 20):entry_bar + 1]) if entry_bar >= 20 else np.mean(dzone_vol)
        result.d_zone_volume_ratio = round(float(np.mean(dzone_vol) / max(vol_ma20, 1e-10)), 4)

        if entry_bar >= 5:
            result.d_zone_momentum = round(float((close[entry_bar] - close[entry_bar - 5]) / max(close[entry_bar - 5], 1e-10)), 4)
        else:
            result.d_zone_momentum = 0.0

        body = abs(close[entry_bar] - opens[entry_bar])
        total_range = high[entry_bar] - low[entry_bar]
        result.d_zone_wick_ratio = round(float((total_range - body) / max(total_range, 1e-10)), 4)

        result.atr_pct = round(float(atr[min(entry_bar, len(atr) - 1)] / entry_price), 6)
        if entry_bar >= 20:
            atr_window = atr[entry_bar - 19:entry_bar + 1] / entry_price
            result.atr_percentile_20 = round(float(np.percentile(atr_window, 80)), 6)
        else:
            result.atr_percentile_20 = result.atr_pct

        if entry_bar >= 14:
            result.ema20_50_direction = 1 if ema20[entry_bar] > ema50[entry_bar] else -1
            result.price_vs_ema20 = round(float(close[entry_bar] / max(ema20[entry_bar], 1e-10) - 1), 6)
        else:
            result.ema20_50_direction = 0; result.price_vs_ema20 = 0.0

        if entry_bar >= 20:
            vol_20 = np.mean(volume[entry_bar - 19:entry_bar + 1])
            vol_5 = np.mean(volume[max(0, entry_bar - 4):entry_bar + 1])
            result.volume_ratio = round(float(volume[entry_bar] / max(vol_20, 1e-10)), 4)
            result.volume_trend = round(float(vol_5 / max(vol_20, 1e-10)), 4)
        else:
            result.volume_ratio = 1.0; result.volume_trend = 1.0

        result.hour_of_day = index_arr[entry_bar].hour
        result.day_of_week = index_arr[entry_bar].dayofweek
