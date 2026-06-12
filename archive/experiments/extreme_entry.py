"""
Walk-Forward D_zone 极值入场引擎

基于 V1 的 3-swing ABCD 检测。改动：
  1. D_zone 触及后跟踪最低点/最高点
  2. 确认极值（小反弹 0.3 ATR）后，以限价单入场在极值价
  3. 风险 = 固定 1.25 ATR（从入场价）
  4. 限价单 N bars 未成交则取消
"""
import numpy as np
import pandas as pd
from collections import deque
from typing import List
from backtest.walk_forward import (
    WalkForwardEngine, ActiveShape, ActiveTrade, TradeResult
)


class ExtremeEntryEngine(WalkForwardEngine):
    """
    D_zone 极值入场引擎

    新增参数:
      extreme_bounce_atr : 确认极值的小反弹 ATR 倍数 (默认 0.3)
      risk_atr_mult : 固定风险 ATR 倍数 (默认 1.25)
      limit_order_bars : 限价单有效 bar 数 (默认 12)
    """

    def __init__(self, extreme_bounce_atr=0.3, risk_atr_mult=1.25, limit_order_bars=12, **kwargs):
        super().__init__(**kwargs)
        self.extreme_bounce_atr = extreme_bounce_atr
        self.risk_atr_mult = risk_atr_mult
        self.limit_order_bars = limit_order_bars

    def run(self, df, symbol, timeframe):
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        opens = df["open"].values
        volume = df["volume"].values
        n = len(df)

        atr = self.compute_atr(high, low, close, self.atr_period)
        ema20 = self.calc_ema(close, 20)
        ema50 = self.calc_ema(close, 50)

        confirmed_swings = deque(maxlen=10)
        all_swings = []

        # dict 格式的活跃形态（追踪 D_zone 状态和极值）
        active_shapes: List[dict] = []
        active_trades: List[ActiveTrade] = []
        completed_trades: List[TradeResult] = []

        start_bar = self.atr_period + 10
        if start_bar >= n:
            return pd.DataFrame()

        direction, extreme_price, extreme_idx = self._init_first_swing(
            high, low, atr, start_bar, confirmed_swings, df
        )
        if direction == 0:
            return pd.DataFrame()
        first = confirmed_swings[-1]
        all_swings.append((first[0], first[1], first[2]))

        for i in range(start_bar, n):
            threshold = self.atr_mult * atr[i]
            adx_i = 0

            # --- Step 1: ZigZag + 形态检测 (V1 3-swing) ---
            if direction == 1:
                if high[i] > extreme_price:
                    extreme_price = high[i]; extreme_idx = i
                retreat = extreme_price - low[i]
                if retreat >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "high"))
                    all_swings.append((extreme_idx, extreme_price, "high"))
                    direction = -1; extreme_price = low[i]; extreme_idx = i
                    self._detect_shapes(confirmed_swings, i, atr, adx_i, symbol, timeframe, active_shapes)
            elif direction == -1:
                if low[i] < extreme_price:
                    extreme_price = low[i]; extreme_idx = i
                rally = high[i] - extreme_price
                if rally >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "low"))
                    all_swings.append((extreme_idx, extreme_price, "low"))
                    direction = 1; extreme_price = high[i]; extreme_idx = i
                    self._detect_shapes(confirmed_swings, i, atr, adx_i, symbol, timeframe, active_shapes)

            # --- Step 2: D_zone 极值追踪 + 限价入场 ---
            for shape in active_shapes:
                if shape.get("entered"):
                    continue
                if shape.get("expired"):
                    continue

                # 首次进入 D_zone
                if not shape.get("in_zone"):
                    if self._in_zone(high[i], low[i], shape):
                        shape["in_zone"] = True
                        shape["zone_bar"] = i
                        shape["extreme_price"] = low[i] if shape["direction"] == "bullish" else high[i]
                        shape["extreme_bar"] = i
                    continue

                # 更新极值
                if shape["direction"] == "bullish":
                    if low[i] < shape["extreme_price"]:
                        shape["extreme_price"] = low[i]
                        shape["extreme_bar"] = i
                else:
                    if high[i] > shape["extreme_price"]:
                        shape["extreme_price"] = high[i]
                        shape["extreme_bar"] = i

                # 检查极值是否已确认（小反弹）
                if not shape.get("extreme_confirmed"):
                    bounce_req = self.extreme_bounce_atr * atr[i]
                    if shape["direction"] == "bullish":
                        if high[i] >= shape["extreme_price"] + bounce_req:
                            shape["extreme_confirmed"] = True
                            shape["confirm_bar"] = i
                            shape["limit_price"] = shape["extreme_price"]  # 限价 = 极值
                    else:
                        if low[i] <= shape["extreme_price"] - bounce_req:
                            shape["extreme_confirmed"] = True
                            shape["confirm_bar"] = i
                            shape["limit_price"] = shape["extreme_price"]
                    continue

                # 限价单等待成交
                limit_bars = i - shape["confirm_bar"]
                if limit_bars > self.limit_order_bars:
                    shape["expired"] = True
                    continue

                # 检查限价单成交
                filled = False
                if shape["direction"] == "bullish":
                    if low[i] <= shape["limit_price"]:
                        filled = True
                else:
                    if high[i] >= shape["limit_price"]:
                        filled = True

                if filled:
                    shape["entered"] = True
                    entry_price = shape["limit_price"]
                    entry_bar = i

                    # 固定 ATR 风险
                    entry_atr = atr[min(i, len(atr) - 1)]
                    risk_amount = entry_atr * self.risk_atr_mult

                    if shape["direction"] == "bullish":
                        sl_price = entry_price - risk_amount
                    else:
                        sl_price = entry_price + risk_amount

                    if risk_amount <= 0:
                        continue

                    ad_bars = shape["ab_bars"] + shape["bc_bars"]
                    timeout_bar = i + int(ad_bars * self.timeout_mult)

                    legacy = ActiveShape(
                        shape_id=shape["shape_id"], direction=shape["direction"],
                        a_price=shape["a_price"], b_price=shape["b_price"],
                        c_price=shape["c_price"], d_projected=shape["d_projected"],
                        bc_ab_ratio=shape["bc_ab_ratio"], quality_score=shape["quality_score"],
                        ab_bars=shape["ab_bars"], bc_bars=shape["bc_bars"],
                        ab_distance_pct=shape.get("ab_distance_pct", 0),
                        c_bar=shape["c_bar"],
                        d_lower=shape["d_lower"], d_upper=shape["d_upper"],
                    )
                    trade = ActiveTrade(
                        shape=legacy, entry_bar=entry_bar, entry_price=entry_price,
                        sl_price=sl_price, be_price=entry_price,
                        tp2_price=shape["a_price"], timeout_bar=timeout_bar,
                    )
                    active_trades.append(trade)

            # --- Step 3: 退出 ---
            still_active = []
            for trade in active_trades:
                result = self._check_exit(trade, i, high, low, close, df)
                if result is not None:
                    self._enrich_features(result, i, high, low, close, opens,
                                          volume, atr, ema20, ema50, df, df.index)
                    result.symbol = symbol; result.timeframe = timeframe
                    completed_trades.append(result)
                else:
                    still_active.append(trade)
            active_trades = still_active

            if i % 100 == 0:
                active_shapes = [s for s in active_shapes if not s.get("entered") and not s.get("expired")]

        # 超时
        for trade in active_trades:
            fb = min(trade.timeout_bar, n - 1)
            r = self._force_close(trade, fb, close[fb], df)
            self._enrich_features(r, fb, high, low, close, opens, volume, atr, ema20, ema50, df, df.index)
            r.symbol = symbol; r.timeframe = timeframe
            completed_trades.append(r)

        return pd.DataFrame([t.__dict__ for t in completed_trades])

    def _detect_shapes(self, swings, i, atr, adx_i, symbol, timeframe, shapes_list):
        """V1 3-swing + D 投影形态检测，产出 dict"""
        if len(swings) < 3:
            return
        a = swings[-3]; b = swings[-2]; c_sw = swings[-1]

        if a[2] == "high" and b[2] == "low" and c_sw[2] == "high":
            if not (c_sw[1] < a[1]): return
            direction = "bullish"
        elif a[2] == "low" and b[2] == "high" and c_sw[2] == "low":
            if not (c_sw[1] > a[1]): return
            direction = "bearish"
        else:
            return

        ab_len = abs(a[1] - b[1]); bc_len = abs(b[1] - c_sw[1])
        if ab_len == 0: return
        bc_ab = bc_len / ab_len
        if not (self.bc_ab_min <= bc_ab <= self.bc_ab_max): return
        b_atr = atr[min(b[0], len(atr) - 1)]
        if ab_len < self.min_atr_mult * b_atr: return

        if direction == "bullish":
            d_proj = c_sw[1] - ab_len * self.cd_ab_ratio
        else:
            d_proj = c_sw[1] + ab_len * self.cd_ab_ratio

        ab_bars = abs(b[0] - a[0]); bc_bars = abs(c_sw[0] - b[0])
        quality = self._quality_score(bc_ab, ab_bars, bc_bars)
        if quality < self.min_quality_score: return

        tp1_dist = abs(c_sw[1] - d_proj)
        tp1_pct = tp1_dist / d_proj if d_proj > 0 else 0
        if self.min_tp1_pct > 0 and tp1_pct < self.min_tp1_pct: return

        d_lower = d_proj * (1 - self.d_zone_tolerance)
        d_upper = d_proj * (1 + self.d_zone_tolerance)

        self._shape_counter += 1
        shapes_list.append({
            "shape_id": f"{symbol}_{timeframe}_ABCD_{direction}_{self._shape_counter}",
            "direction": direction,
            "a_price": a[1], "b_price": b[1], "c_price": c_sw[1],
            "d_projected": d_proj, "d_lower": d_lower, "d_upper": d_upper,
            "bc_ab_ratio": round(bc_ab, 4), "quality_score": round(quality, 1),
            "ab_bars": ab_bars, "bc_bars": bc_bars,
            "ab_distance_pct": round(ab_len / b[1], 6),
            "c_bar": i,
            "in_zone": False, "entered": False,
        })

    @staticmethod
    def _in_zone(high_i, low_i, shape):
        if shape["direction"] == "bullish":
            return low_i <= shape["d_upper"] and high_i >= shape["d_lower"]
        return high_i >= shape["d_lower"] and low_i <= shape["d_upper"]
