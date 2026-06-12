"""
简化版回踩入场引擎

V1 核心逻辑不变，只改入场触发：
  D_zone 首次触及 → 跟踪最低点(看涨)/最高点(看跌)
  → 等价格反向走出 bounce_atr_mult × ATR → 入场
"""
import numpy as np
import pandas as pd
from backtest.walk_forward import WalkForwardEngine


class BounceEntryEngine(WalkForwardEngine):
    """
    在 V1 基础上，入场从「D_zone 触及即入场」改为「触及后等反弹再入场」

    参数:
      bounce_atr_mult : 反弹确认的 ATR 倍数（默认 0.5）
    """

    def __init__(self, bounce_atr_mult: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.bounce_atr_mult = bounce_atr_mult

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

        confirmed_swings = __import__('collections').deque(maxlen=10)
        all_swings = []

        active_shapes = []  # dict-based
        active_trades = []
        completed_trades = []

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
            adx_i = 0  # not using ADX in this test

            # --- Step 1: ZigZag ---
            if direction == 1:
                if high[i] > extreme_price:
                    extreme_price = high[i]; extreme_idx = i
                retreat = extreme_price - low[i]
                if retreat >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "high"))
                    all_swings.append((extreme_idx, extreme_price, "high"))
                    direction = -1; extreme_price = low[i]; extreme_idx = i
                    self._add_shapes(confirmed_swings, i, atr, adx_i, symbol, timeframe, active_shapes)
            elif direction == -1:
                if low[i] < extreme_price:
                    extreme_price = low[i]; extreme_idx = i
                rally = high[i] - extreme_price
                if rally >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "low"))
                    all_swings.append((extreme_idx, extreme_price, "low"))
                    direction = 1; extreme_price = high[i]; extreme_idx = i
                    self._add_shapes(confirmed_swings, i, atr, adx_i, symbol, timeframe, active_shapes)

            # --- Step 2: 追踪 D_zone + 反弹 ---
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
                        # 初始化极值
                        if shape["direction"] == "bullish":
                            shape["extreme_price"] = low[i]
                            shape["extreme_bar"] = i
                        else:
                            shape["extreme_price"] = high[i]
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

                # 检查反弹
                bounce_req = self.bounce_atr_mult * atr[i]
                bounced = False
                if shape["direction"] == "bullish":
                    if high[i] >= shape["extreme_price"] + bounce_req:
                        bounced = True
                else:
                    if low[i] <= shape["extreme_price"] - bounce_req:
                        bounced = True

                if bounced:
                    shape["entered"] = True
                    # 入场价 = 当前 bar open（模拟下一 bar 入场）
                    entry_price = opens[min(i + 1, n - 1)] if i + 1 < n else close[i]
                    entry_bar = min(i + 1, n - 1)

                    # SL = D_zone 下沿 - buffer
                    buffer = atr[min(i, len(atr) - 1)] * self.sl_buffer_atr
                    if shape["direction"] == "bullish":
                        sl_price = shape["d_lower"] - buffer
                    else:
                        sl_price = shape["d_upper"] + buffer

                    risk = abs(entry_price - sl_price)
                    if risk <= 0:
                        continue

                    ad_bars = abs(shape["a_idx"] if "a_idx" in shape else 0) + shape["bc_bars"]
                    timeout_bar = entry_bar + int(ad_bars * self.timeout_mult)

                    from backtest.walk_forward import ActiveShape, ActiveTrade
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
                    trade = __import__('backtest.walk_forward', fromlist=['ActiveTrade']).ActiveTrade(
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
                active_shapes = [s for s in active_shapes if not s.get("entered")]

        # 超时
        for trade in active_trades:
            fb = min(trade.timeout_bar, n - 1)
            r = self._force_close(trade, fb, close[fb], df)
            self._enrich_features(r, fb, high, low, close, opens, volume, atr, ema20, ema50, df, df.index)
            r.symbol = symbol; r.timeframe = timeframe
            completed_trades.append(r)

        return pd.DataFrame([t.__dict__ for t in completed_trades])

    def _add_shapes(self, swings, i, atr, adx_i, symbol, timeframe, active_shapes):
        """添加形态（复制自 V2 逻辑）"""
        if len(swings) < 3:
            return
        a = swings[-3]; b = swings[-2]; c = swings[-1]

        if a[2] == "high" and b[2] == "low" and c[2] == "high" and c[1] < a[1]:
            direction = "bullish"
        elif a[2] == "low" and b[2] == "high" and c[2] == "low" and c[1] > a[1]:
            direction = "bearish"
        else:
            return

        ab_len = abs(a[1] - b[1]); bc_len = abs(b[1] - c[1])
        if ab_len == 0:
            return
        bc_ab = bc_len / ab_len
        if not (self.bc_ab_min <= bc_ab <= self.bc_ab_max):
            return
        b_atr = atr[min(b[0], len(atr) - 1)]
        if ab_len < self.min_atr_mult * b_atr:
            return

        if direction == "bullish":
            d_proj = c[1] - ab_len * self.cd_ab_ratio
        else:
            d_proj = c[1] + ab_len * self.cd_ab_ratio

        ab_bars = abs(b[0] - a[0]); bc_bars = abs(c[0] - b[0])
        quality = self._quality_score(bc_ab, ab_bars, bc_bars)
        if quality < self.min_quality_score:
            return

        tp1_dist = abs(c[1] - d_proj)
        tp1_pct = tp1_dist / d_proj if d_proj > 0 else 0
        if self.min_tp1_pct > 0 and tp1_pct < self.min_tp1_pct:
            return

        d_lower = d_proj * (1 - self.d_zone_tolerance)
        d_upper = d_proj * (1 + self.d_zone_tolerance)

        self._shape_counter += 1
        active_shapes.append({
            "shape_id": f"{symbol}_{timeframe}_ABCD_{direction}_{self._shape_counter}",
            "direction": direction,
            "a_price": a[1], "b_price": b[1], "c_price": c[1],
            "d_projected": d_proj, "d_lower": d_lower, "d_upper": d_upper,
            "bc_ab_ratio": round(bc_ab, 4), "quality_score": round(quality, 1),
            "ab_bars": ab_bars, "bc_bars": bc_bars,
            "a_idx": a[0],
            "ab_distance_pct": round(ab_len / b[1], 6),
            "c_bar": i,
            "in_zone": False, "entered": False,
        })

    @staticmethod
    def _in_zone(high_i, low_i, shape):
        if shape["direction"] == "bullish":
            return low_i <= shape["d_upper"] and high_i >= shape["d_lower"]
        return high_i >= shape["d_lower"] and low_i <= shape["d_upper"]
