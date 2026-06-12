"""
Walk-Forward 回测引擎 v2

继承自 v1，追加：
  1. D 点反转确认 — 进入 D_zone 后等 1.5 ATR 反向才确认 D
  2. D→E 回踩入场 — D 确认后等回踩 0.382 Fib 入场
  3. 固定 ATR 风险 — 风险 = risk_atr_mult × ATR
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from collections import deque
from backtest.walk_forward import (
    WalkForwardEngine, ActiveShape, ActiveTrade, TradeResult
)


class WalkForwardEngineV2(WalkForwardEngine):
    """
    V2: 加入 D 点确认 + DE 回踩入场 + 固定 ATR 风险

    新增参数:
      confirm_atr_mult : D 点反转确认的 ATR 倍数（默认 1.5）
      entry_fib : DE 回踩入场 Fib 比例（0=即时入场）
      risk_atr_mult : 固定 ATR 风险倍数（0=使用 D_zone SL）
      max_confirm_bars : D_zone 触达后最多等多少 bar 确认 D
      max_entry_bars : D 确认后最多等多少 bar 回踩入场
    """

    def __init__(
        self,
        confirm_atr_mult: float = 1.5,
        entry_fib: float = 0.382,
        risk_atr_mult: float = 1.25,
        max_confirm_bars: int = 120,
        max_entry_bars: int = 24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confirm_atr_mult = confirm_atr_mult
        self.entry_fib = entry_fib
        self.risk_atr_mult = risk_atr_mult
        self.max_confirm_bars = max_confirm_bars
        self.max_entry_bars = max_entry_bars

    def run(self, df, symbol, timeframe):
        """覆盖 run()，使用 V2 的状态追踪和入场逻辑"""
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
        all_swings: List = []

        # V2: 活跃形态现在需要追踪 D 确认状态
        active_shapes: List[dict] = []
        self._v2_trades: List[ActiveTrade] = []
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
            adx_i = adx[i] if adx is not None else 0

            # --- Step 1: ZigZag ---
            if direction == 1:
                if high[i] > extreme_price:
                    extreme_price = high[i]; extreme_idx = i
                retreat = extreme_price - low[i]
                if retreat >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "high"))
                    all_swings.append((extreme_idx, extreme_price, "high"))
                    direction = -1; extreme_price = low[i]; extreme_idx = i
                    self._detect_all_shapes(confirmed_swings, i, atr, adx_i,
                                            symbol, timeframe, active_shapes)

            elif direction == -1:
                if low[i] < extreme_price:
                    extreme_price = low[i]; extreme_idx = i
                rally = high[i] - extreme_price
                if rally >= threshold and i > extreme_idx + 1:
                    confirmed_swings.append((extreme_idx, extreme_price, "low"))
                    all_swings.append((extreme_idx, extreme_price, "low"))
                    direction = 1; extreme_price = high[i]; extreme_idx = i
                    self._detect_all_shapes(confirmed_swings, i, atr, adx_i,
                                            symbol, timeframe, active_shapes)

            # --- Step 2: V2 — 追踪 D 确认 ---
            for shape in active_shapes:
                if shape.get("d_confirmed"):
                    continue
                if shape.get("in_zone", False):
                    self._track_d_extreme(shape, i, high, low)
                elif not shape.get("in_zone"):
                    if self._price_in_zone_v2(high[i], low[i], shape):
                        shape["in_zone"] = True
                        shape["zone_entry_bar"] = i
                        shape["d_extreme_price"] = low[i] if shape["direction"] == "bullish" else high[i]
                        shape["d_extreme_bar"] = i

                # 检查 D 确认
                if shape.get("in_zone") and not shape.get("d_confirmed"):
                    self._try_confirm_d(shape, i, atr, high, low)

            # --- Step 3: V2 — DE 回踩入场 ---
            for shape in active_shapes:
                if shape.get("trade_entered"):
                    continue
                if not shape.get("d_confirmed"):
                    continue

                # D 刚确认 → 记录 E 点
                if "e_price" not in shape:
                    shape["e_price"] = high[i] if shape["direction"] == "bullish" else low[i]
                    shape["e_bar"] = i

                # 尝试 DE 回踩入场
                self._try_de_entry(shape, i, high, low, atr, close, df, opens)

            # --- Step 4: 退出检查 ---
            still_active = []
            for trade in self._v2_trades:
                result = self._check_exit(trade, i, high, low, close, df)
                if result is not None:
                    self._enrich_features(result, i, high, low, close, opens,
                                          volume, atr, ema20, ema50, df, df.index)
                    result.symbol = symbol
                    result.timeframe = timeframe
                    if hasattr(trade.shape, 'pattern'):
                        result.pattern = trade.shape.pattern
                    completed_trades.append(result)
                else:
                    still_active.append(trade)
            self._v2_trades = still_active

            # 清理过期形态
            if i % 100 == 0:
                active_shapes = [s for s in active_shapes
                                 if not s.get("trade_entered") and not s.get("expired")]

        # 超时强制平仓
        for trade in self._v2_trades:
            final_bar = min(trade.timeout_bar, n - 1)
            result = self._force_close(trade, final_bar, close[final_bar], df)
            self._enrich_features(result, final_bar, high, low, close, opens,
                                  volume, atr, ema20, ema50, df, df.index)
            result.symbol = symbol
            result.timeframe = timeframe
            completed_trades.append(result)

        return pd.DataFrame([t.__dict__ for t in completed_trades])

    def _detect_all_shapes(self, swings, i, atr, adx_i, symbol, timeframe, active_shapes):
        """检测所有形态类型并添加到活跃列表"""
        # ABCD
        new_shape = self._try_detect_shape_v2(swings, i, atr, adx_i, symbol, timeframe, "ABCD")
        if new_shape:
            active_shapes.append(new_shape)

        # 多谐波
        multi = self._try_multi_patterns_v2(swings, i, atr, adx_i, symbol, timeframe)
        for ms in multi:
            active_shapes.append(ms)

    def _try_detect_shape_v2(self, swings, current_bar, atr, adx_i, symbol, timeframe, pattern_name="ABCD"):
        """V2 版形态检测，返回 dict 而非 ActiveShape"""
        if len(swings) < 3:
            return None
        a=swings[-3]; b=swings[-2]; c=swings[-1]

        if a[2]=="high" and b[2]=="low" and c[2]=="high" and c[1]<a[1]: direction="bullish"
        elif a[2]=="low" and b[2]=="high" and c[2]=="low" and c[1]>a[1]: direction="bearish"
        else: return None

        ab_len=abs(a[1]-b[1]); bc_len=abs(b[1]-c[1])
        if ab_len==0: return None
        bc_ab=bc_len/ab_len
        if not(self.bc_ab_min<=bc_ab<=self.bc_ab_max): return None
        b_atr=atr[min(b[0],len(atr)-1)]
        if ab_len<self.min_atr_mult*b_atr: return None

        if direction=="bullish": d_proj=c[1]-ab_len*self.cd_ab_ratio
        else: d_proj=c[1]+ab_len*self.cd_ab_ratio

        ab_bars=abs(b[0]-a[0]); bc_bars=abs(c[0]-b[0])
        quality=self._quality_score(bc_ab,ab_bars,bc_bars)
        if quality<self.min_quality_score: return None

        tp1_dist=abs(c[1]-d_proj)
        tp1_pct=tp1_dist/d_proj if d_proj>0 else 0
        if self.min_tp1_pct>0 and tp1_pct<self.min_tp1_pct: return None
        if self.max_tp1_pct<1.0 and tp1_pct>self.max_tp1_pct: return None
        if self.max_adx>0 and adx_i>self.max_adx: return None

        d_lower=d_proj*(1-self.d_zone_tolerance)
        d_upper=d_proj*(1+self.d_zone_tolerance)

        self._shape_counter+=1
        return {
            "shape_id":f"{symbol}_{timeframe}_{pattern_name}_{direction}_{self._shape_counter}",
            "pattern":pattern_name,"direction":direction,
            "a_price":a[1],"b_price":b[1],"c_price":c[1],
            "d_projected":d_proj,"d_lower":d_lower,"d_upper":d_upper,
            "bc_ab_ratio":round(bc_ab,4),"quality_score":round(quality,1),
            "ab_bars":ab_bars,"bc_bars":bc_bars,
            "ab_distance_pct":round(ab_len/b[1],6),
            "c_bar":current_bar,
            "in_zone":False,"d_confirmed":False,"trade_entered":False,
        }

    def _try_multi_patterns_v2(self, swings, current_bar, atr, adx_i, symbol, timeframe):
        """V2 版多谐波检测"""
        from abcd_detector.harmonic_patterns import detect_all_patterns
        if len(swings)<4: return []

        a=swings[-3]; b=swings[-2]; c=swings[-1]
        if a[2]=="high" and b[2]=="low" and c[2]=="high" and c[1]<a[1]: direction="bullish"
        elif a[2]=="low" and b[2]=="high" and c[2]=="low" and c[1]>a[1]: direction="bearish"
        else: return []

        ab_len=abs(a[1]-b[1]); bc_len=abs(b[1]-c[1])
        if ab_len==0: return []
        bc_ab=bc_len/ab_len
        cd_ab=self.cd_ab_ratio
        ab_bars=abs(b[0]-a[0]); bc_bars=abs(c[0]-b[0])
        ab_dist=ab_len/b[1]

        if direction=="bullish": d_proj_abcd=c[1]-ab_len*cd_ab
        else: d_proj_abcd=c[1]+ab_len*cd_ab

        detected=detect_all_patterns(list(swings),bc_ab,cd_ab,ab_bars,bc_bars,ab_dist,d_proj_abcd,direction)

        shapes=[]
        for hp in detected:
            if hp.quality_score<self.min_quality_score: continue
            b_atr=atr[min(b[0],len(atr)-1)]
            if ab_len<self.min_atr_mult*b_atr: continue

            tp1_dist=abs(c[1]-hp.d_projected)
            tp1_pct=tp1_dist/hp.d_projected if hp.d_projected>0 else 0
            if self.min_tp1_pct>0 and tp1_pct<self.min_tp1_pct: continue
            if self.max_tp1_pct<1.0 and tp1_pct>self.max_tp1_pct: continue
            if self.max_adx>0 and adx_i>self.max_adx: continue

            d_lower=hp.d_projected*(1-self.d_zone_tolerance)
            d_upper=hp.d_projected*(1+self.d_zone_tolerance)
            self._shape_counter+=1

            shapes.append({
                "shape_id":f"{symbol}_{timeframe}_{hp.pattern}_{direction}_{self._shape_counter}",
                "pattern":hp.pattern,"direction":direction,
                "a_price":a[1],"b_price":b[1],"c_price":c[1],
                "d_projected":hp.d_projected,"d_lower":d_lower,"d_upper":d_upper,
                "bc_ab_ratio":round(bc_ab,4),"quality_score":hp.quality_score,
                "ab_bars":ab_bars,"bc_bars":bc_bars,
                "ab_distance_pct":round(ab_dist,6),
                "c_bar":current_bar,
                "in_zone":False,"d_confirmed":False,"trade_entered":False,
            })
        return shapes

    @staticmethod
    def _price_in_zone_v2(high_i, low_i, shape):
        if shape["direction"]=="bullish":
            return low_i<=shape["d_upper"] and high_i>=shape["d_lower"]
        else:
            return high_i>=shape["d_lower"] and low_i<=shape["d_upper"]

    def _track_d_extreme(self, shape, i, high, low):
        """追踪 D_zone 内的极值"""
        if shape["direction"]=="bullish":
            if low[i]<shape["d_extreme_price"]:
                shape["d_extreme_price"]=low[i]
                shape["d_extreme_bar"]=i
        else:
            if high[i]>shape["d_extreme_price"]:
                shape["d_extreme_price"]=high[i]
                shape["d_extreme_bar"]=i

    def _try_confirm_d(self, shape, i, atr, high, low):
        """检查是否满足 D 点反转确认"""
        if i-shape["zone_entry_bar"]>self.max_confirm_bars:
            shape["expired"]=True; return

        reversal=self.confirm_atr_mult*atr[i]
        confirmed=False
        if shape["direction"]=="bullish":
            if high[i]>=shape["d_extreme_price"]+reversal:
                confirmed=True
        else:
            if low[i]<=shape["d_extreme_price"]-reversal:
                confirmed=True
        
        if confirmed:
            shape["d_confirmed"]=True
            shape["d_price"]=shape["d_extreme_price"]
            shape["d_bar"]=shape["d_extreme_bar"]
            shape["confirm_bar"]=i
            # 记录确认 bar 的 E 点（即时入场用 open，回踩入场用 extreme）
            shape["e_price"]=high[i] if shape["direction"]=="bullish" else low[i]

    def _try_de_entry(self, shape, i, high, low, atr, close, df, opens):
        """DE 回踩入场 或 即时入场"""
        bars_since_confirm = i - shape["confirm_bar"]
        
        if self.entry_fib == 0:
            # 即时入场模式：D 确认后下一 bar open 入场
            if bars_since_confirm == 0:
                return  # 确认 bar 不入场
            entry_price = opens[i]
            self._execute_entry(shape, i, entry_price, atr, close, df, opens)
            return
        
        # 回踩入场模式
        if bars_since_confirm > self.max_entry_bars:
            shape["trade_entered"] = True; return

        if shape["direction"] == "bullish":
            if high[i] > shape.get("e_price", 0):
                shape["e_price"] = high[i]; shape["e_bar"] = i
        else:
            if low[i] < shape.get("e_price", 0):
                shape["e_price"] = low[i]; shape["e_bar"] = i

        de_range = abs(shape["e_price"] - shape["d_price"])
        if shape["direction"] == "bullish":
            entry_price = shape["d_price"] + de_range * self.entry_fib
            if low[i] <= entry_price:
                self._execute_entry(shape, i, entry_price, atr, close, df, opens)
        else:
            entry_price = shape["d_price"] - de_range * self.entry_fib
            if high[i] >= entry_price:
                self._execute_entry(shape, i, entry_price, atr, close, df, opens)

    def _execute_entry(self, shape, i, entry_price, atr, close, df, opens):
        """执行入场 — SL = D点 + buffer, TP2 capped at 2R"""
        shape["trade_entered"]=True
        shape["entry_bar"]=i
        shape["entry_price"]=entry_price

        buffer=atr[min(i,len(atr)-1)]*self.sl_buffer_atr
        
        if shape["direction"]=="bullish":
            sl_price=shape["d_price"]-buffer
        else:
            sl_price=shape["d_price"]+buffer

        risk_amount=abs(entry_price-sl_price)
        if risk_amount<=0: return
        
        # TP2 capped at 2R from entry
        if shape["direction"]=="bullish":
            tp2_cap=entry_price+risk_amount*2.0
            tp2_price=min(shape["a_price"],tp2_cap)
        else:
            tp2_cap=entry_price-risk_amount*2.0
            tp2_price=max(shape["a_price"],tp2_cap)

        ad_bars=shape["ab_bars"]+shape["bc_bars"]
        timeout_bar=i+int(ad_bars*self.timeout_mult)

        # 创建 ActiveTrade（用旧的 ActiveShape 做桥梁）
        legacy_shape=ActiveShape(
            shape_id=shape["shape_id"],
            direction=shape["direction"],
            a_price=shape["a_price"],b_price=shape["b_price"],
            c_price=shape["c_price"],
            d_projected=shape["d_projected"],
            bc_ab_ratio=shape["bc_ab_ratio"],
            quality_score=shape["quality_score"],
            ab_bars=shape["ab_bars"],bc_bars=shape["bc_bars"],
            ab_distance_pct=shape["ab_distance_pct"],
            c_bar=shape["c_bar"],
            d_lower=shape["d_lower"],d_upper=shape["d_upper"],
        )

        trade=ActiveTrade(
            shape=legacy_shape,
            entry_bar=i,
            entry_price=entry_price,
            sl_price=sl_price,
            be_price=entry_price,
            tp2_price=shape["a_price"],
            timeout_bar=timeout_bar,
        )
        # V2 引擎的 active_trades 由 run() 管理
        # 用 _active_trades 属性传递
        if not hasattr(self,'_v2_trades'):
            self._v2_trades=[]
        self._v2_trades.append(trade)

    # 重写 _check_exit 以使用 self._v2_trades
    def _v2_check_exits(self, i, high, low, close, df, atr, ema20, ema50, opens, volume):
        """V2 退出检查，返回完成交易列表"""
        results=[]
        still_active=[]
        for trade in self._v2_trades:
            result=self._check_exit(trade,i,high,low,close,df)
            if result is not None:
                self._enrich_features(result,i,high,low,close,opens,volume,atr,ema20,ema50,df,df.index)
                results.append(result)
            else:
                still_active.append(trade)
        self._v2_trades=still_active
        return results
