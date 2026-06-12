"""
特征提取器 + 标签生成器

为每个 ABCD 形态在 D_zone 入场点提取特征，并计算交易标签。

标签定义：
  1 = 好交易（最终总 R ≥ 0）
  0 = 坏交易（最终总 R < 0）

退出规则：
  - 进场: 价格首次触及 D_zone 的 bar
  - TP1: C 点（50% 仓位）
  - 剩余 50%: BE(入场价) > TP2(A) > SL(X) > Timeout(2×|A-D| bars)
  - 同 bar 多事件 → SL 优先
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LabeledTrade:
    """一笔带标签的交易"""
    shape_id: str
    symbol: str
    timeframe: str
    direction: str
    # 入场信息
    entry_bar: int
    entry_price: float
    entry_time: pd.Timestamp
    # 退出价格
    c_price: float   # TP1 target
    a_price: float   # TP2 target
    sl_price: float  # SL level (D_zone edge + buffer)
    d_projected: float
    # 结果
    exit_reason: str      # 'TP1+TP2', 'TP1+BE', 'TP1+SL', 'TP1+Timeout', 'SL', 'Timeout'
    final_r: float        # 总 R 倍数
    label: int            # 1 or 0
    total_bars: int       # 持仓 bar 数
    # 分阶段细节
    tp1_hit_bar: Optional[int]
    tp1_hit_price: Optional[float]
    final_exit_bar: int
    final_exit_price: float


class FeatureLabelPipeline:
    """
    完整的特征提取 + 标签生成管线

    Parameters
    ----------
    d_zone_tolerance : float
        D_zone = D_projected × (1 ± tolerance)
    sl_buffer_atr : float
        SL = D_zone 边缘 + sl_buffer_atr × ATR
    timeout_mult : float
        超时 = |A-D| bars × timeout_mult
    tp1_pct : float
        TP1 平仓比例（默认 0.5 = 50%）
    atr_period : int
        ATR 计算周期
    """

    def __init__(
        self,
        d_zone_tolerance: float = 0.005,
        sl_buffer_atr: float = 0.3,
        timeout_mult: float = 2.0,
        tp1_pct: float = 0.5,
        atr_period: int = 14,
    ):
        self.d_zone_tolerance = d_zone_tolerance
        self.sl_buffer_atr = sl_buffer_atr
        self.timeout_mult = timeout_mult
        self.tp1_pct = tp1_pct
        self.atr_period = atr_period

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

    def _find_entry_bar(
        self,
        df: pd.DataFrame,
        c_idx: int,
        d_projected: float,
        direction: str,
        max_search_bars: int = 500,
    ) -> Optional[int]:
        """
        找到 D_zone 首次触及的 bar index
        限制搜索范围避免扫描全 dataset
        """
        d_lower = d_projected * (1 - self.d_zone_tolerance)
        d_upper = d_projected * (1 + self.d_zone_tolerance)

        end = min(c_idx + max_search_bars + 1, len(df))
        for i in range(c_idx + 1, end):
            low_i = df["low"].iloc[i]
            high_i = df["high"].iloc[i]

            if direction == "bullish":
                if low_i <= d_upper and high_i >= d_lower:
                    return i
            else:
                if high_i >= d_lower and low_i <= d_upper:
                    return i
        return None

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        entry_bar: int,
        entry_price: float,
        c_price: float,
        a_price: float,
        sl_price: float,
        direction: str,
        timeout_bars: int,
    ) -> LabeledTrade:
        """
        模拟一笔交易从进场到全部退出

        逻辑：
          - 先检查 TP1(C) 是否触及 → 部分平仓 50%
          - 剩余 50%: BE(入场) > TP2(A) > SL(X) > Timeout
          - 同 bar 内 TP+SL 都被触及时 → SL 优先（保守）
        """
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        tp1_hit_bar = None
        tp1_hit_price = None
        remaining = 1.0  # 剩余仓位比例
        exit_reason = "Timeout"
        final_exit_bar = entry_bar
        final_exit_price = entry_price

        tp2_price = a_price
        be_price = entry_price

        # 从下一根 bar 开始
        for i in range(entry_bar + 1, min(entry_bar + timeout_bars + 1, n)):
            high_i = high[i]
            low_i = low[i]
            bar_hit_tp1 = False
            bar_hit_sl = False
            bar_hit_tp2 = False
            bar_hit_be = False

            if direction == "bullish":
                # 看涨：价格向上触及目标
                if tp1_hit_bar is None and high_i >= c_price:
                    bar_hit_tp1 = True
                    tp1_hit_bar = i
                    tp1_hit_price = c_price
                if high_i >= tp2_price:
                    bar_hit_tp2 = True
                if high_i >= be_price:
                    bar_hit_be = True
                if low_i <= sl_price:
                    bar_hit_sl = True
            else:
                # 看跌：价格向下触及目标
                if tp1_hit_bar is None and low_i <= c_price:
                    bar_hit_tp1 = True
                    tp1_hit_bar = i
                    tp1_hit_price = c_price
                if low_i <= tp2_price:
                    bar_hit_tp2 = True
                if low_i <= be_price:
                    bar_hit_be = True
                if high_i >= sl_price:
                    bar_hit_sl = True

            # TP1 优先处理（第一次触及）
            if bar_hit_tp1 and tp1_hit_bar == i:
                remaining -= self.tp1_pct

            # 剩余仓位退出判断（保守：SL 优先于同 bar 其他事件）
            if remaining > 0:
                if bar_hit_sl:
                    exit_reason = "TP1+SL" if tp1_hit_bar is not None else "SL"
                    final_exit_bar = i
                    final_exit_price = sl_price
                    remaining = 0
                elif bar_hit_be and tp1_hit_bar is not None:
                    # BE 只在 TP1 之后才激活
                    exit_reason = "TP1+BE"
                    final_exit_bar = i
                    final_exit_price = be_price
                    remaining = 0
                elif bar_hit_tp2:
                    exit_reason = "TP1+TP2" if tp1_hit_bar is not None else "TP2"
                    final_exit_bar = i
                    final_exit_price = tp2_price
                    remaining = 0

            if remaining == 0:
                break

        # 超时退出
        if remaining > 0:
            final_exit_bar = min(entry_bar + timeout_bars, n - 1)
            final_exit_price = df["close"].iloc[final_exit_bar]
            exit_reason = "TP1+Timeout" if tp1_hit_bar is not None else "Timeout"
            remaining = 0

        # 计算 R
        if direction == "bullish":
            # 做多：价格上涨 = 赚钱
            tp1_r = 0
            if tp1_hit_bar is not None:
                tp1_r = (tp1_hit_price - entry_price) / abs(entry_price - sl_price) * self.tp1_pct

            remaining_pct = 1.0 - self.tp1_pct if tp1_hit_bar is not None else 1.0
            final_r_part = (final_exit_price - entry_price) / abs(entry_price - sl_price) * remaining_pct

            final_r = tp1_r + final_r_part
        else:
            # 做空：价格下跌 = 赚钱
            tp1_r = 0
            if tp1_hit_bar is not None:
                tp1_r = (entry_price - tp1_hit_price) / abs(sl_price - entry_price) * self.tp1_pct

            remaining_pct = 1.0 - self.tp1_pct if tp1_hit_bar is not None else 1.0
            final_r_part = (entry_price - final_exit_price) / abs(sl_price - entry_price) * remaining_pct

            final_r = tp1_r + final_r_part

        return LabeledTrade(
            shape_id="",
            symbol="",
            timeframe="",
            direction=direction,
            entry_bar=entry_bar,
            entry_price=entry_price,
            entry_time=df.index[entry_bar],
            c_price=c_price,
            a_price=a_price,
            sl_price=sl_price,
            d_projected=0,
            exit_reason=exit_reason,
            final_r=round(final_r, 4),
            label=1 if final_r >= 0 else 0,
            total_bars=final_exit_bar - entry_bar,
            tp1_hit_bar=tp1_hit_bar,
            tp1_hit_price=tp1_hit_price,
            final_exit_bar=final_exit_bar,
            final_exit_price=final_exit_price,
        )

    def process_shapes(
        self,
        df: pd.DataFrame,
        shapes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        处理一组形态，返回带标签和特征的 DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 数据（high, low, close, volume），index 为 datetime
        shapes_df : pd.DataFrame
            形态列表（来自 detector.to_dataframe()）

        Returns
        -------
        pd.DataFrame
            每行 = 一个交易样本，含特征 + 标签
        """
        if len(shapes_df) == 0:
            return pd.DataFrame()

        atr = self.compute_atr(
            df["high"].values, df["low"].values, df["close"].values
        )
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values
        opens = df["open"].values

        # 预计算 EMA（避免每形态重复算）
        ema20 = self._calc_ema(close, 20)
        ema50 = self._calc_ema(close, 50)

        results = []
        for _, shape in shapes_df.iterrows():
            c_idx = int(shape["c_idx"])
            d_proj = float(shape["d_projected"])
            direction = str(shape["direction"])
            a_price = float(shape["a_price"])
            c_price = float(shape["c_price"])

            # 1. 找入场点
            entry_bar = self._find_entry_bar(df, c_idx, d_proj, direction)
            if entry_bar is None:
                continue

            # 2. 计算 SL
            entry_atr = atr[min(entry_bar, len(atr) - 1)]
            if direction == "bullish":
                sl_price = d_proj * (1 - self.d_zone_tolerance) - self.sl_buffer_atr * entry_atr
            else:
                sl_price = d_proj * (1 + self.d_zone_tolerance) + self.sl_buffer_atr * entry_atr

            entry_price = d_proj
            ad_bars = abs(int(shape["a_idx"]) - int(shape["c_idx"])) + abs(int(shape["c_idx"]) - int(shape["b_idx"]))
            timeout_bars = int(ad_bars * self.timeout_mult)

            # 3. 模拟交易
            trade = self._simulate_trade(
                df, entry_bar, entry_price,
                c_price, a_price, sl_price,
                direction, timeout_bars,
            )

            trade.shape_id = str(shape["shape_id"])
            trade.symbol = str(shape["symbol"])
            trade.timeframe = str(shape["timeframe"])
            trade.d_projected = d_proj

            # 4. 提取特征（传入预计算数组）
            features = self._extract_features_fast(
                shape, entry_bar, entry_price, high, low, close, opens, volume, atr,
                ema20, ema50, df.index, len(df)
            )
            results.append({**trade.__dict__, **features})

        return pd.DataFrame(results)

    @staticmethod
    def _calc_ema(prices: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[:period] = np.mean(prices[:max(period, 1)])
        for i in range(max(period, 1), len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _extract_features_fast(
        self,
        shape: pd.Series,
        entry_bar: int,
        entry_price: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        opens: np.ndarray,
        volume: np.ndarray,
        atr: np.ndarray,
        ema20: np.ndarray,
        ema50: np.ndarray,
        index_arr,
        n: int,
    ) -> dict:
        """快速特征提取（使用预计算数组）"""
        features = {}

        # === 形态几何 ===
        features["ab_distance_pct"] = abs(float(shape["a_price"]) - float(shape["b_price"])) / float(shape["b_price"])
        features["bc_ab_ratio"] = float(shape["bc_ab_ratio"])
        features["quality_score"] = float(shape["quality_score"])
        features["ab_bars"] = int(shape["ab_bars"])
        features["bc_bars"] = int(shape["bc_bars"])
        features["fib_score"] = float(shape["fib_score"])
        features["price_symmetry_score"] = float(shape["price_symmetry_score"])
        features["time_symmetry_score"] = float(shape["time_symmetry_score"])

        # === D_zone 价格行为 ===
        window = 3
        start_i = max(0, entry_bar - window)
        end_i = min(n, entry_bar + window + 1)
        dzone_vol = volume[start_i:end_i]
        vol_ma20 = np.mean(volume[max(0, entry_bar - 20):entry_bar + 1]) if entry_bar >= 20 else np.mean(dzone_vol)
        features["d_zone_volume_ratio"] = float(np.mean(dzone_vol) / max(vol_ma20, 1e-10))

        if entry_bar >= 5:
            features["d_zone_momentum"] = float((close[entry_bar] - close[entry_bar - 5]) / max(close[entry_bar - 5], 1e-10))
        else:
            features["d_zone_momentum"] = 0.0

        body = abs(close[entry_bar] - opens[entry_bar])
        total_range = high[entry_bar] - low[entry_bar]
        features["d_zone_wick_ratio"] = float((total_range - body) / max(total_range, 1e-10))

        c_idx = int(shape["c_idx"])
        d_proj = float(shape["d_projected"])
        d_lower = d_proj * (1 - self.d_zone_tolerance)
        d_upper = d_proj * (1 + self.d_zone_tolerance)
        zone_bars = 0
        for i in range(c_idx + 1, entry_bar + 1):
            if low[i] <= d_upper and high[i] >= d_lower:
                zone_bars += 1
        features["d_zone_bars"] = zone_bars

        # === 波动率 ===
        features["atr_pct"] = float(atr[min(entry_bar, len(atr) - 1)] / entry_price)
        if entry_bar >= 20:
            atr_window = atr[entry_bar - 19:entry_bar + 1] / entry_price
            features["atr_percentile_20"] = float(np.percentile(atr_window, 80))
        else:
            features["atr_percentile_20"] = features["atr_pct"]

        # === 趋势 ===
        if entry_bar >= 14:
            features["ema20_50_direction"] = 1 if ema20[entry_bar] > ema50[entry_bar] else -1
            features["price_vs_ema20"] = float(close[entry_bar] / max(ema20[entry_bar], 1e-10) - 1)
        else:
            features["ema20_50_direction"] = 0
            features["price_vs_ema20"] = 0.0

        # === 成交量 ===
        if entry_bar >= 20:
            vol_20 = np.mean(volume[entry_bar - 19:entry_bar + 1])
            vol_5 = np.mean(volume[max(0, entry_bar - 4):entry_bar + 1])
            features["volume_ratio"] = float(volume[entry_bar] / max(vol_20, 1e-10))
            features["volume_trend"] = float(vol_5 / max(vol_20, 1e-10))
        else:
            features["volume_ratio"] = 1.0
            features["volume_trend"] = 1.0

        # === 时间 ===
        features["hour_of_day"] = index_arr[entry_bar].hour
        features["day_of_week"] = index_arr[entry_bar].dayofweek

        return features


def generate_dataset(
    shapes_csv: str,
    data_dir: str,
    output_path: str,
    d_zone_tolerance: float = 0.005,
    sl_buffer_atr: float = 0.3,
) -> pd.DataFrame:
    """
    一键生成训练数据集

    Parameters
    ----------
    shapes_csv : str
        scan_all_shapes.py 输出的形态 CSV
    data_dir : str
        OHLCV 数据目录
    output_path : str
        输出 CSV 路径
    """
    pipeline = FeatureLabelPipeline(
        d_zone_tolerance=d_zone_tolerance,
        sl_buffer_atr=sl_buffer_atr,
    )

    shapes_all = pd.read_csv(shapes_csv)
    symbol_tf_groups = shapes_all.groupby(["symbol", "timeframe"])

    all_rows = []
    stats = []

    for (sym, tf), group in symbol_tf_groups:
        parquet_path = f"{data_dir}/binance_{sym}_{tf}.parquet"
        df = pd.read_parquet(parquet_path)

        print(f"处理 {sym} {tf}: {len(group)} 形态...", flush=True)
        result = pipeline.process_shapes(df, group)

        if len(result) > 0:
            all_rows.append(result)
            stats.append({
                "symbol": sym, "tf": tf,
                "shapes": len(group),
                "traded": len(result),
                "pct": round(len(result) / len(group) * 100, 1),
                "win_rate": round(result["label"].mean() * 100, 1),
                "avg_r": round(result["final_r"].mean(), 3),
            })
            print(f"{len(result)} trades (WR={result['label'].mean()*100:.1f}%)", flush=True)
        else:
            print("0 笔（全部未触 D_zone）")

    # 合并保存
    if all_rows:
        dataset = pd.concat(all_rows, ignore_index=True)
        dataset.to_csv(output_path, index=False)

        # 统计报告
        print(f"\n{'='*60}")
        print(f"📊 数据集生成报告")
        print(f"{'='*60}")
        print(f"总形态: {shapes_all.shape[0]}")
        print(f"有效交易: {len(dataset)} ({len(dataset)/shapes_all.shape[0]*100:.1f}%)")
        print(f"跳过: {shapes_all.shape[0] - len(dataset)} (未触 D_zone)")
        print(f"总胜率: {dataset['label'].mean()*100:.1f}%")
        print(f"平均 R: {dataset['final_r'].mean():.3f}")
        print(f"\n分组统计:")
        for s in stats:
            print(f"  {s['symbol']:8s} {s['tf']:4s}: {s['shapes']:>5d} shapes → {s['traded']:>5d} trades "
                  f"({s['pct']:>5.1f}%)  WR={s['win_rate']:>5.1f}%  avgR={s['avg_r']:>6.3f}")

        return dataset
    return pd.DataFrame()


if __name__ == "__main__":
    generate_dataset(
        shapes_csv="outputs/all_shapes_2024_2026.csv",
        data_dir="/mnt/c/Users/12645/Sisie-Quantive/data",
        output_path="outputs/training_dataset.csv",
    )
