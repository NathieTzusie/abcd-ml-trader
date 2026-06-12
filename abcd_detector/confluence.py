"""
Confluence 检测器

Walk-forward 兼容：只使用当前 bar 之前的数据。
支持的 confluence 类型：
  1. 摆动点支撑/阻力 — D_zone 是否落在历史 swing high/low 附近
  2. 成交量节点 — D_zone 是否落在近期高成交量价格区
"""
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class ConfluenceDetector:
    """
    实时 confluence 检测器

    Parameters
    ----------
    sr_tolerance_pct : float
        S/R 容差（价格 %），0.003 = 0.3%
    sr_test_count : int
        价格被「测试」多少次才算有效 S/R 级别
    vol_bins : int
        成交量分布 bins 数量
    vol_lookback_bars : int
        成交量分布回看 bar 数
    """

    def __init__(
        self,
        sr_tolerance_pct: float = 0.003,
        sr_test_count: int = 2,
        vol_bins: int = 10,
        vol_lookback_bars: int = 500,
    ):
        self.sr_tolerance_pct = sr_tolerance_pct
        self.sr_test_count = sr_test_count
        self.vol_bins = vol_bins
        self.vol_lookback_bars = vol_lookback_bars

    def detect_swing_sr(
        self,
        swings: List[Tuple[int, float, str]],
        current_price: float,
    ) -> List[Dict]:
        """
        检测摆动点支撑/阻力

        逻辑：
          - 找出所有历史摆动点（high/low）
          - 聚类：价格相近的摆动点合并为一个「级别」
          - 被多次测试的级别 = 有效 S/R
          - 检查 D_zone 是否落在某个有效级别附近
        """
        if len(swings) < 4:
            return []

        # 收集摆动点价格
        highs = [s[1] for s in swings if s[2] == "high"]
        lows = [s[1] for s in swings if s[2] == "low"]

        # 简单聚类：按容差分组
        def cluster_prices(prices: List[float]) -> List[Dict]:
            if not prices:
                return []
            prices = sorted(prices)
            clusters = []
            current_cluster = [prices[0]]

            for p in prices[1:]:
                ref = current_cluster[0]
                if abs(p - ref) / ref < self.sr_tolerance_pct:
                    current_cluster.append(p)
                else:
                    if len(current_cluster) >= self.sr_test_count:
                        clusters.append({
                            "level": np.mean(current_cluster),
                            "count": len(current_cluster),
                        })
                    current_cluster = [p]

            if len(current_cluster) >= self.sr_test_count:
                clusters.append({
                    "level": np.mean(current_cluster),
                    "count": len(current_cluster),
                })
            return clusters

        sr_levels = cluster_prices(highs) + cluster_prices(lows)

        # 检查当前价格是否靠近任何 S/R 级别
        matches = []
        for sr in sr_levels:
            dist_pct = abs(current_price - sr["level"]) / sr["level"]
            if dist_pct < self.sr_tolerance_pct * 2:
                matches.append({
                    "type": "swing_sr",
                    "level": sr["level"],
                    "distance_pct": round(dist_pct * 100, 2),
                    "test_count": sr["count"],
                })

        return matches

    def detect_volume_node(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        current_bar: int,
        target_price: float,
    ) -> List[Dict]:
        """
        检测成交量节点

        逻辑：
          - 取最近 vol_lookback_bars 的价格数据
          - 分成 vol_bins 个价格区间
          - 每个区间的总成交量 = 该区间活跃度
          - 如果 D_zone 落在高成交量区间 = confluence
        """
        start = max(0, current_bar - self.vol_lookback_bars)
        if current_bar - start < 50:
            return []

        seg_high = high[start:current_bar + 1]
        seg_low = low[start:current_bar + 1]
        seg_close = close[start:current_bar + 1]
        seg_vol = volume[start:current_bar + 1]

        price_min = seg_low.min()
        price_max = seg_high.max()
        if price_max <= price_min:
            return []

        bin_edges = np.linspace(price_min, price_max, self.vol_bins + 1)
        vol_profile = np.zeros(self.vol_bins)

        # 简化：每根 bar 的成交量分配给它跨越的所有价格 bin
        for j in range(len(seg_close)):
            bar_low = seg_low[j]
            bar_high = seg_high[j]
            v = seg_vol[j]
            if bar_high <= bar_low:
                continue

            low_bin = np.searchsorted(bin_edges, bar_low) - 1
            high_bin = np.searchsorted(bin_edges, bar_high) - 1
            low_bin = max(0, min(low_bin, self.vol_bins - 1))
            high_bin = max(0, min(high_bin, self.vol_bins - 1))

            if low_bin == high_bin:
                vol_profile[low_bin] += v
            else:
                bins_spanned = high_bin - low_bin + 1
                for b in range(low_bin, high_bin + 1):
                    vol_profile[b] += v / bins_spanned

        # 找到成交量高于中位数的 bin
        median_vol = np.median(vol_profile[vol_profile > 0]) if np.any(vol_profile > 0) else 0
        if median_vol == 0:
            return []

        # 检查 target_price 落在哪个 bin
        target_bin = np.searchsorted(bin_edges, target_price) - 1
        target_bin = max(0, min(target_bin, self.vol_bins - 1))

        results = []
        if vol_profile[target_bin] > median_vol:
            results.append({
                "type": "volume_node",
                "vol_ratio": round(float(vol_profile[target_bin] / median_vol), 1),
                "bin_low": round(float(bin_edges[target_bin]), 2),
                "bin_high": round(float(bin_edges[target_bin + 1]), 2),
            })

        return results

    def check(
        self,
        d_projected: float,
        direction: str,
        current_bar: int,
        swings: List[Tuple[int, float, str]],
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict:
        """
        一站式 confluence 检查

        Returns
        -------
        dict with:
          has_confluence: bool
          sr_matches: List[Dict]
          vol_matches: List[Dict]
          score: int (0-2, confluence 类型数量)
        """
        sr_matches = self.detect_swing_sr(swings, d_projected)
        vol_matches = self.detect_volume_node(
            high, low, close, volume, current_bar, d_projected
        )

        return {
            "has_confluence": len(sr_matches) > 0 or len(vol_matches) > 0,
            "sr_matches": sr_matches,
            "vol_matches": vol_matches,
            "score": min(len(sr_matches), 1) + min(len(vol_matches), 1),
        }
