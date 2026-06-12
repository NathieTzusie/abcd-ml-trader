"""
多谐波形态检测器

支持: ABCD, Gartley, Butterfly, Bat
基于 ATR-ZigZag 摆动点，walk-forward 兼容
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class HarmonicShape:
    """通用谐波形态"""
    pattern: str = "ABCD"
    direction: str = ""
    shape_id: str = ""
    # 摆动点价格
    a_price: float = 0.0
    b_price: float = 0.0
    c_price: float = 0.0
    d_projected: float = 0.0
    x_price: Optional[float] = None  # X 点（ABCD 没有）
    # 关键比率
    xa_ab_ratio: Optional[float] = None   # B/XA (Gartley/Butterfly/Bat)
    bc_ab_ratio: float = 0.0
    cd_ab_ratio: float = 1.0
    # 元数据
    quality_score: float = 0.0
    ab_bars: int = 0
    bc_bars: int = 0
    ab_distance_pct: float = 0.0


# ================================================================
# 各形态的 Fib 比率要求
# ================================================================

# AB=CD: 已经处理，这里定义其他形态

GARTLEY_RULES = {
    "xa_ab_min": 0.382,    # B retracement of XA
    "xa_ab_max": 0.618,
    "bc_ab_min": 0.382,    # C retracement of AB (same as ABCD)
    "bc_ab_max": 0.886,
    "d_xa": 0.786,         # D retracement of XA (primary)
    "d_xa_tol": 0.05,      # tolerance
}

BUTTERFLY_RULES = {
    "xa_ab_min": 0.618,
    "xa_ab_max": 0.886,    # B at 0.786 of XA (ideal)
    "bc_ab_min": 0.382,
    "bc_ab_max": 0.886,
    "d_xa_min": 1.272,     # D extends beyond X
    "d_xa_max": 1.618,
}

BAT_RULES = {
    "xa_ab_min": 0.382,
    "xa_ab_max": 0.500,
    "bc_ab_min": 0.382,
    "bc_ab_max": 0.886,
    "d_xa": 0.886,         # D at 0.886 of XA
    "d_xa_tol": 0.05,
}


def try_gartley(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                ab_bars, bc_bars, ab_dist_pct):
    """检测 Gartley 形态。D = 0.786 × XA（从 X 投影）"""
    if len(swings) < 4: return None
    x, a, b, c_sw = swings[-4], swings[-3], swings[-2], swings[-1]
    xa_len = abs(x[1] - a[1])
    if xa_len == 0: return None
    
    xa_ab = abs(a[1] - b[1]) / xa_len
    if not (0.382 <= xa_ab <= 0.618): return None
    
    # D 投影 = X + 0.786 × XA（Gartley 特征）
    if direction == "bullish":
        d_proj = x[1] - xa_len * 0.786  # X在上，D在下
    else:
        d_proj = x[1] + xa_len * 0.786
    
    # 验证 D 投影合理（在 X 和 C 之间）
    if direction == "bullish" and not (d_proj < c_sw[1]): return None
    if direction == "bearish" and not (d_proj > c_sw[1]): return None
    
    quality = (1 - min(abs(xa_ab - 0.618) / 0.618, 1)) * 50 + \
              (1 - min(abs(bc_ab - 0.618) / 0.618, 1)) * 50
    
    return HarmonicShape(
        pattern="Gartley", direction=direction,
        x_price=x[1], a_price=a[1], b_price=b[1], c_price=c_sw[1],
        d_projected=d_proj,
        xa_ab_ratio=round(xa_ab, 4),
        bc_ab_ratio=round(bc_ab, 4), cd_ab_ratio=round(cd_ab, 4),
        quality_score=round(quality, 1),
        ab_bars=ab_bars, bc_bars=bc_bars, ab_distance_pct=ab_dist_pct,
    )


def try_butterfly(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                  ab_bars, bc_bars, ab_dist_pct):
    """检测 Butterfly 形态。D = 1.272 或 1.618 × XA（延伸超越 X）"""
    if len(swings) < 4: return None
    x, a, b, c_sw = swings[-4], swings[-3], swings[-2], swings[-1]
    xa_len = abs(x[1] - a[1])
    if xa_len == 0: return None
    
    xa_ab = abs(a[1] - b[1]) / xa_len
    if not (0.618 <= xa_ab <= 0.886): return None
    
    # D 投影 = X + 1.272 × XA
    for d_mult in [1.272, 1.618]:
        if direction == "bullish":
            d_proj = x[1] - xa_len * d_mult
        else:
            d_proj = x[1] + xa_len * d_mult
        
        if direction == "bullish" and d_proj < c_sw[1]:
            quality = (1 - min(abs(xa_ab - 0.786) / 0.786, 1)) * 50 + \
                      (1 - min(abs(bc_ab - 0.618) / 0.618, 1)) * 50
            return HarmonicShape(
                pattern="Butterfly", direction=direction,
                x_price=x[1], a_price=a[1], b_price=b[1], c_price=c_sw[1],
                d_projected=d_proj, cd_ab_ratio=d_mult,
                xa_ab_ratio=round(xa_ab, 4),
                bc_ab_ratio=round(bc_ab, 4),
                quality_score=round(quality, 1),
                ab_bars=ab_bars, bc_bars=bc_bars, ab_distance_pct=ab_dist_pct,
            )
        elif direction == "bearish" and d_proj > c_sw[1]:
            quality = (1 - min(abs(xa_ab - 0.786) / 0.786, 1)) * 50 + \
                      (1 - min(abs(bc_ab - 0.618) / 0.618, 1)) * 50
            return HarmonicShape(
                pattern="Butterfly", direction=direction,
                x_price=x[1], a_price=a[1], b_price=b[1], c_price=c_sw[1],
                d_projected=d_proj, cd_ab_ratio=d_mult,
                xa_ab_ratio=round(xa_ab, 4),
                bc_ab_ratio=round(bc_ab, 4),
                quality_score=round(quality, 1),
                ab_bars=ab_bars, bc_bars=bc_bars, ab_distance_pct=ab_dist_pct,
            )
    return None


def try_bat(swings, d_proj_abcd, direction, bc_ab, cd_ab,
            ab_bars, bc_bars, ab_dist_pct):
    """检测 Bat 形态。D = 0.886 × XA"""
    if len(swings) < 4: return None
    x, a, b, c_sw = swings[-4], swings[-3], swings[-2], swings[-1]
    xa_len = abs(x[1] - a[1])
    if xa_len == 0: return None
    
    xa_ab = abs(a[1] - b[1]) / xa_len
    if not (0.382 <= xa_ab <= 0.5): return None
    
    if direction == "bullish":
        d_proj = x[1] - xa_len * 0.886
    else:
        d_proj = x[1] + xa_len * 0.886
    
    if direction == "bullish" and not (d_proj < c_sw[1]): return None
    if direction == "bearish" and not (d_proj > c_sw[1]): return None
    
    quality = (1 - min(abs(xa_ab - 0.5) / 0.5, 1)) * 40 + \
              (1 - min(abs(bc_ab - 0.618) / 0.618, 1)) * 30 + 30
    
    return HarmonicShape(
        pattern="Bat", direction=direction,
        x_price=x[1], a_price=a[1], b_price=b[1], c_price=c_sw[1],
        d_projected=d_proj,
        xa_ab_ratio=round(xa_ab, 4),
        bc_ab_ratio=round(bc_ab, 4), cd_ab_ratio=round(cd_ab, 4),
        quality_score=round(quality, 1),
        ab_bars=ab_bars, bc_bars=bc_bars, ab_distance_pct=ab_dist_pct,
    )


def try_crab(swings, d_proj_abcd, direction, bc_ab, cd_ab,
             ab_bars, bc_bars, ab_dist_pct):
    """检测 Crab 形态。B 在 0.382-0.618 XA，D = 1.618 XA"""
    if len(swings) < 4: return None
    x, a, b, c_sw = swings[-4], swings[-3], swings[-2], swings[-1]
    xa_len = abs(x[1] - a[1])
    if xa_len == 0: return None
    
    xa_ab = abs(a[1] - b[1]) / xa_len
    if not (0.382 <= xa_ab <= 0.618): return None
    
    if direction == "bullish":
        d_proj = x[1] - xa_len * 1.618
    else:
        d_proj = x[1] + xa_len * 1.618
    
    if direction == "bullish" and not (d_proj < c_sw[1]): return None
    if direction == "bearish" and not (d_proj > c_sw[1]): return None
    
    quality = (1 - min(abs(xa_ab - 0.5) / 0.5, 1)) * 40 + \
              (1 - min(abs(bc_ab - 0.618) / 0.618, 1)) * 30 + 30
    
    return HarmonicShape(
        pattern="Crab", direction=direction,
        x_price=x[1], a_price=a[1], b_price=b[1], c_price=c_sw[1],
        d_projected=d_proj, cd_ab_ratio=1.618,
        xa_ab_ratio=round(xa_ab, 4),
        bc_ab_ratio=round(bc_ab, 4),
        quality_score=round(quality, 1),
        ab_bars=ab_bars, bc_bars=bc_bars, ab_distance_pct=ab_dist_pct,
    )


def try_deep_crab(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                  ab_bars, bc_bars, ab_dist_pct):
    """检测 Deep Crab。B 在 0.886 XA，D = 1.618 XA"""
    if len(swings) < 4: return None
    x, a, b, c_sw = swings[-4], swings[-3], swings[-2], swings[-1]
    xa_len = abs(x[1] - a[1])
    if xa_len == 0: return None
    
    xa_ab = abs(a[1] - b[1]) / xa_len
    if not (0.80 <= xa_ab <= 0.92): return None
    
    if direction == "bullish":
        d_proj = x[1] - xa_len * 1.618
    else:
        d_proj = x[1] + xa_len * 1.618
    
    if direction == "bullish" and not (d_proj < c_sw[1]): return None
    if direction == "bearish" and not (d_proj > c_sw[1]): return None
    
    quality = (1 - min(abs(xa_ab - 0.886) / 0.886, 1)) * 50 + \
              (1 - min(abs(bc_ab - 0.618) / 0.618, 1)) * 50
    
    return HarmonicShape(
        pattern="DeepCrab", direction=direction,
        x_price=x[1], a_price=a[1], b_price=b[1], c_price=c_sw[1],
        d_projected=d_proj, cd_ab_ratio=1.618,
        xa_ab_ratio=round(xa_ab, 4),
        bc_ab_ratio=round(bc_ab, 4),
        quality_score=round(quality, 1),
        ab_bars=ab_bars, bc_bars=bc_bars, ab_distance_pct=ab_dist_pct,
    )


def detect_all_patterns(
    swings: List[Tuple[int, float, str]],
    bc_ab: float, cd_ab: float,
    ab_bars: int, bc_bars: int, ab_dist_pct: float,
    d_proj_abcd: float, direction: str,
) -> List[HarmonicShape]:
    """
    用同一组摆动点检测所有谐波形态
    每个摆动确认时调用
    """
    shapes = []

    # Gartley (需要 4+ 摆动点: X,A,B,C)
    g = try_gartley(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                    ab_bars, bc_bars, ab_dist_pct)
    if g:
        shapes.append(g)

    # Butterfly
    bf = try_butterfly(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                       ab_bars, bc_bars, ab_dist_pct)
    if bf:
        shapes.append(bf)

    # Bat
    bt = try_bat(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                 ab_bars, bc_bars, ab_dist_pct)
    if bt:
        shapes.append(bt)

    # Crab
    c = try_crab(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                 ab_bars, bc_bars, ab_dist_pct)
    if c:
        shapes.append(c)

    # Deep Crab
    dc = try_deep_crab(swings, d_proj_abcd, direction, bc_ab, cd_ab,
                       ab_bars, bc_bars, ab_dist_pct)
    if dc:
        shapes.append(dc)

    return shapes
