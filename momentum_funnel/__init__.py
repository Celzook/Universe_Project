"""
3-Step Momentum Funnel for Korean ETF universe.

섹터(=중카테고리)별 OHLCV → RS/ADX/MFI → Track A(와이드) / B(직렬) / C(스코어).
"""
from .contracts import MarketData, Decision, DataSource, OHLCV_COLS, METRICS_COLS
from .config import FunnelConfig
from .data_adapter import UniverseDataSource
from .pipeline import MomentumFunnel

__all__ = [
    "MarketData", "Decision", "DataSource",
    "OHLCV_COLS", "METRICS_COLS",
    "FunnelConfig",
    "UniverseDataSource",
    "MomentumFunnel",
]
