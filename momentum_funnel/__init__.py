"""
3-Step Momentum Funnel for Korean ETF universe.

섹터(=중카테고리)별 OHLCV → RS/ADX/MFI → Track A(와이드) / B(직렬) / C(스코어).
"""
from .contracts import MarketData, Decision, DataSource, OHLCV_COLS, METRICS_COLS
from .config import FunnelConfig
from .data_adapter import UniverseDataSource
from .pipeline import MomentumFunnel
from .hot_board import (
    compute_hot_metrics, build_mp,
    EXCLUDED_CATEGORIES_DEFAULT,
    CORE_TICKER_DEFAULT, CORE_NAME_DEFAULT,
    HOT_COLS,
)
from .portfolio_tracker import (
    save_mp_local, load_mp, delete_mp,
    push_to_github, compute_mp_performance,
    load_history, append_history_snapshot, push_history_to_github,
)
from .rebalancer import (
    apply_rules, PositionState,
    INITIAL_CAPITAL_DEFAULT,
    BM_50PCT_THRESHOLD, BM_FULL_THRESHOLD,
    MDD_50PCT_THRESHOLD, MDD_FULL_THRESHOLD,
)
from .snapshot import build_snapshot

__all__ = [
    "MarketData", "Decision", "DataSource",
    "OHLCV_COLS", "METRICS_COLS",
    "FunnelConfig",
    "UniverseDataSource",
    "MomentumFunnel",
    "compute_hot_metrics", "build_mp",
    "EXCLUDED_CATEGORIES_DEFAULT",
    "CORE_TICKER_DEFAULT", "CORE_NAME_DEFAULT",
    "HOT_COLS",
    "save_mp_local", "load_mp", "delete_mp",
    "push_to_github", "compute_mp_performance",
    "load_history", "append_history_snapshot", "push_history_to_github",
    "apply_rules", "PositionState",
    "INITIAL_CAPITAL_DEFAULT",
    "BM_50PCT_THRESHOLD", "BM_FULL_THRESHOLD",
    "MDD_50PCT_THRESHOLD", "MDD_FULL_THRESHOLD",
    "build_snapshot",
]
