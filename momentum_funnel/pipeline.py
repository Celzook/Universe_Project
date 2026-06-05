"""
MomentumFunnel — DataSource.load() → compute_metrics() → combine().

단일 진입점: `MomentumFunnel(source, cfg).run(asof=None) -> Decision`
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

from .contracts import DataSource, Decision
from .config import FunnelConfig
from .indicators import compute_metrics
from .tracks import combine


class MomentumFunnel:
    def __init__(self, source: DataSource, cfg: Optional[FunnelConfig] = None):
        self.source = source
        self.cfg = cfg or FunnelConfig()

    def run(self, asof: Optional[pd.Timestamp] = None) -> Decision:
        market = self.source.load()
        metrics = compute_metrics(market, self.cfg, asof=asof)
        decision = combine(metrics, self.cfg, asof=asof)
        # asof가 명시 안 됐다면 공통 인덱스 최종일을 기록
        if decision.asof is None and len(market.common_index) > 0:
            decision.asof = market.common_index[-1]
        return decision

    def market_and_metrics(self, asof: Optional[pd.Timestamp] = None):
        """디버그/검증용 — UI 진단 패널에서 사용."""
        market = self.source.load()
        metrics = compute_metrics(market, self.cfg, asof=asof)
        return market, metrics
