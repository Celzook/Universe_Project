"""
Frozen interface contracts for momentum_funnel.

Do NOT modify after Phase 0. Any change requires Orchestrator approval.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol
import pandas as pd


# ── OHLCV 컬럼 계약 ────────────────────────────────────────────────
# 전부 소문자. DatetimeIndex(거래일, tz=None, KST 영업일 기준).
# 결측 ffill 금지(가격 위조 방지). volume은 거래대금(KRW) 우선, 좌수 fallback.
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ── MetricsFrame 스키마 ─────────────────────────────────────────────
# index = sector(str: 중카테고리 이름)
# valid: min_bars/NaN 워밍업 통과 여부. False면 트랙 단계에서 제외.
# rs: 벤치마크 대비 상대강도 (slope=log-RS 회귀 기울기, disparity=MA 이격도%).
# adx, adx_prev: 당일/전일 ADX(0~100).
# mfi: Money Flow Index(0~100).
METRICS_COLS = ["valid", "rs", "adx", "adx_prev", "mfi"]


@dataclass(frozen=True)
class MarketData:
    """데이터 레이어 → 지표 레이어 절단면."""
    sector_data: Dict[str, pd.DataFrame]      # {sector: DF with OHLCV_COLS}
    benchmark: pd.DataFrame                    # 'close' 컬럼 필수 (KOSPI)
    common_index: pd.DatetimeIndex             # 벤치마크 ∩ 전 섹터 거래일
    # 진단/추적용 메타. 어댑터가 채우고 UI는 참고만.
    meta: Dict[str, dict] = field(default_factory=dict)


class MetricsFrame:
    """METRICS_COLS 컬럼을 가진 DataFrame을 의미하는 마커.
    실 코드는 pd.DataFrame을 그대로 반환·소비한다(런타임 타입은 DataFrame)."""
    pass


class DataSource(Protocol):
    """모든 어댑터가 따르는 단일 인터페이스."""
    def load(self) -> MarketData: ...


@dataclass
class Decision:
    """깔때기 최종 산출."""
    monitor: pd.DataFrame    # Track A 통과 (모니터링 유니버스)
    entry: pd.DataFrame      # Track B 통과 (신규 진입 후보). B가 비면 C 폴백.
    score: pd.DataFrame      # Track C 전체 랭킹 (score 컬럼 내림차순)
    weights: pd.Series       # entry 종목별 비중 (합=1, 점수 비례)
    used_fallback: bool      # B가 비어 C로 대체했는가
    asof: Optional[pd.Timestamp] = None  # 시그널 산출 기준일
