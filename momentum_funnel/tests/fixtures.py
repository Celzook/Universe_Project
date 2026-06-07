"""
공용 더미 생성기 (Phase 0 동결).

세 에이전트(D/I/F)가 상류 완성을 기다리지 않고 독립 테스트하기 위한 합성 데이터.

- make_market(n_days, sectors, drift, vol, seed): 합성 OHLCV MarketData
- make_metrics(sectors, valid_pct): 합성 MetricsFrame (METRICS_COLS)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional

from ..contracts import MarketData, OHLCV_COLS, METRICS_COLS


def _trading_index(n_days: int, end: Optional[pd.Timestamp] = None) -> pd.DatetimeIndex:
    end = end or pd.Timestamp.today().normalize()
    # end 가 주말이면 직전 영업일로 백트래킹 (pandas 2.x 에서 bdate_range 가
    # end=주말 인 경우 periods 보다 1 적게 반환하는 케이스 회피)
    while end.weekday() >= 5:  # Sat=5, Sun=6
        end -= pd.Timedelta(days=1)
    return pd.bdate_range(end=end, periods=n_days)


def _gbm_close(n: int, drift: float, vol: float, s0: float, rng: np.random.Generator) -> np.ndarray:
    """기하 브라운 운동으로 종가 경로."""
    daily_drift = drift / 252.0
    daily_vol = vol / np.sqrt(252.0)
    r = rng.normal(daily_drift, daily_vol, size=n)
    return s0 * np.exp(np.cumsum(r))


def _ohlcv_from_close(close: np.ndarray, vol_mu: float, rng: np.random.Generator) -> pd.DataFrame:
    """종가에서 합성 OHLV 생성. high>=max(o,c), low<=min(o,c) 보장."""
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.001, n - 1))
    span = np.abs(close - open_) + close * 0.003
    high = np.maximum(open_, close) + np.abs(rng.normal(0, span * 0.5))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, span * 0.5))
    volume = np.maximum(rng.lognormal(mean=np.log(vol_mu), sigma=0.3, size=n), 1.0)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def make_market(
    n_days: int = 120,
    sectors: Optional[Iterable[str]] = None,
    drift_map: Optional[Dict[str, float]] = None,
    vol_map: Optional[Dict[str, float]] = None,
    seed: int = 42,
    end: Optional[pd.Timestamp] = None,
) -> MarketData:
    """
    합성 MarketData. 기본: 강추세/중립/약세 3개 섹터 + 중립 벤치마크.

    drift_map: {sector: 연 drift} — 미지정 시 기본값 사용
    vol_map:   {sector: 연 vol}   — 미지정 시 기본값 사용
    """
    rng = np.random.default_rng(seed)
    idx = _trading_index(n_days, end=end)

    if sectors is None:
        sectors = ["반도체", "2차전지", "방어주", "약세섹터"]
    default_drift = {"반도체": 0.40, "2차전지": 0.20, "방어주": 0.05, "약세섹터": -0.25}
    default_vol = {"반도체": 0.28, "2차전지": 0.32, "방어주": 0.15, "약세섹터": 0.30}
    dmap = {**default_drift, **(drift_map or {})}
    vmap = {**default_vol, **(vol_map or {})}

    sector_data: Dict[str, pd.DataFrame] = {}
    for s in sectors:
        drift = dmap.get(s, 0.08)
        vol = vmap.get(s, 0.20)
        close = _gbm_close(n_days, drift, vol, s0=10000.0, rng=rng)
        df = _ohlcv_from_close(close, vol_mu=1e8, rng=rng)
        df.index = idx
        df.index.name = "date"
        sector_data[s] = df[OHLCV_COLS]

    # 벤치마크: 살짝 우상향(KOSPI 가정)
    bench_close = _gbm_close(n_days, drift=0.06, vol=0.16, s0=2500.0, rng=rng)
    bench = pd.DataFrame({"close": bench_close}, index=idx)

    return MarketData(
        sector_data=sector_data,
        benchmark=bench,
        common_index=idx,
        meta={s: {"source": "fixture"} for s in sectors},
    )


def make_metrics(
    sectors: Optional[Iterable[str]] = None,
    valid_pct: float = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    합성 MetricsFrame. 트랙 단계 단독 테스트용.

    valid_pct: 0~1, 무작위로 일부를 valid=False 처리.
    """
    rng = np.random.default_rng(seed)
    if sectors is None:
        sectors = [f"섹터{i:02d}" for i in range(10)]
    sectors = list(sectors)
    n = len(sectors)

    df = pd.DataFrame(index=pd.Index(sectors, name="sector"), columns=METRICS_COLS)
    df["rs"] = rng.normal(0.0, 0.002, n)            # slope 기준
    df["adx"] = rng.uniform(10, 45, n)
    df["adx_prev"] = df["adx"] + rng.normal(0, 2.0, n)
    df["mfi"] = rng.uniform(20, 90, n)
    valid_mask = rng.random(n) < valid_pct
    df["valid"] = valid_mask
    return df
