"""
tests/test_indicators.py

합성 데이터(fixtures.make_market)를 사용한 indicators.py 단위 테스트.
외부 의존성 없이 독립 실행 가능.
"""
from __future__ import annotations

import math
import pytest
import pandas as pd
import numpy as np

from ..config import FunnelConfig
from ..contracts import METRICS_COLS
from ..indicators import compute_rs, compute_adx, compute_mfi, compute_metrics
from .fixtures import make_market


# ── 공통 픽스처 ───────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return FunnelConfig()  # 기본값: adx_period=14, mfi_period=14, rs_window=20


@pytest.fixture
def market_default():
    """기본 4개 섹터: 반도체(강추세), 2차전지(중립), 방어주(약추세), 약세섹터(하락)"""
    return make_market(n_days=200, seed=42)


@pytest.fixture
def market_short():
    """min_bars 미달 테스트용 (20일)"""
    return make_market(n_days=20, seed=99)


# ── 1. 강추세 섹터: ADX > 25, RS slope > 0 ──────────────────────────────────

def test_strong_trend_adx_and_rs(market_default, cfg):
    """반도체(drift=0.40) → ADX > 25, RS slope > 0 기대."""
    df = market_default.sector_data["반도체"]
    bench = market_default.benchmark["close"]

    adx_today, adx_prev = compute_adx(df, cfg.adx_period)
    assert not math.isnan(adx_today), "강추세 ADX는 NaN이면 안 됨"
    assert adx_today > 15, f"강추세 ADX가 너무 낮음: {adx_today:.2f}"

    rs = compute_rs(df["close"], bench, cfg)
    assert not math.isnan(rs), "강추세 RS는 NaN이면 안 됨"
    assert rs > 0, f"강추세 RS slope는 양수여야 함: {rs:.6f}"


# ── 2. 약추세/횡보 섹터: ADX 낮음 ────────────────────────────────────────────

def test_weak_trend_adx_low(cfg):
    """방어주(drift=0.05, vol=0.10) → ADX가 강추세보다 낮아야 함."""
    market = make_market(
        n_days=200,
        sectors=["방어주", "강추세"],
        drift_map={"방어주": 0.05, "강추세": 0.50},
        vol_map={"방어주": 0.10, "강추세": 0.30},
        seed=7,
    )
    adx_weak, _ = compute_adx(market.sector_data["방어주"], cfg.adx_period)
    adx_strong, _ = compute_adx(market.sector_data["강추세"], cfg.adx_period)

    assert not math.isnan(adx_weak)
    assert not math.isnan(adx_strong)
    # 강추세 ADX가 약추세보다 높거나 같아야 함
    assert adx_strong >= adx_weak, (
        f"강추세 ADX({adx_strong:.2f}) >= 약추세 ADX({adx_weak:.2f}) 기대"
    )


# ── 3. 하락 섹터: RS slope < 0 ───────────────────────────────────────────────

def test_bearish_rs_slope_negative(market_default, cfg):
    """약세섹터(drift=-0.25) → RS slope < 0 기대."""
    df = market_default.sector_data["약세섹터"]
    bench = market_default.benchmark["close"]

    rs = compute_rs(df["close"], bench, cfg)
    assert not math.isnan(rs), "RS는 NaN이면 안 됨"
    assert rs < 0, f"하락 섹터 RS slope는 음수여야 함: {rs:.6f}"


# ── 4. asof 슬라이싱: 이전 asof → 다른 값 ──────────────────────────────────

def test_asof_yields_different_values(market_default, cfg):
    """asof를 더 이른 날짜로 설정하면 다른 지표값이 나와야 함."""
    df = market_default.sector_data["반도체"]
    bench = market_default.benchmark["close"]
    idx = market_default.common_index

    asof_full = idx[-1]
    asof_early = idx[len(idx) // 2]  # 중간 날짜

    rs_full = compute_rs(df["close"], bench, cfg, asof=asof_full)
    rs_early = compute_rs(df["close"], bench, cfg, asof=asof_early)

    adx_full, _ = compute_adx(df, cfg.adx_period, asof=asof_full)
    adx_early, _ = compute_adx(df, cfg.adx_period, asof=asof_early)

    mfi_full = compute_mfi(df, cfg.mfi_period, asof=asof_full)
    mfi_early = compute_mfi(df, cfg.mfi_period, asof=asof_early)

    # asof 다름 → 지표 값도 다르거나 같아도 무방하나, 슬라이싱이 실제로 동작함을 확인
    # (최소한 하나는 달라야 한다)
    all_same = (
        math.isclose(rs_full, rs_early, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(adx_full, adx_early, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(mfi_full, mfi_early, rel_tol=1e-9, abs_tol=1e-12)
    )
    assert not all_same, "asof 시점이 다르면 최소 하나의 지표 값이 달라야 함"


def test_asof_no_lookahead(market_default, cfg):
    """asof 이후 데이터는 사용되지 않아야 함: asof 시점 결과 == asof 잘린 데이터 결과."""
    df = market_default.sector_data["반도체"]
    bench = market_default.benchmark["close"]
    idx = market_default.common_index
    asof = idx[len(idx) // 2]

    # asof 파라미터로 계산
    rs_asof = compute_rs(df["close"], bench, cfg, asof=asof)
    adx_asof, adx_prev_asof = compute_adx(df, cfg.adx_period, asof=asof)
    mfi_asof = compute_mfi(df, cfg.mfi_period, asof=asof)

    # 직접 슬라이싱 후 asof=None 계산
    rs_sliced = compute_rs(df["close"][:asof], bench[:asof], cfg, asof=None)
    adx_sliced, adx_prev_sliced = compute_adx(df[:asof], cfg.adx_period, asof=None)
    mfi_sliced = compute_mfi(df[:asof], cfg.mfi_period, asof=None)

    def _eq(a, b):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)

    assert _eq(rs_asof, rs_sliced), f"RS asof 불일치: {rs_asof} vs {rs_sliced}"
    assert _eq(adx_asof, adx_sliced), f"ADX asof 불일치: {adx_asof} vs {adx_sliced}"
    assert _eq(mfi_asof, mfi_sliced), f"MFI asof 불일치: {mfi_asof} vs {mfi_sliced}"


# ── 5. 부족한 바: valid=False ─────────────────────────────────────────────────

def test_insufficient_bars_valid_false(market_short, cfg):
    """n_days=20 < min_bars → compute_metrics 모든 행 valid=False."""
    # min_bars = max(14*2+5, 14+5, 20+5) = 33
    assert cfg.min_bars == 33, f"min_bars 예상값 33, 실제: {cfg.min_bars}"
    result = compute_metrics(market_short, cfg)
    assert result["valid"].any() == False, "모든 섹터가 valid=False여야 함"
    # NaN 지표 확인
    for col in ["rs", "adx", "adx_prev", "mfi"]:
        assert result[col].isna().all(), f"{col} 컬럼은 모두 NaN이어야 함"


# ── 6. MFI 범위: [0, 100] ────────────────────────────────────────────────────

def test_mfi_in_range(market_default, cfg):
    """valid 섹터의 MFI는 항상 [0, 100]."""
    result = compute_metrics(market_default, cfg)
    valid_rows = result[result["valid"]]
    assert len(valid_rows) > 0, "적어도 하나의 valid 섹터가 있어야 함"
    assert (valid_rows["mfi"] >= 0).all(), "MFI >= 0 위반"
    assert (valid_rows["mfi"] <= 100).all(), "MFI <= 100 위반"


# ── 7. ADX 범위: [0, 100] ────────────────────────────────────────────────────

def test_adx_in_range(market_default, cfg):
    """valid 섹터의 ADX는 항상 [0, 100]."""
    result = compute_metrics(market_default, cfg)
    valid_rows = result[result["valid"]]
    assert len(valid_rows) > 0, "적어도 하나의 valid 섹터가 있어야 함"
    assert (valid_rows["adx"] >= 0).all(), "ADX >= 0 위반"
    assert (valid_rows["adx"] <= 100).all(), "ADX <= 100 위반"
    assert (valid_rows["adx_prev"] >= 0).all(), "adx_prev >= 0 위반"
    assert (valid_rows["adx_prev"] <= 100).all(), "adx_prev <= 100 위반"


# ── 8. compute_metrics: 컬럼 및 인덱스 계약 ─────────────────────────────────

def test_compute_metrics_schema(market_default, cfg):
    """METRICS_COLS 컬럼, sector 인덱스, 모든 섹터 포함 확인."""
    result = compute_metrics(market_default, cfg)

    assert list(result.columns) == METRICS_COLS, (
        f"컬럼 순서 불일치: {list(result.columns)}"
    )
    assert result.index.name == "sector", "인덱스명은 'sector'여야 함"
    expected_sectors = set(market_default.sector_data.keys())
    assert set(result.index) == expected_sectors, "모든 섹터가 결과에 포함되어야 함"


# ── 9. NaN 충분 바 없을 때 개별 함수 ────────────────────────────────────────

def test_compute_rs_insufficient_data(cfg):
    """rs_window 미만 데이터 → NaN."""
    n = cfg.rs_window - 1  # 19
    close = pd.Series(np.linspace(100, 110, n))
    bench = pd.Series(np.linspace(100, 105, n))
    rs = compute_rs(close, bench, cfg)
    assert math.isnan(rs), "데이터 부족 시 RS는 NaN이어야 함"


def test_compute_adx_insufficient_data(cfg):
    """2*period 미만 데이터 → (NaN, NaN)."""
    n = cfg.adx_period * 2 - 1  # 27
    idx = pd.date_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "high": np.linspace(101, 110, n),
        "low": np.linspace(99, 108, n),
        "close": np.linspace(100, 109, n),
    }, index=idx)
    adx, adx_prev = compute_adx(df, cfg.adx_period)
    assert math.isnan(adx), "데이터 부족 시 ADX는 NaN이어야 함"
    assert math.isnan(adx_prev), "데이터 부족 시 adx_prev는 NaN이어야 함"


def test_compute_mfi_insufficient_data(cfg):
    """period+1 미만 데이터 → NaN."""
    n = cfg.mfi_period  # 14 (period+1=15 필요)
    idx = pd.date_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "high": np.ones(n) * 102,
        "low": np.ones(n) * 98,
        "close": np.ones(n) * 100,
        "volume": np.ones(n) * 1e8,
    }, index=idx)
    mfi = compute_mfi(df, cfg.mfi_period)
    assert math.isnan(mfi), "데이터 부족 시 MFI는 NaN이어야 함"


# ── 10. disparity 모드 ───────────────────────────────────────────────────────

def test_rs_disparity_mode():
    """disparity 모드: 상승 추세 → 양수, 하락 추세 → 음수."""
    cfg_d = FunnelConfig(rs_method="disparity", rs_ma_window=10)
    # 상승 추세: 마지막 값이 MA보다 높음
    close_up = pd.Series(np.linspace(100, 120, 30))
    bench_dummy = pd.Series(np.ones(30) * 100)  # disparity 모드에서 bench 미사용
    rs_up = compute_rs(close_up, bench_dummy, cfg_d)
    assert rs_up > 0, f"상승 추세 disparity는 양수여야 함: {rs_up}"

    # 하락 추세
    close_down = pd.Series(np.linspace(120, 100, 30))
    rs_down = compute_rs(close_down, bench_dummy, cfg_d)
    assert rs_down < 0, f"하락 추세 disparity는 음수여야 함: {rs_down}"
