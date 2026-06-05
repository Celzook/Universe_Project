"""
momentum_funnel/tests/test_adapter.py — UniverseDataSource 단위 테스트

실제 네트워크 호출 없음. naver_get_ohlcv_history / naver_get_index_history 를
unittest.mock 으로 대체.
"""
from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import patch
from typing import Dict

import numpy as np
import pandas as pd

# 프로젝트 루트를 sys.path 에 추가 (pytest 실행 위치 무관하게 임포트 보장)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from momentum_funnel.contracts import OHLCV_COLS, MarketData
from momentum_funnel.config import FunnelConfig
from momentum_funnel.data_adapter import UniverseDataSource, naver_get_ohlcv_history


# ══════════════════════════════════════════════════════════════════════════════
# 헬퍼: 합성 OHLCV / 벤치마크 생성
# ══════════════════════════════════════════════════════════════════════════════

def _make_ohlcv(n: int, base_close: float = 10000.0,
                seed: int = 0, idx: pd.DatetimeIndex = None) -> pd.DataFrame:
    """단순 합성 OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    if idx is None:
        idx = pd.bdate_range(end='2024-06-28', periods=n)  # 금요일 기준 (bdate 경계 안전)
    # idx 길이에 맞춰 배열 생성
    m = len(idx)
    close = base_close * np.cumprod(1 + rng.normal(0, 0.01, m))
    open_ = close * np.exp(rng.normal(0, 0.001, m))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, m)))
    low  = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, m)))
    volume = rng.lognormal(10, 0.5, m)
    df = pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume},
        index=idx,
    )
    df.index.name = 'date'
    return df[OHLCV_COLS]


def _make_bench_series(n: int, idx: pd.DatetimeIndex = None) -> pd.Series:
    """합성 KOSPI 종가 Series."""
    rng = np.random.default_rng(99)
    if idx is None:
        idx = pd.bdate_range(end='2024-06-28', periods=n)  # 금요일 기준
    m = len(idx)
    close = 2500.0 * np.cumprod(1 + rng.normal(0, 0.008, m))
    return pd.Series(close, index=idx, name='close')


def _make_universe(sectors: Dict[str, list],
                   caps: Dict[str, float] = None) -> pd.DataFrame:
    """
    최소한의 df_universe 생성.
    sectors = {'중카테고리명': [티커, ...]}
    caps    = {티커: 시총(억원)}
    """
    rows = []
    for cat, tickers in sectors.items():
        for t in tickers:
            rows.append({
                '중카테고리': cat,
                '시가총액(억원)': (caps or {}).get(t, 100.0),
            })
    tickers_all = [t for ts in sectors.values() for t in ts]
    df = pd.DataFrame(rows, index=pd.Index(tickers_all, name='ticker'))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 공통 픽스처 날짜 인덱스
# ══════════════════════════════════════════════════════════════════════════════
N = 60  # 충분한 바 수 (FunnelConfig.min_bars = 33)
# 금요일(2024-06-28) 기준 — bdate_range end 가 주말이면 실제 periods 가 줄어드는 문제 방지
IDX_FULL  = pd.bdate_range(end='2024-06-28', periods=N)
IDX_SHORT = pd.bdate_range(end='2024-06-28', periods=10)  # min_bars 미달


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 1: 스키마 준수 — OHLCV_COLS 소문자 + benchmark 'close'
# ══════════════════════════════════════════════════════════════════════════════
class TestSchemaCompliance(unittest.TestCase):
    def test_sector_ohlcv_cols_lowercase(self):
        """sector_data 의 각 DataFrame 컬럼이 OHLCV_COLS(소문자)여야 한다."""
        df_univ = _make_universe({'반도체': ['069500', '091160']})
        ohlcv_A = _make_ohlcv(N, idx=IDX_FULL, seed=1)
        ohlcv_B = _make_ohlcv(N, idx=IDX_FULL, seed=2)
        bench   = _make_bench_series(N, idx=IDX_FULL)

        def mock_ohlcv(ticker, *args, **kwargs):
            return ohlcv_A if ticker == '069500' else ohlcv_B

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', side_effect=mock_ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, use_yf_fallback=False)
            md = src.load()

        self.assertIn('반도체', md.sector_data)
        for col in OHLCV_COLS:
            self.assertIn(col, md.sector_data['반도체'].columns,
                          f"컬럼 '{col}' 누락")
        # 대문자 컬럼이 없어야 함
        for col in md.sector_data['반도체'].columns:
            self.assertEqual(col, col.lower(), f"소문자 아닌 컬럼: {col}")

    def test_benchmark_has_close_column(self):
        """benchmark DataFrame 에 'close' 컬럼이 있어야 한다."""
        df_univ = _make_universe({'채권': ['114260']})
        ohlcv   = _make_ohlcv(N, idx=IDX_FULL)
        bench   = _make_bench_series(N, idx=IDX_FULL)

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', return_value=ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, use_yf_fallback=False)
            md = src.load()

        self.assertIn('close', md.benchmark.columns)

    def test_common_index_is_datetimeindex(self):
        """common_index 가 DatetimeIndex 여야 한다."""
        df_univ = _make_universe({'채권': ['114260']})
        ohlcv   = _make_ohlcv(N, idx=IDX_FULL)
        bench   = _make_bench_series(N, idx=IDX_FULL)

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', return_value=ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, use_yf_fallback=False)
            md = src.load()

        self.assertIsInstance(md.common_index, pd.DatetimeIndex)


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 2: 거래일 교집합 — 한 섹터에 gap 이 있으면 해당 날짜 제외
# ══════════════════════════════════════════════════════════════════════════════
class TestTradingDayIntersection(unittest.TestCase):
    def test_gap_date_excluded_from_common_index(self):
        """섹터 A 에 없는 날짜는 common_index 에서 제외되어야 한다."""
        # IDX_FULL 에서 마지막 5일 제거 → 섹터B 는 짧은 인덱스
        idx_a = IDX_FULL               # 60 영업일
        idx_b = IDX_FULL[:-5]          # 55 영업일 (마지막 5일 없음)
        df_univ = _make_universe({'반도체': ['069500'], '2차전지': ['305720']})

        ohlcv_a = _make_ohlcv(N, idx=idx_a, seed=10)
        ohlcv_b = _make_ohlcv(len(idx_b), idx=idx_b, seed=11)
        bench   = _make_bench_series(N, idx=idx_a)

        def mock_ohlcv(ticker, *args, **kwargs):
            return ohlcv_a if ticker == '069500' else ohlcv_b

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', side_effect=mock_ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, use_yf_fallback=False)
            md = src.load()

        # common_index 는 idx_b 기준이어야 함
        self.assertTrue(len(md.common_index) <= len(idx_b))
        # idx_a 에만 있는 마지막 날짜들이 포함되지 않아야 함
        excluded_dates = set(idx_a[-5:]) - set(idx_b)
        for d in excluded_dates:
            self.assertNotIn(d, md.common_index,
                             f"날짜 {d} 가 common_index 에 잘못 포함됨")


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 3: asof 파라미터 — 해당 날짜 이하만 사용
# ══════════════════════════════════════════════════════════════════════════════
class TestAsofParameter(unittest.TestCase):
    def test_asof_truncates_all_data(self):
        """asof 로 지정한 날짜 이후 데이터가 포함되지 않아야 한다."""
        df_univ = _make_universe({'반도체': ['069500']})
        ohlcv   = _make_ohlcv(N, idx=IDX_FULL)
        bench   = _make_bench_series(N, idx=IDX_FULL)
        asof_ts = IDX_FULL[N // 2]  # 중간 날짜

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', return_value=ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, asof=asof_ts, use_yf_fallback=False)
            md = src.load()

        if not md.common_index.empty:
            self.assertLessEqual(md.common_index.max(), asof_ts,
                                 "asof 이후 날짜가 common_index 에 포함됨")
        if not md.benchmark.empty:
            self.assertLessEqual(md.benchmark.index.max(), asof_ts,
                                 "asof 이후 날짜가 benchmark 에 포함됨")

    def test_asof_no_lookahead(self):
        """asof 이후 날짜는 sector_data 에도 없어야 한다."""
        df_univ = _make_universe({'채권': ['114260']})
        ohlcv   = _make_ohlcv(N, idx=IDX_FULL)
        bench   = _make_bench_series(N, idx=IDX_FULL)
        asof_ts = IDX_FULL[N // 3]

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', return_value=ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, asof=asof_ts, use_yf_fallback=False)
            md = src.load()

        for sector, df in md.sector_data.items():
            if not df.empty:
                self.assertLessEqual(df.index.max(), asof_ts,
                                     f"{sector} 섹터 데이터에 asof 이후 날짜 포함")


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 4: 시총 가중 집계 — 80/20 비중 검증
# ══════════════════════════════════════════════════════════════════════════════
class TestCapWeightedAggregation(unittest.TestCase):
    def test_cap_weighted_close(self):
        """
        티커A cap=80, 티커B cap=20 일 때
        sector_close ≈ 0.8 * norm_A + 0.2 * norm_B (base=100)
        """
        df_univ = _make_universe(
            {'혼합': ['AAA', 'BBB']},
            caps={'AAA': 80.0, 'BBB': 20.0},
        )

        idx = IDX_FULL  # 금요일 기준, 정확히 N개
        m = len(idx)
        # 단조 증가 종가 — 계산 검증 용이
        close_a = np.linspace(10000, 12000, m)
        close_b = np.linspace(5000,  7000, m)

        def _make_flat_ohlcv(close_arr, i):
            """open=high=low=close, volume=1000 의 단순 DF."""
            df = pd.DataFrame({
                'open':   close_arr,
                'high':   close_arr,
                'low':    close_arr,
                'close':  close_arr,
                'volume': np.ones(m) * 1000,
            }, index=idx)
            df.index.name = 'date'
            return df[OHLCV_COLS]

        ohlcv_a = _make_flat_ohlcv(close_a, 0)
        ohlcv_b = _make_flat_ohlcv(close_b, 1)
        bench   = _make_bench_series(m, idx=idx)

        def mock_ohlcv(ticker, *args, **kwargs):
            return ohlcv_a if ticker == 'AAA' else ohlcv_b

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', side_effect=mock_ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, use_yf_fallback=False)
            md = src.load()

        self.assertIn('혼합', md.sector_data, "섹터 '혼합' 이 sector_data 에 없음")
        sector_close = md.sector_data['혼합']['close']

        # 정규화: p_A = close_a / close_a[0] * 100
        norm_a = close_a / close_a[0] * 100.0
        norm_b = close_b / close_b[0] * 100.0
        expected = 0.8 * norm_a + 0.2 * norm_b  # len = m

        np.testing.assert_allclose(
            sector_close.values,
            expected,
            rtol=1e-6,
            err_msg="시총 80/20 가중 섹터 종가가 기대값과 다름",
        )

    def test_equal_weight_fallback_when_no_cap(self):
        """시총 정보가 모두 0 이면 등가 가중이 적용되어야 한다."""
        df_univ = _make_universe({'혼합': ['X', 'Y']}, caps={'X': 0.0, 'Y': 0.0})
        idx = IDX_FULL  # 금요일 기준
        m = len(idx)
        close_x = np.full(m, 10000.0)
        close_y = np.full(m, 10000.0)

        def _flat(c):
            df = pd.DataFrame(
                {'open': c, 'high': c, 'low': c, 'close': c, 'volume': np.ones(m)},
                index=idx,
            )
            df.index.name = 'date'
            return df[OHLCV_COLS]

        def mock_ohlcv(ticker, *args, **kwargs):
            return _flat(close_x) if ticker == 'X' else _flat(close_y)

        bench = _make_bench_series(m, idx=idx)
        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', side_effect=mock_ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, use_yf_fallback=False)
            md = src.load()

        self.assertIn('혼합', md.sector_data)
        # 등가 가중 + 동일 종가 → 섹터 close = 100 (정규화 기준)
        np.testing.assert_allclose(
            md.sector_data['혼합']['close'].values,
            np.full(m, 100.0),
            rtol=1e-6,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 5: min_bars 미달 섹터 스킵 + meta 주석
# ══════════════════════════════════════════════════════════════════════════════
class TestMinBarsFilter(unittest.TestCase):
    def test_short_sector_skipped_with_meta(self):
        """
        데이터 길이 < min_bars 인 섹터는 sector_data 에서 제외되고
        meta[sector]['skipped'] 에 사유가 기록되어야 한다.
        """
        # 짧은 인덱스(10일)와 정상 인덱스(60일) 섹터를 공존
        idx_long  = pd.bdate_range(end='2024-06-30', periods=N)
        idx_short = pd.bdate_range(end='2024-06-30', periods=10)
        # bench는 short에 맞춰 짧게 → common_index 가 10일
        bench = _make_bench_series(10, idx=idx_short)

        df_univ = _make_universe({'정상섹터': ['A'], '짧은섹터': ['B']})
        ohlcv_long  = _make_ohlcv(N,  idx=idx_long,  seed=5)
        ohlcv_short = _make_ohlcv(10, idx=idx_short, seed=6)

        def mock_ohlcv(ticker, *args, **kwargs):
            return ohlcv_long if ticker == 'A' else ohlcv_short

        cfg = FunnelConfig()  # min_bars = 33

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', side_effect=mock_ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, cfg=cfg, use_yf_fallback=False)
            md = src.load()

        # common_index 가 10 이므로 둘 다 min_bars(33) 미달 → 모두 스킵
        for sector in ['정상섹터', '짧은섹터']:
            if sector not in md.sector_data:
                self.assertIn('skipped', md.meta.get(sector, {}),
                              f"스킵된 섹터 '{sector}' 에 meta['skipped'] 없음")

    def test_sufficient_bars_not_skipped(self):
        """충분한 바 수를 가진 섹터는 sector_data 에 남아 있어야 한다."""
        df_univ = _make_universe({'반도체': ['069500']})
        ohlcv   = _make_ohlcv(N, idx=IDX_FULL)
        bench   = _make_bench_series(N, idx=IDX_FULL)
        cfg     = FunnelConfig()  # min_bars = 33, N=60 → 통과

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', return_value=ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, cfg=cfg, use_yf_fallback=False)
            md = src.load()

        self.assertIn('반도체', md.sector_data,
                      "충분한 데이터가 있음에도 섹터가 스킵됨")
        self.assertNotIn('skipped', md.meta.get('반도체', {}))


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 6: naver_get_ohlcv_history 파서 단위 테스트
# ══════════════════════════════════════════════════════════════════════════════
class TestOhlcvParser(unittest.TestCase):
    """_parse_naver_chart_ohlcv 를 간접 검증 (mock 없이 parser 직접 호출)."""

    def test_parse_json_response(self):
        """정상 JSON 응답 파싱."""
        from momentum_funnel.data_adapter import _parse_naver_chart_ohlcv
        raw = '[["20240101",10000,10500,9800,10200,50000],["20240102",10200,10600,10100,10400,60000]]'
        df = _parse_naver_chart_ohlcv(raw)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), OHLCV_COLS)
        self.assertAlmostEqual(df.iloc[0]['close'], 10200.0)
        self.assertAlmostEqual(df.iloc[0]['volume'], 50000.0)

    def test_parse_single_quote_response(self):
        """작은따옴표 응답 파싱."""
        from momentum_funnel.data_adapter import _parse_naver_chart_ohlcv
        raw = "[['20240103',9900,10100,9700,10000,45000]]"
        df = _parse_naver_chart_ohlcv(raw)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]['open'], 9900.0)

    def test_empty_response_returns_empty_df(self):
        """빈 응답 → 빈 DataFrame, OHLCV_COLS 컬럼 유지."""
        from momentum_funnel.data_adapter import _parse_naver_chart_ohlcv
        df = _parse_naver_chart_ohlcv('[]')
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), OHLCV_COLS)

    def test_trailing_comma_handled(self):
        """후행 콤마 처리."""
        from momentum_funnel.data_adapter import _parse_naver_chart_ohlcv
        raw = '[["20240104",10100,10300,9900,10150,30000],]'
        df = _parse_naver_chart_ohlcv(raw)
        self.assertEqual(len(df), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 7: null/빈 카테고리 필터
# ══════════════════════════════════════════════════════════════════════════════
class TestCategoryFilter(unittest.TestCase):
    def test_null_category_tickers_ignored(self):
        """중카테고리가 null 이거나 빈 문자열인 티커는 처리 대상에서 제외."""
        import pandas as pd
        rows = [
            {'중카테고리': '반도체', '시가총액(억원)': 100.0},
            {'중카테고리': None,    '시가총액(억원)': 50.0},
            {'중카테고리': '',      '시가총액(억원)': 30.0},
        ]
        df_univ = pd.DataFrame(rows, index=pd.Index(['A', 'B', 'C'], name='ticker'))

        ohlcv = _make_ohlcv(N, idx=IDX_FULL)
        bench = _make_bench_series(N, idx=IDX_FULL)

        call_count = {'n': 0}
        def mock_ohlcv(ticker, *args, **kwargs):
            call_count['n'] += 1
            return ohlcv

        with patch('momentum_funnel.data_adapter.naver_get_ohlcv_history', side_effect=mock_ohlcv), \
             patch('momentum_funnel.data_adapter.naver_get_index_history', return_value=bench):
            src = UniverseDataSource(df_univ, use_yf_fallback=False)
            md = src.load()

        # 티커 A 만 조회되어야 함
        self.assertEqual(call_count['n'], 1, "null/빈 카테고리 티커까지 조회됨")
        self.assertIn('반도체', md.sector_data)


if __name__ == '__main__':
    unittest.main()
