"""
momentum_funnel/data_adapter.py — 데이터 어댑터 레이어

DISCOVERED:
- 기존 로더 인터페이스: naver_get_price_history (종가 Series), naver_get_index_history (KOSPI),
  _http_get (재시도 백오프 내장), fetch_kr_ohlcv_batch (yfinance .KS, 대문자 컬럼)
- ETF volume(naver) = 좌수(shares); 거래대금 환산 = close*volume (proxy)
- yfinance .KS volume = 좌수, ETF는 종종 0 → MFI 신호 약화 가능
- _parse_naver_chart: JSON 파싱 → 줄 단위 → 정규식 3단 폴백, 종가만 추출
- KOSPI 심볼 후보: ['KOSPI', 'KPI200']
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .contracts import OHLCV_COLS, MarketData
from .config import FunnelConfig

# ── 외부 의존성: 런타임 지연 임포트 (테스트 목(mock) 호환성 + 임포트 사이클 방지)
# etf_universe_builder 는 tqdm 등 무거운 의존성을 가지므로 함수 내부에서 임포트.


def naver_get_index_history(symbol: str, start_date: str, end_date: str):
    """네이버 지수 종가 Series 래퍼 (etf_universe_builder 지연 임포트).

    테스트에서 momentum_funnel.data_adapter.naver_get_index_history 를 mock 할 수 있도록
    모듈 수준에서 노출한다.
    """
    from etf_universe_builder import naver_get_index_history as _ngh
    return _ngh(symbol, start_date, end_date)


# ══════════════════════════════════════════════════════════════════════════════
# 1) 네이버 OHLCV 파서
# ══════════════════════════════════════════════════════════════════════════════

def _parse_naver_chart_ohlcv(text: str) -> pd.DataFrame:
    """네이버 차트 API 응답 → OHLCV DataFrame (소문자 컬럼, DatetimeIndex).

    응답 형식: [["YYYYMMDD", open, high, low, close, volume], ...]
    작은따옴표, 후행 콤마, 헤더 행을 허용한다.
    volume은 좌수(shares) — 호출자가 거래대금 환산.
    """
    text = text.strip()
    rows: List[dict] = []

    # ── 방법 1: 전체 JSON 배열 파싱 ──────────────────────────────────────
    try:
        cleaned = text.replace("'", '"')
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        data = json.loads(cleaned)
        for row in data:
            if isinstance(row, list) and len(row) >= 6:
                date_str = str(row[0]).strip().strip('"').strip()
                digits = re.sub(r'\D', '', date_str)
                if len(digits) == 8:
                    try:
                        rows.append({
                            'date':   pd.Timestamp(digits),
                            'open':   float(row[1]),
                            'high':   float(row[2]),
                            'low':    float(row[3]),
                            'close':  float(row[4]),
                            'volume': float(row[5]),
                        })
                    except (ValueError, TypeError):
                        continue
    except (json.JSONDecodeError, ValueError):
        pass

    # ── 방법 2: 줄 단위 파싱 ─────────────────────────────────────────────
    if not rows:
        for line in text.split('\n'):
            line = line.strip().rstrip(',')
            if not line or line in ('[', ']', '[[', ']]'):
                continue
            if '날짜' in line or 'date' in line.lower():
                continue
            try:
                line = line.replace("'", '"')
                row = json.loads(line)
                if isinstance(row, list) and len(row) >= 6:
                    date_str = str(row[0]).strip().strip('"').strip()
                    digits = re.sub(r'\D', '', date_str)
                    if len(digits) == 8:
                        rows.append({
                            'date':   pd.Timestamp(digits),
                            'open':   float(row[1]),
                            'high':   float(row[2]),
                            'low':    float(row[3]),
                            'close':  float(row[4]),
                            'volume': float(row[5]),
                        })
            except (json.JSONDecodeError, ValueError):
                continue

    # ── 방법 3: 정규식 직접 추출 (최후 수단) ─────────────────────────────
    if not rows:
        pattern = (
            r'\[?\s*["\']?(\d{8})\s*["\']?\s*,'   # date
            r'\s*([\d.]+)\s*,'                      # open
            r'\s*([\d.]+)\s*,'                      # high
            r'\s*([\d.]+)\s*,'                      # low
            r'\s*([\d.]+)\s*,'                      # close
            r'\s*([\d.]+)'                          # volume
        )
        for m in re.finditer(pattern, text):
            try:
                rows.append({
                    'date':   pd.Timestamp(m.group(1)),
                    'open':   float(m.group(2)),
                    'high':   float(m.group(3)),
                    'low':    float(m.group(4)),
                    'close':  float(m.group(5)),
                    'volume': float(m.group(6)),
                })
            except (ValueError, TypeError):
                continue

    if not rows:
        return pd.DataFrame(columns=OHLCV_COLS)

    df = pd.DataFrame(rows).set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index.name = 'date'
    return df[OHLCV_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# 2) 공개 OHLCV 수집 함수
# ══════════════════════════════════════════════════════════════════════════════

def naver_get_ohlcv_history(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """네이버 차트 API → 일별 OHLCV DataFrame.

    Parameters
    ----------
    ticker : str
        ETF 티커 (예: '069500')
    start_date, end_date : str
        'YYYYMMDD' 형식

    Returns
    -------
    pd.DataFrame
        컬럼: OHLCV_COLS(['open','high','low','close','volume']), DatetimeIndex
        volume = 좌수(shares) — 거래대금은 호출자에서 close*volume 으로 환산.
        실패 시 빈 DataFrame 반환.
    """
    url = (
        f"https://fchart.stock.naver.com/siseJson.naver"
        f"?symbol={ticker}&requestType=1"
        f"&startTime={start_date}&endTime={end_date}&timeframe=day"
    )
    try:
        from etf_universe_builder import _http_get  # 지연 임포트 (tqdm 등 무거운 의존성)
        text = _http_get(url, encoding='utf-8')
        return _parse_naver_chart_ohlcv(text)
    except Exception:
        return pd.DataFrame(columns=OHLCV_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# 3) yfinance 폴백 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _yf_ohlcv(tickers: List[str], months: int) -> Dict[str, pd.DataFrame]:
    """fetch_kr_ohlcv_batch 를 래핑, 대문자 → 소문자 컬럼 변환."""
    try:
        # etf_scoring은 streamlit 캐시 데코레이터를 사용하므로 직접 호출
        import yfinance as yf
        from datetime import datetime, timedelta

        end = datetime.today()
        start = end - timedelta(days=months * 31)
        kr_tickers = [f"{t}.KS" for t in tickers]
        ticker_map = {f"{t}.KS": t for t in tickers}
        result: Dict[str, pd.DataFrame] = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(kr_tickers, start=start, end=end,
                              progress=False, auto_adjust=True)

        if raw.empty:
            return result

        for kr_t in kr_tickers:
            orig = ticker_map[kr_t]
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl1 = raw.columns.get_level_values(1)
                    if kr_t not in lvl1:
                        continue
                    df_t = raw.xs(kr_t, axis=1, level=1).copy()
                else:
                    df_t = raw.copy()

                df_t.index = pd.to_datetime(df_t.index)
                if df_t.empty or 'Close' not in df_t.columns:
                    continue

                # 대문자 → 소문자 + OHLCV_COLS 정렬
                df_t.columns = [c.lower() for c in df_t.columns]
                cols_present = [c for c in OHLCV_COLS if c in df_t.columns]
                if 'close' not in cols_present:
                    continue
                # 누락 컬럼은 NaN 으로 채움
                for c in OHLCV_COLS:
                    if c not in df_t.columns:
                        df_t[c] = np.nan
                df_t.index.name = 'date'
                result[orig] = df_t[OHLCV_COLS].copy()
            except Exception:
                continue

        return result
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 4) UniverseDataSource
# ══════════════════════════════════════════════════════════════════════════════

class UniverseDataSource:
    """ETF 유니버스 DataFrame → MarketData 변환 어댑터.

    Parameters
    ----------
    df_universe : pd.DataFrame
        인덱스 = 티커(str), 컬럼 최소 요건: cat_col, cap_col
    lookback_days : int
        오늘 기준 과거 n 영업일 (달력일 기준 요청 — 여유 25% 추가)
    asof : pd.Timestamp, optional
        미래 데이터 차단 기준일. 제공 시 이 날짜 이하 데이터만 사용.
    cap_col : str
        시가총액 컬럼 이름 (기본 '시가총액(억원)')
    cat_col : str
        섹터 분류 컬럼 (기본 '중카테고리')
    use_yf_fallback : bool
        네이버 조회 실패 시 yfinance .KS 폴백 여부
    cfg : FunnelConfig, optional
        min_bars 등 파라미터. None 이면 기본값 사용.
    """

    def __init__(
        self,
        df_universe: pd.DataFrame,
        lookback_days: int = 180,
        asof: Optional[pd.Timestamp] = None,
        cap_col: str = '시가총액(억원)',
        cat_col: str = '중카테고리',
        use_yf_fallback: bool = True,
        cfg: Optional[FunnelConfig] = None,
        max_members_per_sector: int = 5,
        min_members_per_sector: int = 1,
    ) -> None:
        self._df = df_universe
        self._lookback_days = lookback_days
        self._asof = asof
        self._cap_col = cap_col
        self._cat_col = cat_col
        self._use_yf_fallback = use_yf_fallback
        self._cfg = cfg or FunnelConfig()
        self._max_members = max_members_per_sector
        self._min_members = min_members_per_sector

    # ── 날짜 범위 계산 ────────────────────────────────────────────────────
    def _date_range(self):
        """(start_str, end_str) in 'YYYYMMDD' 형식."""
        end_ts = self._asof if self._asof is not None else pd.Timestamp.today()
        # lookback_days 는 영업일 기준이지만 달력일로 요청 (주말·휴장 여유 25%)
        start_ts = end_ts - timedelta(days=int(self._lookback_days * 1.4))
        return start_ts.strftime('%Y%m%d'), end_ts.strftime('%Y%m%d')

    # ── 단일 티커 OHLCV 수집 ─────────────────────────────────────────────
    def _fetch_one(self, ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        df = naver_get_ohlcv_history(ticker, start_str, end_str)
        return df

    # ── asof 트런케이션 ───────────────────────────────────────────────────
    def _truncate(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._asof is not None and not df.empty:
            return df[df.index <= self._asof]
        return df

    def _truncate_series(self, s: pd.Series) -> pd.Series:
        if self._asof is not None and not s.empty:
            return s[s.index <= self._asof]
        return s

    # ── 섹터 집계 (시총 가중 정규화 지수) ────────────────────────────────
    def _aggregate_sector(
        self,
        member_ohlcvs: Dict[str, pd.DataFrame],
        caps: Dict[str, float],
    ) -> pd.DataFrame:
        """
        시총 가중 섹터 OHLCV 합성.

        각 멤버의 가격을 base=100 으로 정규화한 뒤 가중합.
        volume은 좌수 → 거래대금(proxy) 환산(close * volume)의 합.
        """
        # ── 공통 날짜 교집합 ─────────────────────────────────────────────
        indices = [df.index for df in member_ohlcvs.values() if not df.empty]
        if not indices:
            return pd.DataFrame(columns=OHLCV_COLS)

        common = indices[0]
        for idx in indices[1:]:
            common = common.intersection(idx)

        if common.empty:
            return pd.DataFrame(columns=OHLCV_COLS)

        # ── 가중치 계산 ───────────────────────────────────────────────────
        total_cap = sum(caps.get(t, 0.0) for t in member_ohlcvs)
        if total_cap <= 0:
            # 시총 정보 없음 → 등가 가중
            n = len(member_ohlcvs)
            weights = {t: 1.0 / n for t in member_ohlcvs}
        else:
            weights = {t: caps.get(t, 0.0) / total_cap for t in member_ohlcvs}

        # ── 가중 합산 ─────────────────────────────────────────────────────
        agg_open   = pd.Series(0.0, index=common)
        agg_high   = pd.Series(0.0, index=common)
        agg_low    = pd.Series(0.0, index=common)
        agg_close  = pd.Series(0.0, index=common)
        agg_vol    = pd.Series(0.0, index=common)

        for ticker, df in member_ohlcvs.items():
            w = weights[ticker]
            if w == 0.0 or df.empty:
                continue
            df_a = df.reindex(common)
            first_close = df_a['close'].dropna()
            if first_close.empty:
                continue
            base = first_close.iloc[0]
            if base == 0 or np.isnan(base):
                continue

            # 정규화 인덱스 (base=100)
            norm = lambda col: df_a[col] / base * 100.0

            agg_open  += norm('open')  * w
            agg_high  += norm('high')  * w
            agg_low   += norm('low')   * w
            agg_close += norm('close') * w
            # 거래대금 proxy = close(원) * volume(좌수) → 합산
            agg_vol   += (df_a['close'] * df_a['volume']).fillna(0.0)

        result = pd.DataFrame({
            'open':   agg_open,
            'high':   agg_high,
            'low':    agg_low,
            'close':  agg_close,
            'volume': agg_vol,
        }, index=common)
        result.index.name = 'date'
        return result[OHLCV_COLS]

    # ── 메인 load ──────────────────────────────────────────────────────────
    def load(self) -> MarketData:
        df_univ = self._df
        cat_col = self._cat_col
        cap_col = self._cap_col
        cfg = self._cfg
        start_str, end_str = self._date_range()

        # ── 1. 유효 카테고리만 필터 ───────────────────────────────────────
        mask = df_univ[cat_col].notna() & (df_univ[cat_col].astype(str).str.strip() != '')
        df_filtered = df_univ[mask].copy()

        # ── 2. 섹터별 OHLCV 수집 & 집계 ──────────────────────────────────
        sector_data: Dict[str, pd.DataFrame] = {}
        meta: Dict[str, dict] = {}

        for sector, group in df_filtered.groupby(cat_col):
            sector = str(sector)
            # 멤버 수 제한: 시총 상위 N개만 수집 (수집 부담 완화)
            if cap_col in group.columns and self._max_members > 0:
                group = group.sort_values(cap_col, ascending=False).head(self._max_members)
            tickers = list(group.index.astype(str))
            if len(tickers) < self._min_members:
                meta[sector] = {'tickers': tickers, 'skipped': f'멤버 부족 ({len(tickers)} < {self._min_members})'}
                continue
            meta[sector] = {'tickers': tickers, 'source': 'naver'}

            # 시총 맵
            if cap_col in group.columns:
                cap_map = group[cap_col].fillna(0.0).astype(float).to_dict()
            else:
                cap_map = {t: 0.0 for t in tickers}

            # 네이버 OHLCV 수집
            member_ohlcvs: Dict[str, pd.DataFrame] = {}
            for ticker in tickers:
                df_t = self._fetch_one(ticker, start_str, end_str)
                df_t = self._truncate(df_t)
                if not df_t.empty:
                    member_ohlcvs[ticker] = df_t

            # yfinance 폴백 (네이버 실패 티커)
            if self._use_yf_fallback:
                missing = [t for t in tickers if t not in member_ohlcvs]
                if missing:
                    months = max(1, self._lookback_days // 30)
                    yf_result = _yf_ohlcv(missing, months=months)
                    for t, df_yf in yf_result.items():
                        df_yf = self._truncate(df_yf)
                        if not df_yf.empty:
                            member_ohlcvs[t] = df_yf
                    if yf_result:
                        meta[sector]['source'] = 'mixed(naver+yfinance)'

            if not member_ohlcvs:
                meta[sector]['skipped'] = f'모든 멤버({len(tickers)}개) 데이터 없음'
                continue

            # 섹터 집계
            sector_df = self._aggregate_sector(member_ohlcvs, cap_map)
            if sector_df.empty:
                meta[sector]['skipped'] = '공통 날짜 없음 (교집합 공집합)'
                continue

            sector_data[sector] = sector_df

        # ── 3. KOSPI 벤치마크 ─────────────────────────────────────────────
        # 모듈 수준 래퍼 호출 → 테스트에서 mock 가능
        bench_close = naver_get_index_history('KOSPI', start_str, end_str)
        bench_close = self._truncate_series(bench_close)

        if bench_close.empty:
            # 폴백: 첫 번째 섹터 close 를 대용 (최소 기능 보장)
            if sector_data:
                first_sector = next(iter(sector_data.values()))
                bench_close = first_sector['close'].rename('close')
            else:
                bench_close = pd.Series(dtype=float)

        benchmark = pd.DataFrame({'close': bench_close})
        benchmark.index.name = 'date'

        # ── 4. 공통 인덱스 (벤치 ∩ 전 섹터) ─────────────────────────────
        if sector_data and not benchmark.empty:
            common_index = benchmark.index
            for s_df in sector_data.values():
                common_index = common_index.intersection(s_df.index)
        elif sector_data:
            common_index = next(iter(sector_data.values())).index
        else:
            common_index = pd.DatetimeIndex([])

        # ── 5. min_bars 필터 (교집합 기준) ────────────────────────────────
        sectors_to_drop = []
        for sector, s_df in sector_data.items():
            aligned_len = len(common_index)
            if aligned_len < cfg.min_bars:
                meta[sector]['skipped'] = (
                    f'공통 날짜 {aligned_len}개 < min_bars {cfg.min_bars}'
                )
                sectors_to_drop.append(sector)

        for sector in sectors_to_drop:
            del sector_data[sector]

        # ── 6. 공통 인덱스 재계산 (min_bars 제거 후) ─────────────────────
        if sector_data and not benchmark.empty:
            common_index = benchmark.index
            for s_df in sector_data.values():
                common_index = common_index.intersection(s_df.index)
        elif sector_data:
            common_index = next(iter(sector_data.values())).index
        else:
            common_index = pd.DatetimeIndex([])

        # ── 7. 공통 인덱스로 리인덱스 (ffill 금지) ───────────────────────
        final_sector_data: Dict[str, pd.DataFrame] = {}
        for sector, s_df in sector_data.items():
            final_sector_data[sector] = s_df.reindex(common_index)

        if not benchmark.empty:
            benchmark = benchmark.reindex(common_index)

        # 거래대금 환산 메모 기록
        for sector in final_sector_data:
            if 'skipped' not in meta.get(sector, {}):
                meta.setdefault(sector, {})['volume_unit'] = (
                    '거래대금(proxy) = close×volume(좌수) 합산 [KRW 추정]'
                )

        return MarketData(
            sector_data=final_sector_data,
            benchmark=benchmark,
            common_index=common_index,
            meta=meta,
        )
