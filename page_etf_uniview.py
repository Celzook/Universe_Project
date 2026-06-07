"""
ETF Uniview Page
================
- 국내 상장 ETF 유니버스 기반 (유니버스 탐색에서 빌드한 데이터)
- 최대 20개 ETF 다중 선택
- 5×4 그리드 (빈 셀도 placeholder 렌더링)
- 캔들스틱 + 볼린저밴드 + RSI, 마우스 휠 줌 지원 (6개월 데이터, 기본 3개월 뷰)
- C_score 오버레이 (etf_scoring 모듈)
- 하단 랭킹 보드: ETF C_score 상위/하위 15, 카테고리 평균 상위/하위 15
"""
import hashlib
import json
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from etf_scoring import (
    compute_bollinger,
    compute_rsi,
    compute_T_score,
    compute_C_score,
    fetch_macro_scores,
    fetch_kr_ohlcv_batch,
    score_all_kr_etfs,
)

from momentum_funnel import (
    FunnelConfig, UniverseDataSource,
    compute_hot_metrics, build_mp,
    EXCLUDED_CATEGORIES_DEFAULT,
    CORE_TICKER_DEFAULT, CORE_NAME_DEFAULT,
    save_mp_local, load_mp, delete_mp,
    push_to_github, compute_mp_performance,
)

GRID_COLS = 5
GRID_ROWS = 4
MAX_SELECT = GRID_COLS * GRID_ROWS   # 20
FETCH_MONTHS = 6                      # 6개월 데이터 로드 (휠 줌 아웃용)
DEFAULT_VIEW_MONTHS = 3               # 기본 표시 구간

# ── 모멘텀 깔때기: 한국 시장 맞춤 임계값 ──────────────────────────
FUNNEL_DEFAULT_A_ADX_MIN = 18.0
FUNNEL_DEFAULT_B_ADX_MIN = 18.0
FUNNEL_DEFAULT_B_MFI_LOWER = 45.0
FUNNEL_DEFAULT_B_MFI_UPPER = 78.0
FUNNEL_DEFAULT_MEMBERS = 5            # 섹터당 시총 상위 N개만 OHLCV 수집


@st.cache_data(ttl=3600 * 4, show_spinner=False)
def _cached_hot_run(base_date: str, n_etfs: int, max_members: int):
    """캐시: (기준일+유니버스크기+멤버수) → (market_meta, hot_metrics_df).
    market_meta는 hot_metrics index 에 대응되는 tickers 매핑만 추출 (직렬화 가능 형태)."""
    df_uni = st.session_state.df_universe
    cfg = FunnelConfig()
    src = UniverseDataSource(
        df_uni, lookback_days=180, use_yf_fallback=True,
        cfg=cfg, max_members_per_sector=max_members,
    )
    market = src.load()
    hot_metrics = compute_hot_metrics(market, cfg)
    # market.meta → {sector: tickers} 만 추려 캐시(MarketData 통째 직렬화 회피)
    meta_tickers = {s: market.meta.get(s, {}).get('tickers', []) for s in hot_metrics.index}
    asof = market.common_index[-1] if len(market.common_index) else None
    return hot_metrics, meta_tickers, asof


def _saved_mp_section():
    """📌 저장된 MP — 편입일 기준 forward 성과 추적. 없으면 무음 종료."""
    saved = load_mp()
    if not saved:
        return  # 저장본 없음 → 섹션 자체 미표시

    st.subheader("📌 저장된 MP — 편입일 기준 성과 추적")
    inception = saved.get('inception_date', '?')
    method = saved.get('method', '?')
    saved_at = saved.get('saved_at', '?')
    n_pos = len(saved.get('positions', []))

    c0, c1, c2, c3 = st.columns([1.3, 1.0, 1.0, 0.9])
    c0.metric("편입일", inception)
    c1.metric("방법", method)
    c2.metric("포지션", n_pos)
    with c3:
        st.caption(f"저장일시\n{saved_at}")
        if st.button("🗑️ 삭제", key='saved_mp_delete', help="저장된 MP 제거 (로컬만, GitHub 파일은 별도)"):
            if delete_mp():
                st.success("삭제됨. 페이지 새로고침으로 갱신.")
                st.rerun()

    # 가격·KOSPI fetch 어댑터
    try:
        from etf_universe_builder import naver_get_price_history, naver_get_index_history
    except Exception as e:
        st.error(f"가격 어댑터 로드 실패: {e}")
        return

    def _fetch_close(t, s, e):
        return naver_get_price_history(t, s, e)
    def _fetch_kospi(s, e):
        return naver_get_index_history('KOSPI', s, e)

    # 캐시 키: saved 전체 내용 hash 포함 → saved 가 바뀌면 자동 invalidation
    # `_saved` (밑줄 시작 인자) 는 Streamlit 이 해싱하지 않음 → payload 전달용
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_perf(saved_hash: str, inception_key: str, tickers_key: tuple, _saved: dict):
        return compute_mp_performance(_saved, _fetch_close, _fetch_kospi)

    with st.spinner("📡 저장 MP 성과 계산 (편입일 기준 forward)..."):
        try:
            saved_json = json.dumps(saved, ensure_ascii=False, sort_keys=True, default=str)
            saved_hash = hashlib.md5(saved_json.encode('utf-8')).hexdigest()[:12]
            tickers_key = tuple(str(p.get('ticker','')) for p in saved.get('positions', []))
            perf = _cached_perf(saved_hash, inception, tickers_key, saved)
        except Exception as e:
            st.error(f"성과 계산 실패: {e}")
            return

    if perf.empty:
        st.info("성과 산출 결과가 없습니다.")
    else:
        # 포트폴리오 누적 (시작비중 가중)
        port_cum = (perf['시작비중%'] / 100.0 * perf['누적%'].fillna(0)).sum()
        st.metric("포트폴리오 누적 초과수익률 (KOSPI 대비)", f"{port_cum:+.2f}%")

        # 표시용 정수·소수점
        view = perf.copy()
        for c in ['시작비중%','현재비중%']:
            view[c] = view[c].astype(float).round(2)
        for c in ['누적%','+1D','+1W','+1M','+3M','YTD']:
            view[c] = view[c].astype(float).round(2)

        st.dataframe(
            view, width='stretch', hide_index=True, height=420,
            column_config={
                '시작비중%': st.column_config.NumberColumn('시작 비중', format='%.2f%%'),
                '현재비중%': st.column_config.ProgressColumn(
                    '현재 비중 (drift)', min_value=0.0, max_value=100.0, format='%.2f%%',
                ),
                '누적%': st.column_config.NumberColumn('누적 초과 %', format='%+.2f'),
                '+1D':  st.column_config.NumberColumn('+1D %', format='%+.2f'),
                '+1W':  st.column_config.NumberColumn('+1W %', format='%+.2f'),
                '+1M':  st.column_config.NumberColumn('+1M %', format='%+.2f'),
                '+3M':  st.column_config.NumberColumn('+3M %', format='%+.2f'),
                'YTD':  st.column_config.NumberColumn('YTD %',  format='%+.2f'),
            },
        )
        st.caption("**해석**: 모든 수익률은 KOSPI 초과 (ETF − KOSPI). "
                   "**현재 비중**은 편입일 이후 가격 변화 반영한 buy-and-hold drift (합=100). "
                   "Forward 윈도우 (+1D/+1W/+1M/+3M)는 편입일 이후 해당 영업일 도달 시점 시점, 미도달 시 NaN.")

    st.markdown("---")


_HOT_HELP = """
**Hot Sectors** — 시장에서 자금·거래·상대강도가 살아나는 섹터.

**VolRatio** — 최근 3일 거래대금 평균 / 20일 평균. **1.2+ 거래 활성화 / 1.5+ 급증**.

**MoneyFlow** — 5일 누적 자금 유입 강도 (−1 ~ +1).
양수 = 유입 우위, 음수 = 유출. 거래대금 × 가격 방향으로 산출.

**RS (Relative Strength)** — 벤치(KOSPI) 대비 log-RS 회귀 기울기. 양수 = 벤치 초과.

**HotScore** — 위 3지표를 percentile rank 로 환산 → 평균 (0~1).
1.0 = 모든 지표 최상위.

**MP 구성 (코어-새틀라이트)**
- Core: TIGER 200 (102110) @ 87.5% — 코스피200 베타
- Satellite: Hot Top N @ 12.5% — HotScore 비례 가중
- 베타 중복(대형주·코스피200·가치주·배당) 자동 제외
- **Method A** = HotScore 단일 정렬 Top N
- **Method B** = Money Top 10 → 그 중 RS Top N (2단 스크린)
"""


def _hot_board_section(df_uni: pd.DataFrame):
    """🔥 Hot Sectors Board + Model Portfolio (Core-Satellite)."""
    st.subheader("🔥 Hot Sectors + Model Portfolio", help=_HOT_HELP)

    with st.expander("⚙️ 설정", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            n_sat = st.slider("새틀라이트 개수", 3, 5, 5, 1, key='hot_n_sat')
        with s2:
            core_pct = st.slider("코어 비중 %", 80.0, 95.0, 87.5, 0.5, key='hot_core_pct') / 100.0
        with s3:
            max_mem = st.slider("섹터당 멤버 수 (시총 상위)", 3, 10, 5, 1, key='hot_max_mem')
        with s4:
            core_ticker = st.text_input("코어 ETF 티커", value=CORE_TICKER_DEFAULT, key='hot_core_ticker')
        st.caption(f"베타 중복 자동 제외: `{', '.join(EXCLUDED_CATEGORIES_DEFAULT)}`")

    base_date = st.session_state.get('base_date', datetime.today().strftime('%Y%m%d'))

    with st.spinner("📡 섹터 OHLCV 수집 + Hot Score 계산 중... (첫 실행 30~90초, 이후 캐시)"):
        try:
            hot_metrics, meta_tickers, asof = _cached_hot_run(base_date, len(df_uni), max_mem)
        except Exception as e:
            st.error(f"Hot Board 실행 실패: {e}")
            with st.expander("🔍 상세"):
                import traceback as _tb
                st.code(_tb.format_exc())
            return

    asof_str = asof.strftime('%Y-%m-%d') if asof is not None else '?'
    n_total = len(hot_metrics); n_valid = int(hot_metrics['valid'].fillna(False).sum())
    st.caption(f"기준일 **{asof_str}** · 섹터 {n_total} (유효 {n_valid}) · 벤치=KOSPI · 섹터=중카테고리(시총가중)")

    # ── Hot Leaderboard (좌) + MP A/B (우) ────────────────────────────────
    col_l, col_r = st.columns([1.2, 1.8])

    with col_l:
        st.markdown("**🔥 Hot Sectors Leaderboard (Top 15)**")
        view = hot_metrics.head(15)[['vol_ratio','money_flow','rs','hot_score']].copy()
        view.columns = ['VolRatio','MoneyFlow','RS','HotScore']
        view['VolRatio']  = view['VolRatio'].round(2)
        view['MoneyFlow'] = view['MoneyFlow'].round(3)
        view['RS']        = view['RS'].round(4)
        view['HotScore']  = view['HotScore'].round(3)
        st.dataframe(
            view, width='stretch', height=560,
            column_config={
                'HotScore': st.column_config.ProgressColumn(
                    'HotScore', min_value=0.0, max_value=1.0, format='%.3f',
                ),
            },
        )

    # MP 구성을 위한 ticker→name 맵 (cached 컨텍스트 밖이라 직접 생성)
    ticker_to_name = dict(zip(df_uni.index.astype(str), df_uni['ETF명'])) if 'ETF명' in df_uni.columns else {}
    core_name = ticker_to_name.get(core_ticker, CORE_NAME_DEFAULT)

    # market.meta 의 일부만 캐시했으므로 build_mp 에 가짜 MarketData 대신 단순 wrapper
    class _MetaOnly:
        def __init__(self, meta): self.meta = meta
    market_proxy = _MetaOnly({s: {'tickers': tk} for s, tk in meta_tickers.items()})

    with col_r:
        st.markdown(f"**📦 Model Portfolio — 코어 {core_pct*100:.1f}% · 새틀라이트 {(1-core_pct)*100:.1f}%**")

        mp_a = build_mp(
            hot_metrics, market_proxy, method='A',
            core_ticker=core_ticker, core_name=core_name,
            core_weight=core_pct, n_satellites=n_sat,
            ticker_to_name=ticker_to_name,
        )
        mp_b = build_mp(
            hot_metrics, market_proxy, method='B',
            core_ticker=core_ticker, core_name=core_name,
            core_weight=core_pct, n_satellites=n_sat,
            ticker_to_name=ticker_to_name,
        )

        tab_a, tab_b = st.tabs(["A · HotScore Top", "B · Money→RS"])

        def _render_mp(df_mp, label):
            if df_mp.empty:
                st.info("MP 생성 불가 (Hot 후보 없음)"); return
            disp = df_mp.copy()
            disp['hot_score'] = disp['hot_score'].astype(float).round(3)
            disp['weight_pct'] = disp['weight_pct'].astype(float).round(2)
            disp.columns = ['역할','티커','대표 ETF','카테고리','HotScore','비중 %']
            st.dataframe(
                disp, width='stretch', hide_index=True,
                column_config={
                    '비중 %': st.column_config.ProgressColumn(
                        '비중 %', min_value=0.0, max_value=100.0, format='%.2f%%',
                    ),
                },
            )
            # CSV 다운로드 + MP 저장 버튼 가로 배치
            import io
            buf = io.StringIO()
            disp.to_csv(buf, encoding='utf-8-sig', index=False)
            bc1, bc2 = st.columns(2)
            with bc1:
                st.download_button(
                    label=f"📥 MP {label} CSV",
                    data=buf.getvalue().encode('utf-8-sig'),
                    file_name=f"mp_{label}_{asof_str}.csv",
                    mime='text/csv',
                    key=f'mp_csv_{label}',
                )
            with bc2:
                if st.button(f"💾 MP {label} 저장", key=f'mp_save_{label}',
                             help="현재 MP를 편입일 기준으로 LOCK IN (덮어쓰기, GitHub 자동 커밋 시도)"):
                    try:
                        path, payload = save_mp_local(df_mp, inception_date=asof_str, method=label)
                        st.success(f"✅ 저장: `{path}` (편입일 {asof_str}, 방법 {label})")
                        ok, msg = push_to_github(payload)
                        if ok:
                            st.info(f"🔄 GitHub 동기화: {msg} — Streamlit Cloud는 자동 재배포 후 영구 반영")
                        else:
                            st.warning(f"🔒 GitHub 미동기화 (로컬만 저장됨): {msg}")
                    except Exception as e:
                        st.error(f"저장 실패: {e}")

        with tab_a:
            st.caption("HotScore 상위 N개 직접 선택")
            _render_mp(mp_a, 'A')
        with tab_b:
            st.caption("Money Top 10 → 그 중 RS 상위 N개")
            _render_mp(mp_b, 'B')

    st.markdown("---")


# ── AP (Actual Portfolio) 분석 헬퍼 ────────────────────────────────────────

# 인덱스 재간접주식형 펀드코드 (사용자 지정 + D열 '인덱스' 키워드 매치)
INDEX_FUND_CODES = {'V5202E', 'V5304R', 'V6303V', 'V72026'}

# Excel 컬럼 위치 (A=0, B=1, … Z=25)
AP_COL_C_FUND_CODE = 2     # 펀드코드
AP_COL_D_FUND_NAME = 3     # 펀드명
AP_COL_H_STOCK_NAME = 7    # 종목명
AP_COL_M_FACE_QTY = 12     # 액면수량
AP_COL_Q_ORIG_COST = 16    # 원취득가액
AP_COL_S_VALUE = 18        # 적용평가액
AP_COL_X_BUY_DATE = 23     # 최초 매수일자 (편입일)
# 가중평균단가 = SUM(원취득가액) / SUM(액면수량) — ETF 그룹 단위


# 액티브 재간접주식형 — 제외 키워드 (해당 종목명 행은 분석에서 drop)
# '더제이' 한 토큰만으로 '더제이 더행복코리아', '더제이 공모펀드' 등 전부 포함.
AP_ACTIVE_EXCLUDE_KEYWORDS = (
    '더제이',
    '더행복코리아',
    'PLUS ESG 성장주 액티브',
)

# 액티브 재간접주식형 — 종목명 통합 매핑 (보통예금 ↔ 은대 → 은대 하나로)
AP_ACTIVE_NAME_UNIFY = (
    (('보통예금', '은대'), '은대'),
)


def _should_exclude_active(name: str) -> bool:
    """액티브 재간접주식형 행에서 제외할 종목명 판정."""
    if not isinstance(name, str):
        return False
    return any(k in name for k in AP_ACTIVE_EXCLUDE_KEYWORDS)


def _unify_active_name(name: str) -> str:
    """액티브 재간접주식형 — '보통예금' 또는 '은대' 포함 시 '은대' 로 통일."""
    if not isinstance(name, str):
        return name
    for keywords, unified in AP_ACTIVE_NAME_UNIFY:
        if any(k in name for k in keywords):
            return unified
    return name


def _classify_fund(code: str, name: str) -> str:
    """펀드코드 + 펀드명 → '인덱스 재간접주식형' or '액티브 재간접주식형'."""
    code_clean = str(code).strip().upper()
    name_str = str(name) if name is not None else ''
    if code_clean in INDEX_FUND_CODES or '인덱스' in name_str:
        return '인덱스 재간접주식형'
    return '액티브 재간접주식형'


def _resolve_ticker_from_name(stock_name: str, name_to_ticker: dict) -> str:
    """종목명 → 티커. 정확 매칭 우선, 양방향 부분 매칭 fallback."""
    if not isinstance(stock_name, str):
        return ''
    nm = stock_name.strip()
    if not nm:
        return ''
    if nm in name_to_ticker:
        return name_to_ticker[nm]
    for ref_name, tk in name_to_ticker.items():
        if ref_name and (ref_name in nm or nm in ref_name):
            return tk
    return ''


@st.cache_data(ttl=3600 * 2, show_spinner=False)
def _cached_etf_close_series(ticker: str, start_str: str, end_str: str) -> pd.Series:
    """단일 티커 일별 종가 (네이버). 캐시 2h."""
    from etf_universe_builder import naver_get_price_history
    return naver_get_price_history(ticker, start_str, end_str)


@st.cache_data(ttl=3600 * 2, show_spinner=False)
def _cached_kospi200_close(start_str: str, end_str: str) -> pd.Series:
    """KOSPI200 일별 종가. 네이버 KPI200 → yfinance ^KS200 폴백. (실제 지수)"""
    from etf_universe_builder import naver_get_index_history
    for sym in ('KPI200', 'KOSPI200'):
        try:
            s = naver_get_index_history(sym, start_str, end_str)
            if isinstance(s, pd.Series) and not s.empty:
                return s
        except Exception:
            pass
    try:
        import yfinance as yf
        s_iso = f"{start_str[:4]}-{start_str[4:6]}-{start_str[6:8]}"
        e_iso = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}"
        df = yf.download('^KS200', start=s_iso, end=e_iso,
                         progress=False, auto_adjust=True)
        if not df.empty and 'Close' in df.columns:
            return df['Close'].squeeze().dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600 * 2, show_spinner=False)
def _cached_kospi_close(start_str: str, end_str: str) -> pd.Series:
    """KOSPI 종합지수 일별 종가. 네이버 KOSPI → yfinance ^KS11 폴백. (실제 지수)"""
    from etf_universe_builder import naver_get_index_history
    try:
        s = naver_get_index_history('KOSPI', start_str, end_str)
        if isinstance(s, pd.Series) and not s.empty:
            return s
    except Exception:
        pass
    try:
        import yfinance as yf
        s_iso = f"{start_str[:4]}-{start_str[4:6]}-{start_str[6:8]}"
        e_iso = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}"
        df = yf.download('^KS11', start=s_iso, end=e_iso,
                         progress=False, auto_adjust=True)
        if not df.empty and 'Close' in df.columns:
            return df['Close'].squeeze().dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


# 분류 → BM 매핑 (UI 라벨)
_BM_NAME_BY_CATEGORY = {
    '인덱스 재간접주식형': 'KOSPI200',
    '액티브 재간접주식형': 'KOSPI',
}


def _parse_ap_for_analysis(ap_df: pd.DataFrame, df_uni: pd.DataFrame) -> pd.DataFrame:
    """AP DataFrame → 분류·종목별 집계.

    Returns columns:
        분류, 종목명, 티커, 적용평가액, 비중%, 편입일(min), 가중평균단가
    """
    if ap_df is None or not isinstance(ap_df, pd.DataFrame) or ap_df.empty:
        return pd.DataFrame()

    n_cols = ap_df.shape[1]
    needed = max(AP_COL_C_FUND_CODE, AP_COL_D_FUND_NAME, AP_COL_H_STOCK_NAME,
                 AP_COL_M_FACE_QTY, AP_COL_Q_ORIG_COST,
                 AP_COL_S_VALUE, AP_COL_X_BUY_DATE)
    if n_cols <= needed:
        raise ValueError(
            f"AP 파일 컬럼 수({n_cols})가 부족합니다 — 최소 {needed + 1}개 필요."
        )

    base = pd.DataFrame({
        '펀드코드': ap_df.iloc[:, AP_COL_C_FUND_CODE].astype(str).str.strip(),
        '펀드명': ap_df.iloc[:, AP_COL_D_FUND_NAME].astype(str),
        '종목명': ap_df.iloc[:, AP_COL_H_STOCK_NAME].astype(str).str.strip(),
        '액면수량': pd.to_numeric(ap_df.iloc[:, AP_COL_M_FACE_QTY], errors='coerce'),
        '원취득가액': pd.to_numeric(ap_df.iloc[:, AP_COL_Q_ORIG_COST], errors='coerce'),
        '적용평가액': pd.to_numeric(ap_df.iloc[:, AP_COL_S_VALUE], errors='coerce'),
        '편입일': pd.to_datetime(ap_df.iloc[:, AP_COL_X_BUY_DATE], errors='coerce'),
    })
    base = base.dropna(subset=['적용평가액', '편입일', '액면수량', '원취득가액'])
    base = base[(base['적용평가액'] > 0) & (base['액면수량'] > 0) & (base['원취득가액'] > 0)]
    base = base[base['종목명'].str.len() > 0]
    if base.empty:
        return pd.DataFrame()

    base['분류'] = [_classify_fund(c, n) for c, n in zip(base['펀드코드'], base['펀드명'])]

    # ── 액티브 재간접주식형 전용 정제 (인덱스 그룹은 손대지 않음) ─────────
    # (1) 제외 키워드 매치 행 drop
    is_active = base['분류'] == '액티브 재간접주식형'
    if is_active.any():
        excl_mask = is_active & base['종목명'].apply(_should_exclude_active)
        if excl_mask.any():
            base = base[~excl_mask].copy()
    # (2) 보통예금 / 은대 → '은대' 로 종목명 통합 (집계 시 자연스레 합산)
    is_active = base['분류'] == '액티브 재간접주식형'
    if is_active.any():
        base.loc[is_active, '종목명'] = (
            base.loc[is_active, '종목명'].apply(_unify_active_name)
        )

    if base.empty:
        return pd.DataFrame()

    if 'ETF명' in df_uni.columns:
        name_to_ticker = {
            str(n).strip(): str(t)
            for t, n in zip(df_uni.index, df_uni['ETF명'])
            if isinstance(n, str) and n
        }
    else:
        name_to_ticker = {}
    base['티커'] = base['종목명'].apply(lambda n: _resolve_ticker_from_name(n, name_to_ticker))

    rows = []
    for cat in ('인덱스 재간접주식형', '액티브 재간접주식형'):
        sub = base[base['분류'] == cat]
        if sub.empty:
            continue
        total = float(sub['적용평가액'].sum())
        for nm, g in sub.groupby('종목명'):
            value_sum = float(g['적용평가액'].sum())
            cost_sum = float(g['원취득가액'].sum())
            qty_sum = float(g['액면수량'].sum())
            tk_series = g['티커'].replace('', np.nan).dropna()
            ticker = str(tk_series.iloc[0]) if len(tk_series) > 0 else ''
            # 가중평균단가 = SUM(원취득가액) / SUM(액면수량) — 표준 평균단가
            w_price = (cost_sum / qty_sum) if qty_sum > 0 else np.nan
            rows.append({
                '분류': cat,
                '종목명': nm,
                '티커': ticker,
                '적용평가액': value_sum,
                '비중%': (value_sum / total * 100) if total > 0 else np.nan,
                '편입일': g['편입일'].min(),
                '가중평균단가': w_price,
                '원취득가액': cost_sum,
                '액면수량': qty_sum,
            })
    return pd.DataFrame(rows)


def _enrich_with_returns(agg_df: pd.DataFrame,
                         bm_by_category: dict,
                         end_str: str) -> pd.DataFrame:
    """편입일~오늘 ETF 수익률 + 분류별 BM 수익률 + 초과성과 계산.

    Parameters
    ----------
    bm_by_category : dict[str, pd.Series]
        {'인덱스 재간접주식형': KOSPI200_close, '액티브 재간접주식형': KOSPI_close}
        분류별로 다른 BM 사용 — 각 행의 '분류' 값에 따라 매핑.
    """
    if agg_df is None or agg_df.empty:
        return agg_df

    out = agg_df.copy()
    # 분류별 BM 최신가 캐시
    bm_latest_by_cat = {
        cat: float(s.iloc[-1]) if isinstance(s, pd.Series) and not s.empty else np.nan
        for cat, s in bm_by_category.items()
    }

    today_ts = pd.Timestamp.today().normalize()
    cur, ret_pcts, bm_pcts, excess_pcts, bm_names, hold_days = [], [], [], [], [], []
    for _, row in out.iterrows():
        ticker = str(row.get('티커') or '').strip()
        inception = row.get('편입일')
        avg_cost = row.get('가중평균단가')
        category = str(row.get('분류') or '')

        # 보유기간 (일) — 편입일 ~ 오늘
        if pd.isna(inception):
            hold_days.append(np.nan)
        else:
            d = (today_ts - pd.Timestamp(inception)).days
            hold_days.append(int(d) if d >= 0 else np.nan)

        bm_close = bm_by_category.get(category, pd.Series(dtype=float))
        bm_latest = bm_latest_by_cat.get(category, np.nan)
        bm_name = _BM_NAME_BY_CATEGORY.get(category, '?')

        c_now = np.nan
        r_pct = np.nan
        bm_r = np.nan

        if ticker and not pd.isna(inception):
            start_str = pd.Timestamp(inception).strftime('%Y%m%d')
            try:
                close_ser = _cached_etf_close_series(ticker, start_str, end_str)
            except Exception:
                close_ser = pd.Series(dtype=float)
            if isinstance(close_ser, pd.Series) and not close_ser.empty:
                c_now = float(close_ser.iloc[-1])
                if avg_cost and not pd.isna(avg_cost) and avg_cost > 0:
                    r_pct = (c_now / avg_cost - 1.0) * 100

        if not pd.isna(inception) and isinstance(bm_close, pd.Series) \
                and not bm_close.empty and not np.isnan(bm_latest):
            inc_ts = pd.Timestamp(inception)
            after = bm_close.index[bm_close.index >= inc_ts]
            if len(after) > 0:
                bm_at = float(bm_close.loc[after[0]])
                if bm_at > 0:
                    bm_r = (bm_latest / bm_at - 1.0) * 100

        excess = (r_pct - bm_r) if (not np.isnan(r_pct) and not np.isnan(bm_r)) else np.nan

        cur.append(c_now)
        ret_pcts.append(r_pct)
        bm_pcts.append(bm_r)
        excess_pcts.append(excess)
        bm_names.append(bm_name)

    out['보유기간(일)'] = hold_days
    out['현재가'] = cur
    out['수익률%'] = ret_pcts
    out['BM명'] = bm_names
    out['BM수익률%'] = bm_pcts
    out['초과성과%'] = excess_pcts
    return out


def _ap_upload_section():
    """📤 AP (Actual Portfolio) 엑셀 업로드.

    파일을 읽어 `st.session_state['ap_df']` 에 DataFrame 으로 저장한다.
    가공 로직은 추후 추가 (현재는 업로드 + 미리보기까지).
    지원 포맷: .xlsx / .xls / .csv (utf-8, cp949 자동 시도).
    """
    has_ap = 'ap_df' in st.session_state and st.session_state['ap_df'] is not None

    with st.expander("📤 AP (Actual Portfolio) 업로드", expanded=not has_ap):
        col_up, col_meta = st.columns([2, 1])
        with col_up:
            uploaded = st.file_uploader(
                "AP 엑셀 / CSV 선택",
                type=['xlsx', 'xls', 'csv'],
                accept_multiple_files=False,
                key='ap_uploader',
                help="실 보유 포트폴리오(AP) 파일을 업로드하면 session_state['ap_df'] 에 저장됩니다.",
            )
        with col_meta:
            if has_ap:
                meta = st.session_state.get('ap_meta', {})
                st.metric("로드된 행 수", f"{meta.get('rows', '?'):,}")
                st.caption(f"파일: `{meta.get('name', '?')}`")
                if st.button("🗑️ 제거", key='ap_clear', help="세션에서 AP 데이터 제거"):
                    st.session_state['ap_df'] = None
                    st.session_state['ap_meta'] = None
                    st.rerun()

        if uploaded is not None:
            try:
                ext = uploaded.name.rsplit('.', 1)[-1].lower()
                if ext == 'csv':
                    # 인코딩 자동 시도 (utf-8 → utf-8-sig → cp949)
                    raw = uploaded.read()
                    for enc in ('utf-8', 'utf-8-sig', 'cp949', 'euc-kr'):
                        try:
                            from io import BytesIO
                            ap_df = pd.read_csv(BytesIO(raw), encoding=enc)
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    else:
                        st.error("CSV 인코딩 감지 실패 (utf-8/cp949 모두 실패)")
                        return
                else:
                    ap_df = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"파일 파싱 실패: {type(e).__name__}: {e}")
                return

            st.session_state['ap_df'] = ap_df
            st.session_state['ap_meta'] = {
                'name': uploaded.name,
                'rows': len(ap_df),
                'cols': list(ap_df.columns),
            }
            st.success(f"✅ 업로드 완료 — {len(ap_df)}행 × {len(ap_df.columns)}열")

        # 미리보기 — 업로드 후 또는 기존 세션 데이터 있을 때
        ap_df = st.session_state.get('ap_df')
        if isinstance(ap_df, pd.DataFrame) and not ap_df.empty:
            st.caption(f"컬럼: `{', '.join(map(str, ap_df.columns))}`")
            st.dataframe(ap_df.head(20), width='stretch', height=300)
            st.caption("👉 가공 로직은 추후 연결 — `st.session_state['ap_df']` 로 접근 가능.")


# ── 보유 섹터·KOSPI200 이슈/센티먼트 ────────────────────────────────────

SENTIMENT_ICONS = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}

# 뉴스/센티먼트 쿼리에서 제외할 종목명 키워드 (현금성 / 예금성 / 비-증권)
SENTIMENT_EXCLUDE_NAME_KEYWORDS = ('은대', '보통예금', '현금', 'MMF')


def _is_sentiment_exempt(name: str) -> bool:
    """예금·현금 등 뉴스 분석 대상이 아닌 항목인지 판정."""
    if not isinstance(name, str):
        return False
    return any(k in name for k in SENTIMENT_EXCLUDE_NAME_KEYWORDS)


# KOSPI200 ETF 판정 시 다른 지수/규모 키워드 제외 (오탐 방지)
_KOSPI200_EXCLUDE_KEYWORDS = (
    '코스닥', 'KOSDAQ', 'S&P', '500', '나스닥', 'NASDAQ',
    'NIKKEI', 'CSI', 'CHINA', '미국', '글로벌', 'GLOBAL',
    'MSCI', '신흥', 'EMERGING', '인버스', '레버리지',
    '2X', '3X', '곱버스', '일본', '유럽', '베트남', '인도',
)

# 200 토큰 정규식 (단어 경계 또는 비숫자 인접)
_KOSPI200_TOKEN_RE = re.compile(r'(?:^|[^\d])200(?:$|[^\d])')


def _is_kospi200_etf(etf_name: str) -> bool:
    """ETF명 패턴으로 KOSPI200 판정. '200' 토큰 + 타지수/레버리지 키워드 제외."""
    if not isinstance(etf_name, str):
        return False
    n = etf_name.upper()
    if any(k.upper() in n for k in _KOSPI200_EXCLUDE_KEYWORDS):
        return False
    return bool(_KOSPI200_TOKEN_RE.search(n))


def _render_sentiment_card(label: str, result: dict):
    icon = SENTIMENT_ICONS.get(result.get('sentiment', 'neutral'), '🟡')
    with st.container(border=True):
        st.markdown(f"### {icon} {label}")
        if result.get('error'):
            st.warning(result['error'])
        else:
            summary = result.get('summary', '').strip()
            if summary:
                # 1줄 1bullet 으로 변환
                for line in summary.split('\n'):
                    line = line.strip().lstrip('-•').strip()
                    if line:
                        st.markdown(f"- {line}")
            else:
                st.caption("요약 없음")
            kws = result.get('keywords', [])
            if kws:
                tags = ' '.join(f"`{k}`" for k in kws if k)
                st.markdown(f"**키워드**: {tags}")
        headlines = result.get('headlines', [])
        if headlines:
            with st.expander(f"📰 원본 헤드라인 {min(len(headlines), 5)}건"):
                for h in headlines[:5]:
                    title = h.get('title', '')
                    link = h.get('link', '')
                    pub = h.get('pubDate', '')[:16]
                    if title and link:
                        st.markdown(f"- [{title}]({link}) — _{pub}_")


def _collect_sentiment_targets(enriched: pd.DataFrame,
                               df_uni: pd.DataFrame) -> tuple:
    """AP + 저장 MP 보유 종목을 합쳐서 (has_kospi200, sectors_set) 반환."""
    has_k200 = False
    sectors = set()

    cat_map = (df_uni['중카테고리'].astype(str).to_dict()
               if '중카테고리' in df_uni.columns else {})

    if isinstance(enriched, pd.DataFrame) and not enriched.empty:
        for _, row in enriched.iterrows():
            name = row.get('종목명', '') or ''
            ticker = str(row.get('티커', '') or '').strip()
            if _is_sentiment_exempt(name):
                continue  # 은대 / 보통예금 / MMF 등 — 뉴스 분석 대상 아님
            if _is_kospi200_etf(name):
                has_k200 = True
                continue
            if ticker and ticker in cat_map:
                cat = cat_map.get(ticker, '').strip()
                if cat:
                    sectors.add(cat)

    # 저장된 MP 포지션도 포함
    saved = load_mp()
    if saved:
        for pos in saved.get('positions', []):
            name = str(pos.get('representative', '') or '')
            cat = str(pos.get('category', '') or '').strip()
            if _is_sentiment_exempt(name):
                continue
            if _is_kospi200_etf(name):
                has_k200 = True
            elif cat and cat not in ('코스피200 베타',):
                sectors.add(cat)

    return has_k200, sectors


def _ap_sentiment_section(enriched: pd.DataFrame, df_uni: pd.DataFrame):
    """🔎 AP + MP 보유 섹터·KOSPI200 이슈/센티먼트 — 6h 캐시, 버튼 트리거."""
    if not isinstance(enriched, pd.DataFrame) or enriched.empty:
        return
    try:
        from news_sentiment import (
            get_sector_sentiment, get_kospi200_sentiment, keys_configured,
        )
    except ImportError as e:
        st.warning(f"news_sentiment 모듈 로드 실패: {e}")
        return

    st.markdown("---")
    st.subheader("🔎 보유 섹터·KOSPI200 이슈 / 센티먼트")
    st.caption("네이버 뉴스 검색 → Claude(haiku) 3줄 요약. 6시간 캐시.")

    keys = keys_configured()
    if not (keys['naver'] and keys['anthropic']):
        with st.expander("⚙️ API 키 설정 가이드 (펼쳐서 확인)", expanded=True):
            st.markdown(
                f"""
이 기능을 사용하려면 두 가지 API 키가 필요합니다.

**설정 위치** (택 1):
- `.streamlit/secrets.toml` (로컬·Streamlit Cloud 공통)
- 환경 변수

```toml
# .streamlit/secrets.toml
anthropic_api_key = "sk-ant-..."
naver_news_client_id = "..."
naver_news_client_secret = "..."
```

**발급처**
- Anthropic: https://console.anthropic.com (Console → API Keys)
- 네이버 개발자 센터: https://developers.naver.com → 애플리케이션 등록 →
  사용 API 에 **검색** 추가

**현재 상태**
- 네이버 뉴스 키: {'✅ 설정됨' if keys['naver'] else '❌ 미설정'}
- Anthropic 키: {'✅ 설정됨' if keys['anthropic'] else '❌ 미설정'}
                """
            )
        return

    has_k200, sectors = _collect_sentiment_targets(enriched, df_uni)
    if not has_k200 and not sectors:
        st.caption("KOSPI200 ETF 도 매핑된 섹터 ETF 도 발견되지 않았습니다.")
        return

    summary_chip = []
    if has_k200:
        summary_chip.append("KOSPI200 (매크로)")
    summary_chip.extend(sorted(sectors))
    st.caption(f"대상 ({len(summary_chip)}): " + ", ".join(f"`{s}`" for s in summary_chip))

    col_b1, col_b2 = st.columns([1, 4])
    with col_b1:
        run = st.button("🔎 이슈 불러오기", type='primary', key='ap_sent_run',
                        help="6시간 캐시 — 같은 섹터·키워드는 재호출 없음.")
    with col_b2:
        if st.button("🔁 캐시 무시 새로고침", key='ap_sent_refresh',
                     help="강제로 새 뉴스 fetch + LLM 재요약 (캐시 초기화)."):
            get_sector_sentiment.clear()
            get_kospi200_sentiment.clear()
            from news_sentiment import fetch_naver_news
            fetch_naver_news.clear()
            st.session_state['ap_sentiment_loaded'] = True
            run = True

    if run:
        st.session_state['ap_sentiment_loaded'] = True

    if not st.session_state.get('ap_sentiment_loaded'):
        st.caption("👆 버튼을 눌러 뉴스 요약을 불러오세요.")
        return

    if has_k200:
        with st.spinner("KOSPI200 / 한국 매크로 뉴스 요약..."):
            try:
                result = get_kospi200_sentiment()
            except Exception as e:
                result = {'error': f"{type(e).__name__}: {e}",
                          'sentiment': 'neutral', 'summary': '',
                          'keywords': [], 'headlines': []}
        _render_sentiment_card("KOSPI200 / 한국 증시 매크로", result)

    for sector in sorted(sectors):
        with st.spinner(f"'{sector}' 섹터 뉴스 요약..."):
            try:
                result = get_sector_sentiment(sector)
            except Exception as e:
                result = {'error': f"{type(e).__name__}: {e}",
                          'sentiment': 'neutral', 'summary': '',
                          'keywords': [], 'headlines': []}
        _render_sentiment_card(f"{sector} 섹터", result)


def _compute_mp_port_excess():
    """저장된 MP 의 포트폴리오 비중 가중 누적 초과수익률(% vs KOSPI) 계산.

    Returns dict | None.  네트워크 호출이 일어나므로 분석 실행 시점에만 호출.
    """
    saved = load_mp()
    if not saved:
        return None
    try:
        from etf_universe_builder import (
            naver_get_price_history, naver_get_index_history,
        )

        def _fc(t, s, e):
            return naver_get_price_history(t, s, e)

        def _fk(s, e):
            return naver_get_index_history('KOSPI', s, e)

        perf = compute_mp_performance(saved, _fc, _fk)
        if perf.empty:
            return None
        port_cum = float(
            (perf['시작비중%'] / 100.0 * perf['누적%'].fillna(0)).sum()
        )
        return {
            'mp_cum': port_cum,
            'inception': str(saved.get('inception_date', '?')),
            'method': str(saved.get('method', '?')),
            'n_pos': int(len(saved.get('positions', []))),
        }
    except Exception:
        return None


def _render_ap_mp_comparison(enriched: pd.DataFrame):
    """포트폴리오 수준 비교 — AP (KOSPI200 BM) vs 저장 MP (KOSPI BM)."""
    st.markdown("---")
    st.subheader("📊 AP vs MP 포트폴리오 수준 비교 (비중 가중 초과성과)")

    valid = enriched.dropna(subset=['초과성과%'])
    if not valid.empty and valid['적용평가액'].sum() > 0:
        ap_excess = float(
            (valid['초과성과%'] * valid['적용평가액']).sum()
            / valid['적용평가액'].sum()
        )
        # 단순 수익률(비교용)
        valid_r = enriched.dropna(subset=['수익률%'])
        ap_ret = (
            float((valid_r['수익률%'] * valid_r['적용평가액']).sum()
                  / valid_r['적용평가액'].sum())
            if not valid_r.empty and valid_r['적용평가액'].sum() > 0 else np.nan
        )
    else:
        ap_excess = np.nan
        ap_ret = np.nan

    mp_info = st.session_state.get('ap_mp_compare')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "AP 비중 가중 수익률",
        f"{ap_ret:+.2f}%" if not np.isnan(ap_ret) else "N/A",
    )
    c2.metric(
        "AP 비중 가중 초과성과 (혼합 BM)",
        f"{ap_excess:+.2f}%" if not np.isnan(ap_excess) else "N/A",
        help="인덱스 = vs KOSPI200, 액티브 = vs KOSPI 의 비중 가중 평균",
    )
    if mp_info:
        c3.metric(
            f"MP 비중 가중 초과성과 (vs KOSPI)",
            f"{mp_info['mp_cum']:+.2f}%",
            help=f"편입일 {mp_info['inception']} · 방법 {mp_info['method']} · "
                 f"{mp_info['n_pos']}개 포지션",
        )
        diff = (ap_excess - mp_info['mp_cum']) if not np.isnan(ap_excess) else np.nan
        c4.metric(
            "차이 (AP − MP)",
            f"{diff:+.2f}%p" if not np.isnan(diff) else "N/A",
            help="양수면 AP 가 MP 보다 초과수익률 우위",
        )
    else:
        c3.metric("MP 비중 가중 초과성과", "N/A")
        c4.metric("차이 (AP − MP)", "N/A", help="저장된 MP 없음")

    st.caption(
        "AP 분류별 BM: 인덱스 재간접 → KOSPI200 / 액티브 재간접 → KOSPI. "
        "MP BM: KOSPI (저장 MP 기존 계산 유지). 모든 BM 은 실제 지수값."
    )


def _ap_processing_section(df_uni: pd.DataFrame):
    """🔬 AP 초과성과 분석 — 업로드된 ap_df 가 있을 때만 표시."""
    ap_df = st.session_state.get('ap_df')
    if not isinstance(ap_df, pd.DataFrame) or ap_df.empty:
        return

    st.subheader("🔬 AP 초과성과 분석")
    with st.expander("📐 분석 규칙", expanded=False):
        st.markdown("""
- **컬럼 매핑** (Excel 위치 기준): C=펀드코드, D=펀드명, H=종목명,
  M=액면수량, Q=원취득가액, S=적용평가액, X=최초 매수일자(편입일).
- **분류**: 펀드코드 ∈ `V5202E / V5304R / V6303V / V72026` 또는 펀드명에
  '인덱스' 포함 → **인덱스 재간접주식형**, 그 외 → **액티브 재간접주식형**.
- **액티브 정제** (인덱스 그룹은 적용 X):
  - 종목명 `더제이 더행복코리아`·`더행복코리아 공모펀드` 행은 제외.
  - 종목명 `보통예금` 또는 `은대` 포함 행은 `은대` 로 통합 (집계 시 합산).
- **집계**: 분류별 × 종목명별 — 적용평가액 sum, 비중%, 최초 매수일.
- **가중평균단가**: ETF 그룹별 `SUM(원취득가액) / SUM(액면수량)` (표준 평균단가).
- **수익률**: `(현재가 / 가중평균단가 − 1) × 100`
- **BM** (분류별 분리):
  - 인덱스 재간접주식형 → **KOSPI200** (네이버 KPI200 / yfinance `^KS200`)
  - 액티브 재간접주식형 → **KOSPI** (네이버 KOSPI / yfinance `^KS11`)
  - 둘 다 실제 지수 데이터 (EWY 등 ETF 대용 아님). 편입일 직후 첫 영업일
    종가 → 최신 종가 비교.
- **초과성과**: ETF 수익률 − 해당 분류 BM 수익률 (행별 `BM명` 컬럼 참고).
- **포트폴리오 비교**: 전체 AP 비중 가중 초과성과(혼합 BM)를 저장 MP
  비중 가중 초과성과(KOSPI BM)와 대조.
        """)

    run = st.button("🚀 분석 실행", type='primary', key='ap_run',
                    help="네이버 + yfinance 가격 수집 → 종목별 초과성과 계산. 첫 실행 30~90초.")

    if run:
        try:
            with st.spinner("AP 파싱·분류·집계 중..."):
                agg_df = _parse_ap_for_analysis(ap_df, df_uni)
            if agg_df.empty:
                st.warning("AP 데이터에서 유효 행을 찾지 못했습니다 "
                           "(적용평가액·편입일·적용단가 결측 또는 0).")
                st.session_state['ap_analysis'] = None
                return

            end_str = datetime.today().strftime('%Y%m%d')
            min_incept = pd.Timestamp(agg_df['편입일'].min())
            bm_start = (min_incept - pd.Timedelta(days=10)).strftime('%Y%m%d')

            with st.spinner("BM 지수 수집 (KOSPI / KOSPI200)..."):
                bm_kospi200 = _cached_kospi200_close(bm_start, end_str)
                bm_kospi = _cached_kospi_close(bm_start, end_str)
            if bm_kospi200.empty:
                st.error("KOSPI200 가격 수집 실패 → 인덱스 그룹 BM/초과성과 NaN.")
            if bm_kospi.empty:
                st.error("KOSPI 가격 수집 실패 → 액티브 그룹 BM/초과성과 NaN.")
            bm_by_cat = {
                '인덱스 재간접주식형': bm_kospi200,
                '액티브 재간접주식형': bm_kospi,
            }

            with st.spinner(f"{len(agg_df)}개 종목 가격 수집·수익률 계산..."):
                enriched = _enrich_with_returns(agg_df, bm_by_cat, end_str)

            with st.spinner("저장된 MP 포트폴리오 누적 초과수익률 계산..."):
                st.session_state['ap_mp_compare'] = _compute_mp_port_excess()

            st.session_state['ap_analysis'] = enriched
            st.success(f"✅ 분석 완료 — {len(enriched)}개 종목 (인덱스 / 액티브 분류 표시)")
        except Exception as e:
            st.error(f"분석 실패: {type(e).__name__}: {e}")
            return

    enriched = st.session_state.get('ap_analysis')
    if not isinstance(enriched, pd.DataFrame) or enriched.empty:
        st.caption("👆 **분석 실행** 버튼을 눌러주세요.")
        return

    tab_idx, tab_act = st.tabs(['🟢 인덱스 재간접주식형', '🔵 액티브 재간접주식형'])

    def _render_category(label: str):
        sub = enriched[enriched['분류'] == label]
        if sub.empty:
            st.info(f"{label} 카테고리에 해당 종목이 없습니다.")
            return

        total_val = float(sub['적용평가액'].sum())
        valid = sub.dropna(subset=['초과성과%'])
        if not valid.empty and valid['적용평가액'].sum() > 0:
            wavg_excess = float(
                (valid['초과성과%'] * valid['적용평가액']).sum()
                / valid['적용평가액'].sum()
            )
        else:
            wavg_excess = np.nan
        n_unmapped = int((sub['티커'] == '').sum() + sub['현재가'].isna().sum()
                         - ((sub['티커'] == '') & sub['현재가'].isna()).sum())

        bm_name = _BM_NAME_BY_CATEGORY.get(label, '?')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("종목 수", f"{len(sub)}")
        c2.metric("총 적용평가액", f"{total_val:,.0f}")
        c3.metric(f"비중 가중 초과성과 (vs {bm_name})",
                  f"{wavg_excess:+.2f}%" if not np.isnan(wavg_excess) else "N/A")
        c4.metric("매핑/가격 실패", f"{n_unmapped}개")

        view = sub[['종목명', '티커', '적용평가액', '비중%', '편입일', '보유기간(일)',
                    '가중평균단가', '현재가', '수익률%', 'BM명', 'BM수익률%',
                    '초과성과%']].copy()
        view['편입일'] = pd.to_datetime(view['편입일']).dt.strftime('%Y-%m-%d')
        st.dataframe(
            view.sort_values('비중%', ascending=False),
            width='stretch', hide_index=True, height=440,
            column_config={
                '적용평가액': st.column_config.NumberColumn('적용평가액', format='%,.0f'),
                '비중%': st.column_config.ProgressColumn(
                    '비중 %', min_value=0.0, max_value=100.0, format='%.2f%%'),
                '보유기간(일)': st.column_config.NumberColumn(
                    '보유기간 (일)', format='%d',
                    help="편입일 → 오늘. 수익률·BM수익률 모두 이 기간의 누적값입니다."),
                '가중평균단가': st.column_config.NumberColumn('가중평균단가', format='%,.2f'),
                '현재가': st.column_config.NumberColumn('현재가', format='%,.2f'),
                '수익률%': st.column_config.NumberColumn('수익률 %', format='%+.2f'),
                'BM명': st.column_config.TextColumn('BM', width='small'),
                'BM수익률%': st.column_config.NumberColumn('BM 수익률 %', format='%+.2f'),
                '초과성과%': st.column_config.NumberColumn('초과성과 %', format='%+.2f'),
            },
        )

        unmapped = sub[(sub['티커'] == '') | sub['현재가'].isna()]
        if not unmapped.empty:
            with st.expander(f"⚠️ 가격 매핑/수집 실패 종목 {len(unmapped)}개"):
                st.dataframe(
                    unmapped[['종목명', '티커', '적용평가액', '편입일']],
                    width='stretch', hide_index=True,
                )
                st.caption(
                    "원인: 종목명이 국내 유니버스에 없음 (해외 ETF 등) 또는 "
                    "네이버에서 해당 종목 시계열 미제공."
                )

    with tab_idx:
        _render_category('인덱스 재간접주식형')
    with tab_act:
        _render_category('액티브 재간접주식형')

    # ── AP vs MP 포트폴리오 수준 비교 ──────────────────────────────────────
    _render_ap_mp_comparison(enriched)

    # ── 🔎 보유 섹터·KOSPI200 이슈/센티먼트 ───────────────────────────────
    _ap_sentiment_section(enriched, df_uni)


def _score_color(c: float) -> str:
    if np.isnan(c):
        return '#888888'
    if c >= 30:
        return '#2ecc71'
    if c <= -30:
        return '#e74c3c'
    return '#f39c12'


def _make_chart(df_ohlcv: pd.DataFrame, ticker: str, etf_name: str,
                C_score: float, T_score: float) -> go.Figure:
    """
    캔들스틱(Row1) + 볼린저밴드 오버레이 + RSI(Row2).
    마우스 휠 줌: scrollZoom=True + fixedrange=False.
    기본 뷰: 최근 3개월, 데이터: 최근 6개월.
    """
    label = f"{ticker}" + (f" {etf_name[:10]}" if etf_name else "")

    if df_ohlcv.empty or len(df_ohlcv) < 20 or 'Close' not in df_ohlcv.columns:
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", x=0.5, y=0.5,
                           showarrow=False, xref="paper", yref="paper",
                           font=dict(color='#666', size=10))
        fig.update_layout(
            title=dict(text=f"<b>{label}</b>", font=dict(size=10), x=0.03),
            height=290, margin=dict(l=4, r=4, t=26, b=4),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        )
        return fig

    close = df_ohlcv['Close'].squeeze()
    open_ = df_ohlcv['Open'].squeeze() if 'Open' in df_ohlcv.columns else close
    high  = df_ohlcv['High'].squeeze() if 'High' in df_ohlcv.columns else close
    low   = df_ohlcv['Low'].squeeze()  if 'Low'  in df_ohlcv.columns else close

    pb, upper, lower, mid = compute_bollinger(close)
    rsi = compute_rsi(close)

    score_clr = _score_color(C_score)
    score_txt = f"N/A" if np.isnan(C_score) else f"{C_score:+.1f}"

    # 기본 뷰 범위: 최근 DEFAULT_VIEW_MONTHS개월
    view_start = (datetime.today() - timedelta(days=DEFAULT_VIEW_MONTHS * 31)).strftime('%Y-%m-%d')
    view_end   = datetime.today().strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # ── Row 1: 캔들 ──
    fig.add_trace(go.Candlestick(
        x=df_ohlcv.index,
        open=open_, high=high, low=low, close=close,
        increasing_line_color='#e74c3c',
        decreasing_line_color='#3498db',
        showlegend=False,
    ), row=1, col=1)

    # ── 볼린저밴드 ──
    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=upper,
        line=dict(color='rgba(180,180,180,0.5)', width=1),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=lower,
        line=dict(color='rgba(180,180,180,0.5)', width=1),
        fill='tonexty', fillcolor='rgba(150,150,150,0.07)',
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=mid,
        line=dict(color='rgba(150,150,255,0.7)', width=1, dash='dot'),
        showlegend=False,
    ), row=1, col=1)

    # ── Row 2: RSI ──
    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=rsi,
        line=dict(color='#9b59b6', width=1.5),
        showlegend=False,
    ), row=2, col=1)
    for lvl, clr in [(70, 'rgba(231,76,60,0.35)'), (50, 'rgba(200,200,200,0.2)'),
                     (30, 'rgba(52,152,219,0.35)')]:
        fig.add_hline(y=lvl, line_dash='dash', line_color=clr,
                      line_width=1, row=2, col=1)

    # ── C_score 오버레이 ──
    fig.add_annotation(
        x=1, y=1, xref='paper', yref='paper',
        text=f"<b>C:{score_txt}</b>",
        showarrow=False, xanchor='right', yanchor='top',
        font=dict(size=12, color=score_clr),
        bgcolor='rgba(14,17,23,0.80)',
        bordercolor=score_clr, borderwidth=1, borderpad=3,
    )

    fig.update_layout(
        title=dict(text=f"<b>{label}</b>", font=dict(size=10), x=0.03, y=0.97),
        height=290,
        margin=dict(l=4, r=4, t=26, b=4),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ccc', size=8),
        # 마우스 휠 줌: x축 기본 범위를 최근 3개월로 설정
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=False,
            fixedrange=False,             # 휠 줌 허용
            range=[view_start, view_end], # 기본 뷰 3개월
        ),
        xaxis2=dict(
            showgrid=False,
            fixedrange=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#1e2130',
            tickfont=dict(size=7),
            fixedrange=False,
        ),
        yaxis2=dict(
            showgrid=True, gridcolor='#1e2130',
            tickfont=dict(size=7),
            range=[0, 100], dtick=25,
            fixedrange=True,              # RSI 축은 고정
        ),
        dragmode='pan',                   # 기본 조작: pan (휠=줌)
    )
    return fig


def _placeholder_chart(slot_num: int) -> go.Figure:
    """빈 그리드 셀용 placeholder."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"─ ETF {slot_num} ─<br><span style='font-size:10px'>위에서 선택</span>",
        x=0.5, y=0.5, showarrow=False,
        xref="paper", yref="paper",
        font=dict(color='#333', size=12),
        align='center',
    )
    fig.update_layout(
        height=290, margin=dict(l=4, r=4, t=26, b=4),
        paper_bgcolor='#0e1117', plot_bgcolor='#111520',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


# ── 메인 페이지 함수 ────────────────────────────────────────────────────────

def page_etf_uniview():
    st.title("🔭 ETF Uniview")

    # 유니버스 미빌드 체크
    if not st.session_state.get('universe_built', False):
        st.warning("👈 사이드바에서 먼저 **🚀 유니버스 빌드**를 실행하세요.")
        return

    df_uni = st.session_state.df_universe
    if df_uni is None or df_uni.empty:
        st.warning("유니버스 데이터가 없습니다. 다시 빌드해 주세요.")
        return

    # ── 📤 AP (Actual Portfolio) 업로드 + 🔬 분석 ──────────────────────────
    _ap_upload_section()
    _ap_processing_section(df_uni)

    # ── 📌 저장된 MP (편입일 기준 성과 추적) — 저장본 있을 때만 표시 ──────
    _saved_mp_section()

    # ── 🔥 Hot Sectors + Live MP ────────────────────────────────────────────
    _hot_board_section(df_uni)

    # ── ETF 풀: 국내 유니버스 (시총 상위 순 정렬) ──────────────────────────
    pool_df = df_uni.copy()
    if '시가총액(억원)' in pool_df.columns:
        pool_df = pool_df.sort_values('시가총액(억원)', ascending=False)

    pool_tickers = pool_df.index.tolist()
    pool_names   = pool_df['ETF명'].tolist() if 'ETF명' in pool_df.columns else pool_tickers
    pool_cats    = pool_df['중카테고리'].tolist() if '중카테고리' in pool_df.columns else [''] * len(pool_tickers)

    ticker_to_name = dict(zip(pool_tickers, pool_names))
    ticker_to_cat  = dict(zip(pool_tickers, pool_cats))

    pool_options = [f"{t} | {n[:28]}" for t, n in zip(pool_tickers, pool_names)]
    # 첫 진입 시 시총 Top 5만 기본 — 20개 자동 로드로 인한 yfinance 폭주 방지
    default_opts = pool_options[:5]

    st.caption(
        f"국내 상장 ETF 유니버스 {len(pool_tickers)}개 | "
        f"최대 {MAX_SELECT}개 선택 → 5×4 그리드 (빈 칸은 placeholder) | "
        "차트에서 **마우스 휠**: 시계열 줌 / **드래그**: 이동"
    )

    selected_opts = st.multiselect(
        f"📌 ETF 선택 (최대 {MAX_SELECT}개)",
        pool_options,
        default=default_opts,
        max_selections=MAX_SELECT,
        key="uniview_kr_sel",
    )

    selected_tickers = [s.split(' | ')[0] for s in selected_opts]
    n_sel = len(selected_tickers)

    # ── 가격 데이터 + 스코어 계산 ────────────────────────────────────────────
    ohlcv_dict: dict = {}
    scores_map: dict = {}

    if selected_tickers:
        with st.spinner("📡 국내 ETF 가격 데이터 수집 중... (yfinance .KS)"):
            macro_result = fetch_macro_scores()
            M_score = macro_result['M_score']
            ohlcv_dict = fetch_kr_ohlcv_batch(tuple(selected_tickers), months=FETCH_MONTHS)

        for t in selected_tickers:
            df_t = ohlcv_dict.get(t, pd.DataFrame())
            if df_t.empty or 'Close' not in df_t.columns:
                scores_map[t] = (np.nan, np.nan)
            else:
                close = df_t['Close'].squeeze().dropna()
                T = compute_T_score(close)
                C = compute_C_score(T, M_score)
                scores_map[t] = (round(T, 2) if not np.isnan(T) else np.nan,
                                 round(C, 2) if not np.isnan(C) else np.nan)

        # Macro 상세
        with st.expander("🌐 Macro Score 상세"):
            mc = macro_result['scores']
            md = macro_result['details']
            mc_cols = st.columns(6)
            for i, (lbl, key) in enumerate([
                ('Fed(-15)', 'fed'), ('Geo(-7)', 'geo'), ('WTI(-25)', 'oil'),
                ('BEI(+10)', 'bei'), ('VIX(-20)', 'vix'),
            ]):
                mc_cols[i].metric(lbl, f"{mc[key]:+.2f}")
            mc_cols[5].metric("M_score", f"{M_score:+.1f}")
            st.json({k: str(v) for k, v in md.items()})

    st.markdown("---")

    # ── 5×4 그리드 (항상 20칸 렌더링) ──────────────────────────────────────
    for row_i in range(GRID_ROWS):
        cols = st.columns(GRID_COLS)
        for col_j in range(GRID_COLS):
            idx = row_i * GRID_COLS + col_j
            with cols[col_j]:
                if idx >= n_sel:
                    # 빈 셀 → placeholder
                    fig = _placeholder_chart(idx + 1)
                    st.plotly_chart(fig, width='stretch',
                                    config={'displayModeBar': False,
                                            'scrollZoom': True})
                else:
                    ticker = selected_tickers[idx]
                    etf_name = ticker_to_name.get(ticker, '')
                    T_score, C_score = scores_map.get(ticker, (np.nan, np.nan))
                    df_ohlcv = ohlcv_dict.get(ticker, pd.DataFrame())
                    fig = _make_chart(df_ohlcv, ticker, etf_name, C_score, T_score)
                    st.plotly_chart(fig, width='stretch',
                                    config={'displayModeBar': False,
                                            'scrollZoom': True})

    # ── 랭킹 보드 ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 ETF 스코어 랭킹 보드")
    st.caption("전체 국내 ETF 유니버스 기준 (캐시 2시간)")

    all_tickers_tuple = tuple(pool_tickers)

    with st.spinner("전체 ETF C_score 계산 중..."):
        df_scores = score_all_kr_etfs(all_tickers_tuple)

    df_scores['ETF명']   = df_scores.index.map(lambda t: ticker_to_name.get(t, t))
    df_scores['카테고리'] = df_scores.index.map(lambda t: ticker_to_cat.get(t, ''))
    valid = df_scores.dropna(subset=['C_score'])

    disp_cols = ['ticker', 'ETF명', '카테고리', 'T_score', 'C_score']
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🟢 C_score 상위 15**")
        st.dataframe(valid.nlargest(15, 'C_score').reset_index()[disp_cols],
                     width='stretch', hide_index=True)
    with c2:
        st.markdown("**🔴 C_score 하위 15**")
        st.dataframe(valid.nsmallest(15, 'C_score').reset_index()[disp_cols],
                     width='stretch', hide_index=True)

    st.markdown("---")
    st.subheader("📊 카테고리 평균 C_score 랭킹")

    valid_cat = valid[valid['카테고리'].notna() & (valid['카테고리'] != '')]
    if not valid_cat.empty:
        cat_avg = (
            valid_cat.groupby('카테고리')['C_score']
            .agg(평균_C_score='mean', ETF_수='count')
            .sort_values('평균_C_score', ascending=False)
        )
        cat_avg['평균_C_score'] = cat_avg['평균_C_score'].round(2)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**🟢 카테고리 평균 상위 15**")
            st.dataframe(cat_avg.head(15).reset_index(), width='stretch', hide_index=True)
        with c4:
            st.markdown("**🔴 카테고리 평균 하위 15**")
            st.dataframe(cat_avg.tail(15).sort_values('평균_C_score').reset_index(),
                         width='stretch', hide_index=True)
    else:
        st.caption("카테고리 데이터 없음")
