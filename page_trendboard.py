"""
ETF Trend Board Page
====================
이미지 참고: ETF 트렌드보드 + 당일 순위 + 기간별 수익률 TOP

데이터 소스 (Phase A — 외부 fetch 없이 기존 세션 데이터만):
- st.session_state.df_universe : ETF 메타 + 수익률·BM 컬럼
- st.session_state.df_prices_kr : 일별 종가 (당일 등락률·1주 수익률 계산용)

Phase B/C 에서 추가 예정: 기초지수(네이버 파싱) / 자금유입(KRX 설정·환매)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta


# 한국식 색상 (rise=red, fall=blue)
UP_COLOR = '#e74c3c'
DOWN_COLOR = '#3498db'

# ETF 브랜드 prefix → 운용사 매핑
ETF_BRAND_TO_MANAGER = {
    'KODEX': '삼성자산운용',
    'TIGER': '미래에셋자산운용',
    'KOSEF': '키움자산운용',
    'KBSTAR': 'KB자산운용',
    'ARIRANG': '한화자산운용',
    'HANARO': 'NH아문디자산운용',
    'SOL': '신한자산운용',
    'ACE': '한국투자신탁운용',
    'KINDEX': '한국투자신탁운용',
    'WOORI': '우리자산운용',
    'KIWOOM': '키움자산운용',
    'WON': '신영자산운용',
    'TIMEFOLIO': '타임폴리오자산운용',
    'PLUS': '한화자산운용',
    'RISE': 'KB자산운용',
    'BNK': 'BNK자산운용',
    'IBK': 'IBK자산운용',
    '1Q': '하나자산운용',
    'TIME': '타임폴리오자산운용',
    'KOACT': '코액티브자산운용',
    'KCGI': 'KCGI자산운용',
    'UNICORN': '엠지자산운용',
    'TRUSTON': '트러스톤자산운용',
    'MIDAS': '미다스에셋자산운용',
    'HK': '흥국자산운용',
}


# ── 헬퍼 ────────────────────────────────────────────────────────────────
def _get_manager(etf_name: str) -> str:
    """ETF명 첫 단어 → 운용사."""
    if not isinstance(etf_name, str) or not etf_name.strip():
        return ''
    first = etf_name.strip().split()[0].upper()
    return ETF_BRAND_TO_MANAGER.get(first, '')


def _is_lev_inv(etf_name: str) -> bool:
    """레버리지/인버스 ETF 판정."""
    if not isinstance(etf_name, str):
        return False
    n = etf_name.upper()
    return any(k in n for k in ['레버리지', '인버스', '2X', '3X', '곱버스', 'LEVERAGED', 'INVERSE'])


def _today_change(df_close: pd.DataFrame, ticker: str) -> float:
    if df_close is None or ticker not in df_close.columns:
        return np.nan
    s = df_close[ticker].dropna()
    if len(s) < 2:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-2] - 1) * 100)


def _period_change(df_close: pd.DataFrame, ticker: str, n_days: int) -> float:
    if df_close is None or ticker not in df_close.columns:
        return np.nan
    s = df_close[ticker].dropna()
    if len(s) <= n_days:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-1 - n_days] - 1) * 100)


def _vol_ratio(df_close: pd.DataFrame, ticker: str, n_recent: int = 3, n_base: int = 20) -> float:
    """단순 거래량 ratio. (df_close에 거래량 없으므로 다른 시장 데이터로 대체 불가 — 시총 기반 별도 처리)"""
    return np.nan  # 일별 거래량 시계열 없음. 거래대금(억) 단일 값만 있음.


def _color_pct(v: float) -> str:
    if pd.isna(v):
        return '#888'
    return UP_COLOR if v >= 0 else DOWN_COLOR


def _fmt_pct(v: float, signed: bool = True) -> str:
    if pd.isna(v):
        return '-'
    sign = '+' if (signed and v > 0) else ''
    return f"{sign}{v:.2f}%"


def _fmt_num(v) -> str:
    if pd.isna(v):
        return '-'
    try:
        return f"{int(v):,}"
    except Exception:
        return str(v)


# ══════════════════════════════════════════════════════════════════════
# 섹션 1: ETF 트렌드보드 (좌측 카테고리 + 우측 Top 6)
# ══════════════════════════════════════════════════════════════════════
TB_CATEGORIES = ['🚀 RS TOP', '🔥 거래 급증', '⭐ 신규 상장', '💵 배당', '🪙 원자재', '🇺🇸 미국 빅테크']
TB_PERIODS = {'1개월': ('1M', 21), '3개월': ('3M', 63), '6개월': ('6M', 126), '1년': ('1Y', 252)}


def _filter_by_category(df: pd.DataFrame, cat: str, ret_col: str, bm_col: str) -> pd.DataFrame:
    cand = df.copy()
    if cat == '🚀 RS TOP':
        if bm_col in cand.columns:
            return cand.dropna(subset=[bm_col]).sort_values(bm_col, ascending=False)
        return cand.sort_values(ret_col, ascending=False, na_position='last')
    if cat == '🔥 거래 급증':
        # 일별 거래량 시계열 없음 → 빌드시점 거래대금 상위로 대체
        if '거래대금(억)' in cand.columns:
            return cand.dropna(subset=['거래대금(억)']).sort_values('거래대금(억)', ascending=False)
        return cand.head(0)
    if cat == '⭐ 신규 상장':
        if '설정일' not in cand.columns:
            return cand.head(0)
        cand['_listed'] = pd.to_datetime(cand['설정일'], errors='coerce')
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)
        cand = cand[cand['_listed'] >= cutoff]
        # 최근 설정 + 수익률 양호 순
        if ret_col in cand.columns:
            return cand.sort_values([ret_col, '_listed'], ascending=[False, False])
        return cand.sort_values('_listed', ascending=False)
    if cat == '💵 배당':
        mask = pd.Series(False, index=cand.index)
        if '대카테고리' in cand.columns:
            mask |= cand['대카테고리'].astype(str).str.contains('배당', na=False)
        if '중카테고리' in cand.columns:
            mask |= cand['중카테고리'].astype(str).str.contains('배당', na=False)
        if 'ETF명' in cand.columns:
            mask |= cand['ETF명'].astype(str).str.contains('배당|고배당|DIVIDEND', na=False, regex=True, case=False)
        cand = cand[mask]
        return cand.sort_values(ret_col, ascending=False, na_position='last') if ret_col in cand.columns else cand
    if cat == '🪙 원자재':
        mask = pd.Series(False, index=cand.index)
        if '대카테고리' in cand.columns:
            mask |= cand['대카테고리'].astype(str).str.contains('원자재', na=False)
        if '중카테고리' in cand.columns:
            mask |= cand['중카테고리'].astype(str).str.contains('금|은|구리|원유|농산물|원자재|GOLD|SILVER|OIL', na=False, regex=True, case=False)
        cand = cand[mask]
        return cand.sort_values(ret_col, ascending=False, na_position='last') if ret_col in cand.columns else cand
    if cat == '🇺🇸 미국 빅테크':
        mask = pd.Series(False, index=cand.index)
        if 'ETF명' in cand.columns:
            n = cand['ETF명'].astype(str).str.upper()
            mask |= (n.str.contains('미국') | n.str.contains('US')) & n.str.contains(
                '나스닥|빅테크|테크|TECH|FAANG|MAGNIFICENT|M7|NASDAQ', regex=True
            )
        cand = cand[mask]
        return cand.sort_values(ret_col, ascending=False, na_position='last') if ret_col in cand.columns else cand
    return cand.head(0)


def _render_card(slot_num: int, etf_row, ticker, ret_col, period_label, bm_col):
    """Top 6 카드 1개 렌더."""
    name = etf_row.get('ETF명', '')
    cat = etf_row.get('중카테고리', '') or etf_row.get('대카테고리', '')
    manager = etf_row.get('운용사', '')
    close = etf_row.get('종가', np.nan)
    today = etf_row.get('오늘등락(%)', np.nan)
    period_ret = etf_row.get(ret_col, np.nan)
    bm = etf_row.get(bm_col, np.nan) if bm_col in etf_row else np.nan

    period_color = _color_pct(period_ret)
    today_color = _color_pct(today)
    bm_color = _color_pct(bm)

    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        with h1:
            st.markdown(f"**{name}**  \n<span style='color:#888;font-size:11px'>{ticker} · {cat or '-'}</span>",
                        unsafe_allow_html=True)
        with h2:
            st.markdown(
                f"<div style='text-align:right'>"
                f"<span style='color:#888;font-size:10px'>{period_label}</span><br>"
                f"<span style='color:{period_color};font-weight:600;font-size:14px'>{_fmt_pct(period_ret)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<span style='color:#888;font-size:11px'>현재가</span><br><b>{_fmt_num(close)}</b>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<span style='color:#888;font-size:11px'>오늘</span><br>"
                f"<b style='color:{today_color}'>{_fmt_pct(today)}</b>",
                unsafe_allow_html=True,
            )
        with c3:
            label = 'RS' if not pd.isna(bm) else '운용'
            value = _fmt_pct(bm) if not pd.isna(bm) else (manager[:6] if manager else '-')
            color = bm_color if not pd.isna(bm) else '#666'
            st.markdown(
                f"<span style='color:#888;font-size:11px'>{label}</span><br>"
                f"<b style='color:{color}'>{value}</b>",
                unsafe_allow_html=True,
            )


def _section_trendboard(df: pd.DataFrame):
    st.subheader("📈 ETF 트렌드보드")
    period_label = st.radio(
        "기간", list(TB_PERIODS.keys()), horizontal=True, index=0,
        label_visibility='collapsed', key='tb_period',
    )
    period_code, _ = TB_PERIODS[period_label]
    ret_col = f"수익률_{period_code}(%)"
    bm_col = f"BM_{period_code}(%)"

    c_menu, c_cards = st.columns([1, 4])
    with c_menu:
        cat = st.radio(
            "카테고리", TB_CATEGORIES, label_visibility='collapsed', key='tb_cat',
        )

    candidates = _filter_by_category(df, cat, ret_col, bm_col)
    top6 = candidates.head(6)

    with c_cards:
        if top6.empty:
            st.info(f"{cat} 카테고리 후보 없음 (유니버스 빌드 또는 데이터 부족)")
        else:
            for row_i in range(2):
                cc = st.columns(3)
                for col_j in range(3):
                    slot = row_i * 3 + col_j
                    with cc[col_j]:
                        if slot < len(top6):
                            row = top6.iloc[slot]
                            ticker = top6.index[slot]
                            _render_card(slot + 1, row, ticker, ret_col, period_label, bm_col)
                        else:
                            st.empty()


# ══════════════════════════════════════════════════════════════════════
# 섹션 2: 당일 순위
# ══════════════════════════════════════════════════════════════════════
def _render_winner_card(rank: int, row, ticker: str, val: float, val_label: str, color: str):
    name = row.get('ETF명', '')
    brand = name.strip().split()[0] if isinstance(name, str) and name.strip() else ''
    rest = name.replace(brand, '', 1).strip() if brand else name
    with st.container(border=True):
        st.markdown(
            f"<span style='color:#888;font-size:11px'>🏆 {rank}위</span><br>"
            f"<span style='color:#666;font-size:11px'>{brand}</span><br>"
            f"<b style='font-size:14px'>{rest}</b><br>"
            f"<span style='color:#888;font-size:10px'>{ticker}</span><br>"
            f"<span style='color:{color};font-size:22px;font-weight:700'>{_fmt_pct(val) if 'rate' in val_label.lower() or '%' in val_label else _fmt_num(val)}</span>",
            unsafe_allow_html=True,
        )


def _section_today_rank(df: pd.DataFrame):
    st.subheader("📊 당일 순위")
    c1, c2 = st.columns([2.5, 1])
    with c1:
        st.caption("레버리지/인버스 제외 토글 + 정렬 기준 선택")
    with c2:
        excl_lev = st.checkbox("레버리지/인버스 제외", value=True, key='today_excl_lev')

    sort_mode = st.radio("정렬", ['등락률', '거래대금', '시가총액'], horizontal=True, index=0,
                         key='today_sort', label_visibility='collapsed')

    pool = df.copy()
    if excl_lev and 'ETF명' in pool.columns:
        pool = pool[~pool['ETF명'].apply(_is_lev_inv)]

    if sort_mode == '등락률':
        col = '오늘등락(%)'; is_pct = True
    elif sort_mode == '거래대금':
        col = '거래대금(억)' if '거래대금(억)' in pool.columns else None; is_pct = False
    else:
        col = '시가총액(억원)' if '시가총액(억원)' in pool.columns else None; is_pct = False

    if col is None or col not in pool.columns:
        st.info(f"'{sort_mode}' 컬럼이 유니버스 데이터에 없습니다.")
        return

    pool = pool.dropna(subset=[col])
    risers = pool.sort_values(col, ascending=False).head(7)
    fallers = pool.sort_values(col, ascending=True).head(7)

    c_up, c_dn = st.columns(2)

    def _render_side(title: str, df_side: pd.DataFrame, color: str, ascending: bool):
        st.markdown(f"**{title}**")
        if df_side.empty:
            st.info("데이터 없음"); return
        # Top 3 카드
        cc = st.columns(3)
        for i in range(min(3, len(df_side))):
            row = df_side.iloc[i]; ticker = df_side.index[i]; v = row[col]
            with cc[i]:
                _render_winner_card(i+1, row, ticker, v,
                                    'pct' if is_pct else '억', color if is_pct else '#444')
        # 4~7위 리스트
        for i in range(3, min(7, len(df_side))):
            row = df_side.iloc[i]; ticker = df_side.index[i]; v = row[col]
            name = row.get('ETF명', '')
            v_str = _fmt_pct(v) if is_pct else f"{_fmt_num(v)}{'억' if '거래대금' in sort_mode or '시가' in sort_mode else ''}"
            col_str = (UP_COLOR if v >= 0 else DOWN_COLOR) if is_pct else '#444'
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 10px;border-bottom:1px solid #2a2a2a;'>"
                f"<span style='color:#aaa'>{i+1}. {name} <span style='color:#666;font-size:10px'>{ticker}</span></span>"
                f"<span style='color:{col_str};font-weight:600'>{v_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with c_up:
        _render_side("🔴 상승 TOP", risers, UP_COLOR, ascending=False)
    with c_dn:
        _render_side("🔵 하락 TOP", fallers, DOWN_COLOR, ascending=True)


# ══════════════════════════════════════════════════════════════════════
# 섹션 3: 기간별 수익률 TOP 5
# ══════════════════════════════════════════════════════════════════════
PERIOD_RANK = {'1주': ('1W', 5), '1개월': ('1M', 21), '3개월': ('3M', 63),
               '6개월': ('6M', 126), '1년': ('1Y', 252), 'YTD': ('YTD', None)}


def _section_period_rank(df: pd.DataFrame, df_close: pd.DataFrame):
    st.subheader("📈 기간별 수익률 TOP 5")
    period_label = st.radio(
        "기간", list(PERIOD_RANK.keys()), horizontal=True, index=1,
        label_visibility='collapsed', key='period_rank_sel',
    )
    code, n_days = PERIOD_RANK[period_label]
    col = f"수익률_{code}(%)" if code != 'YTD' else '수익률_YTD(%)'

    pool = df.copy()
    if col not in pool.columns:
        # 1W 는 별도 계산
        if code == '1W' and df_close is not None:
            pool['수익률_1W(%)'] = pool.index.map(lambda t: _period_change(df_close, t, 5))
            col = '수익률_1W(%)'
        else:
            st.info(f"{col} 컬럼이 없습니다. (유니버스 빌드 후 사용 가능)")
            return

    pool = pool.dropna(subset=[col])
    if pool.empty:
        st.info("유효 데이터 없음"); return

    risers = pool.sort_values(col, ascending=False).head(5)
    fallers = pool.sort_values(col, ascending=True).head(5)

    c_up, c_dn = st.columns(2)

    def _render_list(title: str, df_side: pd.DataFrame, color: str):
        st.markdown(f"**{title}**")
        for i in range(len(df_side)):
            row = df_side.iloc[i]; ticker = df_side.index[i]; v = row[col]
            name = row.get('ETF명', '')
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:8px 12px;border-bottom:1px solid #2a2a2a;'>"
                f"<span><span style='color:#666;margin-right:8px'>{i+1}</span>"
                f"<b style='color:#ddd'>{name}</b> <span style='color:#666;font-size:10px'>{ticker}</span></span>"
                f"<b style='color:{color}'>{_fmt_pct(v)}</b></div>",
                unsafe_allow_html=True,
            )

    with c_up:
        _render_list("🔴 상승 TOP 5", risers, UP_COLOR)
    with c_dn:
        _render_list("🔵 하락 TOP 5", fallers, DOWN_COLOR)


# ══════════════════════════════════════════════════════════════════════
# 섹션 4: 자금 유입/이탈 (Phase C 예정 — placeholder)
# ══════════════════════════════════════════════════════════════════════
def _section_flow_placeholder():
    st.subheader("📦 자금 유입/이탈 TOP")
    st.info(
        "🚧 정확한 ETF 설정/환매 수량은 KRX 정보시스템에서 별도 수집이 필요합니다. "
        "**Phase C** 작업으로 추가 예정. "
        "현재는 시총 변화 기반 추정만 가능한데, 가격 변동·분배금 조정 등으로 노이즈가 큽니다."
    )


# ══════════════════════════════════════════════════════════════════════
# 메인 페이지
# ══════════════════════════════════════════════════════════════════════
def page_trendboard():
    st.title("🗂️ ETF 트렌드보드")

    if not st.session_state.get('universe_built', False):
        st.warning("👈 사이드바에서 먼저 **🚀 유니버스 빌드**를 실행하세요.")
        return

    df = st.session_state.df_universe
    if df is None or df.empty:
        st.warning("유니버스 데이터가 없습니다. 다시 빌드해 주세요."); return

    df_close = st.session_state.get('df_prices_kr')

    # 작업용 사본 + 파생 컬럼
    df = df.copy()
    if 'ETF명' in df.columns:
        df['운용사'] = df['ETF명'].apply(_get_manager)
    df['오늘등락(%)'] = df.index.map(lambda t: _today_change(df_close, str(t)))

    _section_trendboard(df)
    st.markdown("---")
    _section_today_rank(df)
    st.markdown("---")
    _section_period_rank(df, df_close)
    st.markdown("---")
    _section_flow_placeholder()
