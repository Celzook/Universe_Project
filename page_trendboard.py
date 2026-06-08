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

try:
    from style import UP_COLOR, DOWN_COLOR
except ImportError:
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


def _fmt_vol_ratio(r: float) -> str:
    """전일대비 거래대금 배율 → '+XX.XX%' or '×N배'."""
    if pd.isna(r):
        return '-'
    if r >= 10.0:
        return f"×{int(r)}배"
    pct = (r - 1.0) * 100.0
    sign = '+' if pct >= 0 else ''
    return f"{sign}{pct:.2f}%"


# ── 전일대비 거래대금 비율 캐시 (네이버 OHLCV 재활용) ────────────────────
@st.cache_data(ttl=3600 * 2, show_spinner=False)
def _cached_vol_ratios(tickers: tuple, base_date: str) -> dict:
    """
    티커 리스트에 대해 (오늘 거래대금 / 전일 거래대금) 비율을 계산.
    거래대금 = close × volume (네이버 OHLCV).
    Returns: {ticker: ratio}
    """
    try:
        from momentum_funnel.data_adapter import naver_get_ohlcv_history
    except Exception:
        return {}

    end_dt = pd.Timestamp(base_date) if base_date else pd.Timestamp.today()
    start_dt = end_dt - pd.Timedelta(days=15)  # 휴일 여유
    end_str = end_dt.strftime('%Y%m%d')
    start_str = start_dt.strftime('%Y%m%d')

    out: dict = {}
    for t in tickers:
        try:
            ohlcv = naver_get_ohlcv_history(str(t), start_str, end_str)
            if ohlcv is None or ohlcv.empty or 'close' not in ohlcv.columns or 'volume' not in ohlcv.columns:
                continue
            ohlcv = ohlcv.dropna(subset=['close', 'volume'])
            if len(ohlcv) < 2:
                continue
            amount = (ohlcv['close'] * ohlcv['volume']).astype(float)
            today_amt = float(amount.iloc[-1])
            prev_amt = float(amount.iloc[-2])
            if prev_amt > 0 and today_amt > 0:
                out[str(t)] = today_amt / prev_amt
        except Exception:
            continue
    return out


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
        # 1단계: 거래대금 상위 ~60개 pre-screen (휴면 ETF 제외)
        if '거래대금(억)' not in cand.columns:
            return cand.head(0)
        pre = cand.dropna(subset=['거래대금(억)'])
        pre = pre[pre['거래대금(억)'] > 0].sort_values('거래대금(억)', ascending=False).head(60)
        # 2단계: 네이버 OHLCV로 전일대비 거래대금 비율 계산 (캐시 2h)
        tickers = tuple(pre.index.astype(str))
        base_date = st.session_state.get('base_date') or datetime.today().strftime('%Y%m%d')
        ratios = _cached_vol_ratios(tickers, base_date)
        pre = pre.copy()
        pre['_vol_ratio'] = pre.index.astype(str).map(ratios)
        pre = pre.dropna(subset=['_vol_ratio'])
        pre = pre[pre['_vol_ratio'] > 1.0]   # 증가 ETF만
        return pre.sort_values('_vol_ratio', ascending=False)
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


def _render_card(slot_num: int, etf_row, ticker, ret_col, period_label, bm_col,
                 third_label: str = None, third_value: str = None, third_color: str = None):
    """Top 6 카드 1개 렌더 (테마 적응 + 가독성 강조).

    third_label/value/color : 3번째 메트릭을 RS 대신 강제 표시 (예: 전일대비 거래)."""
    name = etf_row.get('ETF명', '') if isinstance(etf_row.get('ETF명'), str) else ''
    cat = etf_row.get('중카테고리', '') or etf_row.get('대카테고리', '')
    manager = etf_row.get('운용사', '')
    close = etf_row.get('종가', np.nan)
    today = etf_row.get('오늘등락(%)', np.nan)
    period_ret = etf_row.get(ret_col, np.nan)
    bm = etf_row.get(bm_col, np.nan) if bm_col in etf_row else np.nan

    period_color = _color_pct(period_ret)
    today_color = _color_pct(today)
    bm_color = _color_pct(bm)

    # ETF명 길이에 따라 폰트 크기 스케일 (들쭉날쭉 방지)
    name_len = len(name)
    if name_len <= 10:
        name_fs = 19
    elif name_len <= 16:
        name_fs = 17
    elif name_len <= 22:
        name_fs = 15
    else:
        name_fs = 13

    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        with h1:
            # ETF 이름 — 단일 라인 ellipsis + 길이 기반 폰트 스케일
            st.markdown(
                f"<div title='{name}' style='font-size:{name_fs}px;"
                f"font-weight:700;line-height:1.25;letter-spacing:-0.02em;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                f"margin:2px 0 0 0'>{name}</div>",
                unsafe_allow_html=True,
            )
            sub_txt = f"{ticker} · {cat or '-'}{(' · ' + manager) if manager else ''}"
            st.markdown(
                f"<div title='{sub_txt}' style='font-size:11px;color:#888;"
                f"line-height:1.2;white-space:nowrap;overflow:hidden;"
                f"text-overflow:ellipsis;margin:2px 0 0 0'>{sub_txt}</div>",
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                f"<div style='text-align:right;margin-top:6px'>"
                f"<div style='color:#888;font-size:11px'>{period_label}</div>"
                f"<div style='color:{period_color};font-weight:700;font-size:17px'>{_fmt_pct(period_ret)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div style='color:#888;font-size:11px'>현재가</div>"
                f"<div style='font-weight:700;font-size:15px'>{_fmt_num(close)}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div style='color:#888;font-size:11px'>오늘</div>"
                f"<div style='color:{today_color};font-weight:700;font-size:15px'>{_fmt_pct(today)}</div>",
                unsafe_allow_html=True,
            )
        with c3:
            if third_label is not None and third_value is not None:
                # 오버라이드 (예: 전일대비 거래대금)
                st.markdown(
                    f"<div style='color:#888;font-size:11px'>{third_label}</div>"
                    f"<div style='color:{third_color or '#444'};font-weight:700;font-size:15px'>{third_value}</div>",
                    unsafe_allow_html=True,
                )
            elif not pd.isna(bm):
                st.markdown(
                    f"<div style='color:#888;font-size:11px'>RS (BM)</div>"
                    f"<div style='color:{bm_color};font-weight:700;font-size:15px'>{_fmt_pct(bm)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='color:#888;font-size:11px'>운용사</div>"
                    f"<div style='font-weight:600;font-size:13px'>{(manager[:8] if manager else '-')}</div>",
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

    is_vol_surge = (cat == '🔥 거래 급증') and ('_vol_ratio' in top6.columns)

    # 거래 급증 검증 expander — 사용자가 직접 OHLCV 원본 확인
    if is_vol_surge and not top6.empty:
        with st.expander("🔍 거래 급증 데이터 검증 (네이버 OHLCV 원본)"):
            try:
                from momentum_funnel.data_adapter import naver_get_ohlcv_history
                base_date = st.session_state.get('base_date') or datetime.today().strftime('%Y%m%d')
                end = pd.Timestamp(base_date)
                start = (end - pd.Timedelta(days=15)).strftime('%Y%m%d')
                end_s = end.strftime('%Y%m%d')
                check_t = st.selectbox(
                    "검증할 ETF (Top 6 중)",
                    options=[(str(top6.index[i]), top6.iloc[i].get('ETF명','')) for i in range(len(top6))],
                    format_func=lambda x: f"{x[0]} · {x[1]}",
                    key='vol_verify_sel',
                )
                tk = check_t[0]
                raw = naver_get_ohlcv_history(tk, start, end_s)
                if raw.empty:
                    st.warning("네이버 응답 없음")
                else:
                    raw = raw.dropna().tail(7).copy()
                    raw['거래대금(억)'] = (raw['close'] * raw['volume'] / 1e8).round(1)
                    raw['전일대비배율'] = (raw['거래대금(억)'] / raw['거래대금(억)'].shift(1)).round(3)
                    raw.index = raw.index.strftime('%Y-%m-%d')
                    st.dataframe(
                        raw[['close','volume','거래대금(억)','전일대비배율']],
                        width='stretch',
                        column_config={
                            'close': st.column_config.NumberColumn('종가', format='%,.0f'),
                            'volume': st.column_config.NumberColumn('거래량(주)', format='%,.0f'),
                            '거래대금(억)': st.column_config.NumberColumn('거래대금(억원)', format='%,.1f'),
                            '전일대비배율': st.column_config.NumberColumn('전일대비배율', format='%.3fx'),
                        },
                    )
                    st.caption(
                        "거래대금(억원) = close × volume ÷ 1억. "
                        "전일대비배율 = 오늘/전일 거래대금. "
                        "마지막 행이 트렌드보드 카드에 표시되는 값입니다."
                    )
            except Exception as e:
                st.error(f"검증 실패: {e}")

    with c_cards:
        if top6.empty:
            if cat == '🔥 거래 급증':
                st.info("거래 급증 후보 없음 — 빌드 직후이거나 네이버 OHLCV 미수집. 잠시 후 재시도하세요.")
            else:
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
                            # 거래 급증: 3번째 메트릭을 RS 대신 전일대비 거래대금 비율
                            if is_vol_surge:
                                vr = row.get('_vol_ratio')
                                vr_str = _fmt_vol_ratio(vr)
                                vr_color = UP_COLOR if (not pd.isna(vr) and vr > 1.0) else DOWN_COLOR
                                _render_card(slot + 1, row, ticker, ret_col, period_label, bm_col,
                                             third_label='전일대비 거래',
                                             third_value=vr_str,
                                             third_color=vr_color)
                            else:
                                _render_card(slot + 1, row, ticker, ret_col, period_label, bm_col)
                        else:
                            st.empty()


# ══════════════════════════════════════════════════════════════════════
# 섹션 2: 당일 순위
# ══════════════════════════════════════════════════════════════════════
def _render_winner_card(rank: int, row, ticker: str, val: float, val_kind: str, color: str):
    """val_kind: 'pct' | '억' | 'num'."""
    name = row.get('ETF명', '') if isinstance(row.get('ETF명'), str) else ''
    parts = name.strip().split(maxsplit=1)
    brand = parts[0] if parts else ''
    rest = parts[1] if len(parts) > 1 else (brand if parts else '')

    if val_kind == 'pct':
        val_str = _fmt_pct(val); val_color = color
    elif val_kind == '억':
        val_str = f"{_fmt_num(val)} 억원"; val_color = '#444'
    else:
        val_str = _fmt_num(val); val_color = '#444'

    with st.container(border=True):
        st.caption(f"🏆 {rank}위 · {brand or '-'}")
        # ETF 이름: 16px 명시 + bold (테마 색 자동 = 라이트=검정/다크=흰)
        st.markdown(
            f"<div style='font-size:15px;font-weight:700;line-height:1.3;"
            f"margin:2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>"
            f"{rest or brand or '-'}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"`{ticker}`")
        st.markdown(
            f"<div style='font-size:22px;font-weight:800;color:{val_color};"
            f"text-align:right;margin-top:6px'>{val_str}</div>",
            unsafe_allow_html=True,
        )


def _section_today_rank(df: pd.DataFrame):
    st.subheader("📊 당일 순위")
    c1, c2 = st.columns([2.5, 1])
    with c1:
        st.caption("정렬 기준 선택 + 레버리지/인버스 제외 토글 · 거래대금=0 ETF 자동 제외 (휴장·정지·신규)")
    with c2:
        excl_lev = st.checkbox("레버리지/인버스 제외", value=True, key='today_excl_lev')

    sort_mode = st.radio("정렬", ['등락률', '거래대금(억원)', '시가총액(억원)'], horizontal=True, index=0,
                         key='today_sort', label_visibility='collapsed')

    pool = df.copy()
    if excl_lev and 'ETF명' in pool.columns:
        pool = pool[~pool['ETF명'].apply(_is_lev_inv)]

    # 거래대금 0 / NaN ETF 제외 (휴장·신규·거래정지 노이즈 차단) — 모든 정렬 모드에 적용
    if '거래대금(억)' in pool.columns:
        pool = pool[pool['거래대금(억)'].fillna(0) > 0]

    if sort_mode == '등락률':
        col = '오늘등락(%)'; val_kind = 'pct'
    elif sort_mode == '거래대금(억원)':
        col = '거래대금(억)' if '거래대금(억)' in pool.columns else None; val_kind = '억'
    else:
        col = '시가총액(억원)' if '시가총액(억원)' in pool.columns else None; val_kind = '억'

    if col is None or col not in pool.columns:
        st.info(f"'{sort_mode}' 컬럼이 유니버스 데이터에 없습니다."); return

    pool = pool.dropna(subset=[col])
    risers = pool.sort_values(col, ascending=False).head(7)
    fallers = pool.sort_values(col, ascending=True).head(7)

    c_up, c_dn = st.columns(2)

    def _render_side(title: str, df_side: pd.DataFrame, color: str):
        st.markdown(f"### {title}")
        if df_side.empty:
            st.info("데이터 없음"); return

        # Top 3 카드 (큰 글씨)
        cc = st.columns(3)
        for i in range(min(3, len(df_side))):
            row = df_side.iloc[i]; ticker = df_side.index[i]; v = row[col]
            with cc[i]:
                _render_winner_card(i + 1, row, ticker, v, val_kind, color)

        # 4~7위 리스트 — st.dataframe (테마 적응 + 가독성)
        if len(df_side) > 3:
            rest = df_side.iloc[3:7].copy()
            rest.insert(0, '#', range(4, 4 + len(rest)))
            rest['ETF'] = rest['ETF명']
            rest['티커'] = rest.index.astype(str)
            value_label = '등락 %' if val_kind == 'pct' else '값 (억원)'
            rest[value_label] = rest[col].astype(float)
            view = rest[['#', 'ETF', '티커', value_label]]

            if val_kind == 'pct':
                num_cfg = st.column_config.NumberColumn(value_label, format='%+.2f')
            else:
                num_cfg = st.column_config.NumberColumn(value_label, format='%,.0f')

            st.dataframe(
                view, hide_index=True, width='stretch', height=200,
                column_config={
                    '#': st.column_config.NumberColumn('#', width='small', format='%d'),
                    'ETF': st.column_config.TextColumn('ETF명', width='large'),
                    '티커': st.column_config.TextColumn('티커', width='small'),
                    value_label: num_cfg,
                },
            )

    with c_up:
        _render_side("🔴 상승 TOP", risers, UP_COLOR)
    with c_dn:
        _render_side("🔵 하락 TOP", fallers, DOWN_COLOR)


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
    # 하락 TOP 노이즈 차단: 거래대금 0인 ETF 제외 (휴장·정지 ETF가 큰 음수 등장 방지)
    if '거래대금(억)' in pool.columns:
        pool = pool[pool['거래대금(억)'].fillna(0) > 0]
    if pool.empty:
        st.info("유효 데이터 없음"); return

    risers = pool.sort_values(col, ascending=False).head(5)
    fallers = pool.sort_values(col, ascending=True).head(5)

    c_up, c_dn = st.columns(2)

    def _render_list(title: str, df_side: pd.DataFrame):
        st.markdown(f"### {title}")
        if df_side.empty:
            st.info("데이터 없음"); return
        view = df_side.copy()
        view.insert(0, '#', range(1, len(view) + 1))
        view['ETF'] = view['ETF명']
        view['티커'] = view.index.astype(str)
        view['수익률 %'] = view[col].astype(float)
        view = view[['#', 'ETF', '티커', '수익률 %']]
        st.dataframe(
            view, hide_index=True, width='stretch', height=240,
            column_config={
                '#': st.column_config.NumberColumn('#', width='small', format='%d'),
                'ETF': st.column_config.TextColumn('ETF명', width='large'),
                '티커': st.column_config.TextColumn('티커', width='small'),
                '수익률 %': st.column_config.NumberColumn('수익률 %', format='%+.2f'),
            },
        )

    with c_up:
        _render_list("🔴 상승 TOP 5", risers)
    with c_dn:
        _render_list("🔵 하락 TOP 5", fallers)


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
