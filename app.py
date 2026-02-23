"""
==============================================================================
 ETF Universe Explorer — Streamlit Cloud App v3
==============================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import threading, gc, traceback

st.set_page_config(page_title="ETF Universe Explorer", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from etf_universe_builder import build_universe, Config
from global_price_collector import (
    collect_global_prices, calc_period_return, GLOBAL_INDICES, US_ETFS
)


# ============================================================================
# KST 시간 헬퍼
# ============================================================================
def now_kst():
    """현재 한국시간 반환"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        return datetime.utcnow() + timedelta(hours=9)

def today_kst():
    """오늘 날짜 (KST) 반환"""
    return now_kst().date()


# ============================================================================
# 캐시 — base_date도 함께 반환
# ============================================================================
@st.cache_data(ttl=3600*6, show_spinner=False)
def cached_build_universe(min_cap, top_n):
    Config.MIN_MARKET_CAP_BILLIONS = min_cap
    Config.TOP_N_HOLDINGS = top_n
    Config.BASE_DATE = None  # 매번 새로 찾도록 리셋
    df, df_close, df_pdf = build_universe()
    base_date = Config.BASE_DATE  # build 후 설정된 값
    for c in df.select_dtypes(include='float64').columns:
        df[c] = df[c].astype('float32')
    if df_close is not None:
        df_close = df_close.astype('float32')
    gc.collect()
    return df, df_close, df_pdf, base_date  # base_date 포함

@st.cache_data(ttl=3600*6, show_spinner=False)
def cached_global_prices():
    return collect_global_prices(cache_dir=Config.CACHE_DIR, years=3)

# ============================================================================
# 세션 상태
# ============================================================================
def init_session():
    for k, v in {
        'universe_built': False, 'df_universe': None,
        'df_prices_kr': None, 'df_pdf': None, 'base_date': None,
        'global_data': None, 'global_loading': False,
        'global_loaded': False, 'show_global_toast': False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session()

# ============================================================================
# 공통: PDF 구성종목 비교 (최대 3개)
# ============================================================================
def render_pdf_comparison(selected_tickers, df_pdf, df_uni, key_prefix="comp"):
    if not selected_tickers or df_pdf is None:
        return
    n = min(len(selected_tickers), 3)
    tickers = selected_tickers[:n]

    st.markdown("---")
    st.subheader(f"🔬 PDF 구성종목 비교 ({n}개 ETF)")
    cols = st.columns(n)
    for i, ticker in enumerate(tickers):
        with cols[i]:
            name = df_uni.at[ticker, 'ETF명'] if ticker in df_uni.index else ticker
            cap = ''
            if ticker in df_uni.index and '시가총액(억원)' in df_uni.columns:
                c = df_uni.at[ticker, '시가총액(억원)']
                if pd.notna(c) and c != '':
                    cap = f" | {int(c):,}억"
            st.markdown(f"**{name}**{cap}")
            if ticker in df_pdf.index:
                row = df_pdf.loc[ticker].drop('ETF명', errors='ignore')
                vals = pd.to_numeric(row, errors='coerce')
                valid = vals.dropna().sort_values(ascending=False).head(10)
                if not valid.empty:
                    tbl = pd.DataFrame({'종목': valid.index, '비중(%)': [f"{v:.1f}" for v in valid.values]})
                    tbl.index = range(1, len(tbl)+1)
                    tbl.index.name = '#'
                    st.dataframe(tbl, use_container_width=True, height=390)
                else:
                    st.caption("구성종목 없음")
            else:
                st.caption("PDF 데이터 없음")


# ============================================================================
# 사이드바
# ============================================================================
def render_sidebar():
    st.sidebar.title("📊 ETF Universe Explorer")
    st.sidebar.markdown("---")
    st.sidebar.subheader("1️⃣ 유니버스 구축")
    min_cap = st.sidebar.number_input("최소 시가총액 (억원)", value=200, step=50)
    top_n = st.sidebar.number_input("PDF Top N", value=10, min_value=5, max_value=20)
    if st.sidebar.button("🚀 유니버스 빌드", type="primary", use_container_width=True):
        run_universe_build(min_cap, top_n)
    if st.session_state.universe_built:
        st.sidebar.success(f"✅ {len(st.session_state.df_universe)}개 ETF")
        st.sidebar.caption(f"기준일: {st.session_state.base_date}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("2️⃣ 글로벌 가격")
    if st.session_state.global_loaded:
        gd = st.session_state.global_data
        n_idx = len(gd['indices'].columns) if gd and not gd['indices'].empty else 0
        n_us = len(gd['us_etfs'].columns) if gd and not gd['us_etfs'].empty else 0
        st.sidebar.success(f"✅ 지수 {n_idx} + 미국ETF {n_us}")
    elif st.session_state.global_loading:
        st.sidebar.info("⏳ 수집 중...")
    else:
        st.sidebar.caption("유니버스 빌드 후 자동 시작")
    st.sidebar.markdown("---")
    return st.sidebar.radio("📌 메뉴", ["유니버스 탐색", "구성종목(PDF) 분석", "수익률 비교"],
                            label_visibility="collapsed")

def run_universe_build(min_cap, top_n):
    with st.spinner("유니버스 빌드 중... (첫 실행 3~8분, 이후 캐시)"):
        try:
            df, df_close, df_pdf, base_date = cached_build_universe(min_cap, top_n)
            st.session_state.df_universe = df
            st.session_state.df_prices_kr = df_close
            st.session_state.df_pdf = df_pdf
            st.session_state.base_date = base_date or today_kst().strftime("%Y%m%d")
            st.session_state.universe_built = True
            st.success(f"✅ {len(df)}개 ETF 유니버스 빌드 완료! (기준일: {st.session_state.base_date})")
        except Exception as e:
            err_detail = traceback.format_exc()
            st.error(f"빌드 실패: {e}")
            with st.expander("🔍 상세 에러"):
                st.code(err_detail)
            return
    start_global_collection()

def start_global_collection():
    if st.session_state.global_loaded or st.session_state.global_loading:
        return
    st.session_state.global_loading = True
    def _collect():
        try:
            result = cached_global_prices()
            st.session_state.global_data = result
            st.session_state.global_loaded = True
            st.session_state.show_global_toast = True
        except Exception: pass
        finally:
            st.session_state.global_loading = False
    threading.Thread(target=_collect, daemon=True).start()


# ============================================================================
# 페이지 1: 유니버스 탐색
# ============================================================================
def page_universe():
    st.title("📊 ETF 유니버스 탐색")
    if not st.session_state.universe_built:
        st.info("👈 사이드바에서 **🚀 유니버스 빌드** 버튼을 누르세요.")
        return

    df = st.session_state.df_universe.copy()
    df_pdf = st.session_state.df_pdf
    has_cap = '시가총액(억원)' in df.columns and df['시가총액(억원)'].notna().any()

    # 필터
    col1, col2, col3 = st.columns(3)
    with col1:
        cats = ['전체'] + sorted(df['대카테고리'].dropna().unique().tolist()) if '대카테고리' in df.columns else ['전체']
        sel_cat = st.selectbox("대카테고리", cats)
    with col2:
        if sel_cat != '전체' and '중카테고리' in df.columns:
            mids = ['전체'] + sorted(df[df['대카테고리']==sel_cat]['중카테고리'].dropna().unique().tolist())
        else:
            mids = ['전체'] + sorted(df['중카테고리'].dropna().unique().tolist()) if '중카테고리' in df.columns else ['전체']
        sel_mid = st.selectbox("중카테고리", mids)
    with col3:
        search = st.text_input("🔍 ETF명 검색")

    if sel_cat != '전체': df = df[df['대카테고리'] == sel_cat]
    if sel_mid != '전체': df = df[df['중카테고리'] == sel_mid]
    if search: df = df[df['ETF명'].str.contains(search, case=False, na=False)]

    # 메트릭
    m1, m2, m3, m4 = st.columns(4)
    cap_total = df['시가총액(억원)'].sum() if has_cap else 0
    m1.metric("총 시가총액", f"{cap_total/10000:.1f}조원" if has_cap else "N/A")
    m2.metric("ETF 수", f"{len(df)}개")
    m3.metric("평균 YTD", f"{df['수익률_YTD(%)'].mean():+.2f}%" if '수익률_YTD(%)' in df.columns else "N/A")
    m4.metric("평균 BM(YTD)", f"{df['BM_YTD(%)'].mean():+.2f}%" if 'BM_YTD(%)' in df.columns else "N/A")

    # ── [기능4] 테이블 컬럼 순서: ETF명 → 시가총액 → 설정일 ──
    display_cols = [c for c in [
        'ETF명','시가총액(억원)','NAV(억원)','설정일',
        '대카테고리','중카테고리','소카테고리','순위(YTD_BM+)',
        '수익률_1M(%)','수익률_3M(%)','수익률_6M(%)','수익률_1Y(%)','수익률_YTD(%)',
        'BM_1M(%)','BM_3M(%)','BM_6M(%)','BM_1Y(%)','BM_YTD(%)',
        '연간변동성(%)','종가','거래량'
    ] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=500)

    # ETF 선택 → PDF 비교
    etf_options = [f"{t} | {df.at[t,'ETF명'][:30]}" for t in df.index]
    selected = st.multiselect("🔬 PDF 구성종목 비교 (최대 3개 ETF 선택)",
                              etf_options, max_selections=3, key="uni_pdf_comp")
    if selected:
        sel_tickers = [s.split(' | ')[0] for s in selected]
        render_pdf_comparison(sel_tickers, df_pdf, df, key_prefix="uni")

    # 카테고리별 시가총액 바
    if has_cap:
        st.markdown("---")
        st.subheader("📊 카테고리별 시가총액")
        cc = df.groupby('대카테고리')['시가총액(억원)'].sum().sort_values(ascending=True)
        fig = px.bar(x=cc.values, y=cc.index, orientation='h', labels={'x':'시총(억)','y':''},
                    color=cc.values, color_continuous_scale='Viridis')
        fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                        margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # BM 상위/하위
    if 'BM_YTD(%)' in df.columns and len(df) > 0:
        st.subheader("📈 BM(YTD) 상위/하위 15")
        top_cols = ['ETF명','대카테고리','BM_YTD(%)','수익률_YTD(%)']
        if has_cap: top_cols.insert(1, '시가총액(억원)')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 상위**")
            st.dataframe(df.nlargest(15,'BM_YTD(%)')[top_cols].reset_index(), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**🔴 하위**")
            st.dataframe(df.nsmallest(15,'BM_YTD(%)')[top_cols].reset_index(), use_container_width=True, hide_index=True)

    # 도넛차트 맨 아래
    if '대카테고리' in df.columns:
        st.markdown("---")
        st.subheader("🍩 카테고리 분포")
        c1, c2 = st.columns(2)
        with c1:
            cc = df['대카테고리'].value_counts()
            fig = px.pie(values=cc.values, names=cc.index, hole=0.4, title="ETF 수")
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if has_cap:
                cc2 = df.groupby('대카테고리')['시가총액(억원)'].sum()
                fig2 = px.pie(values=cc2.values, names=cc2.index, hole=0.4, title="시가총액")
                fig2.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# 페이지 2: 구성종목(PDF) 분석
# ============================================================================
def page_pdf():
    st.title("🧬 구성종목(PDF) 분석")
    if st.session_state.df_pdf is None:
        st.info("👈 유니버스를 먼저 빌드하세요."); return

    df_pdf = st.session_state.df_pdf
    df_uni = st.session_state.df_universe
    stock_cols = [c for c in df_pdf.columns if c != 'ETF명']
    if not stock_cols:
        st.warning("구성종목 데이터가 없습니다."); return

    st.subheader("🔍 종목별 ETF 보유 현황")
    stock_counts = df_pdf[stock_cols].notna().sum().sort_values(ascending=False)

    c1, c2 = st.columns([2, 1])
    with c1:
        sel = st.selectbox("종목 선택", stock_cols,
                           index=stock_cols.index('삼성전자') if '삼성전자' in stock_cols else 0)
    with c2:
        min_w = st.number_input("최소 비중(%)", value=0.0, step=0.5)

    if sel:
        vals = pd.to_numeric(df_pdf[sel], errors='coerce')
        mask = vals.notna() & (vals > min_w)
        res = df_pdf[mask][['ETF명', sel]].copy()
        res[sel] = vals[mask]
        res = res.sort_values(sel, ascending=False)
        if not res.empty and df_uni is not None:
            for c in ['대카테고리','중카테고리','시가총액(억원)',
                       'BM_1M(%)','BM_3M(%)','BM_6M(%)','BM_1Y(%)','BM_YTD(%)']:
                if c in df_uni.columns:
                    res[c] = res.index.map(lambda t: df_uni.at[t,c] if t in df_uni.index else np.nan)

        st.caption(f"'{sel}' 보유 ETF: **{len(res)}개**")

        # ── [기능3] 3분할 차트: 동일 크기 + BM 기간 선택 ──
        if len(res) > 0:
            _render_stock_analysis_charts(res.head(20), sel, df_uni)

        # 결과 테이블
        st.dataframe(res, use_container_width=True, height=400)

        # ETF 선택 → PDF 비교
        etf_options = [f"{t} | {res.at[t,'ETF명'][:30]}" for t in res.index]
        selected = st.multiselect("🔬 PDF 구성종목 비교 (최대 3개)",
                                  etf_options, max_selections=3, key="pdf_comp")
        if selected:
            sel_tickers = [s.split(' | ')[0] for s in selected]
            render_pdf_comparison(sel_tickers, df_pdf, df_uni, key_prefix="pdf")

    st.markdown("---")
    st.subheader("🏆 보유 ETF 수 상위 종목")
    ts = stock_counts.head(20).reset_index(); ts.columns = ['종목명','보유 ETF 수']
    st.dataframe(ts, use_container_width=True, hide_index=True)
    with st.expander(f"📋 전체 매트릭스 ({len(df_pdf)} × {len(stock_cols)})"):
        st.dataframe(df_pdf, use_container_width=True, height=500)


def _render_stock_analysis_charts(chart_data, stock_name, df_uni):
    """[기능3] 3분할: 비중 | BM 성과 바 | 히트맵 — 동일 높이 + 기간 선택"""

    bm_all = {'1M': 'BM_1M(%)', '3M': 'BM_3M(%)', '6M': 'BM_6M(%)',
              '1Y': 'BM_1Y(%)', 'YTD': 'BM_YTD(%)'}
    available_bm = {k: v for k, v in bm_all.items() if v in chart_data.columns}

    if available_bm:
        sel_periods = st.multiselect(
            "📅 BM 성과 기간 선택",
            list(available_bm.keys()),
            default=list(available_bm.keys()),
            key="bm_period_sel"
        )
    else:
        sel_periods = []

    # ── 라벨 중복 방지: 티커 접미사 추가 ──
    raw_labels = chart_data['ETF명'].str[:15].tolist()
    tickers = chart_data.index.tolist()
    seen = {}
    labels = []
    for lbl, tk in zip(raw_labels, tickers):
        if lbl in seen:
            labels.append(f"{lbl}({tk[-4:]})")
        else:
            labels.append(lbl)
        seen[lbl] = seen.get(lbl, 0) + 1
    chart_h = max(400, len(labels) * 26)  # 동일 높이

    col1, col2, col3 = st.columns(3)

    # ── 좌: 비중 ──
    with col1:
        fig1 = go.Figure(go.Bar(
            x=chart_data[stock_name].values, y=labels, orientation='h',
            marker_color='#3498db',
            text=[f"{v:.1f}%" for v in chart_data[stock_name].values],
            textposition='outside'
        ))
        fig1.update_layout(title=f"'{stock_name}' 비중(%)", height=chart_h,
                          yaxis=dict(autorange='reversed'),
                          margin=dict(l=0, r=40, t=40, b=0))
        st.plotly_chart(fig1, use_container_width=True)

    # ── 중: BM 성과 바 (선택된 기간만) ──
    with col2:
        if sel_periods and available_bm:
            fig2 = go.Figure()
            colors = {'1M':'#e74c3c','3M':'#e67e22','6M':'#f1c40f','1Y':'#2ecc71','YTD':'#3498db'}
            for p in sel_periods:
                bc = available_bm[p]
                v = pd.to_numeric(chart_data[bc], errors='coerce').fillna(0).values
                fig2.add_trace(go.Bar(name=p, x=v, y=labels, orientation='h',
                                     marker_color=colors.get(p, '#999')))
            fig2.update_layout(title="BM 대비 성과(%)", barmode='group', height=chart_h,
                              yaxis=dict(autorange='reversed'),
                              margin=dict(l=0, r=10, t=40, b=0),
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("기간을 선택하세요")

    # ── 우: 히트맵 (선택된 기간만) ──
    with col3:
        if sel_periods and available_bm:
            heat_data = []
            for p in sel_periods:
                bc = available_bm[p]
                heat_data.append(pd.to_numeric(chart_data[bc], errors='coerce').fillna(0).values)
            heat_df = pd.DataFrame(heat_data, index=sel_periods, columns=labels)
            fig3 = px.imshow(heat_df, text_auto='.1f', color_continuous_scale='RdYlGn',
                            aspect='auto', zmin=-20, zmax=20)
            fig3.update_layout(title="BM성과 히트맵(%)", height=chart_h,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("기간을 선택하세요")


# ============================================================================
# 페이지 3: 수익률 비교
# ============================================================================
def page_returns():
    st.title("📈 수익률 비교")
    if not st.session_state.universe_built:
        st.info("👈 유니버스를 먼저 빌드하세요."); return

    kr_close = st.session_state.df_prices_kr
    gd = st.session_state.global_data
    df_uni = st.session_state.df_universe

    # 기간 설정
    c1, c2, c3 = st.columns([1,1,2])
    with c1: start_date = st.date_input("시작일", value=today_kst()-timedelta(days=90))
    with c2: end_date = st.date_input("종료일", value=today_kst())
    with c3:
        q = st.radio("빠른 선택", ['직접입력','1M','3M','6M','YTD','1Y'], horizontal=True)
        if q != '직접입력':
            end_date = today_kst()
            start_date = {'1M': end_date-timedelta(30), '3M': end_date-timedelta(90),
                         '6M': end_date-timedelta(180), '1Y': end_date-timedelta(365),
                         'YTD': datetime(end_date.year,1,1).date()}[q]

    tab1, tab2, tab3 = st.tabs(["🇰🇷 국내 ETF", "🌍 글로벌 지수", "🇺🇸 미국 ETF"])

    with tab1:
        _render_kr_tab(kr_close, df_uni, start_date, end_date)
    with tab2:
        _render_global_tab(gd, start_date, end_date, 'indices')
    with tab3:
        _render_global_tab(gd, start_date, end_date, 'us_etfs')

    # 크로스 비교
    if (kr_close is not None and not kr_close.empty and gd and not gd.get('indices',pd.DataFrame()).empty):
        st.markdown("---"); st.subheader("🔀 크로스 비교")
        c1, c2 = st.columns(2)
        with c1:
            ko = [f"{t} | {df_uni.at[t,'ETF명'][:20]}" if t in df_uni.index else t for t in kr_close.columns[:100]]
            skr = st.multiselect("국내", ko, default=ko[:3], key="xkr")
        with c2:
            ga = list(gd['indices'].columns) + (list(gd['us_etfs'].columns) if not gd['us_etfs'].empty else [])
            sgl = st.multiselect("글로벌", ga, default=ga[:3], key="xgl")
        if skr or sgl:
            comb = pd.DataFrame()
            for s in skr:
                t = s.split(' | ')[0]
                if t in kr_close.columns:
                    nm = df_uni.at[t,'ETF명'][:15] if t in df_uni.index else t
                    comb[f"🇰🇷{nm}"] = kr_close[t]
            for g in sgl:
                if g in gd['indices'].columns: comb[f"🌍{g}"] = gd['indices'][g]
                elif not gd['us_etfs'].empty and g in gd['us_etfs'].columns: comb[f"🇺🇸{g}"] = gd['us_etfs'][g]
            if not comb.empty:
                _draw_return_bar(comb, start_date, end_date, "크로스 비교")


def _render_kr_tab(kr_close, df_uni, start_date, end_date):
    """국내 ETF 탭 — KOSPI 벤치마크 포함"""
    if kr_close is None or kr_close.empty:
        st.warning("가격 데이터 없음"); return

    if df_uni is not None and '대카테고리' in df_uni.columns:
        cats = ['전체'] + sorted(df_uni['대카테고리'].dropna().unique().tolist())
        sc = st.selectbox("카테고리", cats, key="ret_kr_cat")
        vt = df_uni[df_uni['대카테고리']==sc].index.tolist() if sc != '전체' else df_uni.index.tolist()
        vt = [t for t in vt if t in kr_close.columns]
    else:
        vt = kr_close.columns.tolist()

    opts = [f"{t} | {df_uni.at[t,'ETF명'][:25]}" if t in df_uni.index else t for t in vt]
    sel = st.multiselect("ETF 선택", opts, default=opts[:5], key="kr_sel")
    if not sel:
        return

    tks = [s.split(' | ')[0] for s in sel]
    nm = {t: df_uni.at[t,'ETF명'][:15] if t in df_uni.index else t for t in tks}

    # ── 수익률 바 차트 ──
    sub = kr_close[tks]
    _draw_return_bar(sub, start_date, end_date, "국내 ETF", name_map=nm)

    # ── [기능1] 정규화 차트 + KOSPI BM + 커서 툴팁 ──
    mask = (sub.index >= pd.Timestamp(start_date)) & (sub.index <= pd.Timestamp(end_date))
    sp = sub[mask].dropna(how='all')
    if len(sp) > 1:
        norm = (sp / sp.iloc[0] - 1) * 100

        # KOSPI 가져오기 (yfinance 글로벌 데이터 또는 유니버스 빌더의 코스피)
        kospi_norm = None
        gd = st.session_state.global_data
        if gd and not gd.get('indices', pd.DataFrame()).empty:
            idx_df = gd['indices']
            if 'KOSPI' in idx_df.columns:
                k_mask = (idx_df.index >= pd.Timestamp(start_date)) & (idx_df.index <= pd.Timestamp(end_date))
                k_sub = idx_df[k_mask]['KOSPI'].dropna()
                if len(k_sub) > 1:
                    kospi_norm = (k_sub / k_sub.iloc[0] - 1) * 100

        fig = go.Figure()

        # KOSPI BM 라인 (먼저 그려서 뒤로)
        if kospi_norm is not None:
            fig.add_trace(go.Scatter(
                x=kospi_norm.index, y=kospi_norm.values,
                name='KOSPI (BM)', mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate='%{x|%Y-%m-%d}<br>KOSPI: %{y:+.2f}%<extra></extra>'
            ))

        # ETF 라인
        colors_list = px.colors.qualitative.Set2
        for i, t in enumerate(norm.columns):
            label = f"{t} ({nm.get(t, t)})"
            # hover에 KOSPI도 표시
            hover_texts = []
            for idx_val in norm.index:
                etf_val = norm.at[idx_val, t] if idx_val in norm.index else 0
                bm_val = ''
                if kospi_norm is not None:
                    # 가장 가까운 날짜 찾기
                    nearest = kospi_norm.index[kospi_norm.index.get_indexer([idx_val], method='nearest')[0]]
                    bm_val = f"<br>KOSPI(BM): {kospi_norm[nearest]:+.2f}%"
                hover_texts.append(
                    f"{idx_val.strftime('%Y-%m-%d')}<br>{label}: {etf_val:+.2f}%{bm_val}"
                )
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[t],
                name=label, mode='lines+markers',
                marker=dict(size=3),
                line=dict(color=colors_list[i % len(colors_list)], width=2),
                hovertext=hover_texts,
                hoverinfo='text'
            ))

        fig.update_layout(
            title="국내 ETF 추이 (정규화, %) + KOSPI BM",
            height=450, yaxis_title="수익률(%)",
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── [기능2] 수익률 상세 — 일별 수익률 테이블 ──
    with st.expander("📋 수익률 상세 (일별)"):
        if len(sp) > 1:
            daily_ret = sp.pct_change().dropna() * 100  # 일별 수익률 %
            daily_ret.columns = [f"{t} ({nm.get(t,t)})" for t in daily_ret.columns]
            daily_ret.index = daily_ret.index.strftime('%Y-%m-%d')
            daily_ret.index.name = '일자'
            daily_ret = daily_ret.round(2)
            st.dataframe(daily_ret.sort_index(ascending=False), use_container_width=True, height=400)
        else:
            st.caption("데이터 부족")


def _render_global_tab(gd, start_date, end_date, data_key):
    """글로벌 지수 / 미국 ETF 탭"""
    if not gd:
        if st.session_state.global_loading:
            st.info("⏳ 수집 중..."); st.button("🔄 새로고침", key=f"ref_{data_key}")
        else:
            st.warning("유니버스 빌드 후 자동 수집됩니다.")
        return

    df = gd.get(data_key, pd.DataFrame())
    if df.empty:
        st.warning("데이터 없음"); return

    info = gd.get(f"{'index' if data_key=='indices' else 'us_etf'}_info", {})

    # 카테고리 필터 (미국 ETF)
    if data_key == 'us_etfs' and info:
        categories = sorted(set(v.get('category','') for v in info.values()))
        sc = st.selectbox("카테고리", ['전체']+categories, key=f"ret_{data_key}_cat")
        filt = [k for k,v in info.items() if (sc=='전체' or v.get('category')==sc) and k in df.columns]
    else:
        filt = df.columns.tolist()

    if info:
        opts = [f"{t} | {info[t].get('name', info[t].get('country',''))}" for t in filt if t in info]
    else:
        opts = filt

    sel = st.multiselect("선택", opts, default=opts[:8], key=f"{data_key}_sel")
    if not sel: return

    sel_tickers = [s.split(' | ')[0] for s in sel]
    nm = {t: info[t].get('name', info[t].get('country', t))[:18] if t in info else t for t in sel_tickers}
    label = "글로벌 지수" if data_key == 'indices' else "미국 ETF"

    _draw_return_bar(df[sel_tickers], start_date, end_date, label, name_map=nm)

    # 정규화 차트
    mask = (df.index>=pd.Timestamp(start_date)) & (df.index<=pd.Timestamp(end_date))
    sp = df[mask][sel_tickers].dropna(how='all')
    if len(sp) > 1:
        norm = (sp/sp.iloc[0]-1)*100
        if nm: norm.columns = [f"{t} ({nm.get(t,t)})" for t in norm.columns]
        fig = px.line(norm, title=f"{label} 추이 (정규화)")
        fig.update_layout(height=400, yaxis_title="수익률(%)")
        st.plotly_chart(fig, use_container_width=True)

    # 상관관계 (지수만)
    if data_key == 'indices' and len(sel_tickers) >= 2:
        corr = sp.pct_change().dropna().corr()
        if nm: corr.index = [nm.get(t,t) for t in corr.index]; corr.columns = [nm.get(t,t) for t in corr.columns]
        fig3 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)

    # 일별 수익률 상세
    with st.expander("📋 수익률 상세 (일별)"):
        if len(sp) > 1:
            dr = sp.pct_change().dropna() * 100
            if nm: dr.columns = [f"{t} ({nm.get(t,t)})" for t in dr.columns]
            dr.index = dr.index.strftime('%Y-%m-%d')
            dr.index.name = '일자'
            st.dataframe(dr.round(2).sort_index(ascending=False), use_container_width=True, height=400)


def _draw_return_bar(df_prices, start_date, end_date, title, name_map=None):
    """수익률 바 차트"""
    ret = calc_period_return(df_prices, start_date, end_date)
    if ret.empty:
        st.warning("기간 데이터 부족"); return

    if name_map:
        labels = [f"{t} ({name_map.get(t,t)})" for t in ret.index]
    else:
        labels = ret.index.tolist()
    rd = ret.copy(); rd.index = labels; rs = rd.sort_values(ascending=False)
    colors = ['#2ecc71' if v>=0 else '#e74c3c' for v in rs.values]
    fig = go.Figure(go.Bar(x=rs.values, y=rs.index, orientation='h', marker_color=colors,
                           text=[f"{v:+.2f}%" for v in rs.values], textposition='outside'))
    fig.update_layout(title=f"{title} 수익률 ({start_date}~{end_date})",
                     height=max(300,len(rs)*35), xaxis_title="수익률(%)",
                     margin=dict(l=0,r=60,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 메인
# ============================================================================
def main():
    page = render_sidebar()
    if st.session_state.get('show_global_toast'):
        st.toast("🎉 글로벌 가격 데이터 수집 완료!", icon="✅")
        st.session_state.show_global_toast = False
    if page == "유니버스 탐색": page_universe()
    elif page == "구성종목(PDF) 분석": page_pdf()
    elif page == "수익률 비교": page_returns()

if __name__ == "__main__":
    main()
