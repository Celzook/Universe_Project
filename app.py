"""
==============================================================================
 ETF Universe Explorer — Streamlit Cloud App v2
==============================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import threading, gc

st.set_page_config(page_title="ETF Universe Explorer", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from etf_universe_builder import build_universe, Config
from global_price_collector import (
    collect_global_prices, calc_period_return, GLOBAL_INDICES, US_ETFS
)

# ============================================================================
# 캐시
# ============================================================================
@st.cache_data(ttl=3600*6, show_spinner=False)
def cached_build_universe(min_cap, top_n):
    Config.MIN_MARKET_CAP_BILLIONS = min_cap
    Config.TOP_N_HOLDINGS = top_n
    df, df_close, df_pdf = build_universe()
    for c in df.select_dtypes(include='float64').columns:
        df[c] = df[c].astype('float32')
    if df_close is not None:
        df_close = df_close.astype('float32')
    gc.collect()
    return df, df_close, df_pdf

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
# 공통 위젯: PDF 구성종목 비교 (최대 3개 ETF)
# ============================================================================
def render_pdf_comparison(selected_tickers, df_pdf, df_uni, key_prefix="comp"):
    """선택된 ETF들의 PDF 구성종목을 가로 비교로 표시"""
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
                    tbl = pd.DataFrame({
                        '종목': valid.index,
                        '비중(%)': [f"{v:.1f}" for v in valid.values]
                    }).reset_index(drop=True)
                    tbl.index = tbl.index + 1
                    tbl.index.name = '#'
                    st.dataframe(tbl, use_container_width=True, height=390)
                else:
                    st.caption("구성종목 없음 (해외 ETF 등)")
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
            df, df_close, df_pdf = cached_build_universe(min_cap, top_n)
            st.session_state.df_universe = df
            st.session_state.df_prices_kr = df_close
            st.session_state.df_pdf = df_pdf
            st.session_state.base_date = Config.BASE_DATE or datetime.today().strftime("%Y%m%d")
            st.session_state.universe_built = True
            st.success(f"✅ {len(df)}개 ETF 유니버스 빌드 완료!")
        except Exception as e:
            st.error(f"빌드 실패: {e}")
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
        except Exception:
            pass
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

    # ── 필터 ──
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

    # ── 메트릭 ──
    m1, m2, m3, m4 = st.columns(4)
    has_cap = '시가총액(억원)' in df.columns and df['시가총액(억원)'].notna().any()
    cap_total = df['시가총액(억원)'].sum() if has_cap else 0
    m1.metric("총 시가총액", f"{cap_total/10000:.1f}조원" if has_cap else "N/A")
    m2.metric("ETF 수", f"{len(df)}개")
    m3.metric("평균 YTD", f"{df['수익률_YTD(%)'].mean():+.2f}%" if '수익률_YTD(%)' in df.columns else "N/A")
    m4.metric("평균 BM(YTD)", f"{df['BM_YTD(%)'].mean():+.2f}%" if 'BM_YTD(%)' in df.columns else "N/A")

    # ── 데이터 테이블 ──
    display_cols = [c for c in [
        'ETF명','시가총액(억원)','NAV(억원)','설정일',
        '대카테고리','중카테고리','소카테고리','순위(YTD_BM+)',
        '수익률_1M(%)','수익률_3M(%)','수익률_6M(%)','수익률_1Y(%)','수익률_YTD(%)',
        'BM_1M(%)','BM_3M(%)','BM_6M(%)','BM_1Y(%)','BM_YTD(%)',
        '연간변동성(%)','종가','거래량'
    ] if c in df.columns]

    st.dataframe(df[display_cols], use_container_width=True, height=500)

    # ── [기능 1] ETF 선택 → PDF 비교 ──
    etf_options = [f"{t} | {df.at[t,'ETF명'][:30]}" for t in df.index]
    selected = st.multiselect("🔬 PDF 구성종목 비교 (최대 3개 ETF 선택)",
                              etf_options, max_selections=3, key="uni_pdf_comp")
    if selected:
        sel_tickers = [s.split(' | ')[0] for s in selected]
        render_pdf_comparison(sel_tickers, df_pdf, df, key_prefix="uni")

    # ── 카테고리별 시가총액 바 차트 ──
    if has_cap:
        st.markdown("---")
        st.subheader("📊 카테고리별 시가총액")
        cc = df.groupby('대카테고리')['시가총액(억원)'].sum().sort_values(ascending=True)
        fig = px.bar(x=cc.values, y=cc.index, orientation='h',
                    labels={'x':'시총(억)','y':''},
                    color=cc.values, color_continuous_scale='Viridis')
        fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                        margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── BM 상위/하위 ──
    if 'BM_YTD(%)' in df.columns and len(df) > 0:
        st.subheader("📈 BM(YTD) 상위/하위 15")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 상위**")
            top_cols = ['ETF명','대카테고리','BM_YTD(%)','수익률_YTD(%)']
            if has_cap: top_cols.insert(1, '시가총액(억원)')
            st.dataframe(df.nlargest(15,'BM_YTD(%)')[top_cols].reset_index(),
                        use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**🔴 하위**")
            st.dataframe(df.nsmallest(15,'BM_YTD(%)')[top_cols].reset_index(),
                        use_container_width=True, hide_index=True)

    # ── [기능 3] 도넛차트 맨 아래 ──
    if '대카테고리' in df.columns:
        st.markdown("---")
        st.subheader("🍩 카테고리 분포")
        c1, c2 = st.columns(2)
        with c1:
            cc = df['대카테고리'].value_counts()
            fig = px.pie(values=cc.values, names=cc.index, hole=0.4, title="ETF 수 기준")
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if has_cap:
                cc2 = df.groupby('대카테고리')['시가총액(억원)'].sum()
                fig2 = px.pie(values=cc2.values, names=cc2.index, hole=0.4, title="시가총액 기준")
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

    # ── 종목 검색 ──
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

        # 유니버스 정보 추가
        if not res.empty and df_uni is not None:
            for c in ['대카테고리','중카테고리','시가총액(억원)',
                       'BM_1M(%)','BM_3M(%)','BM_6M(%)','BM_1Y(%)','BM_YTD(%)']:
                if c in df_uni.columns:
                    res[c] = res.index.map(lambda t: df_uni.at[t,c] if t in df_uni.index else np.nan)

        st.caption(f"'{sel}' 보유 ETF: **{len(res)}개**")

        # ── [기능 5] 3분할 차트: 비중 + BM성과 ──
        chart_data = res.head(20).copy()
        if len(chart_data) > 0:
            _render_stock_analysis_charts(chart_data, sel, df_uni)

        # ── 결과 테이블 ──
        st.dataframe(res, use_container_width=True, height=400)

        # ── [기능 4] ETF 선택 → PDF 비교 ──
        etf_options = [f"{t} | {res.at[t,'ETF명'][:30]}" for t in res.index]
        selected = st.multiselect("🔬 PDF 구성종목 비교 (최대 3개)",
                                  etf_options, max_selections=3, key="pdf_comp")
        if selected:
            sel_tickers = [s.split(' | ')[0] for s in selected]
            render_pdf_comparison(sel_tickers, df_pdf, df_uni, key_prefix="pdf")

    # ── 인기 종목 ──
    st.markdown("---")
    st.subheader("🏆 보유 ETF 수 상위 종목")
    ts = stock_counts.head(20).reset_index()
    ts.columns = ['종목명','보유 ETF 수']
    st.dataframe(ts, use_container_width=True, hide_index=True)

    # ── 전체 매트릭스 ──
    with st.expander(f"📋 전체 매트릭스 ({len(df_pdf)} × {len(stock_cols)})"):
        st.dataframe(df_pdf, use_container_width=True, height=500)


def _render_stock_analysis_charts(chart_data, stock_name, df_uni):
    """[기능 5] 3분할: 비중 | BM 성과 바 | BM 성과 히트맵"""
    labels = chart_data['ETF명'].str[:15].tolist()
    tickers = chart_data.index.tolist()

    col1, col2, col3 = st.columns(3)

    # ── 좌측: 종목 비중 ──
    with col1:
        fig1 = go.Figure(go.Bar(
            x=chart_data[stock_name].values,
            y=labels, orientation='h',
            marker_color='#3498db',
            text=[f"{v:.1f}%" for v in chart_data[stock_name].values],
            textposition='outside'
        ))
        fig1.update_layout(
            title=f"'{stock_name}' 비중(%)",
            height=max(300, len(labels)*28),
            yaxis=dict(autorange='reversed'),
            margin=dict(l=0, r=40, t=40, b=0),
            xaxis_title=""
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── 중간: BM 대비 성과 (그룹 바) ──
    with col2:
        bm_cols = ['BM_1M(%)','BM_3M(%)','BM_6M(%)','BM_1Y(%)','BM_YTD(%)']
        bm_labels = ['1M','3M','6M','1Y','YTD']
        available_bm = [c for c in bm_cols if c in chart_data.columns]

        if available_bm:
            # Top 10만
            top10 = chart_data.head(10)
            top10_labels = top10['ETF명'].str[:12].tolist()

            fig2 = go.Figure()
            colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
            for j, (bc, bl) in enumerate(zip(available_bm, bm_labels)):
                vals = pd.to_numeric(top10[bc], errors='coerce').fillna(0).values
                fig2.add_trace(go.Bar(
                    name=bl, x=vals, y=top10_labels,
                    orientation='h', marker_color=colors[j % len(colors)],
                ))
            fig2.update_layout(
                title="BM 대비 성과(%)",
                barmode='group',
                height=max(300, len(top10)*28),
                yaxis=dict(autorange='reversed'),
                margin=dict(l=0, r=10, t=40, b=0),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                xaxis_title=""
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("BM 성과 데이터 없음")

    # ── 우측: 히트맵 ──
    with col3:
        if available_bm:
            top10 = chart_data.head(10)
            top10_labels = top10['ETF명'].str[:12].tolist()
            heat_data = []
            for bc in available_bm:
                heat_data.append(pd.to_numeric(top10[bc], errors='coerce').fillna(0).values)

            heat_df = pd.DataFrame(
                heat_data, index=bm_labels[:len(available_bm)],
                columns=top10_labels
            )
            fig3 = px.imshow(heat_df, text_auto='.1f',
                            color_continuous_scale='RdYlGn',
                            aspect='auto',
                            zmin=-20, zmax=20)
            fig3.update_layout(
                title="BM성과 히트맵(%)",
                height=max(300, len(top10)*28),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("BM 성과 데이터 없음")


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
    with c1: start_date = st.date_input("시작일", value=datetime.today()-timedelta(days=90))
    with c2: end_date = st.date_input("종료일", value=datetime.today())
    with c3:
        q = st.radio("빠른 선택", ['직접입력','1M','3M','6M','YTD','1Y'], horizontal=True)
        if q != '직접입력':
            end_date = datetime.today().date()
            start_date = {'1M': end_date-timedelta(30), '3M': end_date-timedelta(90),
                         '6M': end_date-timedelta(180), '1Y': end_date-timedelta(365),
                         'YTD': datetime(end_date.year,1,1).date()}[q]

    tab1, tab2, tab3 = st.tabs(["🇰🇷 국내 ETF", "🌍 글로벌 지수", "🇺🇸 미국 ETF"])

    with tab1:
        if kr_close is not None and not kr_close.empty:
            if df_uni is not None and '대카테고리' in df_uni.columns:
                cats = ['전체'] + sorted(df_uni['대카테고리'].dropna().unique().tolist())
                sc = st.selectbox("카테고리", cats, key="ret_kr_cat")
                vt = df_uni[df_uni['대카테고리']==sc].index.tolist() if sc != '전체' else df_uni.index.tolist()
                vt = [t for t in vt if t in kr_close.columns]
            else: vt = kr_close.columns.tolist()
            opts = [f"{t} | {df_uni.at[t,'ETF명'][:25]}" if t in df_uni.index else t for t in vt]
            sel = st.multiselect("ETF 선택", opts, default=opts[:5], key="kr_sel")
            if sel:
                tks = [s.split(' | ')[0] for s in sel]
                nm = {t: df_uni.at[t,'ETF명'][:15] if t in df_uni.index else t for t in tks}
                _draw_charts(kr_close, tks, start_date, end_date, nm, "국내 ETF")
        else: st.warning("가격 데이터 없음")

    with tab2:
        if gd and not gd.get('indices', pd.DataFrame()).empty:
            di = gd['indices']; ii = gd.get('index_info', {})
            sel = st.multiselect("지수 선택", di.columns.tolist(), default=di.columns.tolist(), key="idx_sel")
            if sel:
                nm = {t: ii[t]['country'] if t in ii else t for t in sel}
                _draw_charts(di, sel, start_date, end_date, nm, "글로벌 지수", show_corr=True)
        elif st.session_state.global_loading:
            st.info("⏳ 수집 중..."); st.button("🔄 새로고침", key="ref_idx")
        else: st.warning("유니버스 빌드 후 자동 수집됩니다.")

    with tab3:
        if gd and not gd.get('us_etfs', pd.DataFrame()).empty:
            du = gd['us_etfs']; ui = gd.get('us_etf_info', {})
            cats = sorted(set(v.get('category','') for v in ui.values()))
            sc = st.selectbox("카테고리", ['전체']+cats, key="us_cat")
            filt = [k for k,v in ui.items() if (sc=='전체' or v.get('category')==sc) and k in du.columns]
            opts = [f"{t} | {ui[t]['name']}" for t in filt if t in ui]
            sel = st.multiselect("ETF 선택", opts, default=opts[:8], key="us_sel")
            if sel:
                tks = [s.split(' | ')[0] for s in sel]
                nm = {t: ui[t]['name'][:18] if t in ui else t for t in tks}
                _draw_charts(du, tks, start_date, end_date, nm, "미국 ETF")
        elif st.session_state.global_loading:
            st.info("⏳ 수집 중..."); st.button("🔄 새로고침", key="ref_us")
        else: st.warning("유니버스 빌드 후 자동 수집됩니다.")

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
                ret = calc_period_return(comb, start_date, end_date)
                if not ret.empty:
                    rs = ret.sort_values(ascending=False)
                    colors = ['#2ecc71' if v>=0 else '#e74c3c' for v in rs.values]
                    fig = go.Figure(go.Bar(x=rs.values, y=rs.index, orientation='h', marker_color=colors,
                                          text=[f"{v:+.2f}%" for v in rs.values], textposition='outside'))
                    fig.update_layout(title=f"크로스 비교 ({start_date}~{end_date})",
                                    height=max(300,len(rs)*40), margin=dict(l=0,r=60,t=40,b=0))
                    st.plotly_chart(fig, use_container_width=True)


def _draw_charts(df_p, tickers, sd, ed, name_map=None, prefix="", show_corr=False):
    valid = [t for t in tickers if t in df_p.columns]
    if not valid: st.warning("가격 데이터 없음"); return
    sub = df_p[valid]; ret = calc_period_return(sub, sd, ed)
    if ret.empty: st.warning("기간 데이터 부족"); return

    labels = [f"{t} ({name_map.get(t,t)})" for t in ret.index] if name_map else ret.index.tolist()
    rd = ret.copy(); rd.index = labels; rs = rd.sort_values(ascending=False)
    colors = ['#2ecc71' if v>=0 else '#e74c3c' for v in rs.values]
    fig = go.Figure(go.Bar(x=rs.values, y=rs.index, orientation='h', marker_color=colors,
                           text=[f"{v:+.2f}%" for v in rs.values], textposition='outside'))
    fig.update_layout(title=f"{prefix} 수익률 ({sd}~{ed})", height=max(300,len(rs)*35),
                     xaxis_title="수익률(%)", margin=dict(l=0,r=60,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    mask = (sub.index>=pd.Timestamp(sd)) & (sub.index<=pd.Timestamp(ed))
    sp = sub[mask].dropna(how='all')
    if len(sp) > 1:
        norm = (sp/sp.iloc[0]-1)*100
        if name_map: norm.columns = [f"{t} ({name_map.get(t,t)})" for t in norm.columns]
        fig2 = px.line(norm, title=f"{prefix} 추이 (정규화)")
        fig2.update_layout(height=400, yaxis_title="수익률(%)")
        st.plotly_chart(fig2, use_container_width=True)

    if show_corr and len(valid) >= 2:
        corr = sp.pct_change().dropna().corr()
        if name_map:
            corr.index = [name_map.get(t,t) for t in corr.index]
            corr.columns = [name_map.get(t,t) for t in corr.columns]
        fig3 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 수익률 상세"):
        st.dataframe(rs.to_frame('수익률(%)'), use_container_width=True)


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
