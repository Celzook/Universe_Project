"""
ETF Uniview Page
================
- 최대 20개 ETF 다중 선택 (US_ETFS 풀)
- 5×4 그리드: 캔들스틱 + 볼린저밴드 + RSI (최근 3개월)
- 각 셀에 C_score 오버레이 표시
- 하단 랭킹 보드: ETF 스코어 상위/하위 15, 카테고리 평균 상위/하위 15
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from etf_scoring import (
    compute_bollinger,
    compute_rsi,
    compute_T_score,
    compute_C_score,
    fetch_macro_scores,
    fetch_ohlcv_batch,
    score_all_etfs,
)
from global_price_collector import US_ETFS

# ── 상수 ───────────────────────────────────────────────────────────────────
GRID_COLS = 5
GRID_ROWS = 4
MAX_SELECT = GRID_COLS * GRID_ROWS   # 20

# ETF 풀: ticker → name
ETF_POOL = {k: v['name'] for k, v in US_ETFS.items()}
ETF_POOL_OPTIONS = sorted([f"{t} | {n}" for t, n in ETF_POOL.items()])
DEFAULT_TICKERS = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'TLT', 'GLD', 'EWY', 'SOXX']
DEFAULT_OPTIONS = [o for o in ETF_POOL_OPTIONS if o.split(' | ')[0] in DEFAULT_TICKERS]


# ── 차트 생성 ───────────────────────────────────────────────────────────────

def _score_color(c_score: float) -> str:
    if np.isnan(c_score):
        return '#888888'
    if c_score >= 30:
        return '#2ecc71'
    if c_score <= -30:
        return '#e74c3c'
    return '#f39c12'


def _make_chart(df_ohlcv: pd.DataFrame, ticker: str,
                C_score: float, T_score: float) -> go.Figure:
    """캔들스틱 + 볼린저밴드(row1) + RSI(row2) 통합 차트."""
    if df_ohlcv.empty or len(df_ohlcv) < 20:
        fig = go.Figure()
        fig.add_annotation(text="데이터 부족", x=0.5, y=0.5,
                           showarrow=False, xref="paper", yref="paper",
                           font=dict(color='#aaa', size=11))
        fig.update_layout(height=280, margin=dict(l=4, r=4, t=28, b=4),
                          paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
        return fig

    close = df_ohlcv['Close'].squeeze()
    open_ = df_ohlcv['Open'].squeeze()
    high  = df_ohlcv['High'].squeeze()
    low   = df_ohlcv['Low'].squeeze()

    pb, upper, lower, mid = compute_bollinger(close)
    rsi = compute_rsi(close)

    score_clr = _score_color(C_score)
    score_txt = f"N/A" if np.isnan(C_score) else f"{C_score:+.1f}"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # ── Row 1: 캔들 + BB ──
    fig.add_trace(go.Candlestick(
        x=df_ohlcv.index,
        open=open_, high=high, low=low, close=close,
        increasing_line_color='#e74c3c',
        decreasing_line_color='#3498db',
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=upper,
        line=dict(color='rgba(180,180,180,0.5)', width=1),
        showlegend=False, name='Upper',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=lower,
        line=dict(color='rgba(180,180,180,0.5)', width=1),
        fill='tonexty', fillcolor='rgba(150,150,150,0.07)',
        showlegend=False, name='Lower',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=mid,
        line=dict(color='rgba(150,150,255,0.7)', width=1, dash='dot'),
        showlegend=False, name='Mid',
    ), row=1, col=1)

    # ── Row 2: RSI ──
    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=rsi,
        line=dict(color='#9b59b6', width=1.5),
        showlegend=False, name='RSI',
    ), row=2, col=1)
    for lvl, clr in [(70, 'rgba(231,76,60,0.35)'), (50, 'rgba(200,200,200,0.2)'),
                     (30, 'rgba(52,152,219,0.35)')]:
        fig.add_hline(y=lvl, line_dash='dash', line_color=clr,
                      line_width=1, row=2, col=1)

    # ── C_score 오버레이 (우상단) ──
    fig.add_annotation(
        x=1, y=1, xref='paper', yref='paper',
        text=f"<b>C:{score_txt}</b>",
        showarrow=False, xanchor='right', yanchor='top',
        font=dict(size=12, color=score_clr),
        bgcolor='rgba(14,17,23,0.80)',
        bordercolor=score_clr, borderwidth=1, borderpad=3,
    )

    # ── 레이아웃 ──
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>", font=dict(size=11), x=0.03, y=0.97),
        height=290,
        margin=dict(l=4, r=4, t=26, b=4),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ccc', size=8),
        xaxis=dict(rangeslider=dict(visible=False), showgrid=False, showticklabels=False),
        xaxis2=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1e2130', tickfont=dict(size=7)),
        yaxis2=dict(showgrid=True, gridcolor='#1e2130', tickfont=dict(size=7),
                    range=[0, 100], dtick=25),
    )
    return fig


# ── 메인 페이지 함수 ────────────────────────────────────────────────────────

def page_etf_uniview():
    st.title("🔭 ETF Uniview")
    st.caption(
        "최대 20개 ETF를 선택하면 **5×4 그리드**로 캔들차트 + 볼린저밴드 + RSI + **C_score**를 한눈에 확인합니다. "
        "하단에서 전체 ETF 스코어 랭킹도 확인할 수 있습니다."
    )

    # ── ETF 선택 ────────────────────────────────────────────────────────────
    selected_opts = st.multiselect(
        f"📌 ETF 선택 (최대 {MAX_SELECT}개)",
        ETF_POOL_OPTIONS,
        default=DEFAULT_OPTIONS,
        max_selections=MAX_SELECT,
        key="uniview_sel",
    )

    if not selected_opts:
        st.info("위에서 ETF를 선택하세요.")
        _render_ranking_board()
        return

    selected_tickers = [s.split(' | ')[0] for s in selected_opts]

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    with st.spinner("📡 가격 데이터 수집 및 C_score 계산 중..."):
        macro_result = fetch_macro_scores()
        M_score = macro_result['M_score']
        ohlcv_dict = fetch_ohlcv_batch(tuple(selected_tickers), months=3)

    # C_score 계산
    scores_map: dict[str, tuple[float, float]] = {}
    for t in selected_tickers:
        df_t = ohlcv_dict.get(t, pd.DataFrame())
        if df_t.empty or 'Close' not in df_t.columns:
            scores_map[t] = (np.nan, np.nan)
        else:
            close = df_t['Close'].squeeze().dropna()
            T = compute_T_score(close)
            C = compute_C_score(T, M_score)
            scores_map[t] = (T, C)

    # ── Macro 정보 ──────────────────────────────────────────────────────────
    with st.expander("🌐 Macro Score 상세 보기"):
        mc = macro_result['scores']
        md = macro_result['details']
        mcols = st.columns(6)
        for i, (lbl, key) in enumerate([
            ('Fed(-15)', 'fed'), ('Geo(-7)', 'geo'), ('WTI(-25)', 'oil'),
            ('BEI(+10)', 'bei'), ('VIX(-20)', 'vix'),
        ]):
            mcols[i].metric(lbl, f"{mc[key]:+.2f}")
        mcols[5].metric("M_score", f"{M_score:+.1f}")
        st.caption("GPR: " + str(md.get('GPR', 'N/A')))

    st.markdown("---")

    # ── 5×4 그리드 ──────────────────────────────────────────────────────────
    n = len(selected_tickers)
    for row_i in range(GRID_ROWS):
        cols = st.columns(GRID_COLS)
        for col_j in range(GRID_COLS):
            idx = row_i * GRID_COLS + col_j
            if idx >= n:
                break
            ticker = selected_tickers[idx]
            T_score, C_score = scores_map.get(ticker, (np.nan, np.nan))
            df_ohlcv = ohlcv_dict.get(ticker, pd.DataFrame())
            with cols[col_j]:
                fig = _make_chart(df_ohlcv, ticker, C_score, T_score)
                st.plotly_chart(fig, width='stretch',
                                config={'displayModeBar': False})

    # ── 랭킹 보드 ────────────────────────────────────────────────────────────
    _render_ranking_board()


def _render_ranking_board():
    """전체 ETF 풀 기준 스코어 상/하위 15 + 카테고리 평균 상/하위 15."""
    st.markdown("---")
    st.subheader("🏆 ETF 스코어 랭킹 보드")
    st.caption("전체 US ETF 풀 기준 (캐시 2시간 유지)")

    all_tickers_tuple = tuple(sorted(ETF_POOL.keys()))

    with st.spinner("전체 ETF 스코어 계산 중... (첫 실행 후 캐시)"):
        df_scores = score_all_etfs(all_tickers_tuple)

    df_scores['ETF명'] = df_scores.index.map(lambda t: ETF_POOL.get(t, t))
    df_scores['카테고리'] = df_scores.index.map(
        lambda t: US_ETFS.get(t, {}).get('category', 'N/A')
    )
    valid = df_scores.dropna(subset=['C_score'])

    # ── 개별 ETF 상위/하위 15 ────────────────────────────────────────────────
    disp_cols = ['ticker', 'ETF명', '카테고리', 'T_score', 'C_score']
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🟢 C_score 상위 15**")
        top15 = valid.nlargest(15, 'C_score').reset_index()[disp_cols]
        st.dataframe(top15, width='stretch', hide_index=True)
    with c2:
        st.markdown("**🔴 C_score 하위 15**")
        bot15 = valid.nsmallest(15, 'C_score').reset_index()[disp_cols]
        st.dataframe(bot15, width='stretch', hide_index=True)

    # ── 카테고리 평균 상위/하위 15 ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 카테고리 평균 C_score 랭킹")

    cat_avg = (
        valid.groupby('카테고리')['C_score']
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
        st.dataframe(
            cat_avg.tail(15).sort_values('평균_C_score').reset_index(),
            width='stretch', hide_index=True,
        )
