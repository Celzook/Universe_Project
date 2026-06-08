"""차트 매트릭스 페이지.

선택한 국내 ETF (최대 20개) 의 캔들 + 볼린저밴드 + RSI 차트를 5×4 그리드로 표시.
- 데이터: yfinance .KS (6개월), 기본 뷰 3개월
- 마우스 휠 줌, 드래그 이동
- C_score 오버레이 + Macro Score 상세 expander

원래 ETF Uniview 페이지에 있던 차트 매트릭스를 분리한 페이지.
공통 의존: app.py 의 session_state.df_universe.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from etf_scoring import (
    compute_bollinger, compute_rsi,
    compute_T_score, compute_C_score,
    fetch_macro_scores, fetch_kr_ohlcv_batch,
)


GRID_COLS = 5
GRID_ROWS = 4
MAX_SELECT = GRID_COLS * GRID_ROWS   # 20
FETCH_MONTHS = 6
DEFAULT_VIEW_MONTHS = 3


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
    """캔들스틱(Row1) + 볼린저밴드 + RSI(Row2).
    마우스 휠 줌: scrollZoom=True + fixedrange=False.
    기본 뷰: 최근 3개월, 데이터: 최근 6개월."""
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
    score_txt = "N/A" if np.isnan(C_score) else f"{C_score:+.1f}"

    view_start = (datetime.today() - timedelta(days=DEFAULT_VIEW_MONTHS * 31)).strftime('%Y-%m-%d')
    view_end   = datetime.today().strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.03,
    )

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

    fig.add_trace(go.Scatter(
        x=df_ohlcv.index, y=rsi,
        line=dict(color='#9b59b6', width=1.5),
        showlegend=False,
    ), row=2, col=1)
    for lvl, clr in [(70, 'rgba(231,76,60,0.35)'), (50, 'rgba(200,200,200,0.2)'),
                     (30, 'rgba(52,152,219,0.35)')]:
        fig.add_hline(y=lvl, line_dash='dash', line_color=clr,
                      line_width=1, row=2, col=1)

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
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=False, fixedrange=False,
            range=[view_start, view_end],
        ),
        xaxis2=dict(showgrid=False, fixedrange=False),
        yaxis=dict(
            showgrid=True, gridcolor='#1e2130',
            tickfont=dict(size=7), fixedrange=False,
        ),
        yaxis2=dict(
            showgrid=True, gridcolor='#1e2130',
            tickfont=dict(size=7),
            range=[0, 100], dtick=25, fixedrange=True,
        ),
        dragmode='pan',
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


def page_chart_matrix():
    st.title("📊 차트 매트릭스")

    if not st.session_state.get('universe_built', False):
        st.warning("👈 사이드바에서 먼저 **🚀 유니버스 빌드**를 실행하세요.")
        return

    df_uni = st.session_state.df_universe
    if df_uni is None or df_uni.empty:
        st.warning("유니버스 데이터가 없습니다. 다시 빌드해 주세요.")
        return

    # ── ETF 풀: 시총 상위 순 정렬 ───────────────────────────────────────
    pool_df = df_uni.copy()
    if '시가총액(억원)' in pool_df.columns:
        pool_df = pool_df.sort_values('시가총액(억원)', ascending=False)

    pool_tickers = pool_df.index.tolist()
    pool_names   = pool_df['ETF명'].tolist() if 'ETF명' in pool_df.columns else pool_tickers
    ticker_to_name = dict(zip(pool_tickers, pool_names))

    pool_options = [f"{t} | {n[:28]}" for t, n in zip(pool_tickers, pool_names)]
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
        key="chart_matrix_sel",
    )

    selected_tickers = [s.split(' | ')[0] for s in selected_opts]
    n_sel = len(selected_tickers)

    # ── 가격 데이터 + 스코어 계산 ───────────────────────────────────────
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

    # ── 5×4 그리드 (항상 20칸 렌더링) ───────────────────────────────────
    for row_i in range(GRID_ROWS):
        cols = st.columns(GRID_COLS)
        for col_j in range(GRID_COLS):
            idx = row_i * GRID_COLS + col_j
            with cols[col_j]:
                if idx >= n_sel:
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
