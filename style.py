"""
Global UI styles — Pretendard 폰트 + 카드 그림자/모서리 + 한국식 색상.

`inject_global_styles()` 를 app 진입점에서 1회 호출.
"""
import streamlit as st

# 한국식 색상 컨벤션 (상승=빨강, 하락=파랑) — 차트/지표 전반에 일관 적용
UP_COLOR = '#e74c3c'    # 빨강 (상승/양수)
DOWN_COLOR = '#3498db'  # 파랑 (하락/음수)


_GLOBAL_CSS = """
<style>
/* ── Pretendard 폰트 (한+영 모던 sans-serif) ───────────────── */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

html, body, [class*="st-"], .stApp, .stMarkdown,
.stButton button, .stTextInput input, .stSelectbox div,
.stRadio label, .stCheckbox label, .stDataFrame,
.stMetricLabel, .stMetricValue {
    font-family: 'Pretendard', 'Pretendard Variable', -apple-system,
                 BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue',
                 Arial, 'Noto Sans KR', sans-serif !important;
    font-feature-settings: 'tnum';
    letter-spacing: -0.01em;
}

/* ── 카드 컨테이너 (st.container border=True) ───────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06),
                0 2px 12px rgba(15, 23, 42, 0.04) !important;
    border: 1px solid rgba(15, 23, 42, 0.06) !important;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08),
                0 8px 24px rgba(15, 23, 42, 0.06) !important;
}

/* ── 헤딩 ─────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Pretendard', 'Pretendard Variable', -apple-system, sans-serif !important;
    letter-spacing: -0.02em !important;
    font-weight: 700 !important;
}

/* ── DataFrame (Streamlit table) 모서리 ────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Radio (탭) 부드럽게 ──────────────────────────────────── */
[role="radiogroup"] label {
    border-radius: 8px !important;
}

/* ── Metric 박스 ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.02);
    padding: 10px 14px;
    border-radius: 10px;
}

/* ── 사이드바 ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
                rgba(248, 250, 252, 0.7),
                rgba(241, 245, 249, 0.7)) !important;
}

/* ── 캡션 (작은 회색 텍스트) ─────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: rgba(100, 116, 139, 0.9) !important;
    font-size: 11.5px !important;
    letter-spacing: 0 !important;
}

/* ── 코드 칩 (`ticker` 같은 inline code) ──────────────────── */
code {
    background: rgba(99, 102, 241, 0.08) !important;
    color: #4f46e5 !important;
    padding: 1px 6px !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

/* ── 차트/이미지도 둥글게 ────────────────────────────────── */
.stPlotlyChart, img {
    border-radius: 10px;
}

/* ── 컨테이너 간 spacing 살짝 ───────────────────────────── */
section.main > div { padding-top: 1rem; }
</style>
"""


def inject_global_styles() -> None:
    """앱 진입점에서 1회 호출. 폰트/카드/배지 글로벌 적용."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
