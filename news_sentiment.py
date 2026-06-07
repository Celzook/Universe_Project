"""
news_sentiment.py — AP/MP 보유 섹터 + KOSPI200 매크로 뉴스 센티먼트 요약.

흐름
----
1) 네이버 뉴스 검색 API 로 섹터/매크로 키워드별 최신 헤드라인 fetch.
2) Anthropic Claude (haiku) 로 3줄 요약 + 센티먼트(긍정/부정/중립) + 키워드 추출.
3) 6시간 캐시 (`@st.cache_data(ttl=3600*6)`).

키 설정 (`.streamlit/secrets.toml` 또는 환경변수)
------------------------------------------------
    anthropic_api_key = "sk-ant-..."          # ANTHROPIC_API_KEY
    naver_news_client_id = "..."              # NAVER_NEWS_CLIENT_ID
    naver_news_client_secret = "..."          # NAVER_NEWS_CLIENT_SECRET

키 미설정 시 모든 진입점은 `error` 키가 채워진 dict 반환 → UI 가 친절한 안내 표시.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import streamlit as st


# ── 시크릿 / 환경변수 헬퍼 ───────────────────────────────────────────────
def _get_secret(name: str, env_name: Optional[str] = None) -> Optional[str]:
    """st.secrets → env var 순으로 키 탐색."""
    try:
        val = st.secrets.get(name)  # type: ignore[attr-defined]
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(env_name or name.upper())


def _anthropic_key() -> Optional[str]:
    return _get_secret('anthropic_api_key', 'ANTHROPIC_API_KEY')


def _naver_credentials() -> tuple:
    cid = _get_secret('naver_news_client_id', 'NAVER_NEWS_CLIENT_ID')
    secret = _get_secret('naver_news_client_secret', 'NAVER_NEWS_CLIENT_SECRET')
    return cid, secret


def keys_configured() -> dict:
    """현재 키 설정 상태 (UI 진단용)."""
    nv_id, nv_sec = _naver_credentials()
    return {
        'naver': bool(nv_id and nv_sec),
        'anthropic': bool(_anthropic_key()),
    }


# ── 네이버 뉴스 fetch ────────────────────────────────────────────────────
_TAG_RE = re.compile(r'<[^>]+>')


def _strip_html(s: str) -> str:
    if not s:
        return ''
    return (_TAG_RE.sub('', s)
            .replace('&quot;', '"')
            .replace('&amp;', '&')
            .replace('&lt;', '<')
            .replace('&gt;', '>')
            .strip())


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def fetch_naver_news(query: str, display: int = 7) -> list:
    """네이버 뉴스 검색 API 호출 → [{title, link, pubDate, description}] list.

    캐시 6시간. 키 미설정 / 응답 실패 시 빈 리스트.
    """
    cid, secret = _naver_credentials()
    if not cid or not secret or not query.strip():
        return []
    try:
        import requests
    except ImportError:
        return []

    url = "https://openapi.naver.com/v1/search/news.json"
    params = {'query': query, 'display': max(1, min(display, 10)), 'sort': 'date'}
    headers = {'X-Naver-Client-Id': cid, 'X-Naver-Client-Secret': secret}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for it in data.get('items', []):
            out.append({
                'title': _strip_html(it.get('title', '')),
                'link': it.get('originallink') or it.get('link', ''),
                'pubDate': it.get('pubDate', ''),
                'description': _strip_html(it.get('description', '')),
            })
        return out
    except Exception:
        return []


# ── Claude 요약 ─────────────────────────────────────────────────────────
_SECTOR_PROMPT = """당신은 한국 ETF 투자자를 위한 섹터 분석가입니다.

다음은 '{sector}' 섹터 관련 최근 뉴스 헤드라인입니다:

{headlines}

이를 바탕으로 정확히 3줄로 요약해 주세요 (각 줄 한국어, 한 줄 60자 이내):
- 1줄: 최근 해당 섹터의 주요 이슈
- 2줄: 실적 / 펀더멘털 동향
- 3줄: 주목 기업 또는 종목

전체 톤(positive/negative/neutral)과 핵심 키워드 3~5개도 함께 추출하세요.

다음 JSON 형식만 응답 (다른 텍스트 금지):
{{
  "summary": "1줄\\n2줄\\n3줄",
  "sentiment": "positive|negative|neutral",
  "keywords": ["키워드1", "키워드2", "키워드3"]
}}"""

_KOSPI200_PROMPT = """당신은 한국 증시 매크로 분석가입니다.

다음은 한국 증시·경제 관련 최근 뉴스 헤드라인입니다:

{headlines}

이를 바탕으로 정확히 3줄로 요약해 주세요 (각 줄 한국어, 한 줄 60자 이내):
- 1줄: 한국 증시 주요 이슈 / 외국인·기관 수급
- 2줄: 한국 경제성장 / 주요 경제지표 (CPI, 수출 등)
- 3줄: 시장금리 / 환율 / 정책 동향

전체 톤(positive/negative/neutral)과 핵심 키워드 3~5개도 함께 추출하세요.

다음 JSON 형식만 응답 (다른 텍스트 금지):
{{
  "summary": "1줄\\n2줄\\n3줄",
  "sentiment": "positive|negative|neutral",
  "keywords": ["키워드1", "키워드2", "키워드3"]
}}"""


_MODEL = "claude-haiku-4-5-20251001"


def _summarize_with_claude(headlines: list,
                           prompt_template: str,
                           sector: str = "") -> Optional[dict]:
    """Claude haiku 호출 → 요약 dict. 실패 시 None.

    `prompt_template` 에 {sector} 가 있으면 채워 넣고, 아니면 헤드라인만 주입.
    """
    key = _anthropic_key()
    if not key or not headlines:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    headlines_text = "\n".join(
        f"{i + 1}. {h.get('title', '')}" for i, h in enumerate(headlines[:10])
    )
    if "{sector}" in prompt_template:
        prompt = prompt_template.format(sector=sector, headlines=headlines_text)
    else:
        prompt = prompt_template.format(headlines=headlines_text)

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        # 응답에서 JSON 블록 추출 (모델이 코드펜스 감쌀 수도 있음)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception:
        return None


# ── 공개 진입점 ─────────────────────────────────────────────────────────
# 섹터명 → 네이버 검색 키워드 매핑. 미등록 시 '<섹터명> 시장 동향' 으로 폴백.
_SECTOR_QUERY_MAP = {
    '반도체': '반도체 산업 동향',
    '2차전지': '2차전지 배터리 시장',
    'IT': 'IT 산업 동향',
    '바이오': '바이오 제약 동향',
    'AI': '인공지능 AI 산업',
    '자동차': '자동차 산업 동향',
    '조선': '조선 산업 동향',
    '철강': '철강 산업 동향',
    '금융': '금융주 시장 동향',
    '은행': '은행주 동향',
    '증권': '증권주 동향',
    '보험': '보험주 동향',
    '건설': '건설업 동향',
    '화학': '화학 산업 동향',
    '에너지': '에너지 산업 동향',
    '반도체장비': '반도체장비 산업',
    '소부장': '소재 부품 장비',
    '게임': '게임 산업 동향',
    '엔터': '엔터테인먼트 K팝',
    '미디어': '미디어 산업 동향',
    '리츠': '리츠 부동산 시장',
    '원자재': '원자재 시장 동향',
}


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def get_sector_sentiment(sector: str) -> dict:
    """섹터별 이슈/센티먼트 요약. 캐시 6시간.

    Returns
    -------
    {summary, sentiment, keywords, headlines, error?}
    """
    if not sector or not sector.strip():
        return {'error': '섹터명 비어있음', 'sentiment': 'neutral',
                'summary': '', 'keywords': [], 'headlines': []}
    query = _SECTOR_QUERY_MAP.get(sector.strip(), f"{sector.strip()} 시장 동향")
    headlines = fetch_naver_news(query, display=7)
    if not headlines:
        return {'error': '뉴스 조회 실패 — 네이버 API 키 미설정 또는 응답 없음.',
                'sentiment': 'neutral', 'summary': '', 'keywords': [], 'headlines': []}
    summary = _summarize_with_claude(headlines, _SECTOR_PROMPT, sector)
    if not summary:
        return {'error': 'LLM 요약 실패 — Anthropic 키 미설정 또는 응답 실패.',
                'sentiment': 'neutral', 'summary': '',
                'keywords': [], 'headlines': headlines}
    return {
        'summary': str(summary.get('summary', '')),
        'sentiment': str(summary.get('sentiment', 'neutral')),
        'keywords': list(summary.get('keywords', [])),
        'headlines': headlines,
        'error': None,
    }


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def get_kospi200_sentiment() -> dict:
    """KOSPI200 / 한국 증시 매크로 센티먼트 요약. 캐시 6시간.

    세 가지 매크로 키워드(증시·지표·금리) 헤드라인을 합쳐서 한 번에 요약.
    """
    headlines = []
    for q in ['한국 증시 코스피', '한국 경제지표 수출 CPI', '한국 금리 환율 정책']:
        h = fetch_naver_news(q, display=4)
        headlines.extend(h)

    # 중복 제거 (link 기준), 최신 우선 12건 사용
    seen, dedup = set(), []
    for h in headlines:
        link = h.get('link', '')
        if link and link not in seen:
            seen.add(link)
            dedup.append(h)
    headlines = dedup[:12]

    if not headlines:
        return {'error': '뉴스 조회 실패 — 네이버 API 키 미설정 또는 응답 없음.',
                'sentiment': 'neutral', 'summary': '', 'keywords': [], 'headlines': []}
    summary = _summarize_with_claude(headlines, _KOSPI200_PROMPT)
    if not summary:
        return {'error': 'LLM 요약 실패 — Anthropic 키 미설정 또는 응답 실패.',
                'sentiment': 'neutral', 'summary': '',
                'keywords': [], 'headlines': headlines}
    return {
        'summary': str(summary.get('summary', '')),
        'sentiment': str(summary.get('sentiment', 'neutral')),
        'keywords': list(summary.get('keywords', [])),
        'headlines': headlines,
        'error': None,
    }
