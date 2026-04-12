"""
ETF 구성종목 수집 진단 앱
----------------------------
Streamlit Cloud 배포용 독립 파일
저장소 루트에 이 파일을 추가하고
Streamlit Cloud에서 '앱 추가' → Main file: diagnose_holdings.py 로 배포
"""
import streamlit as st
import requests
import json
import io
import re
import pandas as pd

st.set_page_config(page_title="ETF 구성종목 진단", layout="wide")
st.title("🔍 ETF 구성종목 수집 진단 (미국 서버 환경)")
st.caption("KRX/네이버 API가 Streamlit Cloud(미국 IP)에서 실제로 어떻게 응답하는지 확인")

TICKER = st.text_input("진단할 티커", value="069500")

if st.button("▶ 진단 실행", type="primary"):

    # ── 테스트 1: KRX ──────────────────────────────────────────────────────
    st.header("① KRX 직접 HTTP")
    try:
        r = requests.post(
            "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
            data={
                'bld': 'dbms/comm/finder/finder_secuprodisu',
                'mktsel': 'ETF',
                'searchText': TICKER,
                'locale': 'ko_KR',
            },
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        st.write(f"**Status:** `{r.status_code}`")
        st.code(r.text[:300])
        if r.status_code == 200:
            st.success("KRX 접근 가능!")
        else:
            st.error("KRX 차단됨 → 네이버 fallback 필수")
    except Exception as e:
        st.error(f"KRX 연결 실패: {e}")

    st.divider()

    # ── 테스트 2: 네이버 모바일 API ────────────────────────────────────────
    st.header("② 네이버 모바일 API `m.stock.naver.com`")
    url2 = f"https://m.stock.naver.com/api/stock/{TICKER}/etf/portfolio"
    st.code(url2)
    try:
        r2 = requests.get(url2, headers={
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)',
            'Referer': f'https://m.stock.naver.com/domestic/stock/{TICKER}/etf'
        }, timeout=10)
        st.write(f"**Status:** `{r2.status_code}`")
        st.write(f"**Content-Type:** `{r2.headers.get('Content-Type', '없음')}`")
        st.write(f"**응답 길이:** {len(r2.text)} bytes")

        if r2.status_code == 200:
            st.success("접근 성공!")
            try:
                data = r2.json()
                st.write(f"**JSON 최상위 타입:** `{type(data).__name__}`")
                if isinstance(data, dict):
                    st.write(f"**최상위 keys:** `{list(data.keys())}`")
                    for k, v in data.items():
                        info = f"type={type(v).__name__}"
                        if isinstance(v, list):
                            info += f", len={len(v)}"
                            if v and isinstance(v[0], dict):
                                info += f"\n    첫 항목 keys: {list(v[0].keys())}"
                                info += f"\n    첫 항목: {v[0]}"
                        elif isinstance(v, dict):
                            info += f", keys={list(v.keys())[:8]}"
                        st.write(f"  `{k}` → {info}")
                elif isinstance(data, list):
                    st.write(f"**리스트 길이:** {len(data)}")
                    if data:
                        st.write(f"**첫 항목:** {data[0]}")
            except Exception as e:
                st.warning(f"JSON 파싱 실패: {e}")
                st.code(r2.text[:800])
        else:
            st.error(f"차단/오류: {r2.text[:300]}")
    except Exception as e:
        st.error(f"연결 실패: {e}")

    st.divider()

    # ── 테스트 3: etfinfo.naver (CU당 구성종목 페이지) ────────────────────
    st.header("③ 네이버 ETF 상세 페이지 `etfinfo.naver` (CU당 구성종목)")
    url3 = f"https://finance.naver.com/item/etfinfo.naver?code={TICKER}"
    st.code(url3)
    try:
        r3 = requests.get(url3, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        st.write(f"**Status:** `{r3.status_code}`")
        st.write(f"**응답 길이:** {len(r3.content)} bytes")

        if r3.status_code == 200:
            try:
                html = r3.content.decode('euc-kr')
            except Exception:
                html = r3.text

            # 주요 키워드 존재 여부
            keywords = ['구성종목', 'CU당', '비중', '종목명', '편입비',
                        'holdingRatio', 'weightedValue', 'portfolioList',
                        'ajax', 'XMLHttpRequest', '<table']
            cols = st.columns(3)
            for i, kw in enumerate(keywords):
                cols[i % 3].write(f"{'✅' if kw.lower() in html.lower() else '❌'} `{kw}`")

            st.write(f"**`<table>` 태그 수:** {html.count('<table')}")

            # '구성종목' 주변 텍스트
            idx = html.lower().find('구성종목')
            if idx >= 0:
                st.write("**'구성종목' 주변 HTML:**")
                st.code(html[max(0, idx-100):idx+500])
            else:
                st.warning("'구성종목' 키워드 없음 → JS 동적 로드 의심")

            # pd.read_html 시도
            st.write("**pd.read_html 결과:**")
            try:
                tables = pd.read_html(io.StringIO(html))
                st.write(f"{len(tables)}개 테이블 발견")
                for i, t in enumerate(tables):
                    is_candidate = any(
                        any(kw in str(c) for kw in ['종목', '비중', '%', '편입'])
                        for c in t.columns
                    )
                    label = "⬅️ **구성종목 후보!**" if is_candidate else ""
                    st.write(f"  [{i}] shape={t.shape} | cols={t.columns.tolist()[:6]} {label}")
                    if is_candidate:
                        st.dataframe(t.head(10))
            except Exception as e:
                st.error(f"pd.read_html 실패: {e}")

            # 페이지 내 AJAX URL 탐색
            ajax_urls = re.findall(
                r'(?:url|fetch|axios\.get)\s*[:(]\s*["\']([^"\']{10,})["\']', html
            )
            if ajax_urls:
                st.write("**페이지 내 API/AJAX 호출 URL:**")
                for u in set(ajax_urls):
                    st.code(u)
        else:
            st.error(f"차단/오류: {r3.text[:200]}")
    except Exception as e:
        st.error(f"연결 실패: {e}")

    st.divider()

    # ── 테스트 4: etfItemList API ──────────────────────────────────────────
    st.header("④ 네이버 ETF 리스트 API (설정일 확인)")
    url4 = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0&targetColumn=market_sum&sortOrder=desc"
    try:
        r4 = requests.get(url4, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        st.write(f"**Status:** `{r4.status_code}`")
        if r4.status_code == 200:
            data4 = r4.json()
            items = data4.get('result', {}).get('etfItemList', [])
            st.write(f"**전체 ETF 수:** {len(items)}")
            if items:
                st.write(f"**첫 항목 keys:** `{list(items[0].keys())}`")
                target = next(
                    (x for x in items if str(x.get('itemcode', '')).strip() == TICKER),
                    None
                )
                if target:
                    st.write(f"**{TICKER} 항목 전체:**")
                    st.json(target)
                else:
                    st.write("해당 티커 없음, 첫 2개:")
                    st.json(items[:2])
    except Exception as e:
        st.error(f"연결 실패: {e}")

    st.divider()
    st.info("""
**결과 해석 가이드**
- ① KRX 403 → 미국 IP 차단 확정 (예상된 결과)
- ② 모바일 API 200 → JSON 구조를 보고 정확한 key 수정 가능
- ② 모바일 API 403/차단 → 네이버도 미국 IP 차단, HTML 스크래핑만 가능
- ③ etfinfo 200 + '구성종목' ✅ + 테이블 발견 → pd.read_html 파싱 코드 수정으로 해결
- ③ etfinfo 200 + '구성종목' ❌ → JS 동적 로드, AJAX URL 확인해서 직접 호출
- ③ etfinfo 403 → 네이버 finance도 차단, 대안 데이터 소스 필요
    """)
