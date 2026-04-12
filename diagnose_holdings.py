"""
holdings 수집 실패 원인 진단 스크립트
Streamlit Cloud에서 직접 실행하거나, app.py에 임시 버튼으로 넣어서 실행

사용법:
  streamlit run diagnose_holdings.py
"""

import streamlit as st
import requests
import json
import io
import re

TICKER = "069500"  # KODEX 200

st.title("ETF 구성종목 수집 진단")

# ── 테스트 1: m.stock.naver.com 모바일 API ──────────────────────────────
st.header("테스트 1: 네이버 모바일 API")
url1 = f"https://m.stock.naver.com/api/stock/{TICKER}/etf/portfolio"
try:
    r = requests.get(url1, headers={
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)',
        'Referer': f'https://m.stock.naver.com/domestic/stock/{TICKER}/etf'
    }, timeout=10)
    st.write(f"**Status:** {r.status_code}")
    st.write(f"**Content-Type:** {r.headers.get('Content-Type', '')}")
    st.write(f"**Response 길이:** {len(r.text)} bytes")
    if r.status_code == 200:
        try:
            data = r.json()
            st.write(f"**JSON type:** {type(data).__name__}")
            if isinstance(data, dict):
                st.write(f"**최상위 keys:** {list(data.keys())}")
                # 중첩 구조 탐색
                for k, v in data.items():
                    st.write(f"  `{k}` → {type(v).__name__}"
                             + (f", len={len(v)}" if hasattr(v, '__len__') else "")
                             + (f", 첫 항목={v[0] if isinstance(v,list) and v else ''}" if isinstance(v, list) else ""))
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        st.write(f"    첫 항목 keys: {list(v[0].keys())}")
                        st.write(f"    첫 항목 값: {v[0]}")
                    elif isinstance(v, dict):
                        st.write(f"    내부 keys: {list(v.keys())[:10]}")
            elif isinstance(data, list):
                st.write(f"**리스트 길이:** {len(data)}")
                if data:
                    st.write(f"**첫 항목:** {data[0]}")
        except Exception as e:
            st.write(f"JSON 파싱 실패: {e}")
            st.write(f"Raw 응답 앞 500자:")
            st.code(r.text[:500])
    else:
        st.write(f"Raw: {r.text[:300]}")
except Exception as e:
    st.error(f"❌ 요청 실패: {e}")

# ── 테스트 2: finance.naver.com etfinfo 페이지 ──────────────────────────
st.header("테스트 2: etfinfo.naver 페이지 (CU당 구성종목)")
url2 = f"https://finance.naver.com/item/etfinfo.naver?code={TICKER}"
try:
    r2 = requests.get(url2, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    st.write(f"**Status:** {r2.status_code}")
    st.write(f"**Content-Type:** {r2.headers.get('Content-Type', '')}")
    st.write(f"**Response 길이:** {len(r2.content)} bytes")
    if r2.status_code == 200:
        # EUC-KR 디코딩
        try:
            html = r2.content.decode('euc-kr')
        except Exception:
            html = r2.text
        
        st.write(f"**디코딩 후 길이:** {len(html)}")
        
        # 주요 키워드 존재 여부
        keywords = ['구성종목', 'CU', '비중', '종목명', 'portfolio', 'holdingRatio',
                    'weightedValue', '비율', 'table', '<table', 'td']
        for kw in keywords:
            found = kw in html
            st.write(f"  `{kw}` in html: {'✅' if found else '❌'}")
        
        # <table> 태그 수
        table_count = html.count('<table')
        st.write(f"**<table> 태그 수:** {table_count}")
        
        # 구성종목 주변 텍스트
        idx = html.find('구성종목')
        if idx >= 0:
            st.write(f"**'구성종목' 주변 300자:**")
            st.code(html[max(0,idx-50):idx+300].replace('\t','  '))
        
        # pd.read_html 시도
        try:
            import pandas as pd
            tables = pd.read_html(io.StringIO(html))
            st.write(f"**pd.read_html 테이블 수:** {len(tables)}")
            for i, t in enumerate(tables):
                st.write(f"  테이블[{i}]: shape={t.shape}, cols={t.columns.tolist()[:6]}")
                if any('종목' in str(c) or '비중' in str(c) or '%' in str(c)
                       for c in t.columns):
                    st.write("  ⬆️ **구성종목 테이블 후보!**")
                    st.dataframe(t.head(5))
        except Exception as e:
            st.write(f"pd.read_html 실패: {e}")
        
        # HTML 앞부분 출력 (JS AJAX 호출 여부 확인)
        st.write("**HTML 앞 2000자 (AJAX 호출 URL 확인):**")
        st.code(html[:2000])
except Exception as e:
    st.error(f"❌ 요청 실패: {e}")

# ── 테스트 3: finance.naver.com 데이터 API 추가 탐색 ────────────────────
st.header("테스트 3: 네이버 금융 추가 API 탐색")
api_candidates = [
    f"https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0&targetColumn=market_sum&sortOrder=desc",
    f"https://finance.naver.com/item/etfinfo.naver?code={TICKER}&targetColumn=etf_detail",
    f"https://api.stock.naver.com/etf/{TICKER}/portfolio",
]
for u in api_candidates:
    try:
        r3 = requests.get(u, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        st.write(f"**{u[:70]}**")
        st.write(f"  Status: {r3.status_code}, Length: {len(r3.text)}")
        if r3.status_code == 200:
            try:
                d = r3.json()
                if isinstance(d, dict):
                    st.write(f"  Keys: {list(d.keys())[:10]}")
                    items = d.get('result', {}).get('etfItemList', [])
                    if items:
                        st.write(f"  항목 수: {len(items)}, 첫 항목 keys: {list(items[0].keys())}")
            except Exception:
                st.write(f"  Raw: {r3.text[:200]}")
    except Exception as e:
        st.error(f"  ❌ {e}")

st.info("""
**결과 해석:**
- 테스트 1이 200이고 JSON 구조가 보이면 → 키 매핑 문제
- 테스트 1이 403/차단이면 → 네이버 모바일 API가 미국 IP 차단
- 테스트 2에서 구성종목 키워드가 없으면 → JS 동적 로드 (requests로 접근 불가)
- 테스트 2에서 키워드는 있는데 pd.read_html이 실패하면 → 테이블 구조 문제
""")
