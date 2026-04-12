"""
ETF 구성종목 URL 탐색 진단 v2
-------------------------------
이전 진단에서 m.stock.naver.com/etfinfo.naver 가 404임이 확인됨
→ 현재 동작하는 올바른 URL을 탐색
"""
import streamlit as st
import requests
import json
import io
import re
import pandas as pd

st.set_page_config(page_title="ETF URL 탐색", layout="wide")
st.title("🔍 ETF 구성종목 URL 탐색 진단 v2")
st.info("이전 진단 결과: m.stock.naver.com(404), etfinfo.naver(404), KRX(403) → 올바른 URL 탐색")

TICKER = st.text_input("티커", value="069500")

HDR_PC     = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
HDR_MOBILE = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15'}

def probe(label, url, method='GET', data=None, headers=None, decode='utf-8'):
    """URL 하나를 테스트하고 결과를 st에 출력"""
    hdr = headers or HDR_PC
    try:
        if method == 'POST':
            r = requests.post(url, data=data, headers=hdr, timeout=8)
        else:
            r = requests.get(url, headers=hdr, timeout=8)
        
        color = "🟢" if r.status_code == 200 else ("🟡" if r.status_code in (301,302) else "🔴")
        st.write(f"{color} **{label}** — Status `{r.status_code}` | {len(r.content)} bytes")
        
        if r.status_code == 200:
            try:
                body = r.content.decode(decode, errors='ignore')
            except Exception:
                body = r.text
            
            # JSON 여부
            ct = r.headers.get('Content-Type', '')
            if 'json' in ct or body.lstrip().startswith('{') or body.lstrip().startswith('['):
                try:
                    d = json.loads(body)
                    if isinstance(d, dict):
                        st.write(f"  JSON keys: `{list(d.keys())[:10]}`")
                        # 구성종목 관련 키 탐색
                        for k, v in d.items():
                            if isinstance(v, list) and v:
                                st.write(f"  `{k}` → list({len(v)}) | 첫항목: `{v[0]}`")
                            elif isinstance(v, dict):
                                st.write(f"  `{k}` → dict keys: `{list(v.keys())[:8]}`")
                    elif isinstance(d, list):
                        st.write(f"  JSON list len={len(d)}")
                        if d: st.write(f"  첫항목: `{d[0]}`")
                    return True, body
                except Exception:
                    pass
            
            # HTML → 구성종목 키워드 체크
            keywords = ['구성종목', '비중', '종목명', 'portfolio', 'holding', 'weight']
            found = [kw for kw in keywords if kw.lower() in body.lower()]
            if found:
                st.write(f"  ✅ HTML 키워드 발견: `{found}`")
            
            st.write(f"  앞 200자: `{body[:200]}`")
            return True, body
        
        elif r.status_code in (301, 302):
            st.write(f"  → 리다이렉트: `{r.headers.get('Location','')}`")
        return False, ''
    except Exception as e:
        st.write(f"🔴 **{label}** — 연결 실패: `{str(e)[:80]}`")
        return False, ''

if st.button("▶ URL 탐색 실행", type="primary"):

    # ── 그룹 1: 네이버 모바일 API 후보 ─────────────────────────────────────
    st.header("① 네이버 모바일 API 후보들")
    mobile_urls = [
        ("mobile /etf/portfolio",      f"https://m.stock.naver.com/api/stock/{TICKER}/etf/portfolio"),
        ("mobile /etf (루트)",          f"https://m.stock.naver.com/api/stock/{TICKER}/etf"),
        ("mobile /etf/component",       f"https://m.stock.naver.com/api/stock/{TICKER}/etf/component"),
        ("mobile /etf/holding",         f"https://m.stock.naver.com/api/stock/{TICKER}/etf/holding"),
        ("mobile /etf/composition",     f"https://m.stock.naver.com/api/stock/{TICKER}/etf/composition"),
        ("mobile stock basic",          f"https://m.stock.naver.com/api/stock/{TICKER}/basic"),
        ("mobile stock detail",         f"https://m.stock.naver.com/api/stock/{TICKER}/detail"),
        ("mobile domestic etf 탭",      f"https://m.stock.naver.com/domestic/stock/{TICKER}/etf"),
    ]
    found_mobile = None
    for label, url in mobile_urls:
        ok, body = probe(label, url, headers=HDR_MOBILE)
        if ok and not found_mobile:
            found_mobile = (label, url, body)

    # ── 그룹 2: 네이버 금융 PC API 후보 ────────────────────────────────────
    st.header("② 네이버 금융 PC 페이지/API 후보")
    pc_urls = [
        ("main.naver",              f"https://finance.naver.com/item/main.naver?code={TICKER}"),
        ("etfinfo.naver (구URL)",   f"https://finance.naver.com/item/etfinfo.naver?code={TICKER}"),
        ("coinfo.naver",            f"https://finance.naver.com/item/coinfo.naver?code={TICKER}"),
        ("new etf detail API",      f"https://finance.naver.com/api/etf/{TICKER}/detail"),
        ("etfDetail.naver",         f"https://finance.naver.com/fund/etfDetail.naver?etfItemId={TICKER}"),
        ("api/etf portfolio",       f"https://finance.naver.com/api/etf/{TICKER}/portfolio"),
        ("naver new real invest",   f"https://new.real.invest.naver.com/api/etf/{TICKER}/portfolio"),
        ("invest naver etf",        f"https://invest.naver.com/api/etf/{TICKER}/detail"),
    ]
    for label, url in pc_urls:
        ok, body = probe(label, url)
        if ok and 'main.naver' in url:
            # main.naver가 200이면 etfinfo 링크 찾기
            etfinfo_links = re.findall(r'etfinfo[^"\']*', body[:3000])
            if etfinfo_links:
                st.write(f"  📎 etfinfo 링크 발견: `{etfinfo_links[:3]}`")

    # ── 그룹 3: 네이버 증권 신규 API ────────────────────────────────────────
    st.header("③ 네이버 증권 신규 도메인 API")
    new_apis = [
        ("api.stock.naver /etf",        f"https://api.stock.naver.com/etf/{TICKER}/portfolio"),
        ("api.stock.naver /etf detail", f"https://api.stock.naver.com/etf/{TICKER}/detail"),
        ("api.stock.naver /stock etf",  f"https://api.stock.naver.com/stock/{TICKER}/etf"),
        ("polling finance naver",       f"https://polling.finance.naver.com/api/realtime/domestic/stock/{TICKER}"),
        ("navercomp wisereport",        f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={TICKER}"),
        ("comp.fnguide /etf",           f"https://comp.fnguide.com/SVO2/etf/etfDetail.aspx?menuSSYY={TICKER}"),
    ]
    for label, url in new_apis:
        probe(label, url)

    # ── 그룹 4: main.naver 응답에서 실제 etfinfo URL 추출 ───────────────────
    st.header("④ main.naver HTML에서 구성종목 URL 직접 추출")
    try:
        r = requests.get(
            f"https://finance.naver.com/item/main.naver?code={TICKER}",
            headers=HDR_PC, timeout=10
        )
        st.write(f"main.naver status: `{r.status_code}`")
        if r.status_code == 200:
            try:
                html = r.content.decode('euc-kr')
            except Exception:
                html = r.text
            
            st.write(f"HTML 길이: {len(html)}")
            
            # 구성종목 관련 모든 URL/링크 추출
            all_links = re.findall(r'(?:href|src|url)[=:\s]+["\']([^"\']+)["\']', html)
            etf_links = [l for l in all_links if any(kw in l.lower()
                         for kw in ['etf', 'portfolio', 'holding', 'component', 'pdf'])]
            if etf_links:
                st.write("**ETF 관련 링크:**")
                for l in set(etf_links)[:20]:
                    st.code(l)
            
            # 페이지 내 iframe 찾기
            iframes = re.findall(r'<iframe[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if iframes:
                st.write("**iframe src:**")
                for fr in iframes:
                    st.code(fr)
            
            # 구성종목 키워드 주변
            for kw in ['구성종목', 'CU당', 'etfinfo', 'portfolio']:
                idx = html.lower().find(kw.lower())
                if idx >= 0:
                    st.write(f"**'{kw}' 주변:**")
                    st.code(html[max(0,idx-50):idx+300])
                    break
    except Exception as e:
        st.error(f"main.naver 실패: {e}")

    st.divider()
    st.success("결과를 복사해서 전달해주세요. 동작하는 URL을 확인하면 바로 코드 수정이 가능합니다.")
