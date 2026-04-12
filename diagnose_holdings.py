"""
ETF 구성종목 테이블 구조 진단 v3
----------------------------------
확인된 사실:
  main.naver (141KB) → '구성종목', '비중', '종목명', 'holding', 'weight' 모두 있음
  navercomp.wisereport.co.kr (72KB) → '구성종목', '비중', '종목명' 있음
  → 이 두 페이지의 실제 테이블 구조를 확인해서 파싱 코드 작성
"""
import streamlit as st
import requests
import io
import re
import pandas as pd

st.set_page_config(page_title="테이블 구조 진단 v3", layout="wide")
st.title("🔍 ETF 구성종목 테이블 구조 진단 v3")
st.info("main.naver(141KB, 키워드 전부 있음) 와 wisereport 의 실제 테이블 구조 확인")

TICKER = st.text_input("티커", value="069500")
HDR = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}

if st.button("▶ 진단 실행", type="primary"):

    # ── main.naver 집중 분석 ────────────────────────────────────────────────
    st.header("① finance.naver.com/item/main.naver")
    url = f"https://finance.naver.com/item/main.naver?code={TICKER}"
    try:
        r = requests.get(url, headers=HDR, timeout=15)
        st.write(f"Status: `{r.status_code}` | {len(r.content)} bytes")

        # UTF-8 (meta에 utf-8 명시됨)
        try:
            html = r.content.decode('utf-8')
        except Exception:
            html = r.content.decode('euc-kr', errors='ignore')

        # 1) '구성종목' 키워드 주변 HTML 500자
        st.subheader("1-A. '구성종목' 주변 HTML")
        for kw in ['CU당 구성종목', '구성종목', 'portfolio']:
            idx = html.find(kw)
            if idx >= 0:
                st.write(f"'{kw}' at index {idx}:")
                st.code(html[max(0, idx-200):idx+600])
                break

        # 2) pd.read_html — 전체 테이블 목록
        st.subheader("1-B. pd.read_html 결과")
        try:
            tables = pd.read_html(io.StringIO(html))
            st.write(f"총 {len(tables)}개 테이블 발견")
            for i, t in enumerate(tables):
                col_strs = [str(c) for c in t.columns]
                is_candidate = any(
                    any(kw in c for kw in ['종목', '비중', '%', '편입', 'weight', 'holding'])
                    for c in col_strs
                )
                tag = "⭐ **구성종목 후보**" if is_candidate else ""
                st.write(f"[{i}] shape={t.shape} | cols=`{col_strs[:6]}` {tag}")
                if is_candidate:
                    st.dataframe(t.head(15))
        except Exception as e:
            st.error(f"pd.read_html 실패: {e}")

        # 3) iframe 탐색
        st.subheader("1-C. iframe/AJAX URL")
        iframes = re.findall(r'<iframe[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
        for fr in iframes:
            st.code(fr)
        # JS 안의 API URL
        api_urls = re.findall(
            r'["\']/((?:api|fund|item)[^"\'?#]{3,60})["\']', html
        )
        if api_urls:
            st.write("JS 내부 API 경로:")
            for u in list(dict.fromkeys(api_urls))[:20]:
                st.code("https://finance.naver.com/" + u)

        # 4) '비중' 주변 HTML
        st.subheader("1-D. '비중' 키워드 주변 HTML (첫 발견)")
        idx2 = html.find('비중')
        if idx2 >= 0:
            st.code(html[max(0, idx2-300):idx2+500])

        # 5) 숫자 비중(%) 패턴 탐색
        st.subheader("1-E. 종목명+비중 패턴 정규식 테스트")
        # 패턴 1: <td>종목명</td><td>비중</td>
        hits1 = re.findall(
            r'<td[^>]*>\s*([가-힣A-Za-z][가-힣A-Za-z0-9\s&().]{1,25}?)\s*</td>\s*<td[^>]*>\s*([\d]{1,3}\.[\d]{1,4})\s*</td>',
            html, re.DOTALL
        )
        if hits1:
            st.write(f"패턴1(td-td): {len(hits1)}개 → 앞 10개:")
            for name, w in hits1[:10]:
                st.write(f"  `{name.strip()}` → `{w}`")
        else:
            st.write("패턴1: 없음")

    except Exception as e:
        st.error(f"main.naver 실패: {e}")

    st.divider()

    # ── navercomp.wisereport 분석 ───────────────────────────────────────────
    st.header("② navercomp.wisereport.co.kr")
    url2 = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={TICKER}"
    try:
        r2 = requests.get(url2, headers=HDR, timeout=15)
        st.write(f"Status: `{r2.status_code}` | {len(r2.content)} bytes")

        try:
            html2 = r2.content.decode('utf-8')
        except Exception:
            html2 = r2.content.decode('euc-kr', errors='ignore')

        # '구성종목' 주변
        st.subheader("2-A. '구성종목' 주변 HTML")
        idx3 = html2.find('구성종목')
        if idx3 >= 0:
            st.code(html2[max(0, idx3-100):idx3+600])
        else:
            st.write("'구성종목' 없음")

        # pd.read_html
        st.subheader("2-B. pd.read_html 결과")
        try:
            tables2 = pd.read_html(io.StringIO(html2))
            st.write(f"총 {len(tables2)}개 테이블")
            for i, t in enumerate(tables2):
                col_strs = [str(c) for c in t.columns]
                is_candidate = any(
                    any(kw in c for kw in ['종목', '비중', '%', '편입'])
                    for c in col_strs
                )
                tag = "⭐ **구성종목 후보**" if is_candidate else ""
                st.write(f"[{i}] shape={t.shape} | cols=`{col_strs[:6]}` {tag}")
                if is_candidate:
                    st.dataframe(t.head(15))
        except Exception as e:
            st.error(f"pd.read_html 실패: {e}")

    except Exception as e:
        st.error(f"wisereport 실패: {e}")

    st.divider()

    # ── coinfo.naver 분석 ───────────────────────────────────────────────────
    st.header("③ coinfo.naver (보조)")
    url3 = f"https://finance.naver.com/item/coinfo.naver?code={TICKER}"
    try:
        r3 = requests.get(url3, headers=HDR, timeout=10)
        st.write(f"Status: `{r3.status_code}` | {len(r3.content)} bytes")

        try:
            html3 = r3.content.decode('utf-8')
        except Exception:
            html3 = r3.content.decode('euc-kr', errors='ignore')

        # iframe 탐색 (coinfo는 iframe 기반일 수 있음)
        iframes3 = re.findall(r'<iframe[^>]+src=["\']([^"\']+)["\']', html3, re.IGNORECASE)
        if iframes3:
            st.write("iframe 발견:")
            for fr in iframes3:
                st.code(fr)
            # iframe URL 중 etf/portfolio 관련 호출
            for fr in iframes3:
                if any(kw in fr.lower() for kw in ['etf', 'portfolio', 'hold', 'fund']):
                    st.write(f"⭐ ETF 관련 iframe: `{fr}`")
                    try:
                        rf = requests.get(
                            fr if fr.startswith('http') else f"https://finance.naver.com{fr}",
                            headers=HDR, timeout=8
                        )
                        st.write(f"  iframe 내용 status: {rf.status_code}, {len(rf.content)} bytes")
                        iframe_html = rf.content.decode('utf-8', errors='ignore')
                        idx_iframe = iframe_html.find('구성종목')
                        if idx_iframe >= 0:
                            st.code(iframe_html[max(0,idx_iframe-50):idx_iframe+400])
                    except Exception as e2:
                        st.write(f"  iframe 요청 실패: {e2}")
        else:
            st.write("iframe 없음")

    except Exception as e:
        st.error(f"coinfo 실패: {e}")

    st.success("이 결과를 공유해주시면 바로 파싱 코드를 작성할 수 있습니다!")
