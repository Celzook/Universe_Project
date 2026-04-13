"""
해외 ETF 구성종목 수집 진단 v1
--------------------------------
국내 ETF는 잘 되는데 해외 ETF(TIGER 미국나스닥100 등)가 안 될 때 실행
Streamlit Cloud 배포 후 실행: Main file → diagnose_overseas_etf.py
"""
import streamlit as st
import requests
import re
import io

st.set_page_config(page_title="해외 ETF PDF 진단", layout="wide")
st.title("🔍 해외 ETF 구성종목 HTML 구조 진단")

TICKERS = {
    "133690": "TIGER 미국나스닥100",
    "449450": "1Q 미국우주항공테크",
    "381170": "TIGER 미국S&P500",
    "069500": "KODEX 200 (국내 비교용)",
}

selected = st.selectbox(
    "진단할 ETF",
    options=list(TICKERS.keys()),
    format_func=lambda t: f"{t} {TICKERS[t]}"
)

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

if st.button("▶ 진단 실행", type="primary"):

    # ── main.naver ──────────────────────────────────────────────────────────
    st.header(f"① main.naver — {selected} {TICKERS[selected]}")
    url = f"https://finance.naver.com/item/main.naver?code={selected}"
    try:
        r = requests.get(url, headers=HDR, timeout=15)
        st.write(f"Status: `{r.status_code}` | {len(r.content)} bytes")
        try:
            html = r.content.decode("utf-8")
            st.write("인코딩: UTF-8")
        except Exception:
            html = r.content.decode("euc-kr", errors="ignore")
            st.write("인코딩: EUC-KR")

        # class="ctg" / class="per" 개수
        ctg = html.count('class="ctg"')
        per = html.count('class="per"')
        st.write(f'`class="ctg"` 개수: **{ctg}** | `class="per"` 개수: **{per}**')

        # 구성종목 섹션
        section_m = re.search(r'구성종목\(구성자산\)(.*?)</tbody>', html, re.DOTALL)
        if section_m:
            section = section_m.group(1)
            st.success(f"구성종목 섹션 발견 ({len(section)} chars)")

            st.subheader("섹션 원문 (앞 1200자)")
            st.code(section[:1200])

            # 이름+비중 패턴 (현재 코드)
            row_full = (
                r'<td\s+class="ctg">'
                r'(?:(?!</td>).)*?'
                r'<a[^>]*>([^<]+)</a>'
                r'(?:(?!</td>).)*?</td>'
                r'(?:(?!</tr>).)*?'
                r'<td\s+class="per">\s*([\d.]+)'
            )
            full_hits = re.findall(row_full, section, re.DOTALL)
            st.write(f"**이름+비중 패턴 매칭:** {len(full_hits)}개")
            if full_hits:
                for n, w in full_hits[:5]:
                    st.write(f"  `{n.strip()}` → `{w}`")

            # 이름만 패턴 (fallback)
            row_name = r'<td\s+class="ctg">(?:(?!</td>).)*?<a[^>]*>([^<]+)</a>'
            name_hits = re.findall(row_name, section, re.DOTALL)
            st.write(f"**이름만 패턴 매칭:** {len(name_hits)}개")
            if name_hits:
                for n in name_hits[:10]:
                    st.write(f"  `{n.strip()}`")

            # class="per" 실제 내용
            per_contents = re.findall(r'<td\s+class="per">(.*?)</td>', section, re.DOTALL)
            st.write(f"**class=\"per\" 셀 내용 ({len(per_contents)}개):**")
            for p in per_contents[:10]:
                st.write(f"  `{repr(p.strip())}`")

        else:
            st.warning("구성종목(구성자산) 섹션 없음")
            for kw in ["구성종목", "portfolio", "holding", "구성자산", "편입종목"]:
                i2 = html.find(kw)
                if i2 >= 0:
                    st.write(f"대체 키워드 `{kw}` at {i2}:")
                    st.code(html[i2:i2+400])
                    break

    except Exception as e:
        st.error(f"실패: {e}")

    st.divider()

    # ── wisereport ──────────────────────────────────────────────────────────
    st.header(f"② wisereport — {selected}")
    url2 = f"https://navercomp.wisereport.co.kr/v2/ETF/index.aspx?cmp_cd={selected}"
    try:
        r2 = requests.get(url2, headers=HDR, timeout=15)
        st.write(f"Status: `{r2.status_code}` | {len(r2.content)} bytes")
        try:
            html2 = r2.content.decode("utf-8")
        except Exception:
            html2 = r2.content.decode("euc-kr", errors="ignore")

        sec2_m = re.search(r'CU당 구성종목(.*?)</table>', html2, re.DOTALL)
        if sec2_m:
            sec2 = sec2_m.group(1)
            st.success(f"CU당 구성종목 섹션 발견 ({len(sec2)} chars)")
            st.code(sec2[:1000])

            # 이름+비중
            rows2 = re.findall(
                r'<td[^>]*>\s*(?:<a[^>]*>)?([가-힣A-Za-z][가-힣A-Za-z0-9&(). ]{1,24}?)(?:</a>)?\s*</td>'
                r'(?:.*?<td[^>]*>\s*([\d]+\.[\d]+)\s*</td>)',
                sec2, re.DOTALL
            )
            st.write(f"**이름+비중:** {len(rows2)}개 → {rows2[:5]}")

            # 이름만
            names2 = re.findall(
                r'<td[^>]*>\s*(?:<a[^>]*>)?([A-Za-z가-힣][A-Za-z가-힣0-9&(). ]{1,24}?)(?:</a>)?\s*</td>',
                sec2
            )
            st.write(f"**이름만:** {len(names2)}개 → {names2[:10]}")
        else:
            st.warning("CU당 구성종목 섹션 없음")
            for kw in ["구성종목", "portfolio", "holding"]:
                i3 = html2.lower().find(kw)
                if i3 >= 0:
                    st.code(html2[i3:i3+400])
                    break

    except Exception as e:
        st.error(f"실패: {e}")

    st.divider()
    st.info("""
**결과 해석:**
- ① 이름+비중=0개, 이름만>0개 → **케이스 A** (비중 없는 해외 ETF) — 코드 수정으로 해결 가능
- ① 이름+비중=0개, 이름만=0개 → **케이스 B** (구조 다름) — wisereport 확인 필요
- ② wisereport 이름+비중>0 → 코드 수정 없이 wisereport만으로 해결 가능
- ② wisereport도 없음 → **케이스 C** (수집 불가)
    """)
