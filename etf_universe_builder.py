"""
==============================================================================
 한국 상장 ETF Managed Portfolio 유니버스 빌더 v5.2
==============================================================================
 [새로운 워크플로우 — 가벼운 필터 먼저, 무거운 수집은 나중에]
  Step 1: 전체 ETF 티커 + 이름 수집 (가벼움)
  Step 2: 유형 필터링 — 키워드 기반 (가벼움)
  Step 3: 시가총액 데이터 수집 → 100억 미만 제외 (중간)
  Step 4: 최종 리스트에 대해 가격/상장일/PDF 수집 (무거움)
  Step 5: 수익률/BM/순위 계산 + 엑셀 저장

 pip install pykrx pandas openpyxl tqdm
==============================================================================
"""

import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from tqdm import tqdm
import time, warnings, os, re, pickle, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError

warnings.filterwarnings("ignore")


# ============================================================================
# 설정
# ============================================================================
class Config:
    BASE_DATE = None
    MIN_MARKET_CAP_BILLIONS = 200      # 클라우드 배포 시 메모리 절약
    PRICE_HISTORY_DAYS = 365
    API_DELAY = 0.05
    MAX_WORKERS = 8                    # 클라우드 안정성
    USE_CACHE = True
    CACHE_DIR = "./etf_cache"
    OUTPUT_DIR = "./etf_universe_output"
    TOP_N_HOLDINGS = 10

    EXCLUDE_KEYWORDS = [
        '채권', '국고채', '통안채', '국채', '회사채', '하이일드', '크레딧',
        '중기채', '장기채', '단기채', '초장기', '금리', '선진국채',
        'KIS국채', '국공채', '우량채',
        '특수채', '전단채', '물가채', '공사채', '은행채', '종금채',
        '카드채', '캐피탈채', '지방채', '도시채',
        '머니마켓', 'CD금리', 'KOFR', '단기자금', '예금',
        '파킹', '단기채권', 'KCD',
        'TDF', 'TRF',
        '혼합', '자산배분',
        '커버드콜', 'COVERED CALL', 'COVERED',
        '레버리지', '인버스', '2X', '곱버스',
    ]


# ============================================================================
# 유틸리티
# ============================================================================
def find_latest_business_date(max_lookback=30):
    """최근 영업일 찾기
    - KST(한국시간) 기준으로 계산 (Streamlit Cloud는 UTC)
    - 주말 자동 건너뛰기
    - 장 마감 전이면 전 영업일 사용
    - 공휴일 대비 최대 30일 뒤로
    """
    # UTC → KST (UTC+9)
    try:
        from zoneinfo import ZoneInfo
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        # Python 3.8 이하 또는 zoneinfo 없는 환경
        now_kst = datetime.utcnow() + timedelta(hours=9)

    today_kst = now_kst.date()
    hour_kst = now_kst.hour

    print(f"  🕐 현재 KST: {now_kst.strftime('%Y-%m-%d %H:%M')}")

    # 장 마감(15:30) 전이면 오늘 데이터 없을 수 있음 → 전일부터 탐색
    start_offset = 0 if hour_kst >= 18 else 1  # 18시 이후면 당일 데이터 확보

    for i in range(start_offset, max_lookback):
        d = today_kst - timedelta(days=i)

        # 주말 건너뛰기 (토=5, 일=6)
        if d.weekday() >= 5:
            continue

        ds = d.strftime("%Y%m%d")
        try:
            tickers = stock.get_etf_ticker_list(ds)
            if tickers is not None and len(tickers) > 0:
                print(f"  ✅ 최근 영업일: {ds}")
                return ds
        except Exception as e:
            print(f"  ⚠️ {ds} 조회 실패: {e}")
            time.sleep(0.5)  # API 부하 방지
            continue

    # 최후 수단: 주말 무시하고 단순 뒤로
    fallback = today_kst - timedelta(days=3)
    while fallback.weekday() >= 5:
        fallback -= timedelta(days=1)
    ds = fallback.strftime("%Y%m%d")
    print(f"  ⚠️ fallback 영업일: {ds}")
    return ds


def _timer(label):
    """Step 타이머 (컨텍스트 매니저)"""
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, *a):
            elapsed = time.time() - self.t0
            print(f"  ⏱️ {label}: {elapsed:.1f}초")
    return Timer()


def _load_cache(name):
    path = os.path.join(Config.CACHE_DIR, name)
    if Config.USE_CACHE and os.path.exists(path):
        try:
            with open(path, 'rb') as f: return pickle.load(f)
        except Exception: pass
    return None

def _save_cache(name, data):
    if Config.USE_CACHE:
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        with open(os.path.join(Config.CACHE_DIR, name), 'wb') as f:
            pickle.dump(data, f)


# ============================================================================
# Step 1: 전체 ETF 티커 + 이름 (가벼움)
# ============================================================================
def step1_get_tickers_and_names(base_date):
    print("\n" + "="*60)
    print(" Step 1: 전체 ETF 티커 + 이름 수집")
    print("="*60)

    with _timer("Step 1"):
        tickers = stock.get_etf_ticker_list(base_date)
        print(f"  → 전체 ETF: {len(tickers)}개")

        # 캐시 확인
        cache_name = f"names_{base_date}.pkl"
        cached = _load_cache(cache_name)
        if cached and len(cached) >= len(tickers) * 0.9:
            print(f"  → 💾 이름 캐시 로드: {len(cached)}개")
            etf_names = cached
        else:
            # 멀티스레드로 이름 수집
            etf_names = {}
            def fetch_name(t):
                try: return t, stock.get_etf_ticker_name(t)
                except Exception: return t, "N/A"

            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as exe:
                futs = {exe.submit(fetch_name, t): t for t in tickers}
                with tqdm(total=len(tickers), desc="  이름 조회") as pbar:
                    for f in as_completed(futs):
                        t, name = f.result()
                        etf_names[t] = name
                        pbar.update(1)

            _save_cache(cache_name, etf_names)

        df = pd.DataFrame({'티커': tickers,
                            'ETF명': [etf_names.get(t, 'N/A') for t in tickers]})
        df = df.set_index('티커')
        print(f"  ✅ {len(df)}개 ETF 이름 수집 완료")
    return df


# ============================================================================
# Step 2: 유형 필터링 — 키워드 기반 (가벼움) + 카테고리 분류
# ============================================================================
def step2_type_filter_and_classify(df):
    print("\n" + "="*60)
    print(" Step 2: 유형 필터링 + 카테고리 분류")
    print("="*60)

    t0 = time.time()
    before = len(df)

    def should_exclude(name):
        if pd.isna(name): return True
        s = str(name)
        for kw in Config.EXCLUDE_KEYWORDS:
            if kw.upper() in s.upper(): return True
        if re.search(r'[가-힣]{1,3}채(?![권])', s): return True
        if re.search(r'TRF\d{4}', s.upper()): return True
        return False

    mask = df['ETF명'].apply(should_exclude)
    excluded = df[mask]
    df = df[~mask].copy()

    print(f"  → 제외: {len(excluded)}개 (채권/머니마켓/커버드콜/레버리지/인버스/TDF 등)")
    if len(excluded) > 0:
        for idx, row in excluded.head(10).iterrows():
            print(f"    - {idx} {row['ETF명']}")
        if len(excluded) > 10:
            print(f"    ... 외 {len(excluded)-10}개")

    # 카테고리 분류
    df = _classify(df)

    print(f"\n  → {before}개 → {len(df)}개")
    print(f"  ⏱️ Step 2: {time.time()-t0:.1f}초")
    return df


# ============================================================================
# Step 3: 시가총액 수집 → 100억 미만 제외
# ============================================================================
def step3_market_cap_filter(df, base_date, min_cap=100):
    print("\n" + "="*60)
    print(f" Step 3: 시가총액 수집 + {min_cap}억 이상 필터")
    print("="*60)

    t0 = time.time()
    before = len(df)

    # 캐시 확인 (v2 — 이전 캐시 무시)
    cache_name = f"mktcap_v2_{base_date}.pkl"
    cached = _load_cache(cache_name)
    if cached is not None and '시가총액(억원)' in cached.columns:
        print(f"  → 💾 시총 캐시 로드: {len(cached)}개")
        df = df.join(cached[['시가총액(억원)', 'NAV(억원)']].dropna(how='all'), how='left')
        df = df[df['시가총액(억원)'].notna() & (df['시가총액(억원)'] >= min_cap)].copy()
        df['시가총액(억원)'] = df['시가총액(억원)'].astype(int)
        print(f"  → {before}개 → {len(df)}개")
        print(f"  ⏱️ Step 3: {time.time()-t0:.1f}초")
        return df

    # ── 시가총액 수집 ──
    cap_series = pd.Series(dtype=float, name='시가총액')
    nav_series = pd.Series(dtype=float, name='NAV')

    # 방법 1: get_etf_price_by_ticker (ETF 전용)
    print("  → [1차] get_etf_price_by_ticker...")
    try:
        df_etf = stock.get_etf_price_by_ticker(base_date)
        print(f"    컬럼: {df_etf.columns.tolist()}")
        print(f"    행 수: {len(df_etf)}, 샘플 인덱스: {df_etf.index[:3].tolist()}")

        # 시가총액 컬럼 찾기 (다양한 이름)
        for c in df_etf.columns:
            cl = str(c).replace(' ','')
            if '시가총액' in cl or '시총' in cl:
                cap_series = df_etf[c]; print(f"    ✅ 시총 컬럼: '{c}'"); break
            if '순자산' in cl or 'NAV' in cl.upper():
                nav_series = df_etf[c]; print(f"    ✅ NAV 컬럼: '{c}'")

        # 종가/거래량도 여기서 가져올 수 있음
        for c in df_etf.columns:
            if '종가' in str(c) and '종가' not in df.columns:
                df = df.join(df_etf[[c]], how='left')
            if '거래량' in str(c) and '거래량' not in df.columns:
                df = df.join(df_etf[[c]], how='left')
    except Exception as e:
        print(f"    ⚠️ 실패: {e}")

    # 방법 2: get_market_cap_by_ticker (주식 시총 — ETF도 포함될 수 있음)
    if cap_series.empty or cap_series.isna().all():
        print("  → [2차] get_market_cap_by_ticker...")
        try:
            df_mc = stock.get_market_cap_by_ticker(base_date)
            print(f"    컬럼: {df_mc.columns.tolist()}, 행: {len(df_mc)}")
            # ETF 인덱스와 겹치는지 확인
            overlap = set(df.index) & set(df_mc.index)
            print(f"    ETF와 겹치는 티커: {len(overlap)}개")
            if len(overlap) > 0 and '시가총액' in df_mc.columns:
                cap_series = df_mc['시가총액']
                print(f"    ✅ 시총 확보: {cap_series.notna().sum()}개")
        except Exception as e:
            print(f"    ⚠️ 실패: {e}")

    # 방법 3: 개별 ETF (느리지만 확실)
    if cap_series.empty or cap_series.isna().all():
        print("  → [3차] 개별 ETF 시총 수집...")
        cap_data = {}
        def fetch_cap(ticker):
            try:
                r = stock.get_market_cap_by_date(base_date, base_date, ticker)
                if not r.empty and '시가총액' in r.columns:
                    return ticker, r['시가총액'].iloc[-1]
            except Exception: pass
            return ticker, np.nan

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as exe:
            futs = {exe.submit(fetch_cap, t): t for t in df.index}
            with tqdm(total=len(df), desc="  시총 수집") as pbar:
                for f in as_completed(futs):
                    t, v = f.result()
                    cap_data[t] = v
                    pbar.update(1)
        cap_series = pd.Series(cap_data, name='시가총액')
        ok = cap_series.notna().sum()
        print(f"    ✅ 개별 수집: {ok}/{len(df)}개")

    # ── 시가총액 적용 ──
    if not cap_series.empty and cap_series.notna().any():
        df['_시가총액_raw'] = cap_series
        valid = df['_시가총액_raw'].notna() & (df['_시가총액_raw'] >= min_cap * 1e8)
        df = df[valid].copy()
        df['시가총액(억원)'] = (df['_시가총액_raw'] / 1e8).round(0).astype(int)
        df = df.drop(columns=['_시가총액_raw'], errors='ignore')

        if not nav_series.empty and nav_series.notna().any():
            df['NAV(억원)'] = (nav_series.reindex(df.index) / 1e8).round(2)

        print(f"  → 시가총액 범위: {df['시가총액(억원)'].min():,} ~ {df['시가총액(억원)'].max():,}억원")

        # 캐시 저장
        cache_df = df[['시가총액(억원)']].copy()
        if 'NAV(억원)' in df.columns:
            cache_df['NAV(억원)'] = df['NAV(억원)']
        _save_cache(cache_name, cache_df)
    else:
        print(f"  ⚠️ 시가총액 수집 실패 — 필터 건너뜀")

    print(f"  → {before}개 → {len(df)}개 (시총 {min_cap}억+ 필터)")

    # 기타 카테고리
    etc = df[df['대카테고리'] == '기타']
    if len(etc) > 0:
        print(f"\n  ⚠️ [기타: {len(etc)}개]")
        for idx, row in etc.iterrows():
            print(f"    - {idx} {row['ETF명']}")

    print(f"  ⏱️ Step 3: {time.time()-t0:.1f}초")
    return df


# ============================================================================
# Step 4: 최종 리스트 → 가격 / 상장일 / PDF 수집 (무거운 작업)
# ============================================================================
def step4_collect_all_data(df, base_date):
    print("\n" + "="*60)
    print(f" Step 4: 가격 / 상장일 / 구성종목 수집 ({len(df)}개 ETF)")
    print("="*60)

    t0_total = time.time()
    tickers = df.index.tolist()

    # 4-A: 가격 + KOSPI
    t0 = time.time()
    df, df_close, kospi_close = _collect_prices(df, tickers, base_date)
    print(f"  ⏱️ Step 4-A (가격): {time.time()-t0:.1f}초")

    # 4-B: 설정일
    t0 = time.time()
    df = _collect_listing_dates(df, tickers, base_date)
    print(f"  ⏱️ Step 4-B (설정일): {time.time()-t0:.1f}초")

    # 4-C: PDF → 별도 df_pdf
    t0 = time.time()
    df_pdf = _collect_pdf_holdings(df, tickers, base_date)
    print(f"  ⏱️ Step 4-C (PDF): {time.time()-t0:.1f}초")

    # 4-D: 수익률 / BM / 순위
    t0 = time.time()
    df = _calc_returns(df, df_close, kospi_close, base_date)
    print(f"  ⏱️ Step 4-D (수익률): {time.time()-t0:.1f}초")

    print(f"  ⏱️ Step 4 전체: {time.time()-t0_total:.1f}초")
    return df, df_close, df_pdf


# ──────────────────────────────────────────────────────────
# 4-A: 가격
# ──────────────────────────────────────────────────────────
def _collect_prices(df, tickers, base_date):
    print("\n  ── 4-A: 가격 데이터 ──")

    base_dt = datetime.strptime(base_date, "%Y%m%d")
    start_date = (base_dt - timedelta(days=Config.PRICE_HISTORY_DAYS)).strftime("%Y%m%d")
    ytd_start = base_dt.replace(month=1, day=1).strftime("%Y%m%d")
    if ytd_start < start_date: start_date = ytd_start

    print(f"  → 기간: {start_date} ~ {base_date}")

    # KOSPI
    print("  → KOSPI 수집...")
    try:
        kdf = stock.get_index_ohlcv_by_date(start_date, base_date, "1001")
        kospi = kdf['종가'] if '종가' in kdf.columns else kdf.iloc[:, 3]
        kospi = kospi.sort_index()
        print(f"  → KOSPI: {len(kospi)}일")
    except Exception as e:
        print(f"  ⚠️  KOSPI 실패: {e}")
        kospi = pd.Series(dtype=float)

    # 캐시
    cache_file = os.path.join(Config.CACHE_DIR, f"price_v5_{base_date}.pkl")
    if Config.USE_CACHE and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                df_close = pickle.load(f)['close']
            common = [t for t in tickers if t in df_close.columns]
            if len(common) / max(len(tickers), 1) > 0.9:
                print(f"  → 💾 캐시: {len(common)}개 ETF")
                return df, df_close[common], kospi
        except Exception: pass

    # 방법 A: 날짜 일괄
    print("  → 날짜 기준 일괄 수집...")
    df_close = _fetch_prices_bulk(tickers, start_date, base_date)

    # 방법 B: fallback
    if df_close.empty or df_close.shape[1] < len(tickers) * 0.5:
        print("  → 개별 티커 수집 전환...")
        df_close = _fetch_prices_by_ticker(tickers, start_date, base_date)

    if not df_close.empty:
        print(f"  → 가격: {df_close.shape[0]}일 × {df_close.shape[1]}개 ETF")

    if Config.USE_CACHE and not df_close.empty:
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'close': df_close}, f)

    return df, df_close, kospi


def _fetch_prices_bulk(tickers, start_date, base_date):
    try:
        sample = stock.get_etf_ohlcv_by_date(start_date, base_date, "069500")
        dates = [d.strftime("%Y%m%d") for d in sample.index]
    except Exception: return pd.DataFrame()

    print(f"  → 영업일: {len(dates)}일 / 스레드: {Config.MAX_WORKERS}")

    def fetch(d):
        try:
            r = stock.get_etf_price_by_ticker(d)
            time.sleep(Config.API_DELAY)
            if not r.empty and '종가' in r.columns: return d, r['종가']
            elif not r.empty: return d, r.iloc[:, 0]
        except Exception: pass
        return d, None

    daily = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as exe:
        futs = {exe.submit(fetch, d): d for d in dates}
        with tqdm(total=len(dates), desc="  날짜별 가격") as pbar:
            for f in as_completed(futs):
                d, p = f.result()
                if p is not None: daily[d] = p
                pbar.update(1)
    print(f"  ⏱️ {time.time()-t0:.1f}초")
    if not daily: return pd.DataFrame()

    out = pd.DataFrame(daily).T
    out.index = pd.to_datetime(out.index, format="%Y%m%d")
    out = out.sort_index()
    common = [t for t in tickers if t in out.columns]
    return out[common].apply(pd.to_numeric, errors='coerce')


def _fetch_prices_by_ticker(tickers, start_date, base_date):
    def fetch(t):
        try:
            o = stock.get_etf_ohlcv_by_date(start_date, base_date, t)
            time.sleep(Config.API_DELAY)
            if not o.empty and '종가' in o.columns: return t, o['종가']
            elif not o.empty: return t, o.iloc[:, 3]
        except Exception: pass
        return t, None

    d = {}
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as exe:
        futs = {exe.submit(fetch, t): t for t in tickers}
        with tqdm(total=len(tickers), desc="  티커별 가격") as pbar:
            for f in as_completed(futs):
                t, s = f.result()
                if s is not None: d[t] = s
                pbar.update(1)
    if not d: return pd.DataFrame()
    return pd.DataFrame(d).sort_index().apply(pd.to_numeric, errors='coerce')


# ──────────────────────────────────────────────────────────
# 4-B: 설정일 (1차 네이버 → 2차 pykrx)
# ──────────────────────────────────────────────────────────
def _collect_listing_dates(df, tickers, base_date):
    print("\n  ── 4-B: 설정일 수집 ──")

    cache_file = os.path.join(Config.CACHE_DIR, "listing_dates_v3.pkl")
    cached = {}
    if Config.USE_CACHE and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f: cached = pickle.load(f)
        except Exception: cached = {}

    to_fetch = [t for t in tickers if t not in cached]
    print(f"  → 신규: {len(to_fetch)}개 / 캐시: {len(tickers)-len(to_fetch)}개")

    if to_fetch:
        # 1차: 네이버
        print("  → [1차] 네이버 금융...")
        naver = _naver_listing_dates(to_fetch)
        cached.update(naver)

        # 2차: pykrx fallback
        missing = [t for t in to_fetch if not cached.get(t)]
        if missing:
            print(f"  → [2차] pykrx fallback: {len(missing)}개...")
            pykrx = _pykrx_listing_dates(missing, base_date)
            cached.update(pykrx)

        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f: pickle.dump(cached, f)

    ok = sum(1 for t in tickers if cached.get(t))
    print(f"  → 설정일 완료: {ok}/{len(tickers)}개")
    df['설정일'] = df.index.map(lambda t: cached.get(t, ''))
    return df


def _naver_listing_dates(tickers):
    results = {}
    def fetch(ticker):
        try:
            url = f"https://finance.naver.com/item/main.naver?code={ticker}"
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urlopen(req, timeout=5).read().decode('euc-kr', errors='ignore')
            m = re.search(r'설정일.*?(\d{4}\.\d{2}\.\d{2})', html, re.DOTALL)
            if not m: m = re.search(r'상장일.*?(\d{4}\.\d{2}\.\d{2})', html, re.DOTALL)
            if m: return ticker, m.group(1).replace('.', '-')
        except Exception: pass
        return ticker, ''

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as exe:
        futs = {exe.submit(fetch, t): t for t in tickers}
        with tqdm(total=len(tickers), desc="  네이버 설정일") as pbar:
            for f in as_completed(futs):
                t, d = f.result()
                if d: results[t] = d
                pbar.update(1)
    print(f"  → 네이버 성공: {len(results)}/{len(tickers)}")
    return results


def _pykrx_listing_dates(tickers, base_date):
    results = {}
    def fetch(ticker):
        try:
            o = stock.get_etf_ohlcv_by_date("20020101", base_date, ticker)
            time.sleep(Config.API_DELAY)
            if not o.empty: return ticker, o.index[0].strftime("%Y-%m-%d")
        except Exception: pass
        return ticker, ''

    with ThreadPoolExecutor(max_workers=8) as exe:
        futs = {exe.submit(fetch, t): t for t in tickers}
        with tqdm(total=len(tickers), desc="  pykrx 설정일") as pbar:
            for f in as_completed(futs):
                t, d = f.result()
                if d: results[t] = d
                pbar.update(1)
    return results


# ──────────────────────────────────────────────────────────
# 4-C: PDF 구성종목 → 피벗 매트릭스
# ──────────────────────────────────────────────────────────
def _collect_pdf_holdings(df, tickers, base_date):
    """구성종목 수집 → 피벗 매트릭스 df_pdf 반환
       행: ETF 티커, 열: 종목명(ㄱㄴㄷ순), 값: 보유비중(%)
    """
    print(f"\n  ── 4-C: 구성종목 Top {Config.TOP_N_HOLDINGS} 수집 ──")

    cache_file = os.path.join(Config.CACHE_DIR, f"holdings_v3_{base_date}.pkl")
    cached = {}
    if Config.USE_CACHE and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f: cached = pickle.load(f)
        except Exception: cached = {}

    to_fetch = [t for t in tickers if t not in cached]
    print(f"  → 신규: {len(to_fetch)}개 / 캐시: {len(tickers)-len(to_fetch)}개")

    if to_fetch:
        print("  → pykrx PDF 조회...")
        pykrx_results = _pykrx_holdings(to_fetch, base_date)
        cached.update(pykrx_results)

        missing_count = sum(1 for t in to_fetch if t not in cached or len(cached.get(t, [])) == 0)
        if missing_count:
            print(f"  → 비중 없는 ETF(해외 등): {missing_count}개 → 빈칸 처리")

        if Config.USE_CACHE:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f: pickle.dump(cached, f)

    ok = sum(1 for t in tickers if cached.get(t) and len(cached[t]) > 0)
    print(f"  → 구성종목 완료: {ok}/{len(tickers)}개")

    # ── 피벗 매트릭스 생성 ──
    # cached: {ticker: [(종목명, 비중%), ...]}
    # → 행=ETF, 열=종목명(ㄱㄴㄷ), 값=비중
    all_stocks = set()
    for t in tickers:
        for name, w in cached.get(t, []):
            all_stocks.add(name)

    # ㄱㄴㄷ 정렬
    sorted_stocks = sorted(all_stocks, key=lambda x: x)
    print(f"  → 전체 고유 종목 수: {len(sorted_stocks)}개")

    # ETF명 컬럼 + 종목별 비중 (빈칸은 NaN → 엑셀에서 정렬 가능)
    pdf_data = {}
    for ticker in tickers:
        row = {'ETF명': df.at[ticker, 'ETF명'] if ticker in df.index else ''}
        holdings_dict = {name: w for name, w in cached.get(ticker, [])}
        for s in sorted_stocks:
            row[s] = holdings_dict.get(s, np.nan)
        pdf_data[ticker] = row

    df_pdf = pd.DataFrame.from_dict(pdf_data, orient='index')
    df_pdf.index.name = '티커'

    # 비중이 있는 셀 수 통계
    stock_cols = [c for c in df_pdf.columns if c != 'ETF명']
    filled = df_pdf[stock_cols].notna().sum().sum()
    print(f"  → 매트릭스: {len(df_pdf)} ETF × {len(sorted_stocks)} 종목 ({filled:,.0f}개 셀 채움)")

    return df_pdf


def _pykrx_holdings(tickers, base_date):
    """pykrx PDF: [(종목명, 비중%), ...] 튜플 리스트 반환"""
    results = {}

    # 종목코드 → 종목명 캐시
    stock_name_cache = {}
    def get_stock_name(code):
        if code in stock_name_cache:
            return stock_name_cache[code]
        try:
            name = stock.get_market_ticker_name(code)
            if name:
                stock_name_cache[code] = name
                return name
        except Exception:
            pass
        return code

    def fetch(ticker):
        try:
            pdf = stock.get_etf_portfolio_deposit_file(ticker, base_date)
            time.sleep(Config.API_DELAY)
            if pdf is None or pdf.empty:
                return ticker, []

            # 비중 컬럼 찾기
            weight_col = None
            for c in pdf.columns:
                if '비중' in str(c) or '구성비' in str(c) or 'weight' in str(c).lower():
                    weight_col = c; break

            # 비중 컬럼 없으면 빈 리스트 (해외 ETF 등)
            if not weight_col:
                return ticker, []

            # ETF 티커 목록 (ETF-in-ETF 제외용)
            try:
                etf_tickers_set = set(stock.get_etf_ticker_list(base_date))
            except Exception:
                etf_tickers_set = set()

            items = []
            pdf_sorted = pdf.sort_values(weight_col, ascending=False)
            for idx, row in pdf_sorted.head(Config.TOP_N_HOLDINGS + 5).iterrows():
                if len(items) >= Config.TOP_N_HOLDINGS:
                    break
                w = row[weight_col]
                if not pd.notna(w) or w <= 0:
                    continue
                code = str(idx)

                # 6자리 숫자 코드인 경우
                if code.isdigit() and len(code) == 6:
                    # ETF-in-ETF → 제외
                    if code in etf_tickers_set:
                        continue
                    name = get_stock_name(code)
                    # 종목명 변환 실패(여전히 코드) → 제외
                    if name == code:
                        continue
                else:
                    # 한글이 아닌 알파벳/숫자 코드 → 제외 (해외종목 등)
                    if not re.search(r'[가-힣]', code):
                        continue
                    name = code

                name = name[:20]
                items.append((name, round(float(w), 2)))

            return ticker, items
        except Exception:
            pass
        return ticker, []

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as exe:
        futs = {exe.submit(fetch, t): t for t in tickers}
        with tqdm(total=len(tickers), desc="  pykrx PDF") as pbar:
            for f in as_completed(futs):
                t, items = f.result()
                if items: results[t] = items
                pbar.update(1)

    print(f"  → pykrx 성공: {len(results)}/{len(tickers)}")
    return results


# ──────────────────────────────────────────────────────────
# 4-D: 수익률 / BM / 순위
# ──────────────────────────────────────────────────────────
def _calc_returns(df, df_close, kospi_close, base_date):
    if df_close.empty or len(df_close) < 2:
        return df

    base_dt = datetime.strptime(base_date, "%Y%m%d")
    ytd_dt = base_dt.replace(month=1, day=1)
    ytd_mask = df_close.index >= pd.Timestamp(ytd_dt)
    ytd_loc = df_close.index.get_loc(df_close.index[ytd_mask][0]) if ytd_mask.any() else 0

    n = len(df_close)
    periods = {'1M': min(21, n-1), '3M': min(63, n-1),
               '6M': min(126, n-1), '1Y': n-1}

    for label, lb in periods.items():
        df[f'수익률_{label}(%)'] = ((df_close.iloc[-1] / df_close.iloc[-1-lb] - 1) * 100).round(2)
    df['수익률_YTD(%)'] = ((df_close.iloc[-1] / df_close.iloc[ytd_loc] - 1) * 100).round(2)

    dr = df_close.pct_change().dropna()
    if len(dr) > 20:
        df['연간변동성(%)'] = (dr.std() * np.sqrt(252) * 100).round(2)

    # KOSPI BM
    if not kospi_close.empty and len(kospi_close) > 1:
        kc = kospi_close.sort_index()
        kn = len(kc)
        bm = {}
        for label, lb in periods.items():
            klb = min(lb, kn-1)
            bm[label] = round((kc.iloc[-1] / kc.iloc[-1-klb] - 1) * 100, 2)

        k_ytd = kc.index >= pd.Timestamp(ytd_dt)
        bm['YTD'] = round((kc.iloc[-1] / kc[k_ytd].iloc[0] - 1) * 100, 2) if k_ytd.any() else 0.0

        print(f"\n  📈 KOSPI BM (검증):")
        print(f"    최종: {kc.iloc[-1]:,.0f} ({kc.index[-1].strftime('%Y-%m-%d')})")
        k1m = -1 - min(21, kn-1)
        print(f"    1M 기준: {kc.iloc[k1m]:,.0f} ({kc.index[k1m].strftime('%Y-%m-%d')})")
        for l, r in bm.items(): print(f"    {l}: {r:+.2f}%")

        for label in ['1M', '3M', '6M', '1Y']:
            df[f'BM_{label}(%)'] = (df[f'수익률_{label}(%)'] - bm[label]).round(2)
        df['BM_YTD(%)'] = (df['수익률_YTD(%)'] - bm['YTD']).round(2)

    # 순위
    if 'BM_YTD(%)' in df.columns:
        df['순위(YTD_BM+)'] = df['BM_YTD(%)'].rank(ascending=False, method='min', na_option='bottom').astype(int)

    return df


# ============================================================================
# 카테고리 분류 (변경 없음)
# ============================================================================
def _classify(df):
    def classify(name):
        n = str(name); u = n.upper()

        # 원자재
        if any(kw in u for kw in ['금현물','금선물','골드','GOLD','국제금','금액티브','금ETF']): return '원자재','금',''
        if any(kw in u for kw in ['은현물','은선물','실버','SILVER']): return '원자재','은',''
        if any(kw in u for kw in ['원유','WTI','브렌트','BRENT','오일']): return '원자재','원유',''
        if any(kw in u for kw in ['천연가스']): return '원자재','천연가스',''
        if any(kw in u for kw in ['구리','팔라듐','백금','플래티넘','비철']): return '원자재','비철금속',''
        if any(kw in u for kw in ['곡물','농산물','옥수수','대두','밀']): return '원자재','농산물',''
        if any(kw in u for kw in ['원자재','커머디티','COMMODITY']): return '원자재','원자재(종합)',''

        # 통화/환율
        if any(kw in u for kw in ['달러','USD','달러선물','미국달러']): return '통화/환율','달러',''
        if any(kw in u for kw in ['엔화','엔선물','JPY']): return '통화/환율','엔화',''
        if any(kw in u for kw in ['유로','EUR']): return '통화/환율','유로',''
        if any(kw in u for kw in ['위안','CNY','CNH']): return '통화/환율','위안',''
        if any(kw in u for kw in ['환헤지','통화','외환','FX']): return '통화/환율','통화(기타)',''

        # 리츠/부동산
        if any(kw in u for kw in ['리츠','REITS','REIT','부동산']): return '리츠/부동산','리츠',''

        # 그룹주
        if '그룹' in n:
            grp = n.split('그룹')[0]
            for pfx in ['TIGER ','KODEX ','ACE ','KBSTAR ','SOL ','HANARO ','ARIRANG ','KOSEF ','PLUS ']:
                grp = grp.replace(pfx.strip(),'').strip()
            return '그룹주', grp+'그룹', ''

        # 해외주식
        if any(kw in u for kw in ['미국','나스닥','NASDAQ','S&P','S&P500','다우','필라델피아','FANG','NYSE','미국빅테크','미국테크']):
            return '해외주식','미국',_sub(u,'미국')
        if any(kw in u for kw in ['일본','니케이','NIKKEI','TOPIX','도쿄']): return '해외주식','일본',_sub(u,'일본')
        if any(kw in u for kw in ['중국','차이나','CSI','항셍','HANG SENG','심천','상해','HSCEI','홍콩','CHINEXT','본토','CHINA']):
            return '해외주식','중국',_sub(u,'중국')
        if any(kw in u for kw in ['인도','니프티','NIFTY','INDIA']): return '해외주식','인도',_sub(u,'인도')
        if any(kw in u for kw in ['베트남','VN30','VIETNAM']): return '해외주식','베트남',_sub(u,'베트남')
        if any(kw in u for kw in ['대만','TAIWAN']): return '해외주식','대만',''
        if any(kw in u for kw in ['유럽','EURO STOXX','유로스탁스','STOXX']): return '해외주식','유럽',_sub(u,'유럽')
        if any(kw in u for kw in ['인도네시아']): return '해외주식','인도네시아',''
        if any(kw in u for kw in ['브라질']): return '해외주식','브라질',''
        if any(kw in u for kw in ['멕시코']): return '해외주식','멕시코',''
        if any(kw in u for kw in ['사우디']): return '해외주식','사우디',''
        if any(kw in u for kw in ['선진국','MSCI WORLD','ACWI','글로벌']): return '해외주식','글로벌/선진국',''
        if any(kw in u for kw in ['신흥국','EM','EMERGING']): return '해외주식','신흥국',''

        # 섹터/테마
        if any(kw in u for kw in ['반도체','팹리스']): return '섹터/테마','반도체',''
        if any(kw in u for kw in ['2차전지','배터리','리튬','양극재','음극재']): return '섹터/테마','2차전지/배터리',''
        if any(kw in u for kw in ['AI','인공지능']): return '섹터/테마','AI',''
        if any(kw in u for kw in ['소프트웨어','SW','클라우드','SAAS']): return '섹터/테마','소프트웨어/클라우드',''
        if any(kw in u for kw in ['게임','GAME']): return '섹터/테마','게임',''
        if any(kw in u for kw in ['엔터','K-POP','KPOP','콘텐츠']): return '섹터/테마','엔터/콘텐츠',''
        if any(kw in u for kw in ['미디어','방송','광고']): return '섹터/테마','미디어',''
        if any(kw in u for kw in ['바이오','헬스케어','제약','의료기기','의료','헬스']): return '섹터/테마','바이오/헬스케어',''
        if any(kw in u for kw in ['자동차','전기차','EV','모빌리티','자율주행']): return '섹터/테마','자동차/모빌리티',''
        if any(kw in u for kw in ['로봇','자동화','로보틱스']): return '섹터/테마','로봇/자동화',''
        if any(kw in u for kw in ['은행']): return '섹터/테마','은행',''
        if any(kw in u for kw in ['증권']): return '섹터/테마','증권',''
        if any(kw in u for kw in ['보험']): return '섹터/테마','보험',''
        if any(kw in u for kw in ['금융']): return '섹터/테마','금융(기타)',''
        if any(kw in u for kw in ['건설','인프라','시멘트']): return '섹터/테마','건설/인프라',''
        if any(kw in u for kw in ['조선']): return '섹터/테마','조선',''
        if any(kw in u for kw in ['해운']): return '섹터/테마','해운',''
        if any(kw in u for kw in ['방산','방위','우주항공','항공우주','우주']): return '섹터/테마','방산/우주항공',''
        if any(kw in u for kw in ['화학','소재','신소재']): return '섹터/테마','화학/소재',''
        if any(kw in u for kw in ['철강','금속']): return '섹터/테마','철강/금속',''
        if any(kw in u for kw in ['에너지','석유']): return '섹터/테마','에너지',''
        if any(kw in u for kw in ['유틸리티','전력','발전']): return '섹터/테마','유틸리티/전력',''
        if any(kw in u for kw in ['필수소비재','음식료','식품','F&B']): return '섹터/테마','필수소비재/식품',''
        if any(kw in u for kw in ['경기소비재','럭셔리','화장품','뷰티']): return '섹터/테마','경기소비재/뷰티',''
        if any(kw in u for kw in ['소비재']): return '섹터/테마','소비재(기타)',''
        if any(kw in u for kw in ['통신','5G','6G','텔레콤']): return '섹터/테마','통신',''
        if any(kw in u for kw in ['운송','물류','택배']): return '섹터/테마','운송/물류',''
        if any(kw in u for kw in ['항공']) and '우주' not in u: return '섹터/테마','항공',''
        if any(kw in u for kw in ['ESG','탄소','그린','친환경']): return '섹터/테마','ESG/친환경',''
        if any(kw in u for kw in ['수소','태양광','풍력','신재생','재생에너지']): return '섹터/테마','수소/신재생에너지',''
        if any(kw in u for kw in ['원자력','원전','우라늄','SMR']): return '섹터/테마','원자력',''
        if any(kw in u for kw in ['사이버보안','보안','시큐리티']): return '섹터/테마','사이버보안',''
        if any(kw in u for kw in ['메타버스','XR','VR','AR']): return '섹터/테마','메타버스/XR',''
        if any(kw in u for kw in ['블록체인','디지털자산','비트코인','가상자산','크립토']): return '섹터/테마','블록체인/가상자산',''
        if any(kw in u for kw in ['플랫폼','인터넷','이커머스','커머스']): return '섹터/테마','플랫폼/인터넷',''
        if any(kw in u for kw in ['IT','테크','기술','ICT']): return '섹터/테마','IT/테크',''

        # 배당/인컴
        if any(kw in u for kw in ['배당','고배당','배당성장','DIVIDEND','프리미엄','월배당','분배','인컴']):
            return '배당/인컴','배당',''

        # 시장대표
        if re.search(r'200(?:TR)?$', n.strip().upper()) or 'KOSPI200' in u or 'KOSPI 200' in u: return '시장대표','KOSPI200',''
        if any(kw in u for kw in ['코스닥','KOSDAQ']): return '시장대표','코스닥',''
        if any(kw in u for kw in ['중소형','중소']): return '시장대표','중소형주',''
        if any(kw in u for kw in ['중형','미드캡','MID']): return '시장대표','중소형주',''
        if any(kw in u for kw in ['소형','스몰캡','SMALL']): return '시장대표','소형주',''
        if any(kw in u for kw in ['코스피','KOSPI','KRX300','KRX 300','TOP10','TOP30','TOP 10','대형']):
            return '시장대표','대형주',''

        # 스마트베타
        if any(kw in u for kw in ['모멘텀','MOMENTUM']): return '스마트베타','모멘텀',''
        if any(kw in u for kw in ['밸류','가치','VALUE']): return '스마트베타','밸류',''
        if any(kw in u for kw in ['퀄리티','QUALITY','우량']): return '스마트베타','퀄리티',''
        if any(kw in u for kw in ['로우볼','저변동','LOW VOL']): return '스마트베타','저변동성',''
        if any(kw in u for kw in ['동일가중','EQUAL']): return '스마트베타','동일가중',''
        if any(kw in u for kw in ['성장','GROWTH']): return '스마트베타','성장',''
        if any(kw in u for kw in ['멀티팩터']): return '스마트베타','멀티팩터',''

        return '기타','기타',''

    results = df['ETF명'].apply(classify)
    df['대카테고리'] = results.apply(lambda x: x[0])
    df['중카테고리'] = results.apply(lambda x: x[1])
    df['소카테고리'] = results.apply(lambda x: x[2])

    print("\n  [대카테고리]")
    for c, n in df['대카테고리'].value_counts().items(): print(f"    {c}: {n}개")
    print("\n  [중카테고리 상위 20]")
    for c, n in df['중카테고리'].value_counts().head(20).items(): print(f"    {c}: {n}개")
    return df


def _sub(u, country):
    if country == '미국':
        if any(kw in u for kw in ['S&P','S&P500']): return 'S&P500'
        if any(kw in u for kw in ['나스닥','NASDAQ']): return '나스닥'
        if any(kw in u for kw in ['다우','DOW']): return '다우'
        if any(kw in u for kw in ['반도체','필라델피아']): return '미국반도체'
        if any(kw in u for kw in ['빅테크','테크','FANG','TECH']): return '미국테크'
        if any(kw in u for kw in ['배당','DIVIDEND']): return '미국배당'
        if any(kw in u for kw in ['헬스','바이오','제약']): return '미국헬스케어'
        if any(kw in u for kw in ['금융','은행']): return '미국금융'
        if any(kw in u for kw in ['성장','GROWTH']): return '미국성장'
        if any(kw in u for kw in ['가치','VALUE']): return '미국가치'
        if any(kw in u for kw in ['AI','인공지능']): return '미국AI'
        if any(kw in u for kw in ['방산','우주항공','방위']): return '미국방산'
        if any(kw in u for kw in ['리츠','REITS']): return '미국리츠'
        if any(kw in u for kw in ['원자력','우라늄','SMR']): return '미국원자력'
        return '미국(기타)'
    if country == '일본':
        if any(kw in u for kw in ['니케이','NIKKEI']): return '니케이225'
        if 'TOPIX' in u: return 'TOPIX'
        if '반도체' in u: return '일본반도체'
        return '일본(기타)'
    if country == '중국':
        if any(kw in u for kw in ['CSI300','CSI 300']): return 'CSI300'
        if any(kw in u for kw in ['항셍','HANG SENG','HSCEI']): return '항셍'
        if any(kw in u for kw in ['심천','CHINEXT']): return '심천/차이넥스트'
        if 'CSI' in u: return 'CSI(기타)'
        return '중국(기타)'
    if country == '인도':
        if any(kw in u for kw in ['니프티','NIFTY']): return '니프티50'
        return '인도(기타)'
    if country == '베트남':
        if 'VN30' in u: return 'VN30'
        return '베트남(기타)'
    if country == '유럽':
        if any(kw in u for kw in ['STOXX','유로스탁스']): return 'EURO STOXX'
        return '유럽(기타)'
    return ''


# ============================================================================
# Step 5: 저장
# ============================================================================
def step5_save(df, df_close, df_pdf, base_date):
    print("\n" + "="*60)
    print(" Step 5: 유니버스 저장")
    print("="*60)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    if '시가총액(억원)' in df.columns:
        df = df.sort_values('시가총액(억원)', ascending=False)

    print(f"\n  📊 최종: {len(df)}개 ETF")
    if '시가총액(억원)' in df.columns:
        total = df['시가총액(억원)'].sum()
        print(f"  💰 시총: {total:,.0f}억원 ({total/10000:.1f}조원)")

    if '대카테고리' in df.columns:
        print("\n  [대카테고리]")
        for cat in df['대카테고리'].value_counts().index:
            sub = df[df['대카테고리']==cat]
            cap = sub['시가총액(억원)'].sum() if '시가총액(억원)' in sub.columns else 0
            print(f"    {cat}: {len(sub)}개 ({cap:,.0f}억)")

    # ── df_universe 컬럼 순서 (Top 없음) ──
    cols = ['ETF명', '시가총액(억원)', 'NAV(억원)', '설정일',
            '대카테고리', '중카테고리', '소카테고리',
            '순위(YTD_BM+)',
            '수익률_1M(%)', '수익률_3M(%)', '수익률_6M(%)',
            '수익률_1Y(%)', '수익률_YTD(%)',
            'BM_1M(%)', 'BM_3M(%)', 'BM_6M(%)',
            'BM_1Y(%)', 'BM_YTD(%)',
            '연간변동성(%)', '종가', '거래량']

    cols = [c for c in cols if c in df.columns]
    df_export = df[cols].copy().fillna('')

    # ── 엑셀: 유니버스 + PDF 시트 분리 ──
    f_master = os.path.join(Config.OUTPUT_DIR, f"etf_universe_{base_date}.xlsx")
    with pd.ExcelWriter(f_master, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='유니버스')
        if df_pdf is not None and not df_pdf.empty:
            # df_pdf도 시가총액 순으로 정렬
            pdf_order = [t for t in df.index if t in df_pdf.index]
            df_pdf_sorted = df_pdf.loc[[t for t in pdf_order if t in df_pdf.index]]
            df_pdf_sorted.to_excel(writer, sheet_name='구성종목(PDF)')
    print(f"\n  📁 유니버스 + PDF: {f_master}")

    # ── 종가 CSV ──
    if df_close is not None and not df_close.empty:
        f_p = os.path.join(Config.OUTPUT_DIR, f"etf_prices_{base_date}.csv")
        df_close.to_csv(f_p, encoding='utf-8-sig')
        print(f"  📁 종가: {f_p}")

    # ── 티커 CSV ──
    f_t = os.path.join(Config.OUTPUT_DIR, f"etf_tickers_{base_date}.csv")
    tc = [c for c in ['ETF명','설정일','대카테고리','중카테고리','소카테고리'] if c in df.columns]
    df[tc].to_csv(f_t, encoding='utf-8-sig')
    print(f"  📁 티커: {f_t}")

    # ── Top 15 출력 ──
    print(f"\n  {'─'*110}")
    print(f"  📋 Top 15 (시가총액)")
    print(f"  {'─'*110}")
    for i, (idx, r) in enumerate(df.head(15).iterrows(), 1):
        nm = str(r.get('ETF명',''))[:24]
        cap = f"{r['시가총액(억원)']:>7,}억" if r.get('시가총액(억원)','') != '' else ""
        c1 = str(r.get('대카테고리',''))[:8]
        c2 = str(r.get('중카테고리',''))[:10]
        ytd = f"{r['수익률_YTD(%)']:+.2f}%" if r.get('수익률_YTD(%)','') != '' else "N/A"
        bm = f"{r['BM_YTD(%)']:+.2f}%" if r.get('BM_YTD(%)','') != '' else "N/A"
        rnk = str(r.get('순위(YTD_BM+)',''))
        # PDF top1 가져오기 (피벗 매트릭스에서 최대 비중 종목)
        top1 = ''
        if df_pdf is not None and idx in df_pdf.index:
            row_pdf = df_pdf.loc[idx].drop('ETF명', errors='ignore')
            num_vals = pd.to_numeric(row_pdf, errors='coerce')
            if num_vals.notna().any():
                max_stock = num_vals.idxmax()
                max_w = num_vals.max()
                top1 = f"{max_stock}({max_w:.1f}%)"
        print(f"  {i:>3}. [{idx}] {nm:<26} {cap} {c1:<8} {c2:<12} YTD:{ytd:>8} BM:{bm:>8} #{rnk:<4} {top1}")

    return df_export


# ============================================================================
# 메인
# ============================================================================
def build_universe():
    print("╔" + "═"*58 + "╗")
    print("║   한국 상장 ETF 유니버스 빌더 v5.3                       ║")
    print("╚" + "═"*58 + "╝")

    t_start = time.time()

    base_date = Config.BASE_DATE or find_latest_business_date()
    Config.BASE_DATE = base_date    # 저장
    print(f"\n  📅 기준일: {base_date}")
    print(f"  💰 최소 시총: {Config.MIN_MARKET_CAP_BILLIONS}억원")
    print(f"  ⚡ 스레드: {Config.MAX_WORKERS} / 캐시: {Config.USE_CACHE}")

    # Step 1: 가벼운 — 티커+이름만
    df = step1_get_tickers_and_names(base_date)

    # Step 2: 가벼운 — 유형 필터 (키워드)
    df = step2_type_filter_and_classify(df)

    # Step 3: 중간 — 시총 데이터 → 필터
    df = step3_market_cap_filter(df, base_date, Config.MIN_MARKET_CAP_BILLIONS)

    # Step 4: 무거운 — 최종 리스트에만 가격/상장일/PDF
    df, df_close, df_pdf = step4_collect_all_data(df, base_date)

    # Step 5: 저장
    step5_save(df, df_close, df_pdf, base_date)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f" ✅ 완료! 총 소요시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
    print(f"{'='*60}")
    return df, df_close, df_pdf


if __name__ == "__main__":
    df_universe, df_prices, df_pdf = build_universe()
