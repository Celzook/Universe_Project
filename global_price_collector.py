"""
==============================================================================
 글로벌 가격 데이터 수집기 (Global Price Collector)
==============================================================================
 - 한국 ETF: pykrx (유니버스 빌더에서 이미 수집된 df_close 활용)
 - 글로벌 지수 + 미국 ETF: yfinance
 - 3년치 일봉 수집 → 캐시 저장
==============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta
import time, os, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# 글로벌 지수 / 미국 ETF 유니버스
# ============================================================================
GLOBAL_INDICES = {
    # ── 한국 ──
    'KOSPI':     {'ticker': '^KS11',   'country': '한국', 'currency': 'KRW'},
    'KOSDAQ':    {'ticker': '^KQ11',   'country': '한국', 'currency': 'KRW'},

    # ── 미국 ──
    'S&P500':    {'ticker': '^GSPC',   'country': '미국', 'currency': 'USD'},
    'NASDAQ':    {'ticker': '^IXIC',   'country': '미국', 'currency': 'USD'},
    'Dow Jones': {'ticker': '^DJI',    'country': '미국', 'currency': 'USD'},

    # ── 일본 ──
    'Nikkei225': {'ticker': '^N225',   'country': '일본', 'currency': 'JPY'},

    # ── 중국 ──
    'CSI300':    {'ticker': '000300.SS','country': '중국', 'currency': 'CNY'},
    'Hang Seng': {'ticker': '^HSI',    'country': '홍콩', 'currency': 'HKD'},

    # ── 유럽 ──
    'EURO STOXX 50': {'ticker': '^STOXX50E', 'country': '유럽', 'currency': 'EUR'},

    # ── 신흥국 ──
    'Nifty50':   {'ticker': '^NSEI',   'country': '인도', 'currency': 'INR'},
    'VN30':      {'ticker': '^VN30',   'country': '베트남', 'currency': 'VND'},
}

# 주요 미국 상장 ETF (섹터/테마/자산군)
US_ETFS = {
    # 시장 대표
    'SPY':  {'name': 'SPDR S&P 500',         'category': '미국/대형'},
    'QQQ':  {'name': 'Invesco NASDAQ 100',    'category': '미국/기술'},
    'DIA':  {'name': 'SPDR Dow Jones',        'category': '미국/대형'},
    'IWM':  {'name': 'iShares Russell 2000',  'category': '미국/소형'},
    'VTI':  {'name': 'Vanguard Total Market', 'category': '미국/전체'},

    # 섹터
    'XLK':  {'name': 'Technology Select',     'category': '미국/기술'},
    'XLF':  {'name': 'Financial Select',      'category': '미국/금융'},
    'XLE':  {'name': 'Energy Select',         'category': '미국/에너지'},
    'XLV':  {'name': 'Health Care Select',    'category': '미국/헬스케어'},
    'XLI':  {'name': 'Industrial Select',     'category': '미국/산업재'},
    'XLP':  {'name': 'Consumer Staples',      'category': '미국/필수소비재'},
    'XLY':  {'name': 'Consumer Disc.',        'category': '미국/경기소비재'},
    'XLU':  {'name': 'Utilities Select',      'category': '미국/유틸리티'},

    # 테마
    'SOXX': {'name': 'iShares Semiconductor', 'category': '미국/반도체'},
    'ARKK': {'name': 'ARK Innovation',        'category': '미국/혁신'},
    'TAN':  {'name': 'Invesco Solar',         'category': '미국/태양광'},
    'LIT':  {'name': 'Global X Lithium',      'category': '미국/리튬배터리'},

    # 채권/인컴
    'TLT':  {'name': 'iShares 20+ Treasury',  'category': '미국/장기국채'},
    'HYG':  {'name': 'iShares High Yield',    'category': '미국/하이일드'},
    'LQD':  {'name': 'iShares IG Corp',       'category': '미국/투자등급'},

    # 원자재
    'GLD':  {'name': 'SPDR Gold',             'category': '원자재/금'},
    'SLV':  {'name': 'iShares Silver',        'category': '원자재/은'},
    'USO':  {'name': 'United States Oil',     'category': '원자재/원유'},

    # 글로벌
    'EEM':  {'name': 'iShares MSCI EM',       'category': '글로벌/신흥국'},
    'EFA':  {'name': 'iShares MSCI EAFE',     'category': '글로벌/선진국'},
    'VWO':  {'name': 'Vanguard FTSE EM',      'category': '글로벌/신흥국'},

    # 리츠
    'VNQ':  {'name': 'Vanguard Real Estate',  'category': '미국/리츠'},

    # 국가 ETF
    'EWJ':  {'name': 'iShares MSCI Japan',    'category': '국가/일본'},
    'FXI':  {'name': 'iShares China LC',      'category': '국가/중국'},
    'INDA': {'name': 'iShares MSCI India',    'category': '국가/인도'},
    'EWY':  {'name': 'iShares MSCI Korea',    'category': '국가/한국'},
    'VGK':  {'name': 'Vanguard FTSE Europe',  'category': '국가/유럽'},
}


# ============================================================================
# 가격 수집 함수
# ============================================================================
def collect_global_prices(cache_dir="./etf_cache", years=3, progress_callback=None):
    """글로벌 지수 + 미국 ETF 3년치 일봉 수집
    
    Args:
        cache_dir: 캐시 디렉토리
        years: 수집 기간 (년)
        progress_callback: 진행상황 콜백 (current, total, message)
    
    Returns:
        dict: {
            'indices': DataFrame (날짜 × 지수),
            'us_etfs': DataFrame (날짜 × ETF),
            'index_info': dict,
            'us_etf_info': dict,
            'collected_at': datetime
        }
    """
    cache_file = os.path.join(cache_dir, "global_prices.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    # 캐시 확인 (당일 수집분이면 재활용)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('collected_at', datetime.min).date() == datetime.today().date():
                if progress_callback:
                    progress_callback(1, 1, "💾 캐시에서 로드 완료")
                return cached
        except Exception:
            pass

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years + 30)

    total = len(GLOBAL_INDICES) + len(US_ETFS)
    current = 0

    # ── 1) 글로벌 지수 ──
    if progress_callback:
        progress_callback(0, total, "📊 글로벌 지수 수집 중...")

    index_data = {}
    for name, info in GLOBAL_INDICES.items():
        try:
            df = yf.download(info['ticker'], start=start_date, end=end_date,
                           progress=False, auto_adjust=True)
            if not df.empty:
                close = df['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                index_data[name] = close
        except Exception:
            pass
        current += 1
        if progress_callback:
            progress_callback(current, total, f"📊 지수: {name}")

    df_indices = pd.DataFrame(index_data)
    df_indices.index = pd.to_datetime(df_indices.index)
    df_indices = df_indices.sort_index()

    # ── 2) 미국 ETF ──
    if progress_callback:
        progress_callback(current, total, "🇺🇸 미국 ETF 수집 중...")

    # yfinance 배치 다운로드 (한 번에)
    us_tickers = list(US_ETFS.keys())
    try:
        df_us_raw = yf.download(us_tickers, start=start_date, end=end_date,
                                progress=False, auto_adjust=True)
        if isinstance(df_us_raw.columns, pd.MultiIndex):
            df_us = df_us_raw['Close']
        else:
            df_us = df_us_raw[['Close']].rename(columns={'Close': us_tickers[0]})
    except Exception:
        df_us = pd.DataFrame()

    current = total
    if progress_callback:
        progress_callback(current, total, "✅ 글로벌 데이터 수집 완료")

    result = {
        'indices': df_indices,
        'us_etfs': df_us if not df_us.empty else pd.DataFrame(),
        'index_info': GLOBAL_INDICES,
        'us_etf_info': US_ETFS,
        'collected_at': datetime.now()
    }

    # 캐시 저장
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


def calc_period_return(df_prices, start_date, end_date):
    """특정 기간 수익률 계산
    
    Args:
        df_prices: 가격 DataFrame (날짜 × 종목)
        start_date: 시작일 (str or datetime)
        end_date: 종료일 (str or datetime)
    
    Returns:
        Series: 종목별 수익률 (%)
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    mask = (df_prices.index >= start_dt) & (df_prices.index <= end_dt)
    sub = df_prices[mask].dropna(how='all')

    if len(sub) < 2:
        return pd.Series(dtype=float)

    ret = ((sub.iloc[-1] / sub.iloc[0]) - 1) * 100
    return ret.round(2)


def get_combined_prices(kr_close, global_data):
    """한국 ETF 종가 + 글로벌 데이터 합치기 (요약 정보용)"""
    frames = {}

    if kr_close is not None and not kr_close.empty:
        # 한국 ETF는 티커가 인덱스 → 그대로
        frames['kr_etfs'] = kr_close

    if global_data:
        if 'indices' in global_data and not global_data['indices'].empty:
            frames['indices'] = global_data['indices']
        if 'us_etfs' in global_data and not global_data['us_etfs'].empty:
            frames['us_etfs'] = global_data['us_etfs']

    return frames
