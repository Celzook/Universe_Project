"""
==============================================================================
 글로벌 가격 데이터 수집기 v2 (간소화)
==============================================================================
 yfinance 배치 다운로드 1회로 모든 데이터 수집
 - 글로벌 지수: ETF로 대체 (SPY=S&P500, EWJ=일본 등)
 - 미국 상장 ETF: 시장/섹터/테마/채권/원자재/국가
==============================================================================
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os, pickle


# ============================================================================
# 미국 상장 ETF 유니버스 (지수 대용 포함)
# ============================================================================
GLOBAL_INDICES = {
    # 지수 대용 ETF (미국 상장)
    'KOSPI':     {'ticker': 'EWY',       'name': 'iShares MSCI Korea',     'country': '한국'},
    'S&P500':    {'ticker': 'SPY',       'name': 'SPDR S&P 500',          'country': '미국'},
    'NASDAQ':    {'ticker': 'QQQ',       'name': 'Invesco NASDAQ 100',    'country': '미국'},
    'Dow Jones': {'ticker': 'DIA',       'name': 'SPDR Dow Jones',        'country': '미국'},
    'Nikkei':    {'ticker': 'EWJ',       'name': 'iShares MSCI Japan',    'country': '일본'},
    'China':     {'ticker': 'FXI',       'name': 'iShares China LC',      'country': '중국'},
    'Europe':    {'ticker': 'VGK',       'name': 'Vanguard FTSE Europe',  'country': '유럽'},
    'India':     {'ticker': 'INDA',      'name': 'iShares MSCI India',    'country': '인도'},
    'EM':        {'ticker': 'EEM',       'name': 'iShares MSCI EM',       'country': '신흥국'},
}

US_ETFS = {
    # 시장
    'SPY':  {'name': 'SPDR S&P 500',         'category': '시장'},
    'QQQ':  {'name': 'Invesco NASDAQ 100',    'category': '시장'},
    'DIA':  {'name': 'SPDR Dow Jones',        'category': '시장'},
    'IWM':  {'name': 'iShares Russell 2000',  'category': '시장'},
    'VTI':  {'name': 'Vanguard Total Market', 'category': '시장'},

    # 섹터
    'XLK':  {'name': 'Technology Select',     'category': '섹터'},
    'XLF':  {'name': 'Financial Select',      'category': '섹터'},
    'XLE':  {'name': 'Energy Select',         'category': '섹터'},
    'XLV':  {'name': 'Health Care Select',    'category': '섹터'},
    'XLI':  {'name': 'Industrial Select',     'category': '섹터'},
    'XLP':  {'name': 'Consumer Staples',      'category': '섹터'},
    'XLY':  {'name': 'Consumer Disc.',        'category': '섹터'},
    'XLU':  {'name': 'Utilities Select',      'category': '섹터'},

    # 테마
    'SOXX': {'name': 'iShares Semiconductor', 'category': '테마'},
    'ARKK': {'name': 'ARK Innovation',        'category': '테마'},
    'TAN':  {'name': 'Invesco Solar',         'category': '테마'},
    'LIT':  {'name': 'Global X Lithium',      'category': '테마'},

    # 채권
    'TLT':  {'name': 'iShares 20+ Treasury',  'category': '채권'},
    'HYG':  {'name': 'iShares High Yield',    'category': '채권'},
    'LQD':  {'name': 'iShares IG Corp',       'category': '채권'},

    # 원자재
    'GLD':  {'name': 'SPDR Gold',             'category': '원자재'},
    'SLV':  {'name': 'iShares Silver',        'category': '원자재'},
    'USO':  {'name': 'United States Oil',     'category': '원자재'},

    # 국가
    'EWY':  {'name': 'iShares MSCI Korea',    'category': '국가'},
    'EWJ':  {'name': 'iShares MSCI Japan',    'category': '국가'},
    'FXI':  {'name': 'iShares China LC',      'category': '국가'},
    'INDA': {'name': 'iShares MSCI India',    'category': '국가'},
    'VGK':  {'name': 'Vanguard FTSE Europe',  'category': '국가'},
    'EEM':  {'name': 'iShares MSCI EM',       'category': '국가'},
    'EFA':  {'name': 'iShares MSCI EAFE',     'category': '국가'},
    'VWO':  {'name': 'Vanguard FTSE EM',      'category': '국가'},

    # 리츠
    'VNQ':  {'name': 'Vanguard Real Estate',  'category': '리츠'},
}


# ============================================================================
# 수집 함수 — 단일 yfinance 배치 호출
# ============================================================================
def collect_global_prices(cache_dir="./etf_cache", years=3, progress_callback=None):
    """미국 상장 ETF 가격 배치 수집 (1회 yfinance 호출)"""

    cache_file = os.path.join(cache_dir, "global_prices_v2.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    # 캐시 확인 (당일 수집분이면 재활용)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('collected_at', datetime.min).date() == datetime.today().date():
                print("  💾 글로벌 캐시 로드")
                return cached
        except Exception:
            pass

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years + 30)

    # 전체 티커 (중복 제거)
    all_tickers = sorted(set(US_ETFS.keys()))
    print(f"  📡 yfinance 배치 다운로드: {len(all_tickers)}개 ETF, {years}년")

    try:
        raw = yf.download(all_tickers, start=start_date, end=end_date,
                         progress=False, auto_adjust=True)

        if raw.empty:
            print("  ⚠️ yfinance 데이터 없음")
            return _empty_result()

        # Close 추출
        if isinstance(raw.columns, pd.MultiIndex):
            df_all = raw['Close']
        else:
            df_all = raw[['Close']].rename(columns={'Close': all_tickers[0]})

        df_all.index = pd.to_datetime(df_all.index)
        df_all = df_all.sort_index()

        # MultiIndex 컬럼 평탄화
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = df_all.columns.get_level_values(-1)

        print(f"  ✅ {df_all.shape[0]}일 × {df_all.shape[1]}개 ETF 수집 완료")

    except Exception as e:
        print(f"  ⚠️ yfinance 실패: {e}")
        return _empty_result()

    # 지수 대용 DataFrame (GLOBAL_INDICES 매핑)
    idx_data = {}
    for name, info in GLOBAL_INDICES.items():
        tk = info['ticker']
        if tk in df_all.columns:
            idx_data[name] = df_all[tk]
    df_indices = pd.DataFrame(idx_data)

    result = {
        'indices': df_indices,
        'us_etfs': df_all,
        'index_info': GLOBAL_INDICES,
        'us_etf_info': US_ETFS,
        'collected_at': datetime.now()
    }

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception:
        pass

    return result


def _empty_result():
    return {
        'indices': pd.DataFrame(),
        'us_etfs': pd.DataFrame(),
        'index_info': GLOBAL_INDICES,
        'us_etf_info': US_ETFS,
        'collected_at': datetime.now()
    }


def calc_period_return(df_prices, start_date, end_date):
    """특정 기간 수익률 계산 (%)"""
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    mask = (df_prices.index >= start_dt) & (df_prices.index <= end_dt)
    sub = df_prices[mask].dropna(how='all')
    if len(sub) < 2:
        return pd.Series(dtype=float)
    ret = ((sub.iloc[-1] / sub.iloc[0]) - 1) * 100
    return ret.round(2)
