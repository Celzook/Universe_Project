"""
Hot Sectors Board + Model Portfolio builder.

핵심 신호 3가지로 섹터 랭킹 + 코어-새틀라이트 MP 구성.
- VolRatio  : 최근 3일 거래대금 / 20일 평균 (거래 급증)
- MoneyFlow : 5일 누적 net money flow / total flow (자금 유입; -1~+1)
- RS        : log(섹터/벤치) 회귀 기울기 (Funnel과 동일)

HotScore = 세 신호의 percentile rank 평균.

MP 구조:
- 코어: TIGER 200 (102110) @ 87.5% — 코스피200 베타
- 새틀라이트: Hot 섹터 N개 @ 12.5% — HotScore 비례 가중
- 베타 중복 카테고리(대형주/코스피200/가치주/배당) 자동 제외
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .contracts import MarketData
from .config import FunnelConfig
from .indicators import compute_rs


# 코어와 베타가 중복되는 카테고리 (새틀라이트에서 자동 제외)
EXCLUDED_CATEGORIES_DEFAULT: List[str] = [
    '대형주', '코스피200', 'KOSPI200',
    '가치주', '배당', '대형가치', '대형성장',
]

# 코어 ETF
CORE_TICKER_DEFAULT = '102110'   # TIGER 200
CORE_NAME_DEFAULT = 'TIGER 200'

HOT_COLS = ['valid', 'vol_ratio', 'money_flow', 'rs', 'hot_score']


# ──────────────────────────────────────────────────────────────────────
# 1. Hot Metrics (섹터별 신호 + HotScore)
# ──────────────────────────────────────────────────────────────────────
def compute_hot_metrics(
    market: MarketData,
    cfg: FunnelConfig,
    asof: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    섹터별 VolRatio · MoneyFlow · RS 계산 + HotScore 합산.

    Returns
    -------
    DataFrame indexed by sector with columns HOT_COLS.
    NaN HotScore 는 valid=False 또는 데이터 부족.
    HotScore 내림차순 정렬.
    """
    vol_window = max(int(getattr(cfg, 'hot_vol_lookback', 20)), 5)
    money_window = max(int(getattr(cfg, 'hot_money_lookback', 5)), 2)
    vol_recent_n = 3   # 최근 3일 평균 vs 기간 평균

    rows: Dict[str, dict] = {}
    bench_close_full = market.benchmark['close'] if 'close' in market.benchmark.columns else pd.Series(dtype=float)

    for sector, df_full in market.sector_data.items():
        df = df_full
        if asof is not None and not df.empty:
            df = df[df.index <= asof]

        min_needed = max(vol_window, cfg.rs_window) + 2
        if len(df) < min_needed:
            rows[sector] = {'valid': False, 'vol_ratio': np.nan,
                            'money_flow': np.nan, 'rs': np.nan, 'hot_score': np.nan}
            continue

        # VolRatio: 최근 3일 거래대금 평균 / 20일 평균
        # NOTE: df['volume'] 은 data_adapter._aggregate_sector 에서 sum(close*shares)
        #       으로 합산된 거래대금(KRW) proxy 이지 raw 좌수가 아님.
        vol_recent = df['volume'].tail(vol_recent_n).mean()
        vol_base = df['volume'].tail(vol_window).mean()
        vol_ratio = float(vol_recent / vol_base) if vol_base and vol_base > 0 else np.nan

        # MoneyFlow: 5일 누적 net flow / total flow ∈ [-1, +1]
        # df['volume'] 이 이미 거래대금(KRW) 단위이므로 TP 를 곱하지 않음 (이중 계상 방지).
        # 방향성은 TP 의 일별 diff 로 양/음수 buckets 분류.
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        raw_mf = df['volume']
        tp_diff = tp.diff()
        recent_raw = raw_mf.tail(money_window)
        recent_diff = tp_diff.tail(money_window)
        pos_flow = float(recent_raw.where(recent_diff > 0, 0).sum())
        neg_flow = float(recent_raw.where(recent_diff < 0, 0).sum())
        total = pos_flow + neg_flow
        money_flow = (pos_flow - neg_flow) / total if total > 0 else 0.0

        # RS (Funnel 공유 구현)
        rs = compute_rs(df['close'], bench_close_full, cfg, asof=asof)

        rows[sector] = {
            'valid': True,
            'vol_ratio': vol_ratio,
            'money_flow': money_flow,
            'rs': rs if rs is not None else np.nan,
            'hot_score': np.nan,
        }

    df_metrics = pd.DataFrame.from_dict(rows, orient='index', columns=HOT_COLS)
    df_metrics.index.name = 'sector'

    # HotScore = percentile rank 3종 평균 (valid 행만)
    valid_mask = df_metrics['valid'] & df_metrics[['vol_ratio', 'money_flow', 'rs']].notna().all(axis=1)
    if valid_mask.any():
        sub = df_metrics.loc[valid_mask, ['vol_ratio', 'money_flow', 'rs']].astype(float)
        ranks = sub.rank(pct=True, method='average')
        df_metrics.loc[valid_mask, 'hot_score'] = ranks.mean(axis=1)

    return df_metrics.sort_values('hot_score', ascending=False, na_position='last')


# ──────────────────────────────────────────────────────────────────────
# 2. MP Builder (Core-Satellite)
# ──────────────────────────────────────────────────────────────────────
def build_mp(
    hot_metrics: pd.DataFrame,
    market: MarketData,
    method: str = 'A',
    core_ticker: str = CORE_TICKER_DEFAULT,
    core_name: str = CORE_NAME_DEFAULT,
    core_weight: float = 0.875,
    n_satellites: int = 5,
    excluded_categories: Optional[List[str]] = None,
    money_pool_size: int = 10,
    ticker_to_name: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Hot 결과로 Core-Satellite MP 구성.

    method:
        'A' = HotScore Top N (단일 정렬)
        'B' = Money Top {money_pool_size} → 그 중 RS Top N (2단 스크린)

    Returns
    -------
    DataFrame with columns [role, ticker, representative, category, hot_score, weight_pct]
    role ∈ {'Core', 'Satellite'}. weight_pct 합 = 100.
    """
    excluded = set(excluded_categories or EXCLUDED_CATEGORIES_DEFAULT)
    ticker_to_name = ticker_to_name or {}

    pool = hot_metrics[
        hot_metrics['valid'].fillna(False)
        & hot_metrics['hot_score'].notna()
        & ~hot_metrics.index.isin(excluded)
    ].copy()

    if method == 'A':
        picks = pool.head(n_satellites)
    elif method == 'B':
        top_money = pool.sort_values('money_flow', ascending=False).head(money_pool_size)
        picks = top_money.sort_values('rs', ascending=False).head(n_satellites)
    else:
        raise ValueError(f"unknown method '{method}', expected 'A' or 'B'")

    # 코어 행
    rows: List[dict] = [{
        'role': 'Core',
        'ticker': core_ticker,
        'representative': core_name,
        'category': '코스피200 베타',
        'hot_score': np.nan,
        'weight_pct': core_weight * 100.0,
    }]

    if picks.empty:
        rows[0]['weight_pct'] = 100.0
        return pd.DataFrame(rows)

    # 새틀라이트 가중치: HotScore 비례
    sat_total = 1.0 - core_weight
    scores = picks['hot_score'].values.astype(float)
    if scores.sum() > 0:
        sat_w = scores / scores.sum() * sat_total
    else:
        sat_w = np.full(len(picks), sat_total / len(picks))

    for (cat, row), w in zip(picks.iterrows(), sat_w):
        tickers = market.meta.get(cat, {}).get('tickers', [])
        rep_ticker = tickers[0] if tickers else ''
        rep_name = ticker_to_name.get(rep_ticker, '')
        rows.append({
            'role': 'Satellite',
            'ticker': rep_ticker,
            'representative': rep_name,
            'category': cat,
            'hot_score': float(row['hot_score']),
            'weight_pct': float(w) * 100.0,
        })

    df = pd.DataFrame(rows)
    # 정확히 100 정규화 (반올림 오차 흡수)
    df['weight_pct'] = df['weight_pct'] * 100.0 / df['weight_pct'].sum()
    return df
