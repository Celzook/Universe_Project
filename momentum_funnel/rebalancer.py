"""MP 자동 리밸런싱 룰 엔진 (Phase 1).

사용자 합의 룰 (mp_rebalance_rules_decisions.md):
  1. **BM 손절** — 편입일 이후 누적 (ETF 수익률 − BM 수익률) ≤ −10% → 50% 매도,
                   ≤ −15% → 전체 매도. (D2: trailing peak 아님, 누적 초과성과)
  2. **교체매매** — 50% 손절분(룰 1)은 즉시 TIGER 200 (`core_ticker`) 으로 흡수.
                   전체 손절분 / 룰 3 매도분도 동일하게 core 로 흡수 (default).
  3. **MDD 매도** — 편입일 이후 peak 대비 ≤ −10% → 50% 매도, ≤ −15% → 전체 매도.
  4. **자본 누적** — 초기 자본 1,000억, 매매시 수익률을 NAV 에 합산.
  5. **새틀라이트 유지** — 활성 satellite 가 3 미만이면 rebalance_fn 으로 최신 top pick 편입.

D5 (룰 1·3 중복): 더 엄격한 액션 우선 (full > 50% > none).
D7: 매매가 = 당일 종가, 즉시 체결.
D8: 매매 비용 0.

호출 인터페이스는 의도적으로 데이터 소스 비독립 — price_data / benchmark / rebalance_fn 을
주입식으로 받음. UI 레이어가 캐시된 시리즈를 넘기고, 테스트는 합성 시리즈로 채움.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd


INITIAL_CAPITAL_DEFAULT = 100_000_000_000.0   # 1,000억
CORE_TICKER_DEFAULT = '102110'                  # TIGER 200
BM_50PCT_THRESHOLD = -0.10
BM_FULL_THRESHOLD = -0.15
MDD_50PCT_THRESHOLD = -0.10
MDD_FULL_THRESHOLD = -0.15
SATELLITE_MIN = 3
SATELLITE_MAX = 5

# 새 편입 시 satellite 1개당 목표 비중 (코어 87.5% / sat 12.5% / sat 5개 가정 → 2.5%)
DEFAULT_NEW_SAT_WEIGHT = 0.025


@dataclass
class PositionState:
    """단일 보유 종목 상태. units × current_price = market value."""
    ticker: str
    representative: str
    category: str
    role: str                      # 'Core' | 'Satellite'
    anchor_date: pd.Timestamp
    anchor_price: float
    peak_price: float
    units: float
    status: str = 'active'          # 'active' | 'partial_cut' | 'full_cut'
    rule1_50_triggered: bool = False
    rule3_50_triggered: bool = False


def _parse_date(s) -> pd.Timestamp:
    if isinstance(s, pd.Timestamp):
        return s.normalize()
    return pd.Timestamp(str(s).replace('-', '').replace('/', '').replace(' ', '')[:8])


def _first_index_on_or_after(idx: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    sub = idx[idx >= target]
    return sub[0] if len(sub) else None


def _last_price_on_or_before(close: pd.Series, day: pd.Timestamp) -> Optional[float]:
    if close.empty:
        return None
    if day in close.index:
        return float(close.loc[day])
    sub = close.index[close.index <= day]
    if len(sub) == 0:
        return None
    return float(close.loc[sub[-1]])


def apply_rules(
    saved_mp: dict,
    today: pd.Timestamp,
    price_data: Dict[str, pd.Series],
    benchmark: pd.Series,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
    core_ticker: str = CORE_TICKER_DEFAULT,
    core_name: str = 'TIGER 200',
    bm_50pct_threshold: float = BM_50PCT_THRESHOLD,
    bm_full_threshold: float = BM_FULL_THRESHOLD,
    mdd_50pct_threshold: float = MDD_50PCT_THRESHOLD,
    mdd_full_threshold: float = MDD_FULL_THRESHOLD,
    rebalance_fn: Optional[Callable[[pd.Timestamp], List[dict]]] = None,
    new_sat_weight: float = DEFAULT_NEW_SAT_WEIGHT,
) -> dict:
    """저장 MP 에 룰 1~5 를 편입일부터 today 까지 소급·forward 시뮬레이션.

    Parameters
    ----------
    saved_mp : dict
        portfolio_tracker.load_mp() 의 결과. positions[*] 는
        {role, ticker, representative, category, hot_score, weight_pct} 가정.
    today : pd.Timestamp
        시뮬 종료일 (inclusive). benchmark 의 거래일만 사용.
    price_data : Dict[ticker → pd.Series]
        각 보유·후보 ticker 의 종가 시리즈 (DatetimeIndex).
        rebalance_fn 으로 추가될 후보도 포함되어 있어야 매수 가능.
    benchmark : pd.Series
        KOSPI 종가 시리즈. trading_days 의 진실 공급원.
    rebalance_fn : Callable[[pd.Timestamp], List[dict]]], optional
        satellite < 3 시 후보 종목 generator. 각 dict 는 ticker/representative/category 필요.
        None 이면 룰 5 비활성 (활성 satellite 가 줄어도 신규 편입 안 함).

    Returns
    -------
    dict 키:
        daily_positions : pd.DataFrame
            index=trading_days, columns=tickers (활성/매수된 적 있는 전체).
            값=당일 종가 평가 기준 weight (%, 합=100).
        cumulative_nav  : pd.Series  (index=trading_days, 값=NAV in KRW)
        trade_log       : list[dict]
            {date, ticker, representative, action, fraction, metric, proceeds, replacement}
            action ∈ {'rule1_50','rule1_full','rule3_50','rule3_full','rule5_new'}
        final_positions : list[dict]  PositionState as_dict()
        inception       : pd.Timestamp  실제 시뮬 시작 거래일
        nav_pct         : float  최종 NAV / initial_capital − 1 (× 100, % 단위)
        active_sat_count: int
    """
    today = _parse_date(today)
    inception_input = _parse_date(saved_mp.get('inception_date'))

    bm = benchmark.dropna().sort_index() if isinstance(benchmark, pd.Series) else pd.Series(dtype=float)
    if bm.empty:
        return _empty_result(inception_input)

    trading_days = bm.index[(bm.index >= inception_input) & (bm.index <= today)]
    if len(trading_days) == 0:
        return _empty_result(inception_input)
    inception_actual = trading_days[0]

    # ── 1. 초기 포지션 구성 ───────────────────────────────────────────────
    positions: Dict[str, PositionState] = {}
    skipped: List[str] = []
    for p in saved_mp.get('positions', []):
        ticker = str(p.get('ticker', ''))
        if not ticker:
            continue
        close = price_data.get(ticker)
        if close is None or close.empty:
            skipped.append(ticker)
            continue
        close = close.dropna().sort_index()
        a_date = _first_index_on_or_after(close.index, inception_actual)
        if a_date is None:
            skipped.append(ticker)
            continue
        a_price = float(close.loc[a_date])
        if a_price <= 0:
            skipped.append(ticker)
            continue
        weight = float(p.get('weight_pct', 0.0))
        if weight <= 0:
            continue
        value0 = weight / 100.0 * initial_capital
        positions[ticker] = PositionState(
            ticker=ticker,
            representative=str(p.get('representative', '')),
            category=str(p.get('category', '')),
            role=str(p.get('role', 'Satellite')),
            anchor_date=a_date,
            anchor_price=a_price,
            peak_price=a_price,
            units=value0 / a_price,
        )

    # ── 2. core (TIGER 200) 보장 — 룰 2 흡수처 ───────────────────────────
    if core_ticker not in positions:
        close = price_data.get(core_ticker)
        if close is not None and not close.empty:
            close = close.dropna().sort_index()
            a_date = _first_index_on_or_after(close.index, inception_actual)
            if a_date is not None:
                a_price = float(close.loc[a_date])
                if a_price > 0:
                    positions[core_ticker] = PositionState(
                        ticker=core_ticker,
                        representative=core_name,
                        category='코스피200 베타',
                        role='Core',
                        anchor_date=a_date,
                        anchor_price=a_price,
                        peak_price=a_price,
                        units=0.0,
                    )

    if not positions:
        return _empty_result(inception_actual)

    bm_anchor = float(bm.loc[inception_actual])

    daily_records: List[dict] = []
    trade_log: List[dict] = []

    # ── 3. 일별 시뮬 루프 ────────────────────────────────────────────────
    for day in trading_days:
        bm_today = float(bm.loc[day])

        # 3-a. peak 갱신 (full_cut 도 갱신은 무의미하지만 cost 0)
        for pos in positions.values():
            if pos.status == 'full_cut':
                continue
            close = price_data.get(pos.ticker)
            if close is None:
                continue
            px = _last_price_on_or_before(close, day)
            if px is None or day < pos.anchor_date:
                continue
            if px > pos.peak_price:
                pos.peak_price = px

        # 3-b. 룰 1·3 평가 (satellite 만, core 는 BM 자체이므로 면제)
        pending_actions: List[tuple] = []
        for ticker, pos in positions.items():
            if pos.status == 'full_cut' or pos.role == 'Core':
                continue
            if day < pos.anchor_date:
                continue
            close = price_data.get(ticker)
            if close is None:
                continue
            px = _last_price_on_or_before(close, day)
            if px is None or pos.anchor_price <= 0:
                continue

            etf_ret = px / pos.anchor_price - 1.0
            bm_ret = bm_today / bm_anchor - 1.0 if bm_anchor > 0 else 0.0
            excess = etf_ret - bm_ret
            mdd = px / pos.peak_price - 1.0 if pos.peak_price > 0 else 0.0

            # D5: 더 엄격한 액션 우선
            action = None
            metric = None
            if excess <= bm_full_threshold:
                action, metric, fraction = 'rule1_full', excess, 1.0
            elif mdd <= mdd_full_threshold:
                action, metric, fraction = 'rule3_full', mdd, 1.0
            elif excess <= bm_50pct_threshold and not pos.rule1_50_triggered:
                action, metric, fraction = 'rule1_50', excess, 0.5
            elif mdd <= mdd_50pct_threshold and not pos.rule3_50_triggered:
                action, metric, fraction = 'rule3_50', mdd, 0.5

            if action is not None:
                pending_actions.append((ticker, action, fraction, metric, px))

        # 3-c. 매도 실행 + 룰 2 흡수
        for ticker, action, fraction, metric, px in pending_actions:
            pos = positions[ticker]
            sold_units = pos.units * fraction
            proceeds = sold_units * px
            pos.units -= sold_units

            if fraction >= 1.0:
                pos.status = 'full_cut'
            else:
                pos.status = 'partial_cut'
                if action.startswith('rule1'):
                    pos.rule1_50_triggered = True
                else:
                    pos.rule3_50_triggered = True

            # TIGER 200 으로 흡수
            replacement = ''
            core_pos = positions.get(core_ticker)
            if core_pos is not None:
                core_close = price_data.get(core_ticker)
                core_px = _last_price_on_or_before(core_close, day) if core_close is not None else None
                if core_px and core_px > 0:
                    core_pos.units += proceeds / core_px
                    replacement = core_ticker
                    if core_pos.status == 'full_cut':
                        core_pos.status = 'active'

            trade_log.append({
                'date': day,
                'ticker': ticker,
                'representative': pos.representative,
                'action': action,
                'fraction': fraction,
                'metric': float(metric) if metric is not None else np.nan,
                'price': float(px),
                'proceeds': float(proceeds),
                'replacement': replacement,
            })

        # 3-d. 룰 5: satellite < 3 시 신규 편입
        active_sat = [p for p in positions.values()
                      if p.role == 'Satellite' and p.status != 'full_cut' and p.units > 0]
        if len(active_sat) < SATELLITE_MIN and rebalance_fn is not None:
            try:
                candidates = rebalance_fn(day) or []
            except Exception:
                candidates = []

            # 현재 NAV 평가
            cur_nav = _portfolio_value(positions, price_data, day)
            target_value = new_sat_weight * cur_nav

            needed = SATELLITE_MIN - len(active_sat)
            core_pos = positions.get(core_ticker)
            core_close = price_data.get(core_ticker) if core_pos else None
            core_px = _last_price_on_or_before(core_close, day) if core_close is not None else None

            for cand in candidates:
                if needed <= 0:
                    break
                tk = str(cand.get('ticker', ''))
                if not tk or tk == core_ticker:
                    continue
                if tk in positions and positions[tk].status != 'full_cut':
                    continue
                cand_close = price_data.get(tk)
                if cand_close is None:
                    continue
                cand_px = _last_price_on_or_before(cand_close, day)
                if cand_px is None or cand_px <= 0:
                    continue

                # core 에서 자금 차감 (코어가 부족하면 skip)
                if core_pos is None or core_px is None or core_px <= 0:
                    break
                core_value = core_pos.units * core_px
                buy_value = min(target_value, core_value)
                if buy_value <= 0:
                    break
                core_pos.units -= buy_value / core_px
                positions[tk] = PositionState(
                    ticker=tk,
                    representative=str(cand.get('representative', '')),
                    category=str(cand.get('category', '')),
                    role='Satellite',
                    anchor_date=day,
                    anchor_price=cand_px,
                    peak_price=cand_px,
                    units=buy_value / cand_px,
                )
                trade_log.append({
                    'date': day,
                    'ticker': tk,
                    'representative': str(cand.get('representative', '')),
                    'action': 'rule5_new',
                    'fraction': float(buy_value / cur_nav) if cur_nav > 0 else 0.0,
                    'metric': np.nan,
                    'price': float(cand_px),
                    'proceeds': float(buy_value),
                    'replacement': core_ticker,
                })
                needed -= 1

        # 3-e. 일말 스냅샷 (weights + NAV)
        snapshot, nav = _portfolio_snapshot(positions, price_data, day)
        snapshot['_date'] = day
        snapshot['_nav'] = nav
        daily_records.append(snapshot)

    # ── 4. 결과 패키징 ───────────────────────────────────────────────────
    df = pd.DataFrame(daily_records).set_index('_date')
    df.index.name = 'date'
    nav_series = df['_nav'].astype(float)
    daily_df = df.drop(columns=['_nav']).fillna(0.0)

    final_nav = float(nav_series.iloc[-1]) if not nav_series.empty else float(initial_capital)
    nav_pct = (final_nav / initial_capital - 1.0) * 100.0

    active_sat_count = sum(
        1 for p in positions.values()
        if p.role == 'Satellite' and p.status != 'full_cut' and p.units > 0
    )

    return {
        'daily_positions': daily_df,
        'cumulative_nav': nav_series,
        'trade_log': trade_log,
        'final_positions': [asdict(p) for p in positions.values()],
        'inception': inception_actual,
        'nav_pct': nav_pct,
        'active_sat_count': active_sat_count,
        'skipped_tickers': skipped,
    }


def _portfolio_value(positions: Dict[str, PositionState],
                     price_data: Dict[str, pd.Series],
                     day: pd.Timestamp) -> float:
    total = 0.0
    for p in positions.values():
        if p.units <= 0:
            continue
        close = price_data.get(p.ticker)
        if close is None:
            continue
        px = _last_price_on_or_before(close, day)
        if px is None:
            continue
        total += p.units * px
    return total


def _portfolio_snapshot(positions: Dict[str, PositionState],
                        price_data: Dict[str, pd.Series],
                        day: pd.Timestamp):
    values = {}
    total = 0.0
    for ticker, p in positions.items():
        if p.units <= 0:
            continue
        close = price_data.get(ticker)
        if close is None:
            continue
        px = _last_price_on_or_before(close, day)
        if px is None:
            continue
        v = p.units * px
        values[ticker] = v
        total += v
    snap = {}
    for ticker, v in values.items():
        snap[ticker] = (v / total * 100.0) if total > 0 else 0.0
    return snap, total


def _empty_result(inception: pd.Timestamp) -> dict:
    return {
        'daily_positions': pd.DataFrame(),
        'cumulative_nav': pd.Series(dtype=float),
        'trade_log': [],
        'final_positions': [],
        'inception': inception,
        'nav_pct': 0.0,
        'active_sat_count': 0,
        'skipped_tickers': [],
    }
