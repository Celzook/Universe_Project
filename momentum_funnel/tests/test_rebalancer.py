"""Unit tests for momentum_funnel.rebalancer.apply_rules.

룰별 발동 시나리오를 합성 가격 시리즈로 검증.
- 룰 1 (BM 손절 50%/full)
- 룰 3 (MDD 50%/full)
- D5 우선순위 (full > 50%)
- 룰 4 NAV 누적 보존성
- 룰 5 신규 편입 (rebalance_fn 주입)
"""
import numpy as np
import pandas as pd
import pytest

from momentum_funnel.rebalancer import apply_rules, INITIAL_CAPITAL_DEFAULT


def _make_bdates(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _flat_series(dates: pd.DatetimeIndex, value: float = 100.0) -> pd.Series:
    return pd.Series([value] * len(dates), index=dates, dtype=float)


def _saved_mp(positions):
    return {
        'saved_at': '2025-01-01 00:00:00',
        'inception_date': '2025-01-02',
        'method': 'A',
        'positions': positions,
    }


@pytest.fixture
def bdates():
    return _make_bdates('2025-01-02', 30)


@pytest.fixture
def flat_bm(bdates):
    return _flat_series(bdates, 100.0)


# ──────────────────────────────────────────────────────────────────────
# 룰 1 — BM 손절
# ──────────────────────────────────────────────────────────────────────
def test_rule1_50_triggers_when_excess_reaches_minus_10(bdates, flat_bm):
    """BM 100 flat, ETF 가 −10% 떨어지면 → rule1_50 발동, units 절반 매도."""
    px = np.linspace(100.0, 88.0, len(bdates))   # −12% drop → excess = −12%
    etf = pd.Series(px, index=bdates)
    core = _flat_series(bdates, 50.0)             # TIGER 200 흡수처

    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999001', 'representative': 'TEST',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    price_data = {'102110': core, '999001': etf}

    result = apply_rules(saved, bdates[-1], price_data, flat_bm)
    actions = [t for t in result['trade_log'] if t['ticker'] == '999001']
    assert len(actions) >= 1
    assert actions[0]['action'] == 'rule1_50'
    assert actions[0]['fraction'] == pytest.approx(0.5)
    # rule1_50 한 번만 발동 (재발동 X)
    assert sum(1 for t in actions if t['action'] == 'rule1_50') == 1


def test_rule1_full_triggers_when_excess_reaches_minus_15(bdates, flat_bm):
    """ETF 가 −16% 떨어지면 → rule1_full 발동 (50% 단계 건너뜀 가능)."""
    px = np.linspace(100.0, 80.0, len(bdates))   # −20% → 도중에 50%, 끝에 full
    etf = pd.Series(px, index=bdates)
    core = _flat_series(bdates, 50.0)

    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999002', 'representative': 'TEST',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    price_data = {'102110': core, '999002': etf}

    result = apply_rules(saved, bdates[-1], price_data, flat_bm)
    actions = [t for t in result['trade_log'] if t['ticker'] == '999002']
    assert any(t['action'] == 'rule1_full' for t in actions)
    final = next(p for p in result['final_positions'] if p['ticker'] == '999002')
    assert final['status'] == 'full_cut'
    assert final['units'] == pytest.approx(0.0)


def test_rule1_no_trigger_when_excess_within_band(bdates, flat_bm):
    """ETF 가 −5% 만 떨어지면 룰 1 비발동."""
    px = np.linspace(100.0, 95.0, len(bdates))
    etf = pd.Series(px, index=bdates)
    core = _flat_series(bdates, 50.0)

    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999003', 'representative': 'TEST',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    price_data = {'102110': core, '999003': etf}

    result = apply_rules(saved, bdates[-1], price_data, flat_bm)
    assert all(t['ticker'] != '999003' or t['action'] == 'rule5_new'
               for t in result['trade_log'])


# ──────────────────────────────────────────────────────────────────────
# 룰 3 — MDD
# ──────────────────────────────────────────────────────────────────────
def test_rule3_50_triggers_after_peak_drop_10pct(bdates, flat_bm):
    """편입 후 peak 까지 +10% 오르고 peak 대비 −11% 떨어지면 rule3_50.

    Excess (vs flat BM) = (peak−11%) − 0% = (1.10 × 0.89 − 1) ≈ −2.1%  → 룰 1 미발동.
    MDD = (0.89 − 1) = −11% → rule3_50 발동.
    """
    n = len(bdates)
    rising = np.linspace(100.0, 110.0, n // 2)
    falling = np.linspace(110.0, 110.0 * 0.89, n - n // 2)
    etf = pd.Series(np.concatenate([rising, falling]), index=bdates)
    core = _flat_series(bdates, 50.0)

    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999004', 'representative': 'TEST',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    price_data = {'102110': core, '999004': etf}

    result = apply_rules(saved, bdates[-1], price_data, flat_bm)
    actions = [t for t in result['trade_log'] if t['ticker'] == '999004']
    assert any(t['action'] == 'rule3_50' for t in actions)


# ──────────────────────────────────────────────────────────────────────
# D5 — 더 엄격한 액션 우선
# ──────────────────────────────────────────────────────────────────────
def test_strict_action_priority_full_over_50(bdates, flat_bm):
    """ETF 가 −16% 떨어지면 (rule1_full 조건) → rule1_50 가 아닌 rule1_full 우선."""
    px = np.array([100.0] * 5 + [84.0] * (len(bdates) - 5))
    etf = pd.Series(px, index=bdates, dtype=float)
    core = _flat_series(bdates, 50.0)

    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999005', 'representative': 'TEST',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    price_data = {'102110': core, '999005': etf}

    result = apply_rules(saved, bdates[-1], price_data, flat_bm)
    actions = [t for t in result['trade_log'] if t['ticker'] == '999005']
    first = actions[0]
    assert first['action'] == 'rule1_full'
    assert first['fraction'] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# 룰 4 — NAV 누적
# ──────────────────────────────────────────────────────────────────────
def test_nav_starts_at_initial_capital(bdates, flat_bm):
    """flat ETF·flat BM 라면 NAV 가 거의 초기자본에 머묾."""
    core = _flat_series(bdates, 50.0)
    sat = _flat_series(bdates, 50.0)
    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999006', 'representative': 'TEST',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    result = apply_rules(saved, bdates[-1], {'102110': core, '999006': sat}, flat_bm)
    nav = result['cumulative_nav']
    assert not nav.empty
    assert nav.iloc[0] == pytest.approx(INITIAL_CAPITAL_DEFAULT, rel=1e-6)
    assert nav.iloc[-1] == pytest.approx(INITIAL_CAPITAL_DEFAULT, rel=1e-6)


def test_partial_cut_transfers_value_to_core(bdates):
    """50% 손절 매매는 value-preserving — proceeds 가 core 매입 가치와 일치.

    setup: BM 만 상승, ETF flat → 룰 3 (MDD) 발동 X, 룰 1 (BM 손절) 만 발동.
    이렇게 해야 매매 1회만 발생해 final units 로 invariant 검증 가능."""
    n = len(bdates)
    bm = pd.Series(np.linspace(100.0, 113.0, n), index=bdates)  # +13% (rule1_50 발동, rule1_full X)
    etf = _flat_series(bdates, 100.0)                            # ETF flat → MDD=0
    core = _flat_series(bdates, 50.0)

    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999007', 'representative': 'TEST',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    result = apply_rules(saved, bdates[-1], {'102110': core, '999007': etf}, bm)
    log = [t for t in result['trade_log'] if t['ticker'] == '999007']
    assert log, 'rule1_50 should fire'
    assert all(t['action'] == 'rule1_50' for t in log), 'only rule1_50 expected'
    assert len(log) == 1, 'rule1_50 must fire exactly once (재발동 방지)'

    trade = log[0]
    assert trade['fraction'] == pytest.approx(0.5)
    assert trade['replacement'] == '102110'

    sat = next(p for p in result['final_positions'] if p['ticker'] == '999007')
    init_sat_units = (12.5 / 100.0 * INITIAL_CAPITAL_DEFAULT) / 100.0
    assert sat['units'] == pytest.approx(init_sat_units * 0.5, rel=1e-6)
    assert sat['status'] == 'partial_cut'

    # core 흡수 검증: 증가분 × core_px(50) == proceeds
    core_pos = next(p for p in result['final_positions'] if p['ticker'] == '102110')
    init_core_units = (87.5 / 100.0 * INITIAL_CAPITAL_DEFAULT) / 50.0
    added_value = (core_pos['units'] - init_core_units) * 50.0
    assert added_value == pytest.approx(trade['proceeds'], rel=1e-6)


# ──────────────────────────────────────────────────────────────────────
# 룰 5 — 신규 편입
# ──────────────────────────────────────────────────────────────────────
def test_rule5_adds_new_sat_when_count_below_three(bdates, flat_bm):
    """satellite 1개가 전체 손절되면 rule 5 가 새 후보 1~2개 보충 (활성 ≥3)."""
    px = np.linspace(100.0, 80.0, len(bdates))      # ETF 전체 손절
    bad = pd.Series(px, index=bdates)
    core = _flat_series(bdates, 50.0)
    new1 = _flat_series(bdates, 30.0)
    new2 = _flat_series(bdates, 40.0)

    saved = _saved_mp([
        {'role': 'Core', 'ticker': '102110', 'representative': 'TIGER 200',
         'category': '코스피200 베타', 'weight_pct': 87.5, 'hot_score': np.nan},
        {'role': 'Satellite', 'ticker': '999008', 'representative': 'BAD',
         'category': '반도체', 'weight_pct': 12.5, 'hot_score': 0.9},
    ])
    price_data = {'102110': core, '999008': bad, 'NEW1': new1, 'NEW2': new2}

    def _rebalance(day):
        return [
            {'ticker': 'NEW1', 'representative': 'NEWONE', 'category': '2차전지'},
            {'ticker': 'NEW2', 'representative': 'NEWTWO', 'category': 'AI'},
        ]

    result = apply_rules(saved, bdates[-1], price_data, flat_bm,
                         rebalance_fn=_rebalance)
    rule5_trades = [t for t in result['trade_log'] if t['action'] == 'rule5_new']
    assert len(rule5_trades) >= 1
    # 활성 satellite ≥ 3 까지 채워야 함 (초기 1개 → 후 cut 0개 → 신규 2개 + ?)
    # 초기 1개가 cut 됐으니 active=0, needed=3 → NEW1, NEW2 추가됨 (후보가 2개뿐이라 2개 추가).
    new_tickers = {t['ticker'] for t in rule5_trades}
    assert {'NEW1', 'NEW2'} <= new_tickers
