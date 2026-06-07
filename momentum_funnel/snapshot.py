"""Headless 스냅샷 생성기 (Phase 3).

UI (streamlit) 의존 없이 saved MP + 룰 1~5 적용 결과를 dict 로 반환.
GitHub Actions cron / 수동 CLI / Streamlit page_history 가 동일 함수를 공유.

진행 상황은 logger 로만 전달 (st.progress 호출 X).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .config import FunnelConfig
from .data_adapter import UniverseDataSource
from .hot_board import (
    compute_hot_metrics, build_mp,
    CORE_TICKER_DEFAULT, CORE_NAME_DEFAULT,
)
from .portfolio_tracker import load_mp
from .rebalancer import apply_rules


log = logging.getLogger(__name__)


def _fetch_close_naver(ticker: str, start_str: str, end_str: str) -> pd.Series:
    from etf_universe_builder import naver_get_price_history
    try:
        s = naver_get_price_history(ticker, start_str, end_str)
        if isinstance(s, pd.Series):
            return s.dropna()
    except Exception as e:
        log.warning("price fetch failed %s: %s", ticker, e)
    return pd.Series(dtype=float)


def _fetch_kospi(start_str: str, end_str: str) -> pd.Series:
    from etf_universe_builder import naver_get_index_history
    try:
        s = naver_get_index_history('KOSPI', start_str, end_str)
        if isinstance(s, pd.Series) and not s.empty:
            return s.dropna()
    except Exception as e:
        log.warning("KOSPI fetch failed: %s", e)
    return pd.Series(dtype=float)


def build_snapshot(
    saved_mp: Optional[dict] = None,
    df_universe: Optional[pd.DataFrame] = None,
    mode: str = 'A',
    enable_rule5: bool = True,
    n_satellites: int = 5,
    today: Optional[pd.Timestamp] = None,
    hot_max_members: int = 5,
) -> Optional[dict]:
    """저장 MP 에 룰 1~5 적용한 오늘 스냅샷 dict 생성. UI 무관.

    Parameters
    ----------
    saved_mp : dict, optional
        portfolio_tracker.load_mp() 결과. None 이면 자동 로드.
    df_universe : pd.DataFrame, optional
        Hot metrics 산출용. None 이거나 빈 DataFrame 이면 룰 5 자동 비활성.
    mode : 'A' | 'B'
        룰 5 신규 편입 후보 산출 방법 (hot_board.build_mp method).
    enable_rule5 : bool
        룰 5 (Hot 신규 편입) 활성 여부. df_universe 없으면 자동으로 False 됨.
    today : pd.Timestamp, optional
        시뮬 종료일 (inclusive). None 이면 오늘.
    hot_max_members : int
        섹터당 OHLCV 수집 멤버 수 (cron 부담 완화 목적).

    Returns
    -------
    dict | None
        None 이면 스냅샷 생성 불가 (saved MP 없음 / 가격 데이터 부족 / 휴장일 등).
        dict 스키마 (history.json entry 와 동일):
            date, nav_krw, nav_pct, active_sat_count, n_trades_total,
            inception_date, method, mode, positions.
    """
    if saved_mp is None:
        saved_mp = load_mp()
    if not saved_mp:
        log.info("saved MP not found — snapshot skipped")
        return None

    today_ts = (today or pd.Timestamp.today()).normalize()

    # ── 1. 보유 + core ticker 수집 ───────────────────────────────────────
    tickers_needed = []
    for p in saved_mp.get('positions', []):
        tk = str(p.get('ticker', ''))
        if tk and tk not in tickers_needed:
            tickers_needed.append(tk)
    if CORE_TICKER_DEFAULT not in tickers_needed:
        tickers_needed.append(CORE_TICKER_DEFAULT)

    # ── 2. Hot 후보 산출 (룰 5) ──────────────────────────────────────────
    candidates: list = []
    use_rule5 = bool(enable_rule5) and isinstance(df_universe, pd.DataFrame) and not df_universe.empty
    if use_rule5:
        try:
            cfg = FunnelConfig()
            src = UniverseDataSource(
                df_universe, lookback_days=180, use_yf_fallback=True,
                cfg=cfg, max_members_per_sector=hot_max_members,
            )
            market = src.load()
            hot_metrics = compute_hot_metrics(market, cfg)
            ticker_to_name = (
                dict(zip(df_universe.index.astype(str), df_universe['ETF명']))
                if 'ETF명' in df_universe.columns else {}
            )
            mp_pick = build_mp(
                hot_metrics, market, method=mode,
                core_ticker=CORE_TICKER_DEFAULT, core_name=CORE_NAME_DEFAULT,
                core_weight=0.875, n_satellites=n_satellites,
                ticker_to_name=ticker_to_name,
            )
            for _, row in mp_pick.iterrows():
                if str(row['role']) == 'Core':
                    continue
                cand_tk = str(row['ticker'])
                if not cand_tk:
                    continue
                candidates.append({
                    'ticker': cand_tk,
                    'representative': str(row['representative']),
                    'category': str(row['category']),
                })
                if cand_tk not in tickers_needed:
                    tickers_needed.append(cand_tk)
        except Exception as e:
            log.warning("Hot 후보 산출 실패 — 룰 5 비활성: %s", e)
            candidates = []
    else:
        log.info("rule 5 disabled (no universe / opt-out)")

    # ── 3. 가격 데이터 수집 ──────────────────────────────────────────────
    inception = str(saved_mp.get('inception_date', ''))
    try:
        start_dt = datetime.strptime(inception.replace('-', ''), '%Y%m%d') - timedelta(days=10)
    except Exception as e:
        log.error("편입일 파싱 실패 (%s): %s", inception, e)
        return None
    start_str = start_dt.strftime('%Y%m%d')
    end_str = today_ts.strftime('%Y%m%d')

    price_data: dict = {}
    for i, tk in enumerate(tickers_needed):
        s = _fetch_close_naver(tk, start_str, end_str)
        if not s.empty:
            price_data[tk] = s
        if (i + 1) % 5 == 0:
            log.info("fetched %d/%d tickers", i + 1, len(tickers_needed))

    benchmark = _fetch_kospi(start_str, end_str)
    if benchmark.empty:
        log.error("KOSPI 벤치마크 비어 있음 — 스냅샷 생성 중단 (휴장일 가능성)")
        return None

    # 휴장일 스킵: 오늘 KOSPI 데이터 없으면 skip (cron 에서 exit 0)
    if benchmark.index[-1].normalize() < today_ts:
        # 마지막 거래일이 오늘보다 이전 → 오늘은 휴장일 가능. cron 에서 skip.
        log.info("today not in benchmark index (last=%s, today=%s) — likely market closed",
                 benchmark.index[-1].date(), today_ts.date())

    # ── 4. 룰 엔진 ───────────────────────────────────────────────────────
    rebalance_fn = (lambda _d: candidates) if (use_rule5 and candidates) else None
    try:
        result = apply_rules(
            saved_mp, today=today_ts,
            price_data=price_data, benchmark=benchmark,
            rebalance_fn=rebalance_fn,
        )
    except Exception as e:
        log.error("apply_rules 실패: %s", e)
        return None

    if result['daily_positions'].empty:
        log.warning("시뮬 결과 비어있음 (편입일 이후 거래일 부족)")
        return None

    nav = result['cumulative_nav']
    final_nav = float(nav.iloc[-1])
    snap_date = nav.index[-1].strftime('%Y-%m-%d')
    last_row = result['daily_positions'].iloc[-1]

    positions = []
    for ticker, weight in last_row.items():
        if weight <= 0.01:
            continue
        state = next((p for p in result['final_positions'] if p['ticker'] == ticker), None)
        positions.append({
            'ticker': str(ticker),
            'representative': state.get('representative', '') if state else '',
            'category': state.get('category', '') if state else '',
            'role': state.get('role', '') if state else '',
            'status': state.get('status', '') if state else '',
            'weight_pct': float(weight),
        })

    return {
        'date': snap_date,
        'nav_krw': final_nav,
        'nav_pct': float(result['nav_pct']),
        'active_sat_count': int(result['active_sat_count']),
        'n_trades_total': int(len(result['trade_log'])),
        'inception_date': str(saved_mp.get('inception_date', '')),
        'method': str(saved_mp.get('method', '')),
        'mode': mode,
        'positions': positions,
    }
