"""GitHub Actions cron entrypoint — 일별 MP 스냅샷 자동 기록 (Phase 3).

흐름:
  1. load_mp() → 저장 MP 없으면 exit 0 (skip).
  2. 휴장일 가드 — KOSPI 최근 데이터의 마지막 날짜가 오늘 이전이면 exit 0 (시장 미개장).
  3. build_universe() — Hot 후보 산출용 유니버스 빌드 (3~8분).
  4. build_snapshot() → entry dict.
  5. append_history_snapshot() — saved_mps/history.json 갱신.
  6. push_history_to_github() — GitHub Contents API 로 자동 커밋.

환경 변수:
  GITHUB_TOKEN — workflow 의 ${{ secrets.GITHUB_TOKEN }} (`permissions: contents: write` 필요).
  HISTORY_MODE — 'A' (default) | 'B'.  룰 5 신규 편입 방법.

로컬 수동 실행 가능:
  GITHUB_TOKEN=ghp_xxx python scripts/record_history.py
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# scripts/ 에서 실행 시 프로젝트 루트 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
)
log = logging.getLogger('record_history')


def _today_kst_date() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('Asia/Seoul'))
    except Exception:
        return datetime.utcnow() + timedelta(hours=9)


def _is_market_open_today() -> bool:
    """KOSPI 최근 종가 데이터 fetch 해서 오늘 거래일인지 확인.

    오늘 데이터가 들어와 있으면 True, 아니면 False (휴장일/주말).
    네이버 인덱스 API 응답 자체가 실패해도 False 로 안전 측 처리.
    """
    today = _today_kst_date().date()
    start = (today - timedelta(days=7)).strftime('%Y%m%d')
    end = today.strftime('%Y%m%d')

    try:
        from etf_universe_builder import naver_get_index_history
        s = naver_get_index_history('KOSPI', start, end)
    except Exception as e:
        log.warning("KOSPI 휴장 가드 fetch 실패: %s", e)
        return False

    if not isinstance(s, pd.Series) or s.empty:
        return False
    last_day = pd.Timestamp(s.index[-1]).date()
    return last_day >= today


def main() -> int:
    from momentum_funnel import (
        load_mp, build_snapshot,
        append_history_snapshot, push_history_to_github,
    )

    saved = load_mp()
    if not saved:
        log.info("저장된 MP 없음 — 스냅샷 skip")
        return 0

    if not _is_market_open_today():
        log.info("오늘 휴장일 (KOSPI 데이터 미입수) — 스냅샷 skip")
        return 0

    mode = os.environ.get('HISTORY_MODE', 'A').strip().upper()
    if mode not in ('A', 'B'):
        log.warning("HISTORY_MODE 잘못됨 (%s) — 'A' 로 fallback", mode)
        mode = 'A'

    log.info("유니버스 빌드 시작 (3~8분)")
    try:
        from etf_universe_builder import build_universe, Config
        Config.MIN_MARKET_CAP_BILLIONS = int(os.environ.get('MIN_CAP', '200'))
        Config.TOP_N_HOLDINGS = int(os.environ.get('TOP_N', '10'))
        Config.BASE_DATE = None
        df_uni, _, _ = build_universe()
    except Exception as e:
        log.error("유니버스 빌드 실패: %s", e)
        return 1
    log.info("유니버스 빌드 완료 (%d개 ETF)", len(df_uni))

    log.info("스냅샷 생성 (mode=%s, rule5=ON)", mode)
    entry = build_snapshot(
        saved_mp=saved, df_universe=df_uni,
        mode=mode, enable_rule5=True,
    )
    if entry is None:
        log.warning("스냅샷 결과 없음 — append skip (exit 0)")
        return 0

    log.info(
        "snapshot OK | date=%s NAV=%.1f억 (%+.2f%%) sat=%d trades=%d",
        entry['date'], entry['nav_krw'] / 1e8, entry['nav_pct'],
        entry['active_sat_count'], entry['n_trades_total'],
    )

    try:
        path, history = append_history_snapshot(entry)
    except Exception as e:
        log.error("history append 실패: %s", e)
        return 1
    log.info("history append OK — %s (%d entries)", path, len(history))

    ok, msg = push_history_to_github(history)
    if ok:
        log.info("GitHub push OK — %s", msg)
        return 0
    else:
        log.error("GitHub push 실패: %s", msg)
        return 1


if __name__ == '__main__':
    sys.exit(main())
