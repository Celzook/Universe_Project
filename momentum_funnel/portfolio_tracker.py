"""
Saved MP persistence + inception-anchored performance tracking.

저장: 로컬 파일 + (선택) GitHub Contents API 자동 커밋.
조회: 가장 최근 저장본의 편입일·포지션·비중 로드.
성과: 편입일 기준 forward 마일스톤 (+1D/+1W/+1M/+3M/YTD) KOSPI 초과수익률 계산.

GitHub 자동 커밋 설정 (선택):
  - 로컬: 환경변수 GITHUB_TOKEN
  - Streamlit Cloud: st.secrets['github_token']
  - 토큰 권한: contents:write (이 repo 한정 fine-grained 권장)
"""
from __future__ import annotations
import base64
import json
import os
from datetime import datetime
from typing import Callable, Optional, Tuple
import numpy as np
import pandas as pd


SAVED_MP_DIR = "saved_mps"
SAVED_MP_FILE = "current.json"
DEFAULT_REPO = "Celzook/Universe_Project"


# ──────────────────────────────────────────────────────────────────────
# 1. 직렬화 / 역직렬화
# ──────────────────────────────────────────────────────────────────────
def _saved_path() -> str:
    return os.path.join(SAVED_MP_DIR, SAVED_MP_FILE)


def save_mp_local(mp_df: pd.DataFrame, inception_date: str, method: str) -> Tuple[str, dict]:
    """
    Save MP to local file `saved_mps/current.json`.

    Parameters
    ----------
    mp_df : DataFrame
        build_mp() 출력. role/ticker/representative/category/hot_score/weight_pct.
    inception_date : str
        편입 기준일 (YYYY-MM-DD).
    method : str
        'A' or 'B' (Hot 선별 방법).

    Returns
    -------
    (path, payload_dict)
    """
    os.makedirs(SAVED_MP_DIR, exist_ok=True)
    payload = {
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'inception_date': str(inception_date),
        'method': str(method),
        'positions': mp_df.to_dict(orient='records'),
    }
    path = _saved_path()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path, payload


def load_mp() -> Optional[dict]:
    """Load saved MP.

    우선순위:
      1) 로컬 파일 `saved_mps/current.json`  (최근 save_mp_local 결과 / 캐시)
      2) GitHub raw URL fallback             (Streamlit Cloud 컨테이너 재배포 등
                                              로컬 ephemeral 디스크가 초기화되어도
                                              repo 에 push 된 MP 가 살아있으면 복원)

    GitHub fetch 성공 시 로컬에도 동일 내용을 즉시 캐싱해서 이후 호출은 O(1).
    """
    path = _saved_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass  # 손상된 파일이면 GitHub 폴백 시도

    data = _fetch_mp_from_github_raw()
    if data is None:
        return None
    # 로컬 복원 캐싱 (다음 호출부터 네트워크 미사용)
    try:
        os.makedirs(SAVED_MP_DIR, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data


def _fetch_mp_from_github_raw(branch: str = 'main',
                              repo: str = DEFAULT_REPO,
                              timeout: float = 10.0) -> Optional[dict]:
    """raw.githubusercontent.com 으로 `saved_mps/current.json` 직접 다운로드.

    공개 repo 면 토큰 불필요. 비공개 repo 라면 404/Unauthorized 로 fallback 실패 →
    호출 측에서 None 처리. Streamlit / requests 어느 쪽이든 사용 가능하도록
    requests 만 의존.
    """
    try:
        import requests
    except ImportError:
        return None
    url = (
        f"https://raw.githubusercontent.com/{repo}/{branch}/"
        f"{SAVED_MP_DIR}/{SAVED_MP_FILE}"
    )
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def delete_mp() -> bool:
    path = _saved_path()
    if os.path.exists(path):
        try:
            os.remove(path); return True
        except Exception:
            return False
    return False


# ──────────────────────────────────────────────────────────────────────
# 2. GitHub Contents API 자동 커밋
# ──────────────────────────────────────────────────────────────────────
def _get_github_token() -> Optional[str]:
    tok = os.environ.get('GITHUB_TOKEN')
    if tok:
        return tok
    try:
        import streamlit as st
        return st.secrets.get('github_token')
    except Exception:
        return None


def push_to_github(payload: dict,
                   repo: str = DEFAULT_REPO,
                   path: str = None,
                   message: str = None) -> Tuple[bool, str]:
    """
    GitHub REST API 로 saved_mps/current.json 업로드.
    토큰 없으면 (False, 'no token') 반환 — 호출 측에서 graceful 처리.
    """
    import requests

    token = _get_github_token()
    if not token:
        return False, 'no GITHUB_TOKEN / st.secrets.github_token configured'

    path = path or f"{SAVED_MP_DIR}/{SAVED_MP_FILE}"
    message = message or f"chore: update saved MP ({payload.get('inception_date','?')})"

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        'Authorization': f"Bearer {token}",
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }

    # 기존 파일 SHA 조회 (있으면 update, 없으면 create)
    sha = None
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            sha = r.json().get('sha')
    except Exception:
        pass

    content_str = json.dumps(payload, ensure_ascii=False, indent=2)
    body = {
        'message': message,
        'content': base64.b64encode(content_str.encode('utf-8')).decode('ascii'),
    }
    if sha:
        body['sha'] = sha

    try:
        r = requests.put(url, headers=headers, json=body, timeout=20)
        if r.status_code in (200, 201):
            commit_sha = r.json().get('commit', {}).get('sha', '')[:7]
            return True, f"커밋 {commit_sha} push 성공"
        return False, f"{r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"네트워크 에러: {e}"


# ──────────────────────────────────────────────────────────────────────
# 3. 성과 계산 (편입일 기준 forward + KOSPI 초과)
# ──────────────────────────────────────────────────────────────────────
WINDOWS_BDAYS = [('+1D', 1), ('+1W', 5), ('+1M', 21), ('+3M', 63)]


def _parse_date(s) -> pd.Timestamp:
    """'YYYY-MM-DD' / 'YYYYMMDD' / Timestamp → Timestamp."""
    if isinstance(s, pd.Timestamp):
        return s.normalize()
    if isinstance(s, datetime):
        return pd.Timestamp(s).normalize()
    s = str(s).replace('-', '').replace('/', '').replace(' ', '')[:8]
    return pd.Timestamp(s)


def _ret_at(close: pd.Series, anchor_price: float, target_date: pd.Timestamp) -> float:
    """target_date 시점 close → anchor 기준 % 수익률. NaN if target > 데이터 끝."""
    if close.empty:
        return np.nan
    pos = close.index.searchsorted(target_date)
    if pos >= len(close):
        return np.nan
    px = float(close.iloc[pos])
    if anchor_price == 0 or np.isnan(px):
        return np.nan
    return (px / anchor_price - 1.0) * 100.0


def compute_mp_performance(
    saved: dict,
    fetch_close: Callable[[str, str, str], pd.Series],
    fetch_kospi: Callable[[str, str], pd.Series],
    today: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    편입일(inception_date) 기준 forward 마일스톤 + 누적 수익률 (KOSPI 초과).

    fetch_close(ticker, start_str, end_str) -> Series(close, DatetimeIndex)
    fetch_kospi(start_str, end_str)         -> Series(close, DatetimeIndex)

    Returns
    -------
    DataFrame 컬럼:
      역할, 티커, 대표 ETF, 카테고리, 편입일, 시작비중%, 현재비중%, 누적%, +1D, +1W, +1M, +3M, YTD
    수익률 단위 모두 % (KOSPI 초과). 윈도우 미도달 셀은 NaN.
    """
    today = (today or pd.Timestamp.today()).normalize()
    inception = _parse_date(saved.get('inception_date'))
    start_str = (inception - pd.Timedelta(days=10)).strftime('%Y%m%d')
    end_str = today.strftime('%Y%m%d')

    # KOSPI 1회 fetch
    kospi = fetch_kospi(start_str, end_str)
    if not isinstance(kospi, pd.Series):
        kospi = pd.Series(dtype=float)
    kospi = kospi.dropna().sort_index()

    # KOSPI anchor (편입일 이후 첫 거래일)
    k_after = kospi.index[kospi.index >= inception]
    if len(k_after) == 0 or kospi.empty:
        k_anchor_date = inception
        k_anchor_px = np.nan
        k_latest_px = np.nan
    else:
        k_anchor_date = k_after[0]
        k_anchor_px = float(kospi.loc[k_anchor_date])
        k_latest_px = float(kospi.iloc[-1])

    def kospi_ret_at(d: pd.Timestamp) -> float:
        return _ret_at(kospi, k_anchor_px, d) if not np.isnan(k_anchor_px) else np.nan

    # YTD 앵커 (편입일 vs Jan 1 of today.year 중 늦은 쪽)
    jan1 = pd.Timestamp(today.year, 1, 1)
    ytd_anchor_inception = max(inception, jan1)

    rows = []
    raw_returns = []

    for pos in saved.get('positions', []):
        ticker = str(pos.get('ticker', ''))
        close = fetch_close(ticker, start_str, end_str)
        if not isinstance(close, pd.Series):
            close = pd.Series(dtype=float)
        close = close.dropna().sort_index()

        # 편입일 이후 첫 거래일 anchor
        valid_idx = close.index[close.index >= inception]
        if len(valid_idx) == 0 or close.empty:
            rows.append({
                '역할': pos.get('role', ''),
                '티커': ticker,
                '대표 ETF': pos.get('representative', ''),
                '카테고리': pos.get('category', ''),
                '편입일': inception.strftime('%Y-%m-%d'),
                '시작비중%': float(pos.get('weight_pct', 0.0)),
                '현재비중%': np.nan,
                '누적%': np.nan,
                **{lbl: np.nan for lbl, _ in WINDOWS_BDAYS},
                'YTD': np.nan,
            })
            raw_returns.append(0.0)
            continue

        anchor_date = valid_idx[0]
        anchor_px = float(close.loc[anchor_date])
        latest_date = close.index[-1]
        latest_px = float(close.iloc[-1])

        # 누적 (KOSPI 초과)
        raw_cum = (latest_px / anchor_px - 1.0) * 100.0
        k_cum = kospi_ret_at(latest_date)
        excess_cum = raw_cum - k_cum if not np.isnan(k_cum) else raw_cum
        raw_returns.append(raw_cum / 100.0)

        # Forward 마일스톤
        forward = {}
        for label, n_bd in WINDOWS_BDAYS:
            tgt = anchor_date + pd.tseries.offsets.BDay(n_bd)
            if tgt > latest_date:
                forward[label] = np.nan
                continue
            etf_ret = _ret_at(close, anchor_px, tgt)
            k_ret = kospi_ret_at(tgt)
            forward[label] = etf_ret - k_ret if not (np.isnan(etf_ret) or np.isnan(k_ret)) else np.nan

        # YTD (편입일 vs Jan 1 중 늦은 쪽)
        ytd_anchor = max(anchor_date, jan1)
        close_after_ytd = close.index[close.index >= ytd_anchor]
        if len(close_after_ytd) == 0:
            ytd = np.nan
        else:
            ytd_first = close_after_ytd[0]
            ytd_first_px = float(close.loc[ytd_first])
            etf_ytd = (latest_px / ytd_first_px - 1.0) * 100.0 if ytd_first_px else np.nan
            # KOSPI YTD
            k_after_ytd = kospi.index[kospi.index >= ytd_anchor]
            if len(k_after_ytd) == 0 or kospi.empty:
                kospi_ytd = np.nan
            else:
                kospi_ytd_first = float(kospi.loc[k_after_ytd[0]])
                kospi_ytd = (k_latest_px / kospi_ytd_first - 1.0) * 100.0 if kospi_ytd_first else np.nan
            ytd = etf_ytd - kospi_ytd if not (np.isnan(etf_ytd) or np.isnan(kospi_ytd)) else etf_ytd

        rows.append({
            '역할': pos.get('role', ''),
            '티커': ticker,
            '대표 ETF': pos.get('representative', ''),
            '카테고리': pos.get('category', ''),
            '편입일': anchor_date.strftime('%Y-%m-%d'),
            '시작비중%': float(pos.get('weight_pct', 0.0)),
            '현재비중%': np.nan,  # 다음 단계에서 채움
            '누적%': excess_cum,
            **forward,
            'YTD': ytd,
        })

    df = pd.DataFrame(rows)

    # 드리프트 비중: start_weight × (1 + raw_return) → 100% 정규화
    if len(df) == len(raw_returns) and len(df) > 0:
        starts = df['시작비중%'].astype(float).values
        growth = 1.0 + np.array(raw_returns, dtype=float)
        drifted = starts * growth
        total = float(drifted.sum())
        if total > 0:
            df['현재비중%'] = drifted * 100.0 / total
        else:
            df['현재비중%'] = starts

    return df
