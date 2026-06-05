"""
momentum_funnel/tracks.py
세 가지 필터 트랙과 combine() 함수.

Track A — 와이드 스크린 (느슨, 모니터링 유니버스)
Track B — 직렬 깔때기 (엄격, 진입 후보)
Track C — 연속 스코어 가중합 (배분 근거)
combine() — A/B/C 통합 → Decision
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .config import FunnelConfig
from .contracts import Decision


# ── Track A ─────────────────────────────────────────────────────────────────

def track_a_wide(metrics: pd.DataFrame, cfg: FunnelConfig) -> pd.DataFrame:
    """
    와이드 스크린(느슨, 모니터링 유니버스).

    Filter: valid=True AND (adx >= a_adx_min OR rs >= a_rs_min).
    Returns subset of metrics sorted by ADX desc.
    """
    valid = metrics[metrics["valid"] == True].copy()  # noqa: E712
    mask = (valid["adx"] >= cfg.a_adx_min) | (valid["rs"] >= cfg.a_rs_min)
    result = valid[mask].sort_values("adx", ascending=False)
    return result


# ── Track B ─────────────────────────────────────────────────────────────────

def track_b_funnel(metrics: pd.DataFrame, cfg: FunnelConfig) -> dict[str, pd.DataFrame]:
    """
    직렬 깔때기(엄격).

    Step 1 — RS > 0
    Step 2 — ADX >= b_adx_min (optionally rising if b_require_rising_adx=True)
    Step 3 — MFI in [b_mfi_lower, b_mfi_upper)

    valid=False 행은 시작부터 제외.
    Returns dict keyed by stage name for traceability.
    """
    # valid 필터
    pool = metrics[metrics["valid"] == True].copy()  # noqa: E712

    # Step 1: RS > 0
    step1 = pool[pool["rs"] > 0]

    # Step 2: ADX >= b_adx_min, optional rising
    adx_mask = step1["adx"] >= cfg.b_adx_min
    if cfg.b_require_rising_adx:
        adx_mask = adx_mask & (step1["adx"] > step1["adx_prev"])
    step2 = step1[adx_mask]

    # Step 3: MFI in [b_mfi_lower, b_mfi_upper)
    step3 = step2[
        (step2["mfi"] >= cfg.b_mfi_lower) & (step2["mfi"] < cfg.b_mfi_upper)
    ]

    return {
        "step1_rs": step1,
        "step2_adx": step2,
        "step3_mfi": step3,
    }


# ── Track C ─────────────────────────────────────────────────────────────────

def _sigmoid(x: "pd.Series | np.ndarray") -> "pd.Series | np.ndarray":
    """요소별 시그모이드. 오버플로 방지."""
    return 1.0 / (1.0 + np.exp(-x))


def track_c_score(metrics: pd.DataFrame, cfg: FunnelConfig) -> pd.DataFrame:
    """
    연속 스코어 [0,1] 가중합.

    rs_norm        : percentile rank of rs within valid universe
    adx_norm       : clip(adx/40, 0, 1)
    adx_rising_norm: sigmoid((adx - adx_prev) / 2)  — equal→0.5, +5→~0.92, -5→~0.08
    mfi_norm       : gaussian sweet-spot centered at 65, sigma=15
                     exp(-((mfi-65)/15)^2)
    score = w_rs*rs_norm + w_adx*adx_norm + w_adx_rising*adx_rising_norm + w_mfi*mfi_norm

    Returns DataFrame indexed by sector with columns:
    ['rs_norm','adx_norm','adx_rising_norm','mfi_norm','score'], sorted by score desc.
    Only valid=True rows.
    """
    valid = metrics[metrics["valid"] == True].copy()  # noqa: E712

    if valid.empty:
        cols = ["rs_norm", "adx_norm", "adx_rising_norm", "mfi_norm", "score"]
        return pd.DataFrame(columns=cols)

    # rs percentile rank (0~1)
    rs_norm = valid["rs"].rank(pct=True)

    # adx 정규화: clip(adx/40, 0, 1)
    adx_norm = valid["adx"].div(40.0).clip(0.0, 1.0)

    # adx 상승 시그모이드
    adx_rising_norm = _sigmoid((valid["adx"] - valid["adx_prev"]) / 2.0)

    # mfi gaussian sweet-spot (center=65, sigma=15)
    mfi_norm = np.exp(-((valid["mfi"] - 65.0) / 15.0) ** 2)

    score = (
        cfg.w_rs * rs_norm
        + cfg.w_adx * adx_norm
        + cfg.w_adx_rising * adx_rising_norm
        + cfg.w_mfi * mfi_norm
    )

    result = pd.DataFrame(
        {
            "rs_norm": rs_norm,
            "adx_norm": adx_norm,
            "adx_rising_norm": adx_rising_norm,
            "mfi_norm": mfi_norm,
            "score": score,
        },
        index=valid.index,
    )
    return result.sort_values("score", ascending=False)


# ── combine ──────────────────────────────────────────────────────────────────

def combine(
    metrics: pd.DataFrame,
    cfg: FunnelConfig,
    asof: Optional[pd.Timestamp] = None,
) -> Decision:
    """
    진입은 좁게(B), 청산은 넓게(A), 배분은 점수로(C).

    - monitor    = track_a_wide (모니터링 유니버스)
    - score_df   = track_c_score (전체 랭킹)
    - entry      = track_b_funnel['step3_mfi']
    - Fallback   : entry 비면 score >= c_fallback_min_score top-5
    - weights    : entry 종목 score 비례, 합=1
    """
    monitor = track_a_wide(metrics, cfg)
    score_df = track_c_score(metrics, cfg)
    b_result = track_b_funnel(metrics, cfg)
    b_entry = b_result["step3_mfi"]

    # entry에 score 컬럼 합류 (UI에서 정렬 가능)
    if not b_entry.empty:
        used_fallback = False
        entry = b_entry.join(score_df[["score"]], how="left")
    else:
        used_fallback = True
        fallback_candidates = score_df[score_df["score"] >= cfg.c_fallback_min_score]
        top5 = fallback_candidates.head(5)
        # entry는 원본 metrics 컬럼 + score 포함 형태로 반환
        if not top5.empty:
            entry = metrics.loc[metrics.index.intersection(top5.index)].join(
                top5[["score"]], how="inner"
            )
        else:
            entry = pd.DataFrame(columns=list(metrics.columns) + ["score"])

    # weights: entry 종목의 score 비례, 합=1
    if entry.empty or "score" not in entry.columns:
        weights = pd.Series(dtype=float)
    else:
        raw = entry["score"].fillna(0.0)
        total = raw.sum()
        if total == 0.0:
            weights = pd.Series(dtype=float)
        else:
            weights = raw / total

    return Decision(
        monitor=monitor,
        entry=entry,
        score=score_df,
        weights=weights,
        used_fallback=used_fallback,
        asof=asof,
    )
