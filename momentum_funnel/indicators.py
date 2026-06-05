"""
momentum_funnel/indicators.py

순수 pandas/numpy 구현 (외부 ta 라이브러리 미사용).
Wilder 스무딩(alpha=1/period) 직접 구현.
룩-어헤드 방지: 모든 함수에서 asof 이전 슬라이스만 사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .contracts import MarketData, METRICS_COLS
from .config import FunnelConfig


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder 스무딩 (EMA with alpha=1/period).
    첫 번째 유효값 = 단순 평균(SMA) 시드 → 이후 재귀 적용.
    """
    n = len(series)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # SMA 시드
    result[period - 1] = np.mean(series[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        result[i] = result[i - 1] * (1 - alpha) + series[i] * alpha
    return result


# ── 공개 함수 ─────────────────────────────────────────────────────────────────

def compute_rs(
    close: pd.Series,
    bench_close: pd.Series,
    cfg: FunnelConfig,
    asof: Optional[pd.Timestamp] = None,
) -> float:
    """
    Relative Strength.

    slope 모드: log(RS) 시계열의 선형회귀 기울기 (rs_window 구간).
      RS_t = close_t / bench_close_t (인덱스 교집합 정렬 후 계산).
    disparity 모드: (close[-1] - MA(close, rs_ma_window)) / MA * 100 (%).

    데이터 부족 시 NaN 반환.
    """
    # ── asof 슬라이싱 (룩-어헤드 방지) ──────────────────────
    if asof is not None:
        close = close[:asof]
        bench_close = bench_close[:asof]

    if cfg.rs_method == "slope":
        # 인덱스 교집합으로 정렬 (ffill 금지)
        common = close.index.intersection(bench_close.index)
        if len(common) < cfg.rs_window:
            return float("nan")
        c = close.loc[common].iloc[-cfg.rs_window:]
        b = bench_close.loc[common].iloc[-cfg.rs_window:]
        rs_series = c.values / b.values
        log_rs = np.log(rs_series)
        n = len(log_rs)
        x = np.arange(n, dtype=float)
        # 선형회귀 기울기
        x_mean = x.mean()
        slope = float(np.sum((x - x_mean) * (log_rs - log_rs.mean())) /
                      np.sum((x - x_mean) ** 2))
        return slope

    elif cfg.rs_method == "disparity":
        if len(close) < cfg.rs_ma_window:
            return float("nan")
        ma = close.iloc[-cfg.rs_ma_window:].mean()
        if ma == 0:
            return float("nan")
        return float((close.iloc[-1] - ma) / ma * 100)

    else:
        raise ValueError(f"Unknown rs_method: {cfg.rs_method!r}")


def compute_adx(
    df: pd.DataFrame,
    period: int,
    asof: Optional[pd.Timestamp] = None,
) -> tuple[float, float]:
    """
    ADX (Wilder 스무딩).

    Returns (adx_today, adx_yesterday).
    2*period 미만 바 → (NaN, NaN).

    df 컬럼: high, low, close (소문자, OHLCV_COLS 계약).
    """
    if asof is not None:
        df = df[:asof]

    if len(df) < 2 * period:
        return (float("nan"), float("nan"))

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    tr = np.empty(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        h, l, pc = high[i], low[i], close[i - 1]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Wilder 스무딩 (ATR, +DMI, -DMI)
    atr = _wilder_smooth(tr, period)
    plus_dmi_smooth = _wilder_smooth(plus_dm, period)
    minus_dmi_smooth = _wilder_smooth(minus_dm, period)

    # +DI, -DI
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(atr > 0, 100.0 * plus_dmi_smooth / atr, 0.0)
        minus_di = np.where(atr > 0, 100.0 * minus_dmi_smooth / atr, 0.0)
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)

    # NaN 마스크 복원 (Wilder 워밍업 전 구간)
    dx[np.isnan(atr)] = np.nan

    # DX의 첫 유효값(index=period-1)부터 Wilder 스무딩 → ADX
    # _wilder_smooth는 입력 배열 내 index=period-1 위치에 첫 값 생성.
    # dx[period-1:] 길이 = n-(period-1). 그 배열에서 index=period-1이 adx의 첫 유효값.
    # → adx_full에서의 절대 위치 = (period-1) + (period-1) = 2*(period-1)
    dx_slice = dx[period - 1:]          # 길이 = n - (period-1)
    adx_slice = _wilder_smooth(dx_slice, period)
    adx_full = np.full(n, np.nan)
    base = period - 1                   # dx_slice의 시작이 원본 배열의 이 위치
    for i, v in enumerate(adx_slice):
        adx_full[base + i] = v

    # 마지막 2개 값
    valid_adx = adx_full[~np.isnan(adx_full)]
    if len(valid_adx) < 2:
        return (float("nan"), float("nan"))

    return (float(valid_adx[-1]), float(valid_adx[-2]))


def compute_mfi(
    df: pd.DataFrame,
    period: int,
    asof: Optional[pd.Timestamp] = None,
) -> float:
    """
    Money Flow Index (MFI).

    TP = (high + low + close) / 3
    money_flow = TP * volume
    positive_flow = sum(MF where TP > TP_prev)  over period bars
    negative_flow = sum(MF where TP < TP_prev)  over period bars
    MFI = 100 - 100 / (1 + positive_flow / negative_flow)

    period+1 미만 바 → NaN.
    """
    if asof is not None:
        df = df[:asof]

    if len(df) < period + 1:
        return float("nan")

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]

    # 최근 period+1개 행 사용 (비교에 period 구간 필요)
    tp_window = tp.iloc[-(period + 1):]
    mf_window = mf.iloc[-(period + 1):]

    tp_vals = tp_window.values
    mf_vals = mf_window.values

    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(1, len(tp_vals)):
        if tp_vals[i] > tp_vals[i - 1]:
            pos_flow += mf_vals[i]
        elif tp_vals[i] < tp_vals[i - 1]:
            neg_flow += mf_vals[i]
        # TP 변화 없으면 양쪽 모두 불포함 (표준 정의)

    if neg_flow == 0:
        # 음수 흐름 없음 → 완전 강세 → MFI=100
        return 100.0 if pos_flow > 0 else 50.0

    mfi = 100.0 - 100.0 / (1.0 + pos_flow / neg_flow)
    return float(np.clip(mfi, 0.0, 100.0))


def compute_metrics(
    market: MarketData,
    cfg: FunnelConfig,
    asof: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    모든 섹터에 대해 RS/ADX/MFI 계산 → MetricsFrame(METRICS_COLS) 반환.

    valid = (충분한 바 수 AND 개별 지표 모두 NaN 아님).
    invalid 섹터도 행은 포함하고 지표는 NaN.
    """
    bench_close = market.benchmark["close"]
    if asof is not None:
        bench_close = bench_close[:asof]

    rows = {}
    for sector, df in market.sector_data.items():
        # asof 슬라이싱 (각 함수 내부에서도 수행하지만 bar 수 확인에도 필요)
        if asof is not None:
            df_slice = df[:asof]
        else:
            df_slice = df

        n_bars = len(df_slice)
        enough_bars = n_bars >= cfg.min_bars

        if not enough_bars:
            rows[sector] = {
                "valid": False,
                "rs": float("nan"),
                "adx": float("nan"),
                "adx_prev": float("nan"),
                "mfi": float("nan"),
            }
            continue

        rs = compute_rs(df_slice["close"], bench_close, cfg, asof=None)
        adx_today, adx_prev = compute_adx(df_slice, cfg.adx_period, asof=None)
        mfi = compute_mfi(df_slice, cfg.mfi_period, asof=None)

        rs_ok = not np.isnan(rs)
        adx_ok = not np.isnan(adx_today)
        mfi_ok = not np.isnan(mfi)
        valid = bool(enough_bars and rs_ok and adx_ok and mfi_ok)

        rows[sector] = {
            "valid": valid,
            "rs": rs if rs_ok else float("nan"),
            "adx": adx_today if adx_ok else float("nan"),
            "adx_prev": adx_prev if adx_ok else float("nan"),
            "mfi": mfi if mfi_ok else float("nan"),
        }

    result = pd.DataFrame.from_dict(rows, orient="index", columns=METRICS_COLS)
    result.index.name = "sector"
    return result
