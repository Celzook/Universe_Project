"""
ETF Scoring Engine — Technical + Macro + Fusion (C_score)

Technical Signal Engine:
  - Bollinger Bands %B, RSI (Wilder), Momentum (ROC percentile rank)
  - Band Walk detection, dynamic weighting → T_score

Macro Score Engine:
  - VIX (z-score), WTI (52w z-score), Fed futures (ZQ=F), BEI proxy (^TNX), GPR=0
  - → M_score

Signal Fusion:
  - damp = clip(0.75 + M_norm*0.45, 0.30, 1.20)
  - C_score = clip(T_score * damp + M_norm*28, -100, 100)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st


# ─── Technical Signal Engine ───────────────────────────────────────────────

def compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Returns (pb, upper, lower, mid). pb clipped to [-0.2, 1.2]."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pb = (close - lower) / (upper - lower)
    pb = pb.clip(-0.2, 1.2)
    return pb, upper, lower, mid


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)


def compute_momentum_normalized(close: pd.Series, roc_period: int = 5,
                                roll_window: int = 252) -> pd.Series:
    """ROC → 252-day rolling percentile rank → normalized to [-3, +3]."""
    roc = (close - close.shift(roc_period)) / close.shift(roc_period) * 100

    def _rank(x):
        if len(x) < 2 or np.isnan(x[-1]):
            return np.nan
        return (x[:-1] < x[-1]).sum() / (len(x) - 1)

    pct = roc.rolling(roll_window, min_periods=max(20, roll_window // 4)).apply(_rank, raw=True)
    return (pct * 6 - 3).clip(-3, 3)


def _score_pb(pb: float) -> float:
    d = pb - 0.5
    return float(np.clip(-np.sign(d) * (abs(d) * 2) ** 1.25 * 100, -100, 100))


def _score_rsi(rsi: float) -> float:
    d = rsi - 50
    return float(np.clip(-np.sign(d) * (abs(d) / 50) ** 0.85 * 100, -100, 100))


def _score_mom(mom_normalized: float) -> float:
    return float(np.clip(mom_normalized / 3 * 100, -100, 100))


def compute_T_score(close: pd.Series) -> float:
    """Compute latest T_score [-100, +100] from a Close price series."""
    clean = close.dropna()
    if len(clean) < 30:
        return np.nan

    pb_s, _, _, _ = compute_bollinger(clean)
    rsi_s = compute_rsi(clean)
    mom_s = compute_momentum_normalized(clean)

    latest_pb  = pb_s.iloc[-1]
    latest_rsi = rsi_s.iloc[-1]
    latest_mom = mom_s.iloc[-1]

    if any(np.isnan(v) for v in [latest_pb, latest_rsi, latest_mom]):
        return np.nan

    # Band Walk: %B >= 1.0 AND RSI 50~75 AND momentum_norm >= 1
    band_walk = (latest_pb >= 1.0) and (50 <= latest_rsi <= 75) and (latest_mom >= 1)

    spb  = _score_pb(latest_pb)
    srsi = _score_rsi(latest_rsi)
    smom = _score_mom(latest_mom)

    if band_walk:
        w_pb, w_rsi, w_mom = 0.18, 0.35, 0.47
    else:
        w_pb, w_rsi, w_mom = 0.45, 0.40, 0.15

    T = spb * w_pb + srsi * w_rsi + smom * w_mom
    return float(np.clip(T, -100, 100))


# ─── Macro Score Engine ────────────────────────────────────────────────────

@st.cache_data(ttl=3600 * 4, show_spinner=False)
def fetch_macro_scores() -> dict:
    """
    Fetch macro indicators via yfinance and compute M_score.
    Falls back gracefully (score=0) if any data is unavailable.

    M_score = clip((fed*-15)+(geo*-7)+(oil*-25)+(bei*10)+(vix*-20), -100, 100)
    """
    end = datetime.today()
    scores = {'fed': 0.0, 'geo': 0.0, 'oil': 0.0, 'bei': 0.0, 'vix': 0.0}
    details = {}

    # VIX → 252-day z-score → [-3, +3]
    try:
        vix_df = yf.download('^VIX', start=end - timedelta(days=365), end=end,
                             progress=False, auto_adjust=True)
        if not vix_df.empty:
            vix_close = vix_df['Close'].squeeze().dropna()
            vix_latest = float(vix_close.iloc[-1])
            vix_std = float(vix_close.std())
            if vix_std > 0:
                vix_z = (vix_latest - float(vix_close.mean())) / vix_std
            else:
                vix_z = 0.0
            scores['vix'] = float(np.clip(vix_z, -3, 3))
            details['VIX'] = round(vix_latest, 2)
    except Exception:
        pass

    # WTI (CL=F) → 52-week z-score → [-3, +3]
    try:
        wti_df = yf.download('CL=F', start=end - timedelta(days=400), end=end,
                             progress=False, auto_adjust=True)
        if not wti_df.empty:
            wti_close = wti_df['Close'].squeeze().dropna()
            wti_latest = float(wti_close.iloc[-1])
            wti_52 = wti_close.tail(252)
            wti_std = float(wti_52.std())
            if wti_std > 0:
                wti_z = (wti_latest - float(wti_52.mean())) / wti_std
            else:
                wti_z = 0.0
            scores['oil'] = float(np.clip(wti_z, -3, 3))
            details['WTI'] = round(wti_latest, 2)
    except Exception:
        pass

    # BEI proxy: 10Y Treasury (^TNX) / 4 → rough inflation expectations [-3, +3]
    try:
        tnx_df = yf.download('^TNX', start=end - timedelta(days=90), end=end,
                             progress=False, auto_adjust=True)
        if not tnx_df.empty:
            tnx = tnx_df['Close'].squeeze().dropna()
            bei_val = float(tnx.iloc[-1]) / 4.0
            scores['bei'] = float(np.clip(bei_val, -3, 3))
            details['10Y_TNX'] = round(float(tnx.iloc[-1]), 3)
    except Exception:
        pass

    # Fed Rate proxy: ZQ=F (30-day Fed funds futures), implied_rate = 100 - price
    try:
        ff_df = yf.download('ZQ=F', start=end - timedelta(days=60), end=end,
                            progress=False, auto_adjust=True)
        if not ff_df.empty:
            ff = ff_df['Close'].squeeze().dropna()
            implied_rate = 100 - float(ff.iloc[-1])
            fed_score = (implied_rate - 5.0) / 2.0   # centred at 5%, ±1.5 per 1%
            scores['fed'] = float(np.clip(fed_score, -3, 3))
            details['Fed_implied_rate'] = round(implied_rate, 3)
    except Exception:
        pass

    # GPR: not available via yfinance → neutral (0)
    scores['geo'] = 0.0
    details['GPR'] = 'N/A (neutral=0)'

    fed, geo, oil, bei, vix = (scores[k] for k in ('fed', 'geo', 'oil', 'bei', 'vix'))
    M = (fed * -15) + (geo * -7) + (oil * -25) + (bei * 10) + (vix * -20)
    M_score = float(np.clip(M, -100, 100))

    return {'M_score': M_score, 'scores': scores, 'details': details}


# ─── Signal Fusion (C_score) ───────────────────────────────────────────────

def compute_C_score(T_score: float, M_score: float) -> float:
    """Fuse T_score and M_score → final C_score [-100, +100]."""
    if np.isnan(T_score):
        return np.nan
    M_norm = M_score / 100.0
    damp = float(np.clip(0.75 + M_norm * 0.45, 0.30, 1.20))
    macro_contrib = M_norm * 28
    return float(np.clip(T_score * damp + macro_contrib, -100, 100))


# ─── OHLCV fetchers ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600 * 2, show_spinner=False)
def fetch_kr_ohlcv_batch(tickers: tuple, months: int = 6) -> dict:
    """
    Batch-fetch OHLCV for Korean ETFs via yfinance (appends .KS suffix).
    Returns dict {original_ticker: DataFrame(Open,High,Low,Close,Volume)}.
    """
    kr_tickers = [f"{t}.KS" for t in tickers]
    ticker_map = {f"{t}.KS": t for t in tickers}
    end = datetime.today()
    start = end - timedelta(days=months * 31)
    result = {}
    try:
        raw = yf.download(kr_tickers, start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return result
        for kr_t in kr_tickers:
            orig = ticker_map[kr_t]
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl1 = raw.columns.get_level_values(1)
                    if kr_t in lvl1:
                        df_t = raw.xs(kr_t, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    df_t = raw.copy()
                df_t.index = pd.to_datetime(df_t.index)
                if not df_t.empty and 'Close' in df_t.columns:
                    result[orig] = df_t
            except Exception:
                pass
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600 * 2, show_spinner=False)
def score_all_kr_etfs(all_tickers: tuple) -> pd.DataFrame:
    """Score Korean ETFs using .KS yfinance data."""
    macro = fetch_macro_scores()
    M_score = macro['M_score']
    ohlcv_dict = fetch_kr_ohlcv_batch(all_tickers, months=6)
    rows = []
    for ticker in all_tickers:
        df_t = ohlcv_dict.get(ticker, pd.DataFrame())
        if df_t.empty or 'Close' not in df_t.columns:
            rows.append({'ticker': ticker, 'T_score': np.nan, 'C_score': np.nan})
            continue
        close = df_t['Close'].squeeze().dropna()
        T = compute_T_score(close)
        C = compute_C_score(T, M_score)
        rows.append({
            'ticker': ticker,
            'T_score': round(T, 2) if not np.isnan(T) else np.nan,
            'C_score': round(C, 2) if not np.isnan(C) else np.nan,
        })
    return pd.DataFrame(rows).set_index('ticker')


@st.cache_data(ttl=3600 * 2, show_spinner=False)
def fetch_ohlcv_batch(tickers: tuple, months: int = 3) -> dict:
    """
    Batch-fetch OHLCV for multiple tickers via one yfinance call.
    Returns dict {ticker: DataFrame(Open,High,Low,Close,Volume)}.
    """
    end = datetime.today()
    start = end - timedelta(days=months * 31)
    result = {}
    try:
        raw = yf.download(list(tickers), start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return result
        for ticker in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl1 = raw.columns.get_level_values(1)
                    if ticker in lvl1:
                        df_t = raw.xs(ticker, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    df_t = raw.copy()
                df_t.index = pd.to_datetime(df_t.index)
                if not df_t.empty:
                    result[ticker] = df_t
            except Exception:
                pass
    except Exception:
        pass
    return result


# ─── Batch scoring ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600 * 2, show_spinner=False)
def score_all_etfs(all_tickers: tuple) -> pd.DataFrame:
    """
    Score every ticker in all_tickers.
    Returns DataFrame indexed by ticker with columns [T_score, C_score].
    """
    macro = fetch_macro_scores()
    M_score = macro['M_score']

    ohlcv_dict = fetch_ohlcv_batch(all_tickers, months=6)

    rows = []
    for ticker in all_tickers:
        df_t = ohlcv_dict.get(ticker, pd.DataFrame())
        if df_t.empty or 'Close' not in df_t.columns:
            rows.append({'ticker': ticker, 'T_score': np.nan, 'C_score': np.nan})
            continue
        close = df_t['Close'].squeeze().dropna()
        T = compute_T_score(close)
        C = compute_C_score(T, M_score)
        rows.append({
            'ticker': ticker,
            'T_score': round(T, 2) if not np.isnan(T) else np.nan,
            'C_score': round(C, 2) if not np.isnan(C) else np.nan,
        })
    return pd.DataFrame(rows).set_index('ticker')
