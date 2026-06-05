"""
FunnelConfig — 모든 임계점의 단일 진실 공급원. 매직넘버 금지.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FunnelConfig:
    # ── 공통 ────────────────────────────────────────────────
    rs_window: int = 20
    rs_method: str = "slope"        # "slope"(log-RS 회귀) | "disparity"(MA 이격도%)
    rs_ma_window: int = 20
    adx_period: int = 14
    mfi_period: int = 14

    # ── Track A (와이드 스크린, 느슨) ──────────────────────
    a_adx_min: float = 20.0
    a_rs_min: float = 0.0           # OR 조건

    # ── Track B (직렬 깔때기, 엄격) ────────────────────────
    b_adx_min: float = 20.0
    b_require_rising_adx: bool = True
    b_mfi_lower: float = 50.0
    b_mfi_upper: float = 80.0

    # ── Track C (스코어 가중치, 합=1.0 권장) ───────────────
    w_rs: float = 0.35
    w_adx: float = 0.30
    w_adx_rising: float = 0.15
    w_mfi: float = 0.20
    c_fallback_min_score: float = 0.70  # B가 빌 때 C 폴백 하한

    # ── Hot Board ──────────────────────────────────────────
    hot_vol_lookback: int = 20      # VolRatio 기간 평균 (영업일)
    hot_money_lookback: int = 5     # MoneyFlow 누적 기간 (영업일)
    hot_top_n: int = 5              # MP 새틀라이트 개수

    # 워밍업: 자동 계산 (init=False)
    min_bars: int = field(init=False)

    def __post_init__(self) -> None:
        self.min_bars = max(
            self.adx_period * 2 + 5,
            self.mfi_period + 5,
            self.rs_window + 5,
        )

    # 편의 메서드 ─────────────────────────────────────────
    def weight_sum(self) -> float:
        return self.w_rs + self.w_adx + self.w_adx_rising + self.w_mfi
