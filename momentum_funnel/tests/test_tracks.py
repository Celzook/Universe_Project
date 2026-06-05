"""
momentum_funnel/tests/test_tracks.py
Track A/B/C 및 combine() 단위 테스트.

픽스처: make_metrics() (fixtures.py) 기반 합성 데이터 사용.
모든 테스트는 상류 indicators/data_adapter 없이 독립 실행 가능.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ..config import FunnelConfig
from ..contracts import METRICS_COLS
from .fixtures import make_metrics
from ..tracks import combine, track_a_wide, track_b_funnel, track_c_score


# ─── 헬퍼 ────────────────────────────────────────────────────────────────────

def _make_controlled(rows: list[dict]) -> pd.DataFrame:
    """테스트용 완전 제어 메트릭 프레임 생성."""
    sectors = [r["sector"] for r in rows]
    df = pd.DataFrame(rows).set_index("sector")
    # METRICS_COLS 순서 보장
    for col in METRICS_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df[METRICS_COLS]


# ─── Track A 테스트 ───────────────────────────────────────────────────────────

class TestTrackAWide:
    def test_passes_adx_condition(self):
        """ADX >= a_adx_min 이면 통과."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "A", "valid": True, "rs": -0.01, "adx": 25.0, "adx_prev": 23.0, "mfi": 40.0},
            {"sector": "B", "valid": True, "rs": -0.01, "adx": 10.0, "adx_prev": 9.0,  "mfi": 40.0},
        ])
        result = track_a_wide(df, cfg)
        assert "A" in result.index
        assert "B" not in result.index

    def test_passes_rs_condition(self):
        """RS >= a_rs_min(0.0)이면 ADX 조건 없어도 통과."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "C", "valid": True, "rs": 0.001, "adx": 5.0, "adx_prev": 4.0, "mfi": 40.0},
        ])
        result = track_a_wide(df, cfg)
        assert "C" in result.index

    def test_excludes_invalid_rows(self):
        """valid=False 행은 ADX/RS 조건 충족해도 제외."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "X", "valid": False, "rs": 0.01, "adx": 30.0, "adx_prev": 28.0, "mfi": 60.0},
            {"sector": "Y", "valid": True,  "rs": 0.01, "adx": 30.0, "adx_prev": 28.0, "mfi": 60.0},
        ])
        result = track_a_wide(df, cfg)
        assert "X" not in result.index
        assert "Y" in result.index

    def test_sorted_by_adx_desc(self):
        """결과는 ADX 내림차순 정렬."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "Low",  "valid": True, "rs": 0.01, "adx": 20.0, "adx_prev": 19.0, "mfi": 60.0},
            {"sector": "High", "valid": True, "rs": 0.01, "adx": 35.0, "adx_prev": 33.0, "mfi": 60.0},
            {"sector": "Mid",  "valid": True, "rs": 0.01, "adx": 28.0, "adx_prev": 26.0, "mfi": 60.0},
        ])
        result = track_a_wide(df, cfg)
        adx_vals = result["adx"].tolist()
        assert adx_vals == sorted(adx_vals, reverse=True)

    def test_empty_when_all_invalid(self):
        """모두 valid=False면 빈 DataFrame."""
        cfg = FunnelConfig()
        df = make_metrics(sectors=["A", "B", "C"], valid_pct=0.0, seed=1)
        result = track_a_wide(df, cfg)
        assert result.empty

    def test_synthetic_fixture(self):
        """make_metrics() 기본 픽스처로 통과 여부 확인 (smoke test)."""
        cfg = FunnelConfig()
        df = make_metrics(seed=42)
        result = track_a_wide(df, cfg)
        # 결과는 valid=True 행만 포함
        assert (result["valid"] == True).all()  # noqa: E712


# ─── Track B 테스트 ───────────────────────────────────────────────────────────

class TestTrackBFunnel:
    def _make_full_pass(self, cfg: FunnelConfig) -> pd.DataFrame:
        """세 단계 모두 통과하는 행 하나."""
        return _make_controlled([{
            "sector": "Pass",
            "valid": True,
            "rs": 0.002,
            "adx": cfg.b_adx_min + 5.0,
            "adx_prev": cfg.b_adx_min + 2.0,  # rising
            "mfi": (cfg.b_mfi_lower + cfg.b_mfi_upper) / 2.0,
        }])

    def test_monotonic_filtering(self):
        """각 단계를 거칠수록 행 수는 단조 감소(비증가)."""
        cfg = FunnelConfig()
        df = make_metrics(seed=0)
        stages = track_b_funnel(df, cfg)
        n1 = len(stages["step1_rs"])
        n2 = len(stages["step2_adx"])
        n3 = len(stages["step3_mfi"])
        assert n1 >= n2 >= n3

    def test_step1_rs_positive(self):
        """step1: RS > 0 필터."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "Pos", "valid": True, "rs":  0.001, "adx": 25.0, "adx_prev": 20.0, "mfi": 60.0},
            {"sector": "Neg", "valid": True, "rs": -0.001, "adx": 25.0, "adx_prev": 20.0, "mfi": 60.0},
            {"sector": "Zer", "valid": True, "rs":  0.000, "adx": 25.0, "adx_prev": 20.0, "mfi": 60.0},
        ])
        stages = track_b_funnel(df, cfg)
        s1 = stages["step1_rs"]
        assert "Pos" in s1.index
        assert "Neg" not in s1.index
        assert "Zer" not in s1.index  # strictly > 0

    def test_step2_adx_threshold(self):
        """step2: ADX >= b_adx_min."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "Above", "valid": True, "rs": 0.001, "adx": cfg.b_adx_min + 1.0, "adx_prev": cfg.b_adx_min - 1.0, "mfi": 60.0},
            {"sector": "Below", "valid": True, "rs": 0.001, "adx": cfg.b_adx_min - 1.0, "adx_prev": cfg.b_adx_min - 2.0, "mfi": 60.0},
        ])
        stages = track_b_funnel(df, cfg)
        # b_require_rising_adx=True: adx_prev도 체크 (Above: adx>adx_prev)
        assert "Above" in stages["step2_adx"].index
        assert "Below" not in stages["step2_adx"].index

    def test_step2_rising_adx_required(self):
        """b_require_rising_adx=True이면 adx > adx_prev 조건 추가."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "Rising",  "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 22.0, "mfi": 60.0},
            {"sector": "Falling", "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 28.0, "mfi": 60.0},
        ])
        stages = track_b_funnel(df, cfg)
        assert "Rising"  in stages["step2_adx"].index
        assert "Falling" not in stages["step2_adx"].index

    def test_step2_rising_adx_disabled(self):
        """b_require_rising_adx=False이면 ADX 방향 무관."""
        cfg = FunnelConfig(b_require_rising_adx=False)
        df = _make_controlled([
            {"sector": "Falling", "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 28.0, "mfi": 60.0},
        ])
        stages = track_b_funnel(df, cfg)
        assert "Falling" in stages["step2_adx"].index

    def test_step3_mfi_range(self):
        """step3: MFI in [b_mfi_lower, b_mfi_upper)."""
        cfg = FunnelConfig()
        lo, hi = cfg.b_mfi_lower, cfg.b_mfi_upper
        df = _make_controlled([
            {"sector": "Inside",    "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 22.0, "mfi": (lo + hi) / 2},
            {"sector": "AtLower",   "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 22.0, "mfi": lo},
            {"sector": "AtUpper",   "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 22.0, "mfi": hi},
            {"sector": "BelowLow",  "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 22.0, "mfi": lo - 0.01},
        ])
        stages = track_b_funnel(df, cfg)
        s3 = stages["step3_mfi"]
        assert "Inside"   in s3.index
        assert "AtLower"  in s3.index       # inclusive lower
        assert "AtUpper"  not in s3.index   # exclusive upper
        assert "BelowLow" not in s3.index

    def test_excludes_invalid(self):
        """valid=False 행은 B 깔때기에 포함되지 않음."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "Bad", "valid": False, "rs": 0.01, "adx": 30.0, "adx_prev": 25.0, "mfi": 60.0},
        ])
        stages = track_b_funnel(df, cfg)
        for stage_df in stages.values():
            assert "Bad" not in stage_df.index

    def test_returns_three_keys(self):
        """반환 dict는 step1_rs / step2_adx / step3_mfi 세 키를 가짐."""
        cfg = FunnelConfig()
        df = make_metrics(seed=5)
        stages = track_b_funnel(df, cfg)
        assert set(stages.keys()) == {"step1_rs", "step2_adx", "step3_mfi"}


# ─── Track C 테스트 ───────────────────────────────────────────────────────────

class TestTrackCScore:
    def test_scores_in_unit_interval(self):
        """모든 score 값은 [0,1] 범위."""
        cfg = FunnelConfig()
        df = make_metrics(seed=99)
        result = track_c_score(df, cfg)
        assert (result["score"] >= 0.0).all()
        assert (result["score"] <= 1.0).all()

    def test_sorted_desc(self):
        """결과는 score 내림차순 정렬."""
        cfg = FunnelConfig()
        df = make_metrics(seed=7)
        result = track_c_score(df, cfg)
        scores = result["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_only_valid_rows(self):
        """valid=False 행은 결과에 포함되지 않음."""
        cfg = FunnelConfig()
        df = make_metrics(valid_pct=0.5, seed=3)
        result = track_c_score(df, cfg)
        # 원본 메트릭에서 valid=False인 섹터가 결과에 없어야 함
        invalid_sectors = df[df["valid"] == False].index  # noqa: E712
        for sec in invalid_sectors:
            assert sec not in result.index

    def test_norm_columns_present(self):
        """필수 컬럼 5개 존재."""
        cfg = FunnelConfig()
        df = make_metrics(seed=1)
        result = track_c_score(df, cfg)
        for col in ["rs_norm", "adx_norm", "adx_rising_norm", "mfi_norm", "score"]:
            assert col in result.columns

    def test_adx_norm_clipped(self):
        """ADX >= 40이면 adx_norm = 1.0, ADX = 0이면 0.0."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "High", "valid": True, "rs": 0.001, "adx": 50.0, "adx_prev": 48.0, "mfi": 65.0},
            {"sector": "Zero", "valid": True, "rs": 0.000, "adx":  0.0, "adx_prev":  0.0, "mfi": 65.0},
        ])
        result = track_c_score(df, cfg)
        assert result.loc["High", "adx_norm"] == pytest.approx(1.0)
        assert result.loc["Zero", "adx_norm"] == pytest.approx(0.0)

    def test_adx_rising_norm_sigmoid(self):
        """
        equal(0 diff) → 0.5, +5 → ~0.92, -5 → ~0.08.
        시그모이드 1/(1+exp(-x/2)) 공식 검증.
        """
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "Equal",   "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 25.0, "mfi": 65.0},
            {"sector": "PlusFive","valid": True, "rs": 0.001, "adx": 30.0, "adx_prev": 25.0, "mfi": 65.0},
            {"sector": "MinusFive","valid": True,"rs": 0.001, "adx": 25.0, "adx_prev": 30.0, "mfi": 65.0},
        ])
        result = track_c_score(df, cfg)
        assert result.loc["Equal",    "adx_rising_norm"] == pytest.approx(0.5, abs=1e-6)
        assert result.loc["PlusFive", "adx_rising_norm"] == pytest.approx(1 / (1 + math.exp(-5/2)), abs=1e-6)
        assert result.loc["MinusFive","adx_rising_norm"] == pytest.approx(1 / (1 + math.exp( 5/2)), abs=1e-6)

    def test_mfi_sweet_spot(self):
        """MFI=65에서 mfi_norm = 1.0 (가우시안 최대)."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "Sweet", "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 23.0, "mfi": 65.0},
            {"sector": "Far",   "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 23.0, "mfi": 10.0},
        ])
        result = track_c_score(df, cfg)
        assert result.loc["Sweet", "mfi_norm"] == pytest.approx(1.0, abs=1e-6)
        assert result.loc["Far",   "mfi_norm"]  < 0.1

    def test_empty_on_all_invalid(self):
        """모두 valid=False면 빈 DataFrame."""
        cfg = FunnelConfig()
        df = make_metrics(valid_pct=0.0, seed=0)
        result = track_c_score(df, cfg)
        assert result.empty

    def test_rs_percentile_rank(self):
        """rs_norm은 percentile rank → [0,1] 균등 분포(정확히 rank/n)."""
        cfg = FunnelConfig()
        df = _make_controlled([
            {"sector": "S1", "valid": True, "rs": 0.001, "adx": 25.0, "adx_prev": 23.0, "mfi": 65.0},
            {"sector": "S2", "valid": True, "rs": 0.003, "adx": 25.0, "adx_prev": 23.0, "mfi": 65.0},
            {"sector": "S3", "valid": True, "rs": 0.002, "adx": 25.0, "adx_prev": 23.0, "mfi": 65.0},
        ])
        result = track_c_score(df, cfg)
        # rs 오름차순 rank: S1→1/3, S3→2/3, S2→3/3
        assert result.loc["S1", "rs_norm"] == pytest.approx(1/3, abs=1e-6)
        assert result.loc["S3", "rs_norm"] == pytest.approx(2/3, abs=1e-6)
        assert result.loc["S2", "rs_norm"] == pytest.approx(3/3, abs=1e-6)


# ─── ADX 경계 테스트 ─────────────────────────────────────────────────────────

class TestBoundaryConditions:
    def test_adx_at_threshold_plus_epsilon(self):
        """ADX = b_adx_min + 0.01 → Track B step2 통과."""
        cfg = FunnelConfig()
        df = _make_controlled([{
            "sector": "JustAbove",
            "valid": True, "rs": 0.001,
            "adx": cfg.b_adx_min + 0.01,
            "adx_prev": cfg.b_adx_min - 1.0,  # rising
            "mfi": (cfg.b_mfi_lower + cfg.b_mfi_upper) / 2,
        }])
        stages = track_b_funnel(df, cfg)
        assert "JustAbove" in stages["step2_adx"].index

    def test_adx_at_threshold_minus_epsilon(self):
        """ADX = b_adx_min - 0.01 → Track B step2 제외."""
        cfg = FunnelConfig()
        df = _make_controlled([{
            "sector": "JustBelow",
            "valid": True, "rs": 0.001,
            "adx": cfg.b_adx_min - 0.01,
            "adx_prev": cfg.b_adx_min - 2.0,
            "mfi": (cfg.b_mfi_lower + cfg.b_mfi_upper) / 2,
        }])
        stages = track_b_funnel(df, cfg)
        assert "JustBelow" not in stages["step2_adx"].index

    def test_mfi_at_lower_bound(self):
        """MFI = b_mfi_lower 정확히 → 통과(inclusive)."""
        cfg = FunnelConfig()
        df = _make_controlled([{
            "sector": "AtLower",
            "valid": True, "rs": 0.001,
            "adx": cfg.b_adx_min + 5.0,
            "adx_prev": cfg.b_adx_min + 2.0,
            "mfi": cfg.b_mfi_lower,
        }])
        stages = track_b_funnel(df, cfg)
        assert "AtLower" in stages["step3_mfi"].index

    def test_mfi_at_upper_bound(self):
        """MFI = b_mfi_upper 정확히 → 제외(exclusive upper)."""
        cfg = FunnelConfig()
        df = _make_controlled([{
            "sector": "AtUpper",
            "valid": True, "rs": 0.001,
            "adx": cfg.b_adx_min + 5.0,
            "adx_prev": cfg.b_adx_min + 2.0,
            "mfi": cfg.b_mfi_upper,
        }])
        stages = track_b_funnel(df, cfg)
        assert "AtUpper" not in stages["step3_mfi"].index


# ─── combine() 테스트 ────────────────────────────────────────────────────────

class TestCombine:
    def _make_b_pass_metrics(self, cfg: FunnelConfig, n: int = 5) -> pd.DataFrame:
        """Track B 통과 보장 행 n개 생성."""
        rows = []
        for i in range(n):
            rows.append({
                "sector": f"Good{i:02d}",
                "valid": True,
                "rs": 0.001 + i * 0.0001,
                "adx": cfg.b_adx_min + 5.0 + i,
                "adx_prev": cfg.b_adx_min + 2.0 + i,  # rising
                "mfi": (cfg.b_mfi_lower + cfg.b_mfi_upper) / 2,
            })
        return _make_controlled(rows)

    def _make_b_fail_metrics(self, cfg: FunnelConfig, n: int = 5) -> pd.DataFrame:
        """Track B 통과 불가(MFI 범위 밖) 행 n개. C 점수는 높을 수 있도록."""
        rows = []
        for i in range(n):
            rows.append({
                "sector": f"NoB{i:02d}",
                "valid": True,
                "rs": 0.003 + i * 0.001,   # RS 높아서 C 스코어는 높음
                "adx": cfg.b_adx_min + 10.0 + i,
                "adx_prev": cfg.b_adx_min + 7.0 + i,
                "mfi": cfg.b_mfi_upper + 5.0,  # MFI 범위 초과 → B 탈락
            })
        return _make_controlled(rows)

    def test_normal_path_used_fallback_false(self):
        """Track B 통과자 있으면 used_fallback=False."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        decision = combine(df, cfg)
        assert decision.used_fallback is False

    def test_normal_path_entry_non_empty(self):
        """Track B 통과자 있으면 entry 비지 않음."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        decision = combine(df, cfg)
        assert not decision.entry.empty

    def test_weights_sum_to_one(self):
        """weights 합계 = 1.0 (entry 비지 않을 때)."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        decision = combine(df, cfg)
        assert decision.weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_weights_proportional_to_score(self):
        """weights는 score 비례. w_i / w_j == score_i / score_j."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg, n=3)
        decision = combine(df, cfg)
        w = decision.weights
        e = decision.entry
        # entry의 score와 weights 비율 일치 확인
        scores = e["score"]
        for i in w.index:
            for j in w.index:
                if i != j and scores[j] != 0:
                    assert w[i] / w[j] == pytest.approx(scores[i] / scores[j], abs=1e-6)

    def test_fallback_path_used_fallback_true(self):
        """Track B 통과자 없으면 used_fallback=True."""
        cfg = FunnelConfig()
        df = self._make_b_fail_metrics(cfg)
        decision = combine(df, cfg)
        assert decision.used_fallback is True

    def test_fallback_entry_from_c(self):
        """폴백 시 entry는 C 스코어 >= c_fallback_min_score."""
        cfg = FunnelConfig()
        df = self._make_b_fail_metrics(cfg, n=8)
        decision = combine(df, cfg)
        score_df = decision.score
        # entry에 있는 모든 섹터는 C 점수 >= c_fallback_min_score OR 총 5개 이하
        if not decision.entry.empty and "score" in decision.entry.columns:
            for sec in decision.entry.index:
                if sec in score_df.index:
                    assert score_df.loc[sec, "score"] >= cfg.c_fallback_min_score

    def test_fallback_max_five_entries(self):
        """폴백 시 entry는 최대 5개."""
        cfg = FunnelConfig()
        df = self._make_b_fail_metrics(cfg, n=20)
        decision = combine(df, cfg)
        if decision.used_fallback:
            assert len(decision.entry) <= 5

    def test_monitor_is_track_a(self):
        """monitor는 Track A 결과와 동일."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        decision = combine(df, cfg)
        expected = track_a_wide(df, cfg)
        pd.testing.assert_frame_equal(decision.monitor, expected)

    def test_score_is_track_c(self):
        """score는 Track C 결과와 동일."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        decision = combine(df, cfg)
        expected = track_c_score(df, cfg)
        pd.testing.assert_frame_equal(decision.score, expected)

    def test_asof_passed_through(self):
        """asof Timestamp이 Decision에 그대로 전달."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        ts = pd.Timestamp("2025-06-05")
        decision = combine(df, cfg, asof=ts)
        assert decision.asof == ts

    def test_empty_weights_when_no_entry(self):
        """entry가 완전히 비면 weights는 빈 Series."""
        cfg = FunnelConfig(c_fallback_min_score=2.0)  # 불가능한 임계값
        df = self._make_b_fail_metrics(cfg, n=3)
        decision = combine(df, cfg)
        assert len(decision.weights) == 0

    def test_entry_has_score_column(self):
        """entry DataFrame에 'score' 컬럼 포함 (UI 정렬용)."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        decision = combine(df, cfg)
        if not decision.entry.empty:
            assert "score" in decision.entry.columns

    def test_entry_contains_metrics_cols(self):
        """entry에 원본 METRICS_COLS가 포함됨 (valid/rs/adx/mfi 표시용)."""
        cfg = FunnelConfig()
        df = self._make_b_pass_metrics(cfg)
        decision = combine(df, cfg)
        if not decision.entry.empty:
            for col in ["rs", "adx", "mfi"]:
                assert col in decision.entry.columns
