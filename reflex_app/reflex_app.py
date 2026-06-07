"""
Reflex 진입점 — ETF Universe Explorer Reflex 버전 (랜딩).

설계 원칙
---------
1) 기존 Streamlit 앱과 완전 격리.
   - import 트리에서 `streamlit` / `etf_scoring` / `momentum_funnel.hot_board`
     등 streamlit 의존 모듈은 사용하지 않는다.
   - 데이터 레이어가 필요해질 때 streamlit-free 인 `etf_universe_builder`
     쪽 헬퍼만 lazy import.

2) 모듈 로드 시 네트워크 / 무거운 연산 금지 (Reflex Cloud cold-start 보호).
   - ETF 리스트 등 외부 데이터는 사용자 버튼 클릭으로만 fetch.

3) Streamlit 쪽에 의존하지 않으므로 Streamlit Cloud 배포에는 영향 0.
"""
from __future__ import annotations

import reflex as rx


# ── 상태 ────────────────────────────────────────────────────────────────
class AppState(rx.State):
    """랜딩 페이지 전역 상태."""

    counter: int = 0
    etf_count: int = 0
    last_fetch: str = ""
    fetching: bool = False
    fetch_error: str = ""

    def inc(self) -> None:
        self.counter += 1

    def reset_counter(self) -> None:
        self.counter = 0

    async def refresh_etf_count(self) -> None:
        """네이버 금융에서 ETF 전종목 카운트만 가져오는 라이트 동작.

        - 무거운 의존성 회피를 위해 streamlit-free 모듈만 lazy import.
        - 실패해도 앱이 죽지 않도록 fetch_error 에만 기록.
        """
        self.fetching = True
        self.fetch_error = ""
        try:
            # 지연 임포트: import 트리에 streamlit / yfinance 가 들어오지 않도록
            from etf_universe_builder import naver_get_all_etfs

            df = naver_get_all_etfs()
            self.etf_count = int(len(df))
            from datetime import datetime
            self.last_fetch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:  # noqa: BLE001
            self.fetch_error = f"{type(e).__name__}: {e}"
        finally:
            self.fetching = False


# ── 컴포넌트 ────────────────────────────────────────────────────────────
def _hero() -> rx.Component:
    return rx.vstack(
        rx.heading("📊 ETF Universe Explorer", size="8"),
        rx.text(
            "한국 상장 ETF 유니버스 + 글로벌 자산 비교 도구 — Reflex 버전",
            size="4",
            color_scheme="gray",
        ),
        rx.text(
            "기존 Streamlit 앱과 별도 배포로 공존합니다.",
            size="2",
            color_scheme="gray",
        ),
        spacing="3",
        align="center",
        padding_y="6",
    )


def _feature_card(icon: str, title: str, desc: str) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading(f"{icon} {title}", size="4"),
            rx.text(desc, size="2", color_scheme="gray"),
            spacing="2",
            align="start",
        ),
        padding="4",
    )


def _features() -> rx.Component:
    items = [
        ("🇰🇷", "국내 ETF 유니버스", "네이버 금융 / KRX 직접 HTTP 기반"),
        ("🌍", "글로벌 가격", "yfinance — 미국 ETF / 글로벌 지수"),
        ("🧬", "구성종목 PDF", "종목별 ETF 보유 비중 매트릭스"),
        ("🔥", "Hot Sectors", "거래대금 / Money Flow / 상대강도"),
    ]
    return rx.grid(
        *[_feature_card(*it) for it in items],
        columns="2",
        spacing="4",
        width="100%",
    )


def _etf_probe() -> rx.Component:
    """네이버 금융 연결 점검 + ETF 카운트."""
    return rx.card(
        rx.vstack(
            rx.heading("🩺 네이버 금융 연결 점검", size="4"),
            rx.text(
                "버튼을 누르면 네이버 ETF 전종목 API 를 1회 호출해 결과 수를 가져옵니다.",
                size="2",
                color_scheme="gray",
            ),
            rx.hstack(
                rx.button(
                    rx.cond(
                        AppState.fetching,
                        "조회 중...",
                        "📡 네이버 ETF 조회",
                    ),
                    on_click=AppState.refresh_etf_count,
                    disabled=AppState.fetching,
                    size="3",
                ),
                rx.cond(
                    AppState.etf_count > 0,
                    rx.text(
                        f"ETF {AppState.etf_count}개 · 최종 ", AppState.last_fetch,
                        size="2",
                    ),
                    rx.fragment(),
                ),
                spacing="3",
                align="center",
            ),
            rx.cond(
                AppState.fetch_error != "",
                rx.callout(
                    AppState.fetch_error,
                    icon="triangle_alert",
                    color_scheme="red",
                    size="1",
                ),
                rx.fragment(),
            ),
            spacing="3",
            align="start",
        ),
        padding="4",
    )


def _counter_demo() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Reflex 상태 데모", size="4"),
            rx.text(
                "Reflex 백엔드 상태가 살아있는지 카운터로 확인.",
                size="2",
                color_scheme="gray",
            ),
            rx.hstack(
                rx.text("카운터:", size="3"),
                rx.text(AppState.counter, size="5", weight="bold"),
                spacing="2",
                align="center",
            ),
            rx.hstack(
                rx.button("+1", on_click=AppState.inc, size="3"),
                rx.button(
                    "Reset",
                    on_click=AppState.reset_counter,
                    size="3",
                    variant="soft",
                ),
                spacing="2",
            ),
            spacing="3",
            align="start",
        ),
        padding="4",
    )


def index() -> rx.Component:
    return rx.container(
        _hero(),
        _features(),
        rx.box(height="2em"),
        _etf_probe(),
        rx.box(height="1em"),
        _counter_demo(),
        size="3",
        padding_y="6",
    )


# ── 앱 정의 ─────────────────────────────────────────────────────────────
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="indigo",
        radius="medium",
    ),
)
app.add_page(index, title="ETF Universe Explorer — Reflex")
