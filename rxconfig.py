"""
Reflex 설정 — 기존 Streamlit 앱 (app.py) 과 독립.

배포 / 실행:
- Reflex Cloud: `reflex deploy` (GitHub Actions: .github/workflows/deploy.yml)
- 로컬: `pip install -r reflex_requirements.txt && reflex run`
  → frontend: http://localhost:3000 / backend: http://localhost:8000
  (Streamlit 8501 과 포트 충돌 없음)

Streamlit 쪽은 영향 없음:
- 기존 requirements.txt 는 streamlit 전용 — Reflex 미설치 환경에서도 동작.
- 이 파일은 Streamlit Cloud 가 무시.
"""
import reflex as rx

config = rx.Config(
    app_name="reflex_app",
)
