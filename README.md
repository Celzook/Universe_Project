# 📊 ETF Universe Explorer

한국 상장 ETF 유니버스 구축 + 글로벌 자산 수익률 비교 도구

## 🚀 바로 실행
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> 위 배지의 URL은 배포 후 수정하세요

## 기능

| 메뉴 | 설명 |
|------|------|
| **유니버스 탐색** | 국내 ETF 필터링, 카테고리별 분석, BM 상위/하위 |
| **구성종목(PDF)** | 종목별 ETF 보유 비중 피벗 매트릭스 |
| **수익률 비교** | 국내 ETF / 글로벌 지수 / 미국 ETF 기간별 비교 |

## 데이터 소스
- **한국 ETF**: 네이버 금융 API + KRX 직접 HTTP (pykrx 의존성 제거)
- **글로벌 지수/미국 ETF**: yfinance (Yahoo Finance)
- **상장일/설정일**: 네이버 금융
- **구성종목(PDF)**: KRX 직접 HTTP

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 파일 구조
```
├── app.py                      # Streamlit 메인 앱
├── etf_universe_builder.py     # 한국 ETF 유니버스 엔진
├── global_price_collector.py   # 글로벌 가격 수집기
├── requirements.txt            # 의존성
└── .streamlit/
    └── config.toml             # Streamlit 설정
```

## Streamlit Cloud 배포

1. 이 레포를 GitHub에 push
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. GitHub 레포 연결
4. Main file: `app.py` 선택
5. Deploy 클릭
