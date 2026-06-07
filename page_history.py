"""MP History Page (Phase 2)

저장된 MP 의 일별 스냅샷 (NAV·활성 satellite·룰 누적·구성) 기록·열람.

소스: `saved_mps/history.json` (load_history → 로컬 → GitHub raw fallback)
저장: 수동 「오늘 스냅샷 저장」 버튼 (Phase 3 에서 cron 자동화 예정)

표시:
- 누적 NAV 라인차트
- 일별 스냅샷 표 (date, NAV, NAV%, sats, trades_total)
- 가장 최근 스냅샷의 구성 표
"""
from __future__ import annotations
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from momentum_funnel import (
    load_mp, load_history, append_history_snapshot, push_history_to_github,
    build_snapshot, INITIAL_CAPITAL_DEFAULT,
)


def page_history():
    st.title("📜 MP History")

    saved = load_mp()
    history = load_history()

    if not saved and not history:
        st.warning("저장된 MP 도, history 도 없습니다. 먼저 ETF Uniview 에서 MP 를 저장하세요.")
        return

    if saved:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("편입일", saved.get('inception_date', '?'))
        c2.metric("방법", saved.get('method', '?'))
        c3.metric("포지션 수", len(saved.get('positions', [])))
        c4.metric("History 스냅샷", f"{len(history)}건")

    st.markdown("---")

    # ── 오늘 스냅샷 저장 (수동) ─────────────────────────────────────────
    st.subheader("🟢 오늘 스냅샷 저장")
    df_uni = st.session_state.get('df_universe')
    universe_ready = isinstance(df_uni, pd.DataFrame) and not df_uni.empty

    if not universe_ready:
        st.info("👈 사이드바에서 먼저 **🚀 유니버스 빌드**를 실행해야 룰 5 신규 편입 후보가 산출됩니다.")

    s1, s2, s3 = st.columns([1.5, 1.5, 2])
    with s1:
        mode = st.radio(
            "신규 편입 방법 (룰 5)",
            options=['A', 'B'],
            format_func=lambda x: 'A · HotScore Top' if x == 'A' else 'B · Money→RS',
            horizontal=True, key='hist_mode',
        )
    with s2:
        enable_rule5 = st.checkbox(
            "룰 5 활성", value=universe_ready and bool(saved),
            disabled=not universe_ready, key='hist_enable_rule5',
        )
    with s3:
        push_gh = st.checkbox(
            "GitHub 동기화", value=True, key='hist_push_github',
            help="동일 history.json 을 GitHub repo 에 자동 커밋 (token 필요)",
        )

    if st.button("📸 오늘 스냅샷 생성·저장", type='primary',
                 disabled=not saved, key='hist_snap_save',
                 help="저장된 MP 에 룰 1~5 적용한 결과를 history.json 에 append"):
        with st.spinner("📊 스냅샷 생성 중 (룰 시뮬 → 저장, 1~2분)..."):
            try:
                entry = build_snapshot(
                    saved_mp=saved,
                    df_universe=df_uni if universe_ready else None,
                    mode=mode, enable_rule5=enable_rule5,
                )
            except Exception as e:
                st.error(f"스냅샷 생성 실패: {e}")
                entry = None

        if entry is None:
            st.error("스냅샷 생성 실패 (saved MP·가격 데이터 확인 필요).")
        else:
            try:
                path, new_history = append_history_snapshot(entry)
                st.success(
                    f"✅ 저장: `{path}` · 날짜 {entry['date']} · "
                    f"NAV {entry['nav_krw'] / 1e8:.1f}억 "
                    f"({entry['nav_pct']:+.2f}%)"
                )
                if push_gh:
                    ok, msg = push_history_to_github(new_history)
                    if ok:
                        st.info(f"🔄 GitHub 동기화: {msg}")
                    else:
                        st.warning(f"🔒 GitHub 미동기화 (로컬만): {msg}")
                history = new_history
            except Exception as e:
                st.error(f"저장 실패: {e}")

    st.markdown("---")

    # ── 누적 NAV 라인차트 ──────────────────────────────────────────────
    if not history:
        st.info("아직 스냅샷이 없습니다. 위 📸 버튼으로 첫 스냅샷을 만드세요.")
        return

    hist_df = pd.DataFrame(history)
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date').reset_index(drop=True)

    st.subheader("📈 누적 NAV 추이")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df['date'], y=hist_df['nav_krw'].astype(float) / 1e8,
        mode='lines+markers', name='NAV (억원)',
        line=dict(color='#4FC3F7', width=2),
        marker=dict(size=6),
    ))
    fig.add_hline(
        y=INITIAL_CAPITAL_DEFAULT / 1e8, line_dash='dash', line_color='#888',
        annotation_text='초기자금 1,000억', annotation_position='top right',
    )
    fig.update_layout(
        title='MP 가상 NAV 일별 스냅샷',
        height=360, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor='#0e1117', plot_bgcolor='#111520',
        xaxis=dict(title='', showgrid=True, gridcolor='#1e2130'),
        yaxis=dict(title='NAV (억원)', showgrid=True, gridcolor='#1e2130'),
        hovermode='x unified',
    )
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    # ── 일별 스냅샷 표 ──────────────────────────────────────────────────
    st.subheader("🗓️ 일별 스냅샷")
    disp = hist_df.copy()
    disp['NAV (억)'] = (disp['nav_krw'].astype(float) / 1e8).round(2)
    disp['NAV %'] = disp['nav_pct'].astype(float).round(2)
    disp['활성 sat'] = disp['active_sat_count'].astype(int)
    disp['누적 매매'] = disp['n_trades_total'].astype(int)
    disp['편입일'] = disp['inception_date']
    disp['방법/모드'] = disp['method'].astype(str) + ' / ' + disp.get('mode', pd.Series(['?'] * len(disp))).astype(str)
    show = disp[['date', 'NAV (억)', 'NAV %', '활성 sat', '누적 매매', '편입일', '방법/모드']].copy()
    show['date'] = show['date'].dt.strftime('%Y-%m-%d')
    show = show.sort_values('date', ascending=False).reset_index(drop=True)
    st.dataframe(
        show, width='stretch', hide_index=True, height=320,
        column_config={
            'NAV %': st.column_config.NumberColumn('NAV %', format='%+.2f'),
        },
    )

    # ── 최근 스냅샷 구성 ────────────────────────────────────────────────
    st.subheader("📋 가장 최근 스냅샷 — 구성")
    latest = history[-1]
    pos_list = latest.get('positions', [])
    if not pos_list:
        st.caption("구성 정보 없음.")
    else:
        pos_df = pd.DataFrame(pos_list)
        if 'weight_pct' in pos_df.columns:
            pos_df = pos_df.sort_values('weight_pct', ascending=False)
            pos_df['weight_pct'] = pos_df['weight_pct'].astype(float).round(2)
        col_map = {
            'role': '역할', 'ticker': '티커', 'representative': '대표 ETF',
            'category': '카테고리', 'status': '상태', 'weight_pct': '비중 %',
        }
        rename_cols = {k: v for k, v in col_map.items() if k in pos_df.columns}
        pos_df = pos_df.rename(columns=rename_cols)
        order = [v for v in col_map.values() if v in pos_df.columns]
        st.dataframe(
            pos_df[order], width='stretch', hide_index=True,
            column_config={
                '비중 %': st.column_config.ProgressColumn(
                    '비중 %', min_value=0.0, max_value=100.0, format='%.2f%%',
                ),
            },
        )
    st.caption(f"기준일자: {latest.get('date', '?')} · 스냅샷 총 {len(history)}건")
