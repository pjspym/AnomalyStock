from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from symbols import SYMBOLS
from symbol_name import SYMBOL_NAMES

API_URL = "http://localhost:8000/predict"
MAX_WATCHLIST = 6
LOGO_DIR = Path("logos_images")

st.set_page_config(page_title="Nasdaq 100 이상치 탐지", layout="wide")

st.title("Nasdaq 100 이상치 탐지")
st.caption("30분 캔들 패턴 이상 여부 판단")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []       # 관심종목 순서 리스트 (최대 6개)
if "results" not in st.session_state:
    st.session_state.results = {}         # {symbol: API 응답 결과}


def logo_path(symbol: str) -> str | None:
    """logos_images/{symbol}.png 경로를 반환 (없으면 None)."""
    p = LOGO_DIR / f"{symbol}.png"
    return str(p) if p.exists() else None


def fetch_predict(symbol: str) -> dict | None:
    try:
        resp = requests.get(API_URL, params={"symbol": symbol}, timeout=90)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        detail = None
        try:
            detail = resp.json().get("detail")
        except Exception:
            pass
        st.error(f"[{symbol}] 백엔드 요청 실패: {detail or e}")
        return None


def build_candlestick_figure(candles: list[dict], anomaly_times: list[str], symbol: str) -> go.Figure:
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])

    x_labels = df["datetime"].dt.strftime("%m-%d %H:%M")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_labels,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbol,
            )
        ]
    )

    if anomaly_times:
        anomaly_dt = pd.to_datetime(anomaly_times)
        mask = df["datetime"].isin(anomaly_dt)

        fig.add_trace(
            go.Scatter(
                x=x_labels[mask],
                y=df.loc[mask, "high"] * 1.003,   # 캔들 위쪽에 표시
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red"),
                name="이상치",
            )
        )

    fig.update_xaxes(type="category", nticks=15, tickangle=-45)

    fig.update_layout(
        title=f"{symbol} - 최근 3거래일 (▽ 이상치)",
        xaxis_title="datetime",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(t=40, b=40),
    )

    return fig


# 관심종목 추가
st.subheader(f"관심종목 ({len(st.session_state.watchlist)}/{MAX_WATCHLIST})")

available = [s for s in sorted(SYMBOLS.keys()) if s not in st.session_state.watchlist]

st.sidebar.write("**왓치 리스트**")
with st.sidebar:
    candidate = st.selectbox(
        "종목 추가",
        available,
        label_visibility="collapsed",
        disabled=len(st.session_state.watchlist) >= MAX_WATCHLIST or not available,
    )
    add_clicked = st.button(
        "추가",
        use_container_width=True,
        disabled=len(st.session_state.watchlist) >= MAX_WATCHLIST or not available,
    )
 
    if st.session_state.watchlist:
        for sym in st.session_state.watchlist:
            with st.container():
                logo_col, name_col, x_col = st.columns([1, 4, 1])

                with logo_col:
                    path = logo_path(sym)
                    if path:
                        st.image(path, width=28)

                with name_col:
                    name = SYMBOL_NAMES.get(sym, sym)
                    st.markdown(f"**{name} ({sym})**")

                with x_col:
                    if st.button("✕", key=f"remove_{sym}"):
                        st.session_state.watchlist.remove(sym)
                        st.session_state.results.pop(sym, None)
                        st.rerun()
    else:
        st.info("왓치리스트가 비어 있습니다. 위에서 종목을 선택하고 '추가'를 눌러주세요.")

if add_clicked and candidate:
    if len(st.session_state.watchlist) >= MAX_WATCHLIST:
        st.warning(f"관심종목은 최대 {MAX_WATCHLIST}개까지만 추가할 수 있습니다.")
    else:
        st.session_state.watchlist.append(candidate)
        with st.spinner(f"{candidate} 데이터 조회 및 예측 중..."):
            result = fetch_predict(candidate)
        if result:
            st.session_state.results[candidate] = result
        st.rerun()

if len(st.session_state.watchlist) >= MAX_WATCHLIST:
    st.caption(f"관심종목이 최대 개수({MAX_WATCHLIST}개)에 도달했습니다. 추가하려면 먼저 하나를 삭제하세요.")




st.divider() #수평 구분선

if st.session_state.watchlist:
    if st.button("전체 새로고침"):
        for sym in st.session_state.watchlist:
            with st.spinner(f"{sym} 데이터 조회 및 예측 중..."):
                result = fetch_predict(sym)
            if result:
                st.session_state.results[sym] = result


# 관심종목별 결과 (최대 6개, 2열 그리드)
if st.session_state.watchlist:
    cols = st.columns(2)

    for idx, sym in enumerate(st.session_state.watchlist):
        result = st.session_state.results.get(sym)

        with cols[idx % 2]:
            logo_col, title_col = st.columns([0.1, 2], vertical_alignment="top", gap="small")

            with logo_col:
                path = logo_path(sym)
                if path:
                    st.image(path, width=32)

            with title_col:
                st.markdown(
                    f"""
                    <div style="
                        font-size:28px;
                        font-weight:700;
                        line-height:32px;
                        margin-left:-15px;
                        padding:0;
                    ">
                        {sym}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if result is None:
                st.warning("아직 예측 결과가 없습니다. '전체 새로고침'을 눌러주세요.")
                continue

            if result["has_anomaly"]:
                st.error(result["message"])
                with st.expander(f"발견된 이상치 시점 ({len(result['anomaly_times'])}건)"):
                    for t in result["anomaly_times"]:
                        st.write(t)
            else:
                st.success(result["message"])

            fig = build_candlestick_figure(result["candles"], result["anomaly_times"], sym)
            st.plotly_chart(fig, use_container_width=True)