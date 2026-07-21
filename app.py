# """
# app.py

# Streamlit 프론트엔드.
# 종목을 선택하고 "확인"을 누르면 FastAPI 백엔드(/predict)에 요청을 보내
# - 이상패턴 발견 여부 메시지
# - 최근 3거래일 캔들차트 (이상치 표시)
# 를 화면에 보여준다.

# 실행 전 필요한 패키지:
#     pip install streamlit requests

# 실행 (터미널 2개 필요):
#     터미널 1) uvicorn api:app --reload
#     터미널 2) streamlit run app.py
# """
# import base64

# import requests
# import streamlit as st

# from symbols import SYMBOLS

# API_URL = "http://localhost:8000/predict"

# st.set_page_config(page_title="나스닥100 이상치 탐지", layout="centered")

# st.title("나스닥100 이상치 탐지")
# st.caption("Isolation Forest 모델로 30분봉 패턴 이상 여부를 확인합니다.")

# symbol = st.selectbox("종목 선택", sorted(SYMBOLS.keys()))

# if st.button("확인", type="primary"):
#     with st.spinner(f"{symbol} 데이터 조회 및 예측 중..."):
#         try:
#             resp = requests.get(API_URL, params={"symbol": symbol}, timeout=90)
#             resp.raise_for_status()
#             result = resp.json()
#         except requests.exceptions.RequestException as e:
#             detail = None
#             try:
#                 detail = resp.json().get("detail")
#             except Exception:
#                 pass
#             st.error(f"백엔드 요청 실패: {detail or e}")
#             st.stop()

#     if result["has_anomaly"]:
#         st.error(result["message"])
#         with st.expander(f"발견된 이상치 시점 ({len(result['anomaly_times'])}건)"):
#             for t in result["anomaly_times"]:
#                 st.write(t)
#     else:
#         st.success(result["message"])

#     image_bytes = base64.b64decode(result["image_base64"])
#     st.image(image_bytes, caption=f"{symbol} 최근 3거래일 캔들 (▽ 이상치 표시)")
    
#====================================================================================================
"""
app.py

Streamlit 프론트엔드.
종목을 선택하고 "확인"을 누르면 FastAPI 백엔드(/predict)에 요청을 보내
- 이상패턴 발견 여부 메시지
- 최근 3거래일 캔들차트 (이상치 표시)
를 화면에 보여준다.

실행 전 필요한 패키지:
    pip install streamlit requests plotly

실행 (터미널 2개 필요):
    터미널 1) uvicorn api:app --reload
    터미널 2) streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from symbols import SYMBOLS

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="나스닥100 이상치 탐지", layout="centered")

st.title("나스닥100 이상치 탐지")
st.caption("Isolation Forest 모델로 30분봉 패턴 이상 여부를 확인합니다.")

symbol = st.selectbox("종목 선택", sorted(SYMBOLS.keys()))


def build_candlestick_figure(candles: list[dict], anomaly_times: list[str], symbol: str) -> go.Figure:
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"])

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["datetime"],
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
        anomaly_df = df[df["datetime"].isin(anomaly_dt)]

        fig.add_trace(
            go.Scatter(
                x=anomaly_df["datetime"],
                y=anomaly_df["high"] * 1.003,   # 캔들 위쪽에 표시
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red"),
                name="이상치",
            )
        )

    fig.update_layout(
        title=f"{symbol} - 최근 3거래일 (▽ 이상치)",
        xaxis_title="datetime",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        height=500,
    )

    return fig


if st.button("확인", type="primary"):
    with st.spinner(f"{symbol} 데이터 조회 및 예측 중..."):
        try:
            resp = requests.get(API_URL, params={"symbol": symbol}, timeout=90)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            detail = None
            try:
                detail = resp.json().get("detail")
            except Exception:
                pass
            st.error(f"백엔드 요청 실패: {detail or e}")
            st.stop()

    if result["has_anomaly"]:
        st.error(result["message"])
        with st.expander(f"발견된 이상치 시점 ({len(result['anomaly_times'])}건)"):
            for t in result["anomaly_times"]:
                st.write(t)
    else:
        st.success(result["message"])

    fig = build_candlestick_figure(result["candles"], result["anomaly_times"], symbol)
    st.plotly_chart(fig, use_container_width=True)