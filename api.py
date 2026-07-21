# """
# api.py

# FastAPI 백엔드.
# - 종목을 받으면 KIS API에서 최근 30분봉을 가져온다.
# - 캔들 간 시간 간격(gap)으로 거래 세션을 나눈다.
#   -> 가장 최근 세션이 자연스럽게 "오늘(장 시작~현재)" 또는
#      (아직 장이 시작 안 했으면) "직전 거래일 전체" 가 된다.
#      주말이면 KIS에 토/일 데이터가 없으므로 자동으로 금요일 세션이 잡힌다.
# - 그 세션 데이터를 train_isolation_forest.py와 동일한 특징으로 변환해
#   저장된 모델로 이상치를 예측한다.
# - 최근 3세션(3거래일) 캔들차트에 이상치를 표시해 base64 이미지로 반환한다.

# 실행 전 필요한 패키지:
#     pip install fastapi uvicorn mplfinance

# 실행:
#     uvicorn api:app --reload
# """
# from __future__ import annotations

# import base64
# import io
# from typing import List

# import joblib
# import matplotlib
# matplotlib.use("Agg")
# import mplfinance as mpf
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# import collector
# import train_isolation_forest as tif
# from symbols import SYMBOLS

# app = FastAPI(title="나스닥100 Isolation Forest 이상치 탐지 API")

# MODEL_PATH = tif.MODEL_DIR / "isolation_forest.pkl"

# FETCH_DAYS = 6          # rolling window 워밍업 + 3세션 차트를 위해 넉넉히 가져온다
# SESSION_GAP_HOURS = 3    # 이 시간 이상 비면 다른 거래세션(거래일)로 간주
# CHART_SESSIONS = 3       # 차트에 보여줄 최근 세션(거래일) 수

# # 세션 간 캐시 (앱이 켜져있는 동안 재사용 - 토큰 재발급/재로그인 비용 절감)
# _session = None
# _collector = None
# _model = None


# def get_collector() -> collector.OverseasCollector:
#     global _session, _collector
#     if _collector is None:
#         _session = collector.KisSession()
#         _collector = collector.OverseasCollector(_session)
#     return _collector


# def get_model():
#     global _model
#     if _model is None:
#         if not MODEL_PATH.exists():
#             raise HTTPException(
#                 status_code=500,
#                 detail=(
#                     f"모델 파일이 없습니다: {MODEL_PATH}. "
#                     "train_isolation_forest.py를 먼저 실행해 모델을 학습하세요."
#                 ),
#             )
#         _model = joblib.load(MODEL_PATH)
#     return _model


# class PredictResponse(BaseModel):
#     symbol: str
#     message: str
#     has_anomaly: bool
#     anomaly_times: List[str]
#     image_base64: str


# def assign_sessions(df: pd.DataFrame, gap_hours: float = SESSION_GAP_HOURS) -> pd.DataFrame:
#     """캔들 사이 시간 간격이 gap_hours보다 크면 새로운 거래세션으로 구분한다."""
#     df = df.sort_values("datetime").reset_index(drop=True)
#     gaps = df["datetime"].diff() > pd.Timedelta(hours=gap_hours) # bool타입
#     #df = df.copy()
#     df["session_id"] = gaps.cumsum()
#     return df


# def make_candle_image(df: pd.DataFrame, symbol: str, anomaly_times) -> str:
#     """최근 N세션 캔들차트 + 이상치 표시 이미지를 base64 문자열로 만든다."""

#     plot_df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]

#     anomaly_marks = pd.Series(index=plot_df.index, dtype=float)
#     for t in anomaly_times:
#         if t in anomaly_marks.index:
#             anomaly_marks.loc[t] = plot_df.loc[t, "high"] * 1.002  # 캔들 위쪽에 표시

#     addplots = []
#     if anomaly_marks.notna().any():
#         addplots.append(
#             mpf.make_addplot(
#                 anomaly_marks, type="scatter", markersize=90, marker="v", color="red"
#             )
#         )

#     buf = io.BytesIO()
#     mpf.plot(
#         plot_df,
#         type="candle",
#         style="yahoo",
#         addplot=addplots if addplots else None,
#         volume=True,
#         title=f"{symbol} - 최근 {CHART_SESSIONS}거래일 (▽ 이상치)",
#         savefig=dict(fname=buf, dpi=120, bbox_inches="tight"),
#     )
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode("utf-8")


# @app.get("/predict", response_model=PredictResponse)
# def predict(symbol: str):
#     try:
#         return _predict(symbol)
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# def _predict(symbol: str) -> PredictResponse:
#     symbol = symbol.upper()

#     if symbol not in SYMBOLS:
#         raise HTTPException(status_code=404, detail=f"{symbol}은 등록된 종목이 아닙니다.")

#     market = SYMBOLS[symbol]
#     coll = get_collector()
#     model = get_model()

#     raw_df = coll.get_30min_candle(symbol=symbol, market=market, days=FETCH_DAYS)

#     if raw_df.empty:
#         raise HTTPException(status_code=502, detail=f"{symbol} 데이터를 가져오지 못했습니다.")

#     raw_df = assign_sessions(raw_df)
#     session_ids = sorted(raw_df["session_id"].unique())

#     target_session_id = session_ids[-1]           # 가장 최근 세션 = 오늘(또는 직전 거래일)
#     chart_session_ids = session_ids[-CHART_SESSIONS:]

#     # train_isolation_forest.py의 add_features와 동일한 로직으로 특징 생성
#     # (학습 때와 추론 때 특징 계산 방식이 다르면 모델이 무의미해지므로 반드시 재사용한다)
#     feat_df = tif.add_features(raw_df.drop(columns=["session_id"]))
    
#     # 주의: add_features는 rolling 계산으로 생긴 NaN 행을 제거하기 때문에,
#     # feat_df에서 세션을 다시 계산(assign_sessions)하면 빠진 행 때문에
#     # 시간 간격 패턴이 달라져 session_id가 target_session_id와 어긋난다.
#     # 반드시 raw_df에서 이미 매긴 session_id를 그대로 병합해서 써야 한다.
#     feat_df = feat_df.merge(
#         raw_df[["symbol", "datetime", "session_id"]],
#         on=["symbol", "datetime"],
#         how="left",
#     )

#     target_df = feat_df[feat_df["session_id"] == target_session_id]

#     if target_df.empty:
#         raise HTTPException(
#             status_code=502,
#             detail=(
#                 f"{symbol}의 최신 세션 데이터가 충분하지 않습니다 "
#                 f"(rolling window 워밍업에 데이터가 더 필요합니다)."
#             ),
#         )

#     X = target_df[tif.FEATURE_COLS].values
#     preds = model.predict(X)  # -1: 이상치, 1: 정상

#     target_df = target_df.copy()
#     target_df["is_anomaly"] = preds == -1
#     anomalies = target_df[target_df["is_anomaly"]]

#     if not anomalies.empty:
#         last_anomaly_time = anomalies["datetime"].max()
#         message = f"{last_anomaly_time:%Y-%m-%d %H:%M} 이상패턴 발견"
#         has_anomaly = True
#     else:
#         message = "이상 패턴이 발견되지 않았습니다"
#         has_anomaly = False

#     chart_df = raw_df[raw_df["session_id"].isin(chart_session_ids)]
#     image_b64 = make_candle_image(chart_df, symbol, anomalies["datetime"])

#     return PredictResponse(
#         symbol=symbol,
#         message=message,
#         has_anomaly=has_anomaly,
#         anomaly_times=[f"{t:%Y-%m-%d %H:%M}" for t in anomalies["datetime"]],
#         image_base64=image_b64,
#     )


# @app.get("/health")
# def health():
#     return {"status": "ok"}


#============================================================================================================

"""
api.py

FastAPI 백엔드.
- 종목을 받으면 KIS API에서 최근 30분봉을 가져온다.
- 캔들 간 시간 간격(gap)으로 거래 세션을 나눈다.
  -> 가장 최근 세션이 자연스럽게 "오늘(장 시작~현재)" 또는
     (아직 장이 시작 안 했으면) "직전 거래일 전체" 가 된다.
     주말이면 KIS에 토/일 데이터가 없으므로 자동으로 금요일 세션이 잡힌다.
- 그 세션 데이터를 train_isolation_forest.py와 동일한 특징으로 변환해
  저장된 모델로 이상치를 예측한다.
- 최근 3세션(3거래일) 캔들차트에 이상치를 표시해 base64 이미지로 반환한다.

실행 전 필요한 패키지:
    pip install fastapi uvicorn

실행:
    uvicorn api:app --reload
"""
from __future__ import annotations

from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import collector
import train_isolation_forest as tif
from symbols import SYMBOLS

app = FastAPI(title="나스닥100 Isolation Forest 이상치 탐지 API")

MODEL_PATH = tif.MODEL_DIR / "isolation_forest.pkl"

FETCH_DAYS = 6          # rolling window 워밍업 + 3세션 차트를 위해 넉넉히 가져온다
SESSION_GAP_HOURS = 3    # 이 시간 이상 비면 다른 거래세션(거래일)로 간주
CHART_SESSIONS = 3       # 차트에 보여줄 최근 세션(거래일) 수

# 세션 간 캐시 (앱이 켜져있는 동안 재사용 - 토큰 재발급/재로그인 비용 절감)
_session = None
_collector = None
_model = None


def get_collector() -> collector.OverseasCollector:
    global _session, _collector
    if _collector is None:
        _session = collector.KisSession()
        _collector = collector.OverseasCollector(_session)
    return _collector


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    f"모델 파일이 없습니다: {MODEL_PATH}. "
                    "train_isolation_forest.py를 먼저 실행해 모델을 학습하세요."
                ),
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictResponse(BaseModel):
    symbol: str
    message: str
    has_anomaly: bool
    anomaly_times: List[str]
    candles: List[Dict]   # plotly candlestick용 원본 OHLCV (프론트에서 직접 그림)


def assign_sessions(df: pd.DataFrame, gap_hours: float = SESSION_GAP_HOURS) -> pd.DataFrame:
    """캔들 사이 시간 간격이 gap_hours보다 크면 새로운 거래세션으로 구분한다."""
    df = df.sort_values("datetime").reset_index(drop=True)
    gaps = df["datetime"].diff() > pd.Timedelta(hours=gap_hours)
    df = df.copy()
    df["session_id"] = gaps.cumsum()
    return df


def build_candle_records(df: pd.DataFrame) -> List[Dict]:
    """plotly.graph_objects.Candlestick에 바로 넣을 수 있는 레코드 리스트로 변환."""
    out = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict(orient="records")


@app.get("/predict", response_model=PredictResponse)
def predict(symbol: str):
    try:
        return _predict(symbol)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def _predict(symbol: str) -> PredictResponse:
    symbol = symbol.upper()

    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"{symbol}은 등록된 종목이 아닙니다.")

    market = SYMBOLS[symbol]
    coll = get_collector()
    model = get_model()

    raw_df = coll.get_30min_candle(symbol=symbol, market=market, days=FETCH_DAYS)

    if raw_df.empty:
        raise HTTPException(status_code=502, detail=f"{symbol} 데이터를 가져오지 못했습니다.")

    raw_df = assign_sessions(raw_df)
    session_ids = sorted(raw_df["session_id"].unique())

    target_session_id = session_ids[-1]           # 가장 최근 세션 = 오늘(또는 직전 거래일)
    chart_session_ids = session_ids[-CHART_SESSIONS:]

    # train_isolation_forest.py의 add_features와 동일한 로직으로 특징 생성
    # (학습 때와 추론 때 특징 계산 방식이 다르면 모델이 무의미해지므로 반드시 재사용한다)
    feat_df = tif.add_features(raw_df.drop(columns=["session_id"]))

    # 주의: add_features는 rolling 계산으로 생긴 NaN 행을 제거하기 때문에,
    # feat_df에서 세션을 다시 계산(assign_sessions)하면 빠진 행 때문에
    # 시간 간격 패턴이 달라져 session_id가 target_session_id와 어긋난다.
    # 반드시 raw_df에서 이미 매긴 session_id를 그대로 병합해서 써야 한다.
    feat_df = feat_df.merge(
        raw_df[["symbol", "datetime", "session_id"]],
        on=["symbol", "datetime"],
        how="left",
    )

    target_df = feat_df[feat_df["session_id"] == target_session_id]

    if target_df.empty:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{symbol}의 최신 세션 데이터가 충분하지 않습니다 "
                f"(rolling window 워밍업에 데이터가 더 필요합니다)."
            ),
        )

    X = target_df[tif.FEATURE_COLS].values
    preds = model.predict(X)  # -1: 이상치, 1: 정상

    target_df = target_df.copy()
    target_df["is_anomaly"] = preds == -1
    anomalies = target_df[target_df["is_anomaly"]]

    if not anomalies.empty:
        last_anomaly_time = anomalies["datetime"].max()
        message = f"{last_anomaly_time:%Y-%m-%d %H:%M} 이상패턴 발견"
        has_anomaly = True
    else:
        message = "이상 패턴이 발견되지 않았습니다"
        has_anomaly = False

    chart_df = raw_df[raw_df["session_id"].isin(chart_session_ids)]
    candles = build_candle_records(chart_df)

    return PredictResponse(
        symbol=symbol,
        message=message,
        has_anomaly=has_anomaly,
        anomaly_times=[f"{t:%Y-%m-%d %H:%M}" for t in anomalies["datetime"]],
        candles=candles,
    )


@app.get("/health")
def health():
    return {"status": "ok"}