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

    feat_df = tif.add_features(raw_df.drop(columns=["session_id"]))

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