
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
PLOT_DIR = Path("plots")

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# 이상치 탐지에 사용할 특징(feature) 컬럼
FEATURE_COLS = [
    "log_return",
    "hl_range",
    "oc_range",
    "volume_change",
    "volume_zscore",
    "price_zscore",
    "rolling_volatility",
]

ROLLING_WINDOW = 20     # 롤링 통계량 윈도우 (30분봉 기준 20개 = 약 10시간)
CONTAMINATION = 0.015    # 전체 데이터 중 이상치로 볼 비율 (도메인 지식에 맞게 조정 가능)


# 1. 데이터 로드
def load_all_symbol_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """data/ 폴더의 종목별 CSV를 전부 읽어 하나의 DataFrame으로 합친다."""

    csv_files = sorted(
        f for f in data_dir.glob("*.csv")
        if not f.name.startswith("_")   # _summary.csv 등 제외
    )

    if not csv_files:
        raise FileNotFoundError(
            f"{data_dir} 에서 종목 CSV를 찾지 못했습니다. collector.py를 먼저 실행하세요."
        )

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, parse_dates=["datetime"])
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    print(f"종목 {len(csv_files)}개, 총 {len(all_df)}건 로드 완료")

    return all_df

# 2. 특징 생성
def add_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    종목별로 스케일에 독립적인 특징을 만든다.
    (가격대가 다른 종목을 하나의 모델로 같이 학습하려면
     원가격이 아니라 수익률/비율/z-score 형태로 변환해야 한다.)
    """

    parts = []

    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("datetime").copy()

        g["log_return"] = np.log(g["close"] / g["close"].shift(1))
        g["hl_range"] = (g["high"] - g["low"]) / g["close"]
        g["oc_range"] = (g["close"] - g["open"]).abs() / g["close"]

        g["volume_change"] = g["volume"].pct_change()

        vol_mean = g["volume"].rolling(window).mean()
        vol_std = g["volume"].rolling(window).std()
        g["volume_zscore"] = (g["volume"] - vol_mean) / vol_std

        price_mean = g["close"].rolling(window).mean()
        price_std = g["close"].rolling(window).std()
        g["price_zscore"] = (g["close"] - price_mean) / price_std

        g["rolling_volatility"] = g["log_return"].rolling(window).std()

        parts.append(g)

    result = pd.concat(parts, ignore_index=True)

    # 롤링 계산으로 생긴 초반 NaN 행 제거 + inf 방지 (거래정지 등으로 volume=0인 구간)
    result = result.replace([np.inf, -np.inf], np.nan)
    before = len(result)
    result = result.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"특징 생성 후 {before}건 -> {len(result)}건 (NaN/inf 제거)")

    return result


# 3. 모델 학습
def train_isolation_forest(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    contamination: float = CONTAMINATION,
) -> tuple[IsolationForest, pd.DataFrame]:
    """전 종목 데이터를 합쳐 하나의 Isolation Forest를 학습한다."""

    X = df[feature_cols].values

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    df = df.copy()
    df["anomaly_score"] = model.decision_function(X)   # 낮을수록 이상치에 가까움
    df["is_anomaly"] = model.predict(X) == -1           # True면 이상치

    return model, df


# 4. 결과 요약
def summarize_results(df: pd.DataFrame) -> None:
    total = len(df)
    n_anomaly = int(df["is_anomaly"].sum())

    print("\n" + "=" * 60)
    print("Isolation Forest 학습 결과")
    print("=" * 60)
    print(f"전체 데이터 건수     : {total}")
    print(f"이상치로 탐지된 건수 : {n_anomaly} ({n_anomaly / total:.2%})")

    print("\n종목별 이상치 개수 (상위 10개):")
    by_symbol = (
        df.groupby("symbol")["is_anomaly"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
    )
    print(by_symbol.to_string())

    print("\n이상치 점수가 가장 낮은(=가장 이상한) 상위 15건:")
    top = df.sort_values("anomaly_score").head(15)
    print(
        top[["symbol", "datetime", "close", "volume", "anomaly_score"]]
        .to_string(index=False)
    )


# 5. (선택) 시각화
def plot_symbol_anomalies(df: pd.DataFrame, symbol: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib이 설치되어 있지 않아 시각화를 건너뜁니다. "
              "`pip install matplotlib` 후 다시 실행해보세요.")
        return

    g = df[df["symbol"] == symbol].sort_values("datetime")

    if g.empty:
        print(f"{symbol}에 대한 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(g["datetime"], g["close"], label="close", linewidth=1)

    anomalies = g[g["is_anomaly"]]
    ax.scatter(
        anomalies["datetime"], anomalies["close"],
        color="red", label="anomaly", zorder=5, s=25
    )

    ax.set_title(f"{symbol} - Isolation Forest Anomalies")
    ax.set_xlabel("datetime")
    ax.set_ylabel("close")
    ax.legend()
    fig.tight_layout()

    out_path = PLOT_DIR / f"{symbol}_anomalies.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"{symbol} 이상치 시각화 저장: {out_path} (이상치 {len(anomalies)}건)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", type=str, default=None,
        help="시각화할 종목 코드 (예: --plot AAPL)"
    )
    parser.add_argument(
        "--contamination", type=float, default=CONTAMINATION,
        help="이상치 비율 (기본 0.02)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="시각화를 아예 생략하고 싶을 때"
    )
    args = parser.parse_args()

    raw_df = load_all_symbol_data()
    feat_df = add_features(raw_df)

    model, result_df = train_isolation_forest(
        feat_df, contamination=args.contamination
    )

    summarize_results(result_df)

    # 결과 저장
    result_path = DATA_DIR / "isolation_forest_results.csv"
    result_df.to_csv(result_path, index=False)
    print(f"\n전체 결과가 {result_path} 에 저장되었습니다.")

    model_path = MODEL_DIR / "isolation_forest.pkl"
    joblib.dump(model, model_path)
    print(f"학습된 모델이 {model_path} 에 저장되었습니다.")

    if args.no_plot:
        print("--no-plot 지정됨: 시각화 생략")
        return

    target_symbol = args.plot.upper() if args.plot else None

    if target_symbol is None:
        # --plot을 안 줬으면 이상치가 가장 많이 나온 종목을 자동으로 그려서 보여준다.
        counts = result_df.groupby("symbol")["is_anomaly"].sum()
        if counts.empty or counts.max() == 0:
            print("이상치가 하나도 없어 자동 시각화를 생략합니다. "
                  "--plot SYMBOL 로 특정 종목을 직접 지정해보세요.")
            return
        target_symbol = counts.idxmax()
        print(f"\n--plot 미지정 -> 이상치가 가장 많은 종목({target_symbol})을 자동으로 시각화합니다.")

    plot_symbol_anomalies(result_df, target_symbol)


if __name__ == "__main__":
    main()