"""
collector.py

한국투자증권(KIS) OpenAPI를 이용해 나스닥100 구성종목의
30분봉 데이터를 30일치 수집하고, 수집 결과를 검증한다.
(KIS 해외주식 분봉조회 API는 공식적으로 최대 약 1개월까지 지원)

1. 한국투자증권 OAuth
2. Header 생성
3. NASDAQ100 종목별 30분봉 수집 (30일치)
4. 수집 결과 저장 (CSV) + 검증
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

import os
import pandas as pd
import requests

from symbols import SYMBOLS  # {"AAPL": "NAS", "ABNB": "NAS", ...} 형태의 dict

# ==========================================================
# Config
# ==========================================================
load_dotenv()
BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_URL = f"{BASE_URL}/oauth2/tokenP"
CHART_URL = f"{BASE_URL}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"

TR_ID = "HHDFS76950200"

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")

TIMEOUT = 30

NMIN = 30                      # 분봉 단위 (분)
DAYS_TO_COLLECT = 30           # 목표 수집 기간 (KIS 분봉 API는 최대 약 1개월까지 지원)
MAX_PAGES_PER_SYMBOL = 60      # 무한루프 방지용 안전장치 (120건 * 60페이지면 30일치에 충분)
SLEEP_BETWEEN_REQUESTS = 0.7   # 페이지 요청 간 대기 (500 에러/초당 거래건수 제한 대응)
SLEEP_BETWEEN_SYMBOLS = 1.0    # 종목 간 대기
MAX_RETRIES = 3                # 요청 실패 시 재시도 횟수
RETRY_BACKOFF_SEC = 1.5        # 재시도 대기 시간 (시도마다 배수 증가)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ERROR_LOG_PATH = DATA_DIR / "_errors.log"


def log_error(message: str) -> None:
    """실패 원인을 파일로 남겨서 나중에 원인 진단이 가능하게 한다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


# ==========================================================
# OAuth
# ==========================================================

class KisAuthenticator:

    def __init__(self, app_key: str, app_secret: str):
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = None

    def issue_token(self) -> str:
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        response = requests.post(TOKEN_URL, json=body, timeout=TIMEOUT)
        response.raise_for_status()

        token = response.json()["access_token"]
        self.access_token = token

        return token


# ==========================================================
# Header
# ==========================================================

def create_header(token: str) -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": TR_ID,
    }


# ==========================================================
# Session
# ==========================================================

class KisSession:

    def __init__(self):
        self.auth = KisAuthenticator(APP_KEY, APP_SECRET)
        token = self.auth.issue_token()
        self.headers = create_header(token)


# ==========================================================
# Collector
# ==========================================================

class OverseasCollector:

    def __init__(self, session: KisSession):
        self.session = requests.Session()
        self.session.headers.update(session.headers)

    def get_30min_candle(
        self,
        symbol: str,
        market: str,
        days: int = DAYS_TO_COLLECT,
    ) -> pd.DataFrame:
        """
        symbol의 30분봉을 과거 방향으로 페이지네이션하며 수집한다.
        - days만큼의 기간이 채워지거나
        - API가 더 이상 데이터가 없다고(more != "Y") 응답하거나
        - MAX_PAGES_PER_SYMBOL 페이지에 도달하면 종료한다.
        """

        all_rows: List[dict] = []

        next_flag = ""
        keyb = ""

        for page in range(MAX_PAGES_PER_SYMBOL):

            params = {
                "AUTH": "",
                "EXCD": market,
                "SYMB": symbol,
                "NMIN": str(NMIN),
                "PINC": "1",
                "NEXT": next_flag,
                "NREC": "120",
                "FILL": "",
                "KEYB": keyb,
            }

            data = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = self.session.get(CHART_URL, params=params, timeout=TIMEOUT)
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.RequestException as e:
                    log_error(f"[{symbol}] page={page} attempt={attempt} 요청 예외: {e}")
                    data = None

                if data is not None and data.get("rt_cd") == "0":
                    break

                # 실패했거나 rt_cd != "0" (ex. 초당 거래건수 초과)인 경우 재시도
                reason = data.get("msg1") if data else "요청 예외"
                log_error(f"[{symbol}] page={page} attempt={attempt} 실패: {reason}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * attempt)

            if data is None or data.get("rt_cd") != "0":
                print(f"  [{symbol}] 재시도 {MAX_RETRIES}회 후에도 실패 (상세: data/_errors.log 참고)")
                break

            raw_output1 = data.get("output1")

            if isinstance(raw_output1, list):
                info = raw_output1[0] if raw_output1 else {}
            elif isinstance(raw_output1, dict):
                info = raw_output1
            else:
                info = {}

            candles = data.get("output2") or []

            # 실제 API 응답은 문서(Y/N)와 다르게 '1'/'0'으로 내려온다.
            # 또한 페이지네이션 지속 여부는 'more'(추가데이터여부)가 아니라
            # 'next'(다음가능여부) 필드를 봐야 한다.
            next_possible = str(info.get("next", "")).strip() == "1"

            debug_line = (
                f"  [{symbol}] page={page} candles={len(candles)} "
                f"more='{info.get('more')}' next='{info.get('next')}' "
                f"nrec='{info.get('nrec')}' rsym='{info.get('rsym')}'"
            )
            print(debug_line)
            log_error(debug_line)

            if not candles:
                break

            for c in candles:
                all_rows.append({
                    "symbol": symbol,
                    "datetime": pd.to_datetime(
                        c["kymd"] + c["khms"], format="%Y%m%d%H%M%S"
                    ),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["last"]),
                    "volume": float(c["evol"]),
                })

            # 목표 기간을 이미 충분히 채웠으면 조기 종료
            oldest_so_far = min(r["datetime"] for r in all_rows)
            newest_so_far = max(r["datetime"] for r in all_rows)
            if (newest_so_far - oldest_so_far).days >= days:
                break

            if not next_possible:
                break

            next_flag = "1"
            last = candles[-1]

            # KIS 문서: "이전 조회 결과의 마지막 분봉 데이터를 이용하여,
            # n분 전의 시간을 입력" → 마지막 캔들 시간을 그대로 넣으면 안 되고
            # NMIN만큼 빼줘야 다음 페이지로 정상적으로 넘어간다.
            last_dt = datetime.strptime(last["kymd"] + last["khms"], "%Y%m%d%H%M%S")
            keyb_dt = last_dt - timedelta(minutes=NMIN)
            keyb = keyb_dt.strftime("%Y%m%d%H%M%S")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        df = pd.DataFrame(all_rows)

        if df.empty:
            return df

        df = (
            df.sort_values("datetime")
              .drop_duplicates("datetime")
              .reset_index(drop=True)
        )

        start = df["datetime"].max() - pd.Timedelta(days=days)

        return df[df["datetime"] >= start].reset_index(drop=True)


# ==========================================================
# 전체 종목 수집
# ==========================================================

def collect_all(
    collector: OverseasCollector,
    symbols: Dict[str, str],
    days: int = DAYS_TO_COLLECT,
) -> Dict[str, pd.DataFrame]:
    """
    symbols(dict: {"AAPL": "NAS", ...})의 전 종목에 대해 데이터를 수집하고,
    종목별로 data/{symbol}.csv 로 저장한다.
    """

    results: Dict[str, pd.DataFrame] = {}
    total = len(symbols)

    for i, (symbol, market) in enumerate(symbols.items(), start=1):
        print(f"[{i}/{total}] {symbol} ({market}) 수집 중...")

        try:
            df = collector.get_30min_candle(symbol=symbol, market=market, days=days)
        except requests.exceptions.RequestException as e:
            print(f"  [{symbol}] 요청 실패: {e} (상세: data/_errors.log 참고)")
            log_error(f"[{symbol}] collect_all 단계 예외: {e}")
            continue

        if df.empty:
            print(f"  [{symbol}] 수집된 데이터 없음 (상세: data/_errors.log 참고)")
            continue

        out_path = DATA_DIR / f"{symbol}.csv"
        df.to_csv(out_path, index=False)

        results[symbol] = df
        print(f"  [{symbol}] {len(df)}건 저장 완료 "
              f"({df['datetime'].min()} ~ {df['datetime'].max()})")

        time.sleep(SLEEP_BETWEEN_SYMBOLS)

    return results


# ==========================================================
# 검증
# ==========================================================

def verify_collection(results: Dict[str, pd.DataFrame], symbols: Dict[str, str]) -> None:
    """
    수집 결과를 요약 출력한다.
    - 종목별 건수 / 기간 / 결측 여부
    - 목표 대비 누락된 종목
    """

    print("\n" + "=" * 60)
    print("수집 결과 검증")
    print("=" * 60)

    summary_rows = []

    for symbol in symbols:
        df = results.get(symbol)

        if df is None or df.empty:
            summary_rows.append({
                "symbol": symbol,
                "rows": 0,
                "start": None,
                "end": None,
                "span_days": None,
                "null_count": None,
            })
            continue

        summary_rows.append({
            "symbol": symbol,
            "rows": len(df),
            "start": df["datetime"].min(),
            "end": df["datetime"].max(),
            "span_days": (df["datetime"].max() - df["datetime"].min()).days,
            "null_count": int(df.isnull().sum().sum()),
        })

    summary = pd.DataFrame(summary_rows)

    missing = summary[summary["rows"] == 0]["symbol"].tolist()
    short = summary[
        (summary["rows"] > 0) & (summary["span_days"] < DAYS_TO_COLLECT * 0.9)
    ]

    print(f"\n총 대상 종목 수     : {len(symbols)}")
    print(f"수집 성공 종목 수   : {(summary['rows'] > 0).sum()}")
    print(f"수집 실패(0건) 종목 : {len(missing)}")
    if missing:
        print(f"  -> {missing}")
    print(f"기간 부족(목표의 90% 미만) 종목: {len(short)}")
    if not short.empty:
        print(short[["symbol", "rows", "span_days"]].to_string(index=False))

    print("\n종목별 요약 (상위 10개):")
    print(summary.head(10).to_string(index=False))

    summary.to_csv(DATA_DIR / "_summary.csv", index=False)
    print(f"\n전체 요약이 {DATA_DIR / '_summary.csv'} 에 저장되었습니다.")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    session = KisSession()
    print("Access Token 발급 완료")

    collector = OverseasCollector(session)

    # 전체 나스닥100 종목 대상 (테스트하려면 아래처럼 일부만 슬라이싱해서 사용)
    # target_symbols = dict(list(SYMBOLS.items())[:5])
    target_symbols = SYMBOLS

    results = collect_all(collector, target_symbols, days=DAYS_TO_COLLECT)
    verify_collection(results, target_symbols)