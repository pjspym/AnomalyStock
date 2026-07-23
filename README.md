# 2026 인공지능 부트캠프 프로젝트
# 나스닥100 이상치(Anomaly) 탐지

한국투자증권(KIS) OpenAPI로 나스닥100 종목의 30분봉 데이터를 수집하고,
Isolation Forest로 이상 패턴을 탐지해 웹에서 확인하는 프로젝트.

- 데이터 수집: `collector.py` (KIS OpenAPI)
- 모델 학습: `train_isolation_forest.py` (scikit-learn IsolationForest)
- 백엔드: `api.py` (FastAPI)
- 프론트엔드: `app.py` (Streamlit, 관심종목 위시리스트 + Plotly 캔들차트)

## 사용 기술
- Python
- 한국투자증권(KIS) OpenAPI - 데이터 수집
- pandas - 데이터 처리
- scikit-learn - Isolation Forest
- FastAPI - 백엔드 API
- Streamlit - 프론트엔드
- Plotly - 캔들차트 시각화

## 파일 구성

```
symbols.py                  나스닥100 종목코드-거래소코드 매핑
symbol_name.py              나스닥100 종목코드-기업이름 매핑
collector.py                KIS OAuth, 30분봉 수집기
train_isolation_forest.py   특징 생성 + Isolation Forest 학습
api.py                      FastAPI 백엔드 (/predict)
app.py                      Streamlit 프론트엔드
data/                       수집된 종목별 CSV, 결과 CSV (자동 생성)
models/                     학습된 모델 pkl (자동 생성)
requirements.txt            프로젝트 실행에 필요한 Python 패키지 목록
.env                        KIS API 인증 정보 및 환경 변수 설정
```

## 준비

```bash
pip install -r requirements.txt
```

`.env` 에 본인의 KIS OpenAPI 키를 입력한다.

```python
APP_KEY = "발급받은 appkey"
APP_SECRET = "발급받은 appsecret"
```

## 실행 순서

### 1. 데이터 수집

```bash
python collector.py
```

- `symbols.py`에 정의된 전 종목의 30분봉을 최대 30일치(실측상 KIS가 실제로 보유한
  기간은 약 28일 정도) 수집해 `data/{종목}.csv`로 저장한다.
- 페이지네이션(KEYB)으로 과거 데이터를 이어받고, 요청 실패 시 자동 재시도한다.
- 실패/재시도 로그는 `data/_errors.log`, 종목별 수집 요약은 `data/_summary.csv`에 남는다.
- 토큰은 `.kis_token_cache.json`에 캐싱되어 23시간 동안 재사용된다 (짧은 시간에
  반복 발급 시 KIS가 403으로 막는 문제 방지).

### 2. 모델 학습

```bash
python train_isolation_forest.py
```

- `data/*.csv`를 모두 읽어 종목별로 스케일에 독립적인 특징을 만든다:
  로그수익률, 고저폭, 시종가폭, 거래량 변화율, 거래량/가격 롤링 z-score,
  롤링 변동성 (윈도우 20개 = 30분봉 기준 약 10시간).
- 전 종목 데이터를 합쳐 **하나의** `IsolationForest`를 학습한다 (기본
  `contamination=0.02`).
- 결과: `data/isolation_forest_results.csv`, 모델: `models/isolation_forest.pkl`.
- 이상치가 가장 많이 나온 종목의 캔들차트를 `plots/`에 자동 저장한다
  (`--plot SYMBOL`로 특정 종목 지정 가능, `--no-plot`으로 생략 가능).

### 3. 웹 서비스 실행

터미널 2개 필요:

```bash
# 터미널 1 - 백엔드
uvicorn api:app --reload

# 터미널 2 - 프론트엔드
streamlit run app.py
```

## 웹 UI 사용법

1. 종목을 선택하고 "추가"를 누르면 **관심종목(위시리스트)** 에 담긴다
   (최대 6개, 중복 추가 불가).
2. 추가 즉시 해당 종목의 예측이 실행되어 결과가 표시된다.
3. 관심종목 이름 옆의 **✕** 버튼으로 개별 삭제 가능.
4. "전체 새로고침"으로 관심종목 전체를 다시 예측할 수 있다 (실시간 데이터라
   시간이 지나면 결과가 바뀔 수 있음).
5. 각 종목마다 최대 2열 그리드로 다음이 표시된다:
   - `"YYYY-MM-DD HH:MM 이상패턴 발견"` 또는 `"이상 패턴이 발견되지 않았습니다"`
   - 최근 3거래일 캔들차트 (Plotly, 이상치는 ▽ 빨간 마커로 표시, 데이터
     없는 구간(주말/장마감~개장 사이)은 x축을 카테고리로 처리해 제거)

## 예측 로직 (`api.py`)

- 종목 요청이 들어오면 KIS API로 최근 6일치 30분봉을 실시간으로 가져온다.
- 캔들 사이 시간 간격이 3시간 이상 벌어지는 지점을 기준으로 "거래세션"을 나눈다.
  - 가장 최근 세션이 자동으로 "오늘 장 시작 ~ 요청 시점" 이 된다.
  - 아직 오늘 장이 시작하지 않았다면 자동으로 "직전 거래일 전체"가 된다.
  - 주말에는 KIS에 토·일 데이터가 없으므로 자동으로 금요일 세션이 잡힌다.
  - 별도의 날짜/요일 분기 코드 없이 데이터 자체로 판별하는 방식이라, 별도
    공휴일 캘린더 없이도 대체로 맞아떨어진다.
- 학습 때와 동일한 `add_features()` 로직을 그대로 재사용해 특징을 계산하고,
  저장된 모델로 최근 세션의 각 캔들이 이상치인지 예측한다.
- 이상치가 하나라도 있으면 그중 가장 최근 시점을 메시지에 사용한다.

## 일자별 진행 내용
### 1일차 - 아이디어 구상 및 데이터 확보

나스닥100 종목의 이상 패턴을 자동 탐지하는 아이디어를 구상하고, 한국 투자증권 OpenAPI를 데이터 소스로 선정했습니다. OAuth인증, 헤더 생성, 종목 리스트 관리 등 기본 골격을 구성했습니다. 해외주식 분봉 조회 API로 30분봉 데이터를 수집하는 기능을 개발했습니다. 페이지네이션 파라미터 해석 오류 등을 수정하며 실제 동작하는 수집 파이프라인을 완성했습니다.

### 2일차 - 프로토타입 개발 및 API설계

수집된 데이터를 바탕으로 로그수익률, 변동성, 거래량, z-score 등 특징을 설계하고 Isolation Forest 학습 스크립트를 개발했습니다. 전 종목 데이터를 통합해 하나의 모델로 학습하는 구조를 확정했습니다. 학습된 모델을 서빙하기 위한 FastAPI 백엔드를 설계하고 `/predict` 엔드포인트를 구현했습니다. 실시간 요청 시점 기준으로 최신 거래세션을 판별하는 로직을 설계했습니다

### 3일차 - UI연동 및 통합 테스트

Streamlit으로 프론트엔드를 구현해 백엔드 API와 연동했습니다. mplfinance/matplotlib 에서 Plotly캔들차트로 시각화 방식을 전환하고, 데이터 공백구간(주말/장외시간)이 표시되지 않도록 개선했습니다. 관심종목을 추가·삭제할 수 있는 위시리스트 UI(최대6개)를 구현했습니다. 전체 파이프라인(수집→학습→예측→시각화)을 통합해 실제 동작을 테스트했습니다

### 4일차 - 하이퍼파라미터 튜닝 및 배포

Isolation Forest의 `contamination`값을 여러차례 조정하며 이상치 탐지 민감도를 튜닝했습니다. FastAPI백엔드와 Streamlit프론트엔드를 각각 Docker이미지로 빌드해 실행환경을 일관되게 구성했습니다. Render를 통해 컨테이너를 배포하여 별도 서버 관리 없이 서비스를 외부에 노출했습니다. 배포 과정에서 발생한 의존성/포트 설정 이슈를 점검하며 안정적으로 서비스가 구동되도록 마무리 했습니다.

## 알려진 제약 / 참고사항

- KIS 해외주식 분봉 조회 API는 문서상 "최대 약 1개월" 이지만, 실측상 약 28일
  전후에서 데이터가 끊긴다. 30일을 요청해도 그보다 조금 적게 모일 수 있다.
- 거래세션 구분은 3시간 gap 휴리스틱이라, 거래가 뜸한 종목은 정상 거래일
  중에도 세션이 잘못 쪼개질 수 있다 (`api.py`의 `SESSION_GAP_HOURS`로 조정).
- `IsolationForest`의 `contamination`(이상치 비율) 은 도메인 지식에 맞게
  `train_isolation_forest.py --contamination` 값으로 튜닝 필요.
- KIS 토큰 재발급을 짧은 시간에 반복하면 403이 날 수 있다
  (`.kis_token_cache.json` 캐시로 대부분 방지됨).
