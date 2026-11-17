# 머신러닝 분류 모형 실습

분류 모형 비교, 스태킹, 로지스틱 회귀와 랜덤 포레스트 해석을 다룬 분석 코드

- 대학원 과정에서 직접 작성한 분석 코드
- 원자료와 보고서 원문은 비공개
- 실행 결과는 코드와 같은 이름의 `.md` 파일 (코드, 출력, 그림을 함께 표시)

| 폴더 | 내용 | 방법 | 도구 |
|---|---|---|---|
| [datamining-stacking](datamining-stacking) | 이진 분류 예측 | 11개 분류기 비교, 스태킹 앙상블 | R |
| [turnover-logistic-ml](turnover-logistic-ml) | MZ세대 공무원 이직의도 | 로지스틱 회귀, 상호작용, 이상치 진단, 머신러닝 비교 | R |

## 방법 설명

- **스태킹 앙상블**: 여러 모형의 예측값을 다시 입력으로 써서 최종 예측을 만드는 방법
- **LASSO**: 예측에 기여가 작은 변수의 계수를 0으로 만들어 변수를 골라 주는 회귀 방법
- **랜덤 포레스트**: 의사결정나무 여러 개의 예측을 모아 최종 예측을 내는 머신러닝 모형
- **부분의존도(PDP)**: 한 변수의 값이 바뀔 때 예측이 평균적으로 어떻게 달라지는지 보여 주는 그림

## 스태킹 분류 모형

`stacking_classification.Rmd`

- 주어진 학습, 시험 자료로 이진 분류 예측
- kNN, 나이브 베이즈, 로지스틱 회귀, GLM, Elastic Net, LDA, QDA, SVM(선형, 방사형, 다항), 랜덤 포레스트, XGBoost를 10겹 5회 반복 교차검증으로 학습해 비교
- 개별 모형의 예측을 입력으로 쓰는 스태킹 모형을 메타 학습기 11종으로 각각 구성해 성능 비교

## MZ세대 공무원 이직의도

`01_logistic_regression.Rmd`, `02_ml_model_comparison.Rmd`, `03_ranger_pdp.Rmd`

- 2023 공직생활실태조사에서 MZ세대(1980년 이후 출생) 응답자를 골라 이직의도(이진)를 예측
- 역문항 처리, 요인 점수 계산, 신뢰도 분석 후 LASSO로 예측변수를 선택해 로지스틱 회귀 적합
- 상호작용 항(평균 중심화)을 넣은 모형과 비교하고, 표준화 잔차와 Cook's distance로 이상치를 찾아 재분석
- caret으로 여러 머신러닝 모형을 비교하고, 랜덤 포레스트(ranger)의 부분의존도(PDP)로 주요 변수 효과 확인
- 실행 결과: [`01_logistic_regression.md`](turnover-logistic-ml/01_logistic_regression.md), [`02_ml_model_comparison.md`](turnover-logistic-ml/02_ml_model_comparison.md), [`03_ranger_pdp.md`](turnover-logistic-ml/03_ranger_pdp.md)

**로지스틱 회귀 결과**

![로지스틱 회귀 모형 비교](turnover-logistic-ml/figures/logistic_model_comparison.png)

- 상호작용 항을 넣으면 AIC가 낮아지고 Hosmer-Lemeshow 검정도 유의하지 않게 되지만(p = .287), 판별력(AUC)은 거의 같음
- 표준화 잔차와 Cook's distance 기준을 모두 넘는 이상치 9개를 제거한 뒤 재분석

**머신러닝 모형 15개 비교**

![머신러닝 모형 비교](turnover-logistic-ml/figures/ml_model_comparison.png)

- 트리 기반 앙상블(랜덤 포레스트, GBM, XGBoost)의 교차검증 AUC가 높고, 평가 자료 정확도는 선형 모형과 큰 차이 없음 (.71~.73)

**랜덤 포레스트(ranger) 해석** (10겹 5회 반복 교차검증 AUC .828, 평가 자료 AUC .799, 정확도 .729)

![랜덤 포레스트 변수 중요도](turnover-logistic-ml/figures/rf_variable_importance.png)

![부분의존도](turnover-logistic-ml/figures/rf_partial_dependence.png)

- 공직 만족과 조직몰입이 3점 안팎보다 낮아지면 이직의도 있음 확률이 크게 높아지고, 그 위 구간에서는 거의 평평
- 보수와 인센티브, 성과관리제도는 같은 방향이지만 변화 폭이 작음

- 이 분석의 자료로 이어 간 랜덤 포레스트와 SHAP 해석: [public-sector-turnover-shap](https://github.com/wls0kim/public-sector-turnover-shap)
