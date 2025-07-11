# SKN15-2nd-3Team


# 1. 팀 소개


<p align="center">
  <img src="./images/team_image_ex.png" alt="이적하지말아조 팀 이미지" width="300"/>
</p>


<div align="center">

## 이적하지말아조



| 박진우 | 권주연 | 서혜선 | 정민철 | 한승희 |
|:---:|:---:|:---:|:---:|:---:|
| [@pjw876](https://github.com/pjw876) | [@juyeonkwon](https://github.com/juyeonkwon) | [@hyeseon](https://github.com/hyeseon7135) | [@jeong-mincheol](https://github.com/jeong-mincheol) | [@seunghee-han](https://github.com/seunghee-han) |



</div>




# 2. 프로젝트 기간

- 2025.07.10 ~ 2025.07.11 (총 2일)





# 3. 프로젝트 개요


## 📕 프로젝트명

**FootTrade(축구선수 이적률 예측 시스템)**

## ✅ 프로젝트 배경 및 목적

현대 축구에서 선수단 관리는 단순한 영입을 넘어, 핵심 인재의 잔류와 효율적인 이직 관리가 구단의 장기적인 성공에 필수적인 요소가 되었습니다. 이 프로젝트는 선수의 경기 성적, 계약 상황, 개인 특성, 팀 내 역할 등 다양한 데이터를 종합적으로 분석하여 선수의 **구단 이탈 가능성(이직률)**을 예측하는 데 목적이 있습니다. 이를 통해 구단은 잠재적 이탈 선수를 사전에 파악하고, 핵심 선수 잔류를 위한 맞춤형 전략을 수립하며, 보다 안정적이고 효율적인 선수단 운영을 도모할 수 있습니다.

## 🖐️ 프로젝트 소개

- 축구선수 개인 데이터(출전 경기 수, 득점, 도움, 계약 기간, 부상 이력, 연봉 추정치 등)를 활용하여 선수의 구단 잔류 또는 이탈 가능성을 분류합니다.
- 최신 머신러닝 알고리즘을 적용하여 선수의 이직 여부를 확률값으로 예측하고, 주요 영향 요소를 도출합니다.
- 데이터 수집 및 전처리부터 모델링, 성능 평가, 결과 시각화에 이르는 전체 데이터 분석 파이프라인을 체계적으로 구현하였습니다.
- Streamlit 기반의 사용자 친화적인 인터페이스를 구축하여 구단 관계자가 예측 결과를 직관적으로 확인하고 활용할 수 있도록 지원합니다.

## ❤️ 기대효과

- 선수단 관리 효율성 증대: 잠재적 이탈 선수를 조기에 식별하여 선제적인 관리 및 협상 전략 수립에 기여합니다.
- 핵심 선수 잔류율 향상: 데이터 기반의 인사이트를 통해 선수 개개인에게 최적화된 잔류 유인책 마련을 돕습니다.
- 장기적인 구단 운영 안정화: 선수단 재편성 리스크를 최소화하고, 안정적인 팀 전력 유지를 위한 의사결정을 지원합니다.
- 데이터 기반 인력 관리 역량 강화: 구단 내 데이터 분석 및 예측 모델 활용 역량을 높이는 실질적인 기회를 제공합니다.

## 👤 대상 사용자

- 구단 운영진 및 경영진
- 선수단 관리팀 및 스카우팅 부서
- 인력 관리 및 재무 담당자 등 구단 내부 관계자





# 4. 기술 스택

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white">







# 5. 수행결과(분석 및 예측 결과)

## ✅ EDA

| <img width="1920" height="1440" alt="타겟분포도" src="https://github.com/user-attachments/assets/6d12c86e-2dc6-40f2-8541-9bb816dfcf6f" /> |
|:-------------------------------------:|
| Target Distribution |

| <img width="5400" height="4500" alt="히스토그램" src="https://github.com/user-attachments/assets/334e6010-a3fa-438f-a206-47c4b817e4ec" /> |
|:-------------------------------------:|
| Histogram |

| <img width="2400" height="6300" alt="박스플롯" src="https://github.com/user-attachments/assets/bc1744c1-1824-4de0-92f6-ee6a9b2055ad" /> |
|:-------------------------------------:|
| BoxPlot |

| <img width="6000" height="4500" alt="히트맵" src="https://github.com/user-attachments/assets/f0bd4209-9808-46e8-8a9a-02266f357305" /> |
|:-------------------------------------:|
| Heatmap |

## ✅ 머신러닝 성능 분석
| **Random Forest**  |
| <img width="1200" height="1200" alt="Random Forest_confusion_matrix" src="https://github.com/user-attachments/assets/1920a8ef-6fac-4fc5-9ecb-9938c00b14ca" /> | <img width="1800" height="1200" alt="Random Forest_precision_recall_curve" src="https://github.com/user-attachments/assets/ffc276a0-d4c5-49b6-82fc-c0a949219b9d" /> |
|:-----------------------------------:|:-------------------------------------:|
| confusion_matrix | precision_recall_curve |
| <img width="1800" height="1200" alt="Random Forest_roc_curve" src="https://github.com/user-attachments/assets/fd6f9527-8fe6-436c-ab3e-2c8aeccb9614" /> | <img width="1800" height="1200" alt="Random Forest_threshold_f1_recall" src="https://github.com/user-attachments/assets/c413da57-0089-4c0b-bc4b-0126ea6f1259" /> |
| roc_curve | threshold_f1_recall  |

## ✅ 딥러닝 성능 분석
| **DeepLearning**  |
| <img width="1200" height="1200" alt="DeepLearning_confusion_matrix" src="https://github.com/user-attachments/assets/72fa5fb0-1515-471f-8c1f-7615bee9f00b" /> | <img width="1800" height="1200" alt="DeepLearning_precision_recall_curve" src="https://github.com/user-attachments/assets/cd0a3a3c-9202-4610-a9e1-c4436e43b296" /> |
|:-----------------------------------:|:-------------------------------------:|
| confusion_matrix | precision_recall_curve |
| <img width="1800" height="1200" alt="DeepLearning_roc_curve" src="https://github.com/user-attachments/assets/dc0437ae-a29f-4196-b9ee-2eeb7100063a" /> | <img width="1800" height="1200" alt="DeepLearning_threshold_f1_recall" src="https://github.com/user-attachments/assets/b4868ab9-ddd0-4b91-9fe7-61a2f60e3027" /> |
| roc_curve | threshold_f1_recall  |


 

# 6. 한 줄 회고

<p align="center" width="100%">

|박진우|권주연|서혜선|
|----|---|---|
|박진우님의 회고|권주연님의 회고|서혜선님의 회고|



|정민철|한승희|
|----|---|
|데이터 분석부터 모델 선정,학습 및 평가까지 서로서로 맡은 부분에 대해서 열심히 해준 팀원들 너무 고생하셨습니다.|한승희님의 회고|

</p>


