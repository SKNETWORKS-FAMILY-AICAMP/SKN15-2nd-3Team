# SKN15-2nd-3Team


# 1. 팀 소개


<p align="center">
  <img width="225" height="225" alt="9k" src="https://github.com/user-attachments/assets/96d63267-a1ac-4f45-9dbc-7c7f4f91c0b6" />
</p>


<div align="center">

## 비상이조 ✈️



| 박진우 | 권주연 | 서혜선 | 정민철 | 한승희 |
|:---:|:---:|:---:|:---:|:---:|
| [@pjw876](https://github.com/pjw876) | [@juyeonkwon](https://github.com/juyeonkwon) | [@hyeseon](https://github.com/hyeseon7135) | [@jeong-mincheol](https://github.com/jeong-mincheol) | [@seunghee-han](https://github.com/seunghee-han) |



</div>

<br />


# 2. 프로젝트 기간

- 2025.07.10 ~ 2025.07.11 (총 2일)



<br />

# 3. 프로젝트 개요


## 📕 프로젝트명

**FootTrade (축구선수 이적률 예측 시스템)**

## ✅ 프로젝트 배경 및 목적

현대 축구에서 선수단 관리는 단순한 영입을 넘어, 핵심 인재의 잔류와 효율적인 이직 관리가 구단의 장기적인 성공에 필수적인 요소가 되었습니다. 이 프로젝트는 선수의 경기 성적, 계약 상황, 개인 특성, 팀 내 역할 등 다양한 데이터를 종합적으로 분석하여 선수의 **구단 이탈 가능성(이직률)** 을 예측하는 데 목적이 있습니다. 이를 통해 구단은 잠재적 이탈 선수를 사전에 파악하고, 핵심 선수 잔류를 위한 맞춤형 전략을 수립하며, 보다 안정적이고 효율적인 선수단 운영을 도모할 수 있습니다.



## 🖐️ 프로젝트 소개

- 축구선수 개인 데이터(출전 경기 수, 득점, 도움, 계약 기간, 부상 이력, 연봉 추정치 등)를 활용, 선수의 구단 잔류 또는 이탈 가능성을 분류
- 최신 머신러닝 알고리즘을 적용하여 선수의 이직 여부를 확률값으로 예측하고, 주요 영향 요소를 도출
- 데이터 수집 및 전처리부터 모델링, 성능 평가, 결과 시각화에 이르는 전체 데이터 분석 파이프라인을 체계적으로 구현
- Streamlit 기반의 사용자 친화적인 인터페이스를 구축, 구단 관계자의 예측 결과 직관적 확인 및 활용 지원

  

## ❤️ 기대효과

- 선수단 관리 효율성 증대: 잠재적 이탈 선수 조기 식별, 선제적인 관리 및 협상 전략 수립에 기여
- 핵심 선수 잔류율 향상: 데이터 기반 인사이트를 통해 선수 개개인에게 최적화된 잔류 유인책 마련 지원
- 장기적인 구단 운영 안정화: 선수단 재편성 리스크 최소화, 안정적인 팀 전력 유지를 위한 의사결정 지원
- 데이터 기반 인력 관리 역량 강화: 구단 내 데이터 분석 및 예측 모델 활용 역량 증진

  

## 👤 대상 사용자

- 구단 운영진 및 경영진
- 선수단 관리팀 및 스카우팅 부서
- 인력 관리 및 재무 담당자 등 구단 내부 관계자


<br />


# 4. 기술 스택

 - 프로그래밍 언어: Python
 - 데이터 분석: pandas, numpy, matplotlib, seaborn
 - 머신러닝 모델: Scikit-learn (Random Forest, Logistic Regression)
 - 딥러닝 모델: TensorFlow, Keras
 - 버전 관리: GitHub
 - 기타 도구: Jupyter Notebook, Google Colab, Kaggle





<br />

# 5. 수행결과(분석 및 예측 결과)

<br />

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


<br />


## ✅ 딥러닝 모델 아키텍처


<img width="2937" height="2697" alt="Image" src="https://github.com/user-attachments/assets/da9fa2bf-b833-45cc-94ad-a3cd13056bce" />
  <br />
  
- 1D Convolution 기반의 **Residual Block** 구조와 Fully Connected **MLP**를 결합하여 설계
- Conv1D 기반의 잔차 연결(Residual Connection)을 통해 **특성 손실을 최소화** 및 **계층 간 깊이를 확보**하며 학습 안정성을 확보
- Feature 추출 후에는 3층의 MLP를 거쳐 **이진 분류를 수행**
- 입력 데이터는 StandardScaler로 정규화된 수치형 탭형 데이터, **채널을 추가한 후 Conv1D로 처리**

  <br />
  

## ✅ 머신러닝 성능 분석

| <img width="1200" height="1200" alt="Random Forest_confusion_matrix" src="https://github.com/user-attachments/assets/1920a8ef-6fac-4fc5-9ecb-9938c00b14ca" /> | <img width="1800" height="1200" alt="Random Forest_precision_recall_curve" src="https://github.com/user-attachments/assets/ffc276a0-d4c5-49b6-82fc-c0a949219b9d" /> |
|:-----------------------------------:|:-------------------------------------:|
| confusion_matrix | precision_recall_curve |
| <img width="1800" height="1200" alt="Random Forest_roc_curve" src="https://github.com/user-attachments/assets/fd6f9527-8fe6-436c-ab3e-2c8aeccb9614" /> | <img width="1800" height="1200" alt="Random Forest_threshold_f1_recall" src="https://github.com/user-attachments/assets/c413da57-0089-4c0b-bc4b-0126ea6f1259" /> |
| roc_curve | threshold_f1_recall  |


<br />


## ✅ 딥러닝 성능 분석


| <img width="1200" height="1200" alt="DeepLearning_confusion_matrix" src="https://github.com/user-attachments/assets/72fa5fb0-1515-471f-8c1f-7615bee9f00b" /> | <img width="1800" height="1200" alt="DeepLearning_precision_recall_curve" src="https://github.com/user-attachments/assets/cd0a3a3c-9202-4610-a9e1-c4436e43b296" /> |
|:-----------------------------------:|:-------------------------------------:|
| confusion_matrix | precision_recall_curve |
| <img width="1800" height="1200" alt="DeepLearning_roc_curve" src="https://github.com/user-attachments/assets/dc0437ae-a29f-4196-b9ee-2eeb7100063a" /> | <img width="1920" height="1440" alt="Image" src="https://github.com/user-attachments/assets/33fae3ba-deb7-4a8f-adcd-a4325f4397b3" /> |
| roc_curve | threshold_f1_recall  |
|<img width="1650" height="1200" alt="Image" src="https://github.com/user-attachments/assets/be5f2e5f-e124-41bb-91d2-15bc5bbec7f5" />| <img width="579" height="146" alt="Image" src="https://github.com/user-attachments/assets/e870ade0-3ee3-4e1e-b8e3-c96cf1de218d" /> |
|Epoch-wise Training and Validation Accuracy |result |


<br />



<br />


# 6. 아쉬운 점
- 이번 이적 예측 모델은 경기 내 스탯만을 기반으로 만들어졌기 때문에, 실제 이적에 영향을 주는 다양한 **외부 요인들(구단 사정, 선수 의지 등)** 을 반영하지 못함
- 또한, 포지션별로 중요한 지표가 다름에도 불구하고 이를 세분화하지 못하고 동일한 기준으로 예측한 점도 한계
 



<br />


# 7. 한 줄 회고

<p align="center" width="100%">

|박진우|권주연|서혜선|
|----|---|---|
|EDA, 시각화, 모델링의 과정을 거치면서 전반적인 AI 프로젝트가 어떻게 이루어지는지에 대해 배울 수 있어서 값진 경험이 되었습니다.|모델링과 시각화를 직접 맡으며 시행착오를 반복했고, 그 과정에서 데이터 균형화부터 임계값 조정, 성능 해석까지 전반적인 흐름을 알게되었습니다. 맡은 파트에 최선을 다해준 우리 팀원 최고입니다❤️|아직 머신러닝과 딥러닝을 잘 모르는 상태여서 어떻게 할 수 있을까 걱정이 많았는데 다들 도와주시고 모르는 부분들은 설명도 잘 해주셔서 몰랐던 부분 등을 알게 되어 뜻깊은 시간이였습니다. 부족한 저를 같이 이끌어 가느라 고생하셨을텐데 수고 많으셨습니다. 감사합니다.|



|정민철|한승희|
|----|---|
|데이터 분석부터 모델 선정,학습 및 평가까지 해보면서 배워갈 수 있었고 서로서로 맡은 부분에 대해서 열심히 해준 팀원들 너무 고생하셨습니다.|실제 데이터를 활용해 전처리, 시각화, 모델링 까지 직접 수행해보며 빅데이터 분석가 맛보기를 해본것 같아서 즐거운 시간이였습니다. 끝까지 최선을 다해 열심히해준 팀원들과 함께해 성공적으로 프로젝트를 마무리할 수 있었던 것 같습니다. 수고하셨습니다❤️|

</p>


