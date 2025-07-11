import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 머신러닝 모델
def show_model_results(model_name: str, model_display: str):
    st.subheader(f"🔍 {model_display}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='text-align: center;font-size: 21px;font-weight: bold;'> Confusion Matrix  </div>",unsafe_allow_html=True)
        st.image(f"image/{model_name}_confusion_matrix.png", width=600)
    with col2:
        st.markdown("<div style='text-align: center;font-size: 21px;font-weight: bold;'> Precision-Recall Curve</div>",unsafe_allow_html=True)

        st.image(f"image/{model_name}_precision_recall_curve.png", width=700)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div style='text-align: center;font-size: 21px;font-weight: bold;'> ROC Curve</div>",unsafe_allow_html=True)
        st.image(f"image/{model_name}_roc_curve.png", width=700)
    with col4:
        st.markdown("<div style='text-align: center;font-size: 21px;font-weight: bold;'> Threshold vs F1/Recall</div>",unsafe_allow_html=True)
        st.image(f"image/{model_name}_threshold_f1_recall.png", width=700)

    # 트리 기반 모델만 feature importance 표시
    if model_name != "DeepLearning":
        st.markdown("<div style='text-align: center;font-size: 21px;font-weight: bold;'> Feature Importance</div>",unsafe_allow_html=True)
        st.image(f"image/{model_name}_feature_importance.png", width=700)



# 페이지 설정
st.set_page_config(page_title="프로젝트 대시보드", layout="wide")

# 사이드바 메뉴
menu = st.sidebar.radio(
    "메뉴 선택",
    ("EDA", "머신러닝", "딥러닝")
)

if menu == "EDA":
    st.title("탐색적 데이터 분석 (EDA) 결과")
    st.subheader("데이터셋에 대한 분석 결과 및 시각화")

    # 첫 번째 행: 타겟 분포도 / 히스토그램
    #col1, col2 = st.columns(2)
    tab1, tab2, tab3,tab4 = st.tabs(['Target Variable Distribution','Histogram of Numerical Features','Boxplot of Key Features','Correlation Heatmap'])
    with tab1:
        st.markdown("### 🎯 Target Variable Distribution")
        st.markdown("타겟 변수(이적 여부)의 클래스 분포를 시각화한 그래프입니다.")
        st.image("image/타겟분포도.png", width=700)

    with tab2:
        st.markdown("### 📊 Histogram of Numerical Features")
        st.markdown("연속형 수치 데이터의 분포와 편향성을 확인합니다.")
        st.image("image/히스토그램.png", width=1000)

    with tab3:
        st.markdown("### 📦 Boxplot of Key Features")
        st.markdown("각 주요 피처별 분포와 이상치(outlier) 여부를 확인할 수 있습니다.")
        st.image("image/박스플롯.png", width=700)

    with tab4:
        st.markdown("### 🔥 Correlation Heatmap")
        st.markdown("수치형 변수 간의 상관관계를 시각화한 히트맵입니다.")
        st.image("image/히트맵.png", width=1000)

elif menu == "머신러닝":
    st.title("머신러닝 모델 분석 및 결과")
    tab1, tab2, tab3 = st.tabs(['Random Forest','Logistic Regression','XGBoost'])

    with tab1:
        show_model_results("Random Forest", "Random Forest")
    with tab2:
        show_model_results("Logistic Regression", "Logistic Regression")

    with tab3:
        show_model_results("XGBoost", "XGBoost")

elif menu == "딥러닝":
    st.title("딥러닝 모델 분석 및 결과")
    st.subheader("CNN + MLP 결과")
    tab1,tab2 = st.tabs(['summary','model architecture'])

    # 딥러닝 모델 결과 표시
    with tab1:
        show_model_results("DeepLearning", "CNN + MLP Model")
        st.markdown("### 🔥 train loss & validation loss")
        st.image("image/loss.png", width=750)
    with tab2:
        st.markdown("### ⭐ Model Architecture")
        st.markdown("##### 저희가 제안하는 모델 구조입니다.")
        st.image("image/architecture.png", width=750)

