import streamlit as st
import pandas as pd
import joblib
import torch
from utils import ImprovedCNN1DClassifier  # 반드시 클래스 정의 필요
import os
import numpy as np

st.set_page_config(page_title="가상 선수 예측", layout="centered")
st.title("🎯 가상 선수 이적 예측")

# 📦 사용자 입력 받기 전 데이터프레임 준비
input_df = None

# 🎯 사용자 입력
with st.form("prediction_form"):
    age = st.slider("나이", 16, 40, 24)
    apps = st.number_input("출장 경기 수 (apps)", 0, 38, 1)
    goals = st.number_input("골 수", 0, 200, 0)
    assists = st.number_input("도움 수", 0, 200, 0)
    shots = st.number_input("슛 수", 0, 10000, 0)
    rating = st.slider("평점", 5.0, 10.0, 7.0)
    tackles = st.number_input("태클 수", 0, 1000, 0)
    interceptions = st.number_input("인터셉트 수", 0, 1000, 0)
    clearances = st.number_input("클리어 수", 0, 1000, 0)
    key_passes = st.number_input("키패스 수", 0, 1000, 0)
    dribblings = st.number_input("드리블 수", 0, 1000, 0)
    avg_passes = st.number_input("경기당 평균 패스 수", 0, 1000, 0)
    club_position = st.slider("소속팀 리그 순위", 1, 20, 1)

    submitted = st.form_submit_button("예측하기")

    input_df = pd.DataFrame([{
        "age": age,
        "apps": apps,
        "goals": goals,
        "assists": assists,
        "shots": shots,
        "rating": rating,
        "tackles": tackles,
        "interceptions": interceptions,
        "clearances": clearances,
        "key_passes": key_passes,
        "dribblings": dribblings,
        "avg_passes": avg_passes,
        "club_position": 21 - club_position
    }])

# 📦 모델 선택 (입력 이후로 이동)
model_name = st.selectbox("사용할 모델 선택", [f for f in os.listdir("model") if f.endswith((".pkl", ".pth"))])
model = None

# 모델 로드 함수 정의
@st.cache_resource
def load_model(model_name, input_data):
    if model_name.endswith(".pkl"):
        return joblib.load(f"model/{model_name}")
    elif model_name.endswith(".pth"):
        model = ImprovedCNN1DClassifier(input_dim=13)
        state_dict = torch.load(f"model/{model_name}", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model

# 예측 실행
if input_df is not None:
    model = load_model(model_name, input_df)

if submitted:
    if model_name.endswith(".pkl"):
        scaler = joblib.load("./scaler/Random_Forest_scaler_reverse.pkl")
        input_df.iloc[:, :] = scaler.transform(input_df)
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
    else:
        scaler = joblib.load("./scaler/cnn_mlp_scaler.pkl")
        input_df.iloc[:, :] = scaler.transform(input_df)
        pred = model.predict(input_df)
        prob = model.predict_proba(input_df)

        # ✅ 확률이 ndarray인 경우 float로 변환
        if isinstance(prob, (np.ndarray, list)):
            prob = float(prob[0])
        if isinstance(pred, (np.ndarray, list)):
            pred = int(pred[0])

    st.success(f"📊 이적 확률: **{prob * 100:.1f}%**")
    st.info("예상 결과: " + ("🟢 이적" if pred else "🔵 잔류"))