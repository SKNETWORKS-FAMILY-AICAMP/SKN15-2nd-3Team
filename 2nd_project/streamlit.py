import streamlit as st
import pandas as pd
import joblib
import time
import base64
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="EA SPORTS Intro", layout="centered")

@st.cache_resource
def load_model(model_name):
    if model_name.endswith(".pth"):
        pass # 딥러닝 모델로드
    elif model_name.endswith(".pkl"):
        return joblib.load(f"model/{model_name}")
    else:
        raise ValueError("지원하지 않는 모델 형식입니다.")

def predict_dl_model(model, input_df, threshold=0.5):
    pass # 딥러닝 예측

def play_ea_intro():
    if st.session_state.get("intro_played"):
        return

    img_placeholder = st.empty()
    loading = st.empty()
    audio_placeholder = st.empty()

    with open("ea_sports_logo.png", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    html_img = f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{img_base64}' width='400'/>
    </div>
    """
    img_placeholder.markdown(html_img, unsafe_allow_html=True)

    with audio_placeholder:
        st.audio("ea_sports_intro.ogg", format="audio/ogg", autoplay=True)

    for i in range(3):
        loading.markdown(
            f"<h3 style='text-align:center;'>LOADING{'.' * (i % 4)}</h3>",
            unsafe_allow_html=True
        )
        time.sleep(1)

    img_placeholder.empty()
    loading.empty()
    audio_placeholder.empty()

    st.session_state.intro_played = True

def show_team_dashboard(team: str):
    team_display_name = {
        "manchestercity": "맨시티",
        "arsenalfc": "아스날",
        "manchesterunited": "맨유",
        "newcastleunited": "뉴캐슬",
        "liverpoolfc": "리버풀",
        "brighton": "브라이튼",
        "astonvilla": "아스톤 빌라",
        "tottenhamhotspur": "토트넘",
        "brentfordfc": "브렌트포드",
        "fulhamfc": "풀럼",
        "crystalpalace": "크리스탈 팰리스",
        "chelsea": "첼시",
        "wolverhamptonwanderers": "울버햄튼",
        "westhamunited": "웨스트햄",
        "bournemouth": "본머스",
        "nottinghamforest": "노팅엄 포레스트",
        "evertonfc": "에버튼",
        "leicestercity": "레스터 시티",
        "leedsunited": "리즈 유나이티드",
        "southampton": "사우샘프턴"
    }.get(team, team)

    team_file_mapping = {
        "manchestercity": "Manchester City_2023.csv",
        "arsenalfc": "Arsenal FC_2023.csv",
        "manchesterunited": "Manchester United_2023.csv",
        "newcastleunited": "Newcastle United_2023.csv",
        "liverpoolfc": "Liverpool FC_2023.csv",
        "brighton": "Brighton_2023.csv",
        "astonvilla": "Aston Villa_2023.csv",
        "tottenhamhotspur": "Tottenham Hotspur_2023.csv",
        "brentfordfc": "Brentford FC_2023.csv",
        "fulhamfc": "Fulham FC_2023.csv",
        "crystalpalace": "Crystal Palace_2023.csv",
        "chelsea": "Chelsea_players_2023.csv",
        "wolverhamptonwanderers": "Wolverhampton Wanderers_2023.csv",
        "westhamunited": "West Ham United_2023.csv",
        "bournemouth": "AFC Bournemouth_2023.csv",
        "nottinghamforest": "Nottingham Forest_2023.csv",
        "evertonfc": "Everton FC_2023.csv",
        "leicestercity": "Leicester City_2023.csv",
        "leedsunited": "Leeds United_2023.csv",
        "southampton": "Southampton_2023.csv"
    }

    csv_path = f"./data/{team_file_mapping.get(team)}"
    try:
        df = pd.read_csv(csv_path)
        st.header(f"📋 {team_display_name} 선수 명단")
        st.dataframe(df)

        if st.button(f"{team_display_name} 이적 예측 실행"):
            input_data = df.drop(columns=["name"], errors="ignore")
            # st.write("예측에 사용되는 컬럼:", input_data.columns.tolist())

            model = load_model(st.session_state.get("selected_model", "transfer_model.pkl"))

            if st.session_state.selected_model.endswith(".pkl"):
                preds = model.predict(input_data)
                probs = model.predict_proba(input_data)[:, 1]
            else:
                preds, probs = predict_dl_model(model, input_data, threshold=0.5)

            df["이적확률(%)"] = (probs * 100).round(1)
            df["예측"] = preds

            st.subheader("🚨 이적 예상 선수 리스트")
            st.subheader("📊 전체 선수 이적 확률")
            df_sorted = df.sort_values("이적확률(%)", ascending=False)
            st.dataframe(df_sorted[["name", "이적확률(%)"]])

        if st.button("🔙 뒤로가기"):
            del st.session_state.selected_team
            st.rerun()
    except FileNotFoundError:
        st.error(f"{team_display_name} 데이터 파일이 없습니다.")

def main():
    st.markdown(
        """
        <style>
        html, body, .stApp, .block-container, header, [data-testid="stToolbar"] {
            background-color: #FAF9F8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    play_ea_intro()

    model_options = [f for f in os.listdir("model") if f.endswith(".pkl") or f.endswith(".pth")]
    selected_model = st.selectbox("사용할 모델 선택:", model_options)
    st.session_state.selected_model = selected_model

    if "selected_team" in st.session_state:
        show_team_dashboard(st.session_state.selected_team)
        return

    st.title("⚽ AI 이적 예측 시스템")
    st.markdown("**아래 팀 중 하나를 선택하세요:**")

    teams = [
        "manchestercity", "arsenalfc", "manchesterunited", "newcastleunited",
        "liverpoolfc", "brighton", "astonvilla", "tottenhamhotspur",
        "brentfordfc", "fulhamfc", "crystalpalace", "chelsea",
        "wolverhamptonwanderers", "westhamunited", "bournemouth",
        "nottinghamforest", "evertonfc", "leicestercity",
        "leedsunited", "southampton"
    ]

    team_display_name = {
        "manchestercity": "맨시티",
        "arsenalfc": "아스날",
        "manchesterunited": "맨유",
        "newcastleunited": "뉴캐슬",
        "liverpoolfc": "리버풀",
        "brighton": "브라이튼",
        "astonvilla": "아스톤 빌라",
        "tottenhamhotspur": "토트넘",
        "brentfordfc": "브렌트포드",
        "fulhamfc": "풀럼",
        "crystalpalace": "크리스탈 팰리스",
        "chelsea": "첼시",
        "wolverhamptonwanderers": "울버햄튼",
        "westhamunited": "웨스트햄",
        "bournemouth": "본머스",
        "nottinghamforest": "노팅엄 포레스트",
        "evertonfc": "에버튼",
        "leicestercity": "레스터 시티",
        "leedsunited": "리즈 유나이티드",
        "southampton": "사우샘프턴"
    }

    rows = [teams[i:i + 4] for i in range(0, len(teams), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for i, team in enumerate(row):
            if cols[i].button(team_display_name[team]):
                st.session_state.selected_team = team
                st.rerun()

if __name__ == "__main__":
    main()
