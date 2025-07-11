import streamlit as st
import pandas as pd
import joblib
import time
import base64
import os
from utils import ImprovedCNN1DClassifier  # 반드시 클래스 정의 필요
import torch

st.set_page_config(page_title="Home - EA SPORTS", layout="wide")

# 사이드바 자동 접기
st.markdown("""
    <script>
        window.addEventListener('DOMContentLoaded', (event) => {
            const sidebarToggle = window.parent.document.querySelector('[data-testid="baseButton-header"]');
            if (sidebarToggle) {
                sidebarToggle.click();  // 사이드바 접기
            }
        });
    </script>
""", unsafe_allow_html=True)

# 모델 로드
@st.cache_resource
def load_model(model_name, input_data):
    if model_name.endswith(".pkl"):
       return joblib.load(f"model/{model_name}")
    elif model_name.endswith(".pth"):
        model = ImprovedCNN1DClassifier(input_dim=13)  # ✅ 모델 구조 선언
        state_dict = torch.load(f"model/{model_name}", map_location=torch.device("cpu"))  # ✅ 파라미터 로딩
        model.load_state_dict(state_dict)
        model.eval()  # ✅ 반드시 추론 모드로 전환
        return model

def predict_model(model_name, model, input_df, threshold=0.5):
    # club_position 컬럼을 21 - 값으로 변환
    input_df = input_df.copy()
    if 'club_position' in input_df.columns:
        input_df['club_position'] = 21 - input_df['club_position']

    if model_name.endswith(".pkl"):
        scaler = joblib.load("./scaler/Random_Forest_scaler_reverse.pkl")
        input_data = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        return model.predict(input_data), model.predict_proba(input_data)[:, 1]

    elif model_name.endswith(".pth"):
        scaler = joblib.load("./scaler/cnn_mlp_scaler.pkl")
        input_data = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        return model.predict(input_data), model.predict_proba(input_data)


# EA intro
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
    audio_placeholder.audio("ea_sports_intro.ogg", format="audio/ogg", autoplay=True)

    for i in range(4):
        loading.markdown(
            f"<h3 style='text-align:center;'>LOADING{'.' * (i % 4)}</h3>",
            unsafe_allow_html=True
        )
        time.sleep(1)

    img_placeholder.empty()
    loading.empty()
    audio_placeholder.empty()

    st.session_state.intro_played = True

    # ✅ CSS 제거 위해 강제 리렌더
    st.rerun()

# 팀 대시보드 표시
def show_team_dashboard(team: str):
    display_name = team_display_name.get(team, team)
    csv_file = f"data/{team}_2023.csv"
    try:
        df = pd.read_csv(csv_file)
        st.header(f"📋 {display_name} 선수 명단")
        st.dataframe(df)

        if st.button(f"{display_name} 이적 예측 실행"):
            input_data = df.drop(columns=["name"], errors="ignore")
            
            # 현재 상태
            # model = st.session_state.get("selected_model", "transfer_model.pkl")

            # 올바른 코드
            model_name = st.session_state.get("selected_model", "transfer_model.pkl")
            model = load_model(model_name, input_data)  
            preds, probs = predict_model(model_name, model, input_data)


            df["이적확률(%)"] = (probs * 100).round(1)
            df["예측"] = preds

            st.subheader("🚨 이적 예상 선수 리스트")
            df_sorted = df.sort_values("이적확률(%)", ascending=False)
            st.dataframe(df_sorted[["name", "이적확률(%)"]])

        if st.button("🔙 뒤로가기"):
            del st.session_state.selected_team
            st.rerun()
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")

# 팀 이름과 로고 매핑
team_display_name = {
    "arsenal": "아스날", "aston": "아스톤 빌라", "bournemouth": "본머스",
    "brentford": "브렌트포드", "brighton": "브라이튼", "chelsea": "첼시",
    "crystal": "크리스탈 팰리스", "everton": "에버튼", "fulham": "풀럼",
    "leeds_united": "리즈 유나이티드", "leicester": "레스터 시티", "liverpool": "리버풀",
    "manchester_city": "맨시티", "manchester_united": "맨유", "newcastle": "뉴캐슬",
    "nottingham": "노팅엄 포레스트", "southampton": "사우샘프턴", "tottenham": "토트넘",
    "westham": "웨스트햄", "wolves": "울버햄튼"
}
team_logo_mapping = {team: f"{team}_logo.png" for team in team_display_name}

# Main
def main():
     # ✅ 인트로 화면에서만 배경색 지정
    if not st.session_state.get("intro_played"):
        st.markdown("""
            <style>
            html, body, .stApp, .block-container, header, [data-testid="stToolbar"] {
                background-color: #FAF9F8 !important;
            }
            </style>
        """, unsafe_allow_html=True)

    # 쿼리 파라미터 처리
    query_params = st.query_params
    if query_params.get("selected_team"):
        st.session_state.selected_team = query_params["selected_team"]
        st.rerun()

    # ✅ 인트로 실행
    play_ea_intro()

    model_options = [f for f in os.listdir("model") if f.endswith((".pkl", ".pth"))]
    selected_model = st.selectbox("사용할 모델 선택:", model_options)
    st.session_state.selected_model = selected_model

    if "selected_team" in st.session_state:
        show_team_dashboard(st.session_state.selected_team)
        return

    st.title("⚽ 프리미어리그 팀")
    # st.markdown("**팀 로고를 클릭하세요:**")

    teams = list(team_display_name.keys())
    rows = [teams[i:i + 5] for i in range(0, len(teams), 5)]
    for row in rows:
        cols = st.columns(len(row))
        for i, team in enumerate(row):
            logo_file = team_logo_mapping.get(team)
            logo_path = f"soccer_team_logo/{logo_file}"

            with cols[i]:
                if os.path.exists(logo_path):
                    with st.form(key=f"{team}_form"):
                        st.markdown(f"""
                            <div style="text-align:center;">
                                <img src="data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}"
                                    style="width:100px; height:100px; object-fit:contain;" />
                            </div>
                        """, unsafe_allow_html=True)

                        # 버튼을 안 보이게 넣고, 로고 전체가 클릭되는 느낌
                        submitted = st.form_submit_button(label=f"{team_display_name[team]}", use_container_width=True)
                        if submitted:
                            st.session_state.selected_team = team
                            st.rerun()



if __name__ == "__main__":
    main()
