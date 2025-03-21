import sys
import os

# 현재 파일(app.py)의 디렉토리 경로를 기준으로 상위 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ai_tutoring import analyze_code
from models.vector_search import build_faiss_index, VectorDB
from models.data_processing import load_and_preprocess_data
import streamlit as st

# ✅ 현재 `app.py` 파일이 위치한 `streamlit_app` 폴더의 상위 디렉토리를 기준으로 `BASE_DIR` 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ 절대 경로로 데이터 파일 경로 설정
file_paths = [
    os.path.join(BASE_DIR, "data", "code_data-1.csv"),
    os.path.join(BASE_DIR, "data", "code_data-2.csv")
]
df_correct, df_error = load_and_preprocess_data(file_paths)

# 유사 코드 검색을 위한 FAISS 벡터화 및 DB 생성
index, vectorizer = build_faiss_index(df_correct["code"])
vector_db = VectorDB(index, vectorizer, df_correct)

st.set_page_config(page_title="AI 코딩 튜터", layout="wide")
st.title("💡 AI 코딩 튜터: 당신만을 위한 알고리즘 가이드")
st.write("AI가 사용자의 코드를 분석하고, 단계별 힌트와 최적화 피드백을 제공합니다.")

# 사용자 코드 입력
user_code = st.text_area("✍️ 코드 입력 (Python)", height=200, placeholder="여기에 코드를 입력하세요...")

if st.button("🔍 분석 요청"):
    if len(user_code.strip()) == 0:
        st.warning("코드를 입력해주세요!")
    else:
        response = analyze_code(user_code, vector_db=vector_db)
        st.subheader("🛠 AI 분석 결과")
        st.write(response)
