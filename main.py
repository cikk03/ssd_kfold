import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import SSD         # 수정된 SSD.py
import ensemble1   # 수정된 ensemble1.py

# ----------------------------
# 최신 Streamlit 컨테이너 선택자를 사용해 배경 이미지 적용
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: url("data_science_bg.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ----------------------------

# 세션 상태 초기화 (이미지 저장용)
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

def save_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.session_state.uploaded_images[uploaded_file.name] = image

def main():
    # 사이드바 상단: 모델 선택 섹션
    st.sidebar.header("✨ 모델 선택 ✨")
    choice = st.sidebar.radio("실행할 기능 선택", ("SSD 분석", "Ensemble 실행"))
    
    # 섹션 구분선
    st.sidebar.markdown("---")
    
    # 사이드바 하단: 이미지 불러오기 섹션
    st.sidebar.header("✨ 이미지 불러오기 ✨")
    
    # 파일 업로더 (사이드바에 배치)
    uploaded_file = st.sidebar.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        save_uploaded_image(uploaded_file)
        st.sidebar.success(f"'{uploaded_file.name}' 파일이 저장되었습니다.")
    
    # 저장된 이미지가 있으면 선택할 수 있도록 함 (미리보기 없이)
    if st.session_state.uploaded_images:
        selected_image_name = st.sidebar.selectbox(
            "불러올 이미지를 선택하세요",
            list(st.session_state.uploaded_images.keys())
        )
        image = st.session_state.uploaded_images[selected_image_name]
    else:
        image = None

    # 선택한 기능에 따라 해당 모듈에 이미지를 전달
    if image is not None:
        if choice == "SSD 분석":
            SSD.main(image)
        elif choice == "Ensemble 실행":
            ensemble1.main(image)

if __name__ == "__main__":
    main()
