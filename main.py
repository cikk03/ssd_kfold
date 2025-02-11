import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import SSD
import ensemble1

st.markdown(
    """
    <style>
    body {
        background: url("https://github.com/cikk03/ssd_kfold/blob/main/data_science_bg.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# (나머지 코드는 그대로 사용)
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

def save_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.session_state.uploaded_images[uploaded_file.name] = image

def main():
    st.sidebar.header("✨ 모델 선택 ✨")
    choice = st.sidebar.radio("실행할 기능 선택", ("SSD 분석", "Ensemble 실행"))
    st.sidebar.markdown("---")
    st.sidebar.header("✨ 이미지 불러오기 ✨")
    
    uploaded_file = st.sidebar.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        save_uploaded_image(uploaded_file)
        st.sidebar.success(f"'{uploaded_file.name}' 파일이 저장되었습니다.")
    
    if st.session_state.uploaded_images:
        selected_image_name = st.sidebar.selectbox(
            "불러올 이미지를 선택하세요",
            list(st.session_state.uploaded_images.keys())
        )
        image = st.session_state.uploaded_images[selected_image_name]
    else:
        image = None

    if image is not None:
        if choice == "SSD 분석":
            SSD.main(image)
        elif choice == "Ensemble 실행":
            ensemble1.main(image)

if __name__ == "__main__":
    main()
