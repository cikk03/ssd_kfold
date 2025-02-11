import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import SSD         # 수정된 SSD.py
import ensemble1   # 수정된 ensemble1.py

# 세션 상태 초기화 (이미지 저장)
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

def save_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.session_state.uploaded_images[uploaded_file.name] = image

def main():
    # 제목에 이모티콘을 추가하여 예쁘게 변경
    st.sidebar.title("✨ 모델 선택 및 이미지 불러오기 ✨")
    choice = st.sidebar.radio("실행할 기능 선택", ("SSD 분석", "Ensemble 실행"))
    
    # 메인 영역에서 이미지 업로드
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        save_uploaded_image(uploaded_file)
        st.success(f"'{uploaded_file.name}' 파일이 저장되었습니다.")
    
    # 저장된 이미지 목록을 사이드바에서 선택 (미리보기 없이)
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
