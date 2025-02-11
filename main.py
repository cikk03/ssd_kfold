import streamlit as st
import SSD         # SSD.py 모듈 (예: main() 또는 ssd_run() 등)
import ensemble1   # ensemble1.py 모듈 (예: main() 또는 ensemble_run() 등)
import cv2
import numpy as np

# 세션 상태 초기화
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}

def save_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.session_state.uploaded_images[uploaded_file.name] = image

def main():
    st.sidebar.title("모델 선택 및 이미지 불러오기")
    choice = st.sidebar.radio("실행할 기능 선택", ("SSD 분석", "Ensemble 실행"))

    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        save_uploaded_image(uploaded_file)
        st.success(f"'{uploaded_file.name}' 파일이 저장되었습니다.")

    # 저장된 이미지 목록 표시
    if st.session_state.uploaded_images:
        selected_image_name = st.sidebar.selectbox(
            "불러올 이미지를 선택하세요",
            list(st.session_state.uploaded_images.keys())
        )
        image = st.session_state.uploaded_images[selected_image_name]
        st.image(image, caption=selected_image_name, use_container_width=True)
    else:
        st.sidebar.info("아직 저장된 이미지가 없습니다.")
        image = None

    # 선택한 모델 기능 실행
    if image is not None:
        if choice == "SSD 분석":
            # SSD.py에서 처리할 때 이미지 인자로 넘겨줄 수 있도록 함
            SSD.main(image)  # 또는 SSD.ssd_run(image) 등으로 수정
        elif choice == "Ensemble 실행":
            ensemble1.main(image)  # 또는 ensemble1.ensemble_run(image)

if __name__ == "__main__":
    main()
