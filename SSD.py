import torch
from torchvision.models.detection import ssd300_vgg16
import cv2
import numpy as np
import streamlit as st
from io import BytesIO

####################################
# 모델 정의 및 로딩 함수
####################################
def get_ssd_model(num_classes):
    """
    SSD300 VGG16 모델을 생성합니다.
    pretrained=False로 설정하여 기본 가중치 다운로드를 막고,
    커스텀 가중치(여기서는 best_model_ssd.pth)를 로드할 예정입니다.
    """
    model = ssd300_vgg16(pretrained=False)
    model.head.classification_head.num_classes = num_classes
    return model

device = torch.device("cpu")
st.write(f"Using device: {device}")

def load_model(path, num_classes=6):
    """
    가중치 파일(path)을 로드하여 모델에 적용합니다.
    클래스 구성: 0: 배경, 1: normal, 2: Extruded, 3: Crack, 4: Cutting, 5: Side_stamp
    """
    model = get_ssd_model(num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# 캐시를 이용해 모델을 한 번만 로드 (Streamlit 1.18 이상)
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("best_model_ssd.pth", num_classes=6)

####################################
# 이미지 전처리 함수
####################################
def preprocess_image_from_array(image):
    """
    업로드된 RGB 이미지 배열을 SSD300 모델 입력에 맞게 전처리합니다.
    1. 300x300으로 리사이즈
    2. [0,1] 범위로 정규화
    3. 채널, 높이, 너비 순서로 변환 후 텐서 생성
    """
    image_resized = cv2.resize(image, (300, 300))
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device)
    return tensor, image

####################################
# 객체 탐지 및 결과 시각화 함수
####################################
def detect_objects(image, model, score_thr=0.5):
    """
    전처리된 이미지로부터 모델 예측을 수행하고,
    점수 임계값 및 배경(0) 제거를 적용한 후 결과 박스를 원본 이미지에 그립니다.
    """
    image_tensor, original_image = preprocess_image_from_array(image)
    h, w = original_image.shape[:2]

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    # 배경(0) 제거 및 score threshold 적용
    valid_idx = (outputs['scores'] > score_thr) & (outputs['labels'] != 0)
    boxes = outputs['boxes'][valid_idx].cpu().numpy()
    scores = outputs['scores'][valid_idx].cpu().numpy()
    labels = outputs['labels'][valid_idx].cpu().numpy()

    # 만약 좌표가 정규화되어 있다면 원본 이미지 크기로 변환
    if boxes.size > 0 and boxes.max() <= 1.0:
        boxes = boxes * np.array([w, h, w, h])
    boxes = np.round(boxes).astype(int)

    # 라벨 매핑 (배경은 제외)
    label_mapping = {
        1: 'normal',
        2: 'Extruded',
        3: 'Crack',
        4: 'Cutting',
        5: 'Side_stamp'
    }

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = label_mapping.get(int(label), 'Unknown')
        cv2.putText(original_image, f"{label_text} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return original_image

####################################
# Streamlit UI
####################################
def main():
    st.title("🔍 SSD Object Detection")
    st.write("💡 best_ssd_model.pth 가중치를 사용한 객체 탐지 앱")

    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # 업로드된 파일을 넘파이 배열로 디코딩
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="업로드된 이미지", use_container_width=True)

        if st.button("🔎 탐지 실행"):
            with st.spinner("모델 실행 중..."):
                model = get_model()
                result_image = detect_objects(image, model, score_thr=0.5)
            st.image(result_image, caption="탐지 결과", use_container_width=True)

            # 결과 이미지 다운로드 기능
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".jpg", result_bgr)
            if is_success:
                st.download_button(
                    label="📥 결과 다운로드",
                    data=BytesIO(buffer.tobytes()),
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()
