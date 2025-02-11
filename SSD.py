import torch
from torchvision.models.detection import ssd300_vgg16
import cv2
import numpy as np
import streamlit as st
from io import BytesIO

####################################
# 모델 정의 및 로딩 함수 (기존 코드 그대로)
####################################
def get_ssd_model(num_classes):
    model = ssd300_vgg16(pretrained=False)
    model.head.classification_head.num_classes = num_classes
    return model

device = torch.device("cpu")
st.write(f"Using device: {device}")

def load_model(path, num_classes=6):
    model = get_ssd_model(num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("best_model_ssd.pth", num_classes=6)

####################################
# 이미지 전처리 및 탐지 함수 (기존 코드 그대로)
####################################
def preprocess_image_from_array(image):
    image_resized = cv2.resize(image, (300, 300))
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device)
    return tensor, image

def detect_objects(image, model, score_thr=0.5):
    image_tensor, original_image = preprocess_image_from_array(image)
    h, w = original_image.shape[:2]
    with torch.no_grad():
        outputs = model(image_tensor)[0]

    valid_idx = (outputs['scores'] > score_thr) & (outputs['labels'] != 0)
    boxes = outputs['boxes'][valid_idx].cpu().numpy()
    scores = outputs['scores'][valid_idx].cpu().numpy()
    labels = outputs['labels'][valid_idx].cpu().numpy()

    if boxes.size > 0 and boxes.max() <= 1.0:
        boxes = boxes * np.array([w, h, w, h])
    boxes = np.round(boxes).astype(int)

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
def main(image=None):
    st.title("🔍 SSD Object Detection")
    st.write("💡 best_ssd_model.pth 가중치를 사용한 객체 탐지 앱")
    
    # 이미지 업로드 처리 (이미지가 전달되지 않은 경우)
    if image is None:
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 업로드 후 바로 이미지는 출력하지 않고, 저장 메시지만 출력
            st.success(f"'{uploaded_file.name}' 파일이 저장되었습니다.")
    
    # 전달받은 이미지가 있다면 (이미 미리 업로드되어 저장된 경우)
    if image is not None:
        # 탐지 실행 버튼을 눌렀을 때만 결과 이미지를 보여줌
        if st.button("🔎 탐지 실행"):
            with st.spinner("모델 실행 중..."):
                model = get_model()
                result_image = detect_objects(image, model, score_thr=0.5)
            st.image(result_image, caption="탐지 결과", use_container_width=True)
            # 결과 이미지 다운로드 버튼 (필요시)
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
