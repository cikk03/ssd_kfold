import streamlit as st
import torch
import numpy as np
import cv2
from torchvision.models.detection import ssd300_vgg16
import matplotlib.pyplot as plt
from io import BytesIO

# ✅ 모델 정의 함수
def get_ssd_model(num_classes):
    model = ssd300_vgg16(pretrained=False)
    model.head.classification_head.num_classes = num_classes
    return model

# ✅ 모델 파일 경로 (로컬에 저장된 모델 파일 경로로 변경)
model_paths = [
    "best_model_ssd_fold1.pth",
    "best_model_ssd_fold2.pth",
    "best_model_ssd_fold3.pth",
    "best_model_ssd_fold4.pth",
    "best_model_ssd_fold5.pth"
]

# ✅ GPU 설정 (Streamlit에서는 CPU 사용)
device = torch.device("cpu")

# ✅ 모델 로드 함수
def load_model(path, num_classes=5):
    model = get_ssd_model(num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ✅ 모델 로딩 캐싱 (한번만 로드하도록 캐싱)
@st.cache_resource(show_spinner=False)  # Streamlit 1.18 이상 사용
def get_models():
    return [load_model(path) for path in model_paths]

# ✅ 이미지 전처리 함수
def preprocess_image(image):
    image_resized = cv2.resize(image, (300, 300))
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    return torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device), image

# ✅ NMS 적용 함수
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = torch.ops.torchvision.nms(
        torch.tensor(boxes, dtype=torch.float),
        torch.tensor(scores, dtype=torch.float),
        iou_threshold
    )
    return indices.numpy()

# ✅ 바운딩 박스 평균화 함수
def ensemble_boxes_mean(boxes_list, scores_list, labels_list):
    if len(boxes_list) == 0:
        return [], [], []
    
    boxes = np.mean(boxes_list, axis=0)
    scores = np.mean(scores_list, axis=0)
    labels = np.array(labels_list)
    
    return boxes, scores, labels

# ✅ 앙상블 수행 함수
def ensemble_predictions(image):
    image_tensor, original_image = preprocess_image(image)
    h, w = original_image.shape[:2]

    boxes_list, scores_list, labels_list = [], [], []

    # ✅ 캐싱된 모델 로딩 (매 세션당 최초 1회 실행)
    models = get_models()

    # ✅ 각 모델의 예측 수행
    for model in models:
        with torch.no_grad():
            outputs = model(image_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        boxes_list.extend(boxes.tolist())
        scores_list.extend(scores.tolist())
        labels_list.extend(labels.tolist())

    # ✅ NMS 적용
    indices = non_max_suppression(boxes_list, scores_list, iou_threshold=0.4)
    
    # ✅ 박스 평균화 적용
    boxes, scores, labels = ensemble_boxes_mean(boxes_list, scores_list, labels_list)
    
    scores = np.array(scores_list)[indices]
    labels = np.array(labels_list)[indices]

    # ✅ 원본 이미지 크기로 복원
    if len(boxes) > 0 and boxes.max() <= 1.0:
        boxes = np.round(boxes * [w, h, w, h]).astype(int)
        boxes = np.clip(boxes, 0, [w-1, h-1, w-1, h-1])
    
    # ✅ 결과 시각화
    for box, score, label in zip(np.array(boxes).reshape(-1, 4), scores, labels):
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 + 1, y2 + 1
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label_mapping = {1: "Extruded", 2: "Crack", 3: "Cutting", 4: "Side_stamp"}
        label_text = label_mapping.get(int(label), "Unknown")
        
        cv2.putText(original_image, f"{label_text} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return original_image

# ✅ Streamlit UI
def main():
    st.title("🔍 SSD Object Detection Ensemble")
    st.write("💡 SSD300 VGG16 모델 앙상블을 사용한 객체 탐지")
    
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # ✅ 업로드된 이미지 읽기
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ✅ 원본 이미지 출력
        st.image(image, caption="📷 업로드된 이미지", use_container_width=True)
        
        if st.button("🔎 탐지 실행"):
            with st.spinner("모델 실행 중... ⏳"):
                result_image = ensemble_predictions(image)
                
                # ✅ 결과 이미지 표시
                st.image(result_image, caption="🔍 탐지 결과", use_container_width=True)
                
                # ✅ 이미지 다운로드 기능 추가
                img_rgb = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                is_success, buffer = cv2.imencode(".jpg", img_rgb)
                if is_success:
                    st.download_button(
                        label="📥 결과 다운로드",
                        data=BytesIO(buffer.tobytes()),
                        file_name="detection_result.jpg",
                        mime="image/jpeg"
                    )

if __name__ == "__main__":
    main()
