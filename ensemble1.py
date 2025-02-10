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

def ensemble_predictions(image):
    image_tensor, original_image = preprocess_image(image)
    h, w = original_image.shape[:2]

    boxes_list, scores_list, labels_list = [], [], []
    models = get_models()  # 캐싱된 모델 불러오기

    # 각 모델의 예측 결과 수집
    for model in models:
        with torch.no_grad():
            outputs = model(image_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()   # 300×300 기준 좌표
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        # 각 모델의 결과를 리스트에 추가 (extend 대신 append로 개별 배열을 저장한 후 concatenate)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # 각 모델의 예측 결과를 모두 합치기
    if len(boxes_list) > 0:
        all_boxes = np.concatenate(boxes_list, axis=0)
        all_scores = np.concatenate(scores_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
    else:
        return original_image

    # NMS 적용
    indices = non_max_suppression(all_boxes.tolist(), all_scores.tolist(), iou_threshold=0.4)
    final_boxes = all_boxes[indices]
    final_scores = all_scores[indices]
    final_labels = all_labels[indices]

    # (이미지가 300x300이므로 좌표 변환은 필요 없음)

    # 결과 시각화
    for box, score, label in zip(final_boxes, final_scores, final_labels):
        x1, y1, x2, y2 = map(int, box)
        # (옵션) 경계를 약간 확장
        x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 + 1, y2 + 1
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label_mapping = {1: "Extruded", 2: "Crack", 3: "Cutting", 4: "Side_stamp"}
        label_text = label_mapping.get(int(label), "Unknown")
        cv2.putText(original_image, f"{label_text} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return original_image


    # NMS 적용 (리스트가 아닌 np.array를 리스트로 변환하여 사용)
    indices = non_max_suppression(all_boxes.tolist(), all_scores.tolist(), iou_threshold=0.4)

    final_boxes = all_boxes[indices]
    final_scores = all_scores[indices]
    final_labels = all_labels[indices]

    # 만약 박스 좌표가 정규화된 값이라면 원본 이미지 크기로 복원
    if final_boxes.shape[0] > 0 and final_boxes.max() <= 1.0:
        final_boxes = np.round(final_boxes * [w, h, w, h]).astype(int)
        final_boxes = np.clip(final_boxes, 0, [w-1, h-1, w-1, h-1])
    
    # 결과 시각화
    for box, score, label in zip(final_boxes, final_scores, final_labels):
        x1, y1, x2, y2 = map(int, box)
        # 경계를 약간 확장해서 표시
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
