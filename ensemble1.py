import os
import glob
import io
import torch
import cv2
import numpy as np
import streamlit as st
from torchvision.models.detection import ssd300_vgg16
from matplotlib import pyplot as plt
from torchvision.ops import nms


# -------------------------
# 1. 모델 및 추론 관련 함수 정의
# -------------------------

# GPU/CPU 설정 (여기서는 CPU 사용)
device = torch.device('cpu')
st.write(f"Using device: {device}")

# SSD300 모델 정의 (클래스 수에 맞게 head 수정)
def get_ssd_model(num_classes):
    model = ssd300_vgg16(pretrained=False)
    model.head.classification_head.num_classes = num_classes
    return model

# 단일 모델 로드 함수
def load_model(path, num_classes=5):
    model = get_ssd_model(num_classes).to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Repository 내에 있는 모델 파일들을 glob로 불러오기 (예: model_ssd_kfold1.pth, model_ssd_kfold2.pth, ...)
model_paths = sorted(glob.glob("model_ssd_kfold*.pth"))
st.write(f"Found model weight files: {model_paths}")

# 여러 모델 로드 (Streamlit 캐시를 이용해 한 번만 로드)
@st.cache_resource
def load_models():
    models = [load_model(path) for path in model_paths]
    return models

models = load_models()

# 이미지 전처리 함수 (업로드된 파일 객체를 받아 cv2를 이용해 디코딩)
def preprocess_image(file_obj):
    # 파일 객체에서 바이트 읽기
    file_bytes = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("이미지 디코딩에 실패하였습니다.")
        return None, None
    # OpenCV는 기본 BGR이므로 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 원본 이미지는 추후 결과 시각화를 위해 따로 저장
    original_image = image.copy()
    # 모델 입력 크기(300x300)에 맞게 리사이즈 및 정규화
    image_resized = cv2.resize(image, (300, 300))
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device)
    return image_tensor, original_image

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    boxes_tensor = torch.tensor(boxes, dtype=torch.float)
    scores_tensor = torch.tensor(scores, dtype=torch.float)
    indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    return indices.numpy()


# 바운딩 박스 평균화 함수 (여러 박스들의 좌표, 점수, 레이블의 평균 계산)
def ensemble_boxes_mean(boxes_list, scores_list, labels_list):
    if len(boxes_list) == 0:
        return [], [], []
    boxes = np.mean(boxes_list, axis=0)  # 바운딩 박스 좌표 평균
    scores = np.mean(scores_list, axis=0)  # 점수 평균
    labels = np.array(labels_list)         # 레이블 리스트 변환
    return boxes, scores, labels

# 앙상블 예측 함수  
def ensemble_predictions(file_obj, iou_thr=0.4, score_thr=0.5):
    image_tensor, original_image = preprocess_image(file_obj)
    if image_tensor is None:
        return None
    h, w = original_image.shape[:2]

    boxes_list, scores_list, labels_list = [], [], []

    # 각 모델에 대해 예측 수행 및 결과 누적
    for model in models:
        with torch.no_grad():
            outputs = model(image_tensor)[0]
        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        boxes_list.extend(boxes.tolist())
        scores_list.extend(scores.tolist())
        labels_list.extend(labels.tolist())

    # NMS 적용
    indices = non_max_suppression(boxes_list, scores_list, iou_threshold=iou_thr)
    st.write(f"After NMS, remaining {len(indices)} boxes")
    
    # 단순 평균 앙상블 (원래 코드의 ensemble_boxes_mean 방식 사용)
    boxes, scores, labels = ensemble_boxes_mean(boxes_list, scores_list, labels_list)
    scores = np.array(scores_list)[indices]
    labels = np.array(labels_list)[indices]

    # 모델 입력은 300x300이므로, 박스 좌표가 [0,1] 범위인 경우 원본 이미지 크기로 복원
    if len(boxes) > 0 and boxes.max() <= 1.0:
        boxes = np.round(boxes * np.array([w, h, w, h])).astype(int)  # 반올림 적용
        boxes = np.clip(boxes, 0, np.array([w - 1, h - 1, w - 1, h - 1]))  # 이미지 경계 보정
    st.write(f"Final Scaled Boxes (int type): {boxes.astype(int) if isinstance(boxes, np.ndarray) else boxes}")

    # 결과 이미지에 바운딩 박스 및 레이블 그리기
    for box, score, label in zip(np.array(boxes).reshape(-1, 4), scores, labels):
        x1, y1, x2, y2 = map(int, box)
        # 미세 조정을 위해 좌표값 조정
        x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 + 1, y2 + 1  
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_mapping = {1: 'Extruded', 2: 'Crack', 3: 'Cutting', 4: 'Side_stamp'}
        label_text = label_mapping.get(int(label), 'Unknown')
        cv2.putText(original_image, f'{label_text} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return original_image

# -------------------------
# 2. Streamlit UI 구성
# -------------------------

st.title("Ensemble SSD Object Detection")

# 이미지 파일 업로드 (jpg, jpeg, png)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # 업로드된 파일의 바이트 데이터를 BytesIO에 담아서 전달 (파일 객체 소진 문제 해결)
    file_bytes = io.BytesIO(uploaded_file.read())
    file_bytes.seek(0)
    
    # 예측 수행
    result_img = ensemble_predictions(file_bytes)
    if result_img is not None:
        st.image(result_img, caption="Detection Result", channels="RGB", use_column_width=True)
