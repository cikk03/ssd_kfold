import torch
from torchvision.models.detection import ssd300_vgg16
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ✅ 모델 정의
def get_ssd_model(num_classes):
    model = ssd300_vgg16(pretrained=True)
    model.head.classification_head.num_classes = num_classes
    return model

# 모델 경로 리스트 (Google Drive 경로로 수정)
model_paths = [
    'https://github.com/cikk03/ssd_kfold/raw/main/best_model_ssd_fold1.pth',
    'https://github.com/cikk03/ssd_kfold/raw/main/best_model_ssd_fold2.pth',
    'https://github.com/cikk03/ssd_kfold/raw/main/best_model_ssd_fold3.pth',
    'https://github.com/cikk03/ssd_kfold/raw/main/best_model_ssd_fold4.pth',
    'https://github.com/cikk03/ssd_kfold/raw/main/best_model_ssd_fold5.pth'
]

# GPU 사용 설정
device = torch.device('cpu')
print(f"Using device: {device}")

# 모델 로드 함수
import requests

def download_model(path, save_path):
    response = requests.get(path, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def load_model(path, num_classes=5):
    local_model_path = path.split('/')[-1]  # 파일 이름 추출
    download_model(path, local_model_path)  # 모델 다운로드
  # num_classes를 실제 클래스 수로 설정
    model = get_ssd_model(num_classes).to(device)
    model.load_state_dict(torch.load(local_model_path, map_location=device))
    model.eval()
    return model



# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (300, 300))  # SSD300 입력 크기에 맞춤
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    return torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device), image

# NMS 적용
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = torch.ops.torchvision.nms(torch.tensor(boxes, dtype=torch.float), torch.tensor(scores, dtype=torch.float), iou_threshold)
    return indices.numpy()

# 바운딩 박스 평균화 함수
def ensemble_boxes_mean(boxes_list, scores_list, labels_list):
    if len(boxes_list) == 0:
        return [], [], []
    
    boxes = np.mean(boxes_list, axis=0)  # 바운딩 박스 좌표 평균
    scores = np.mean(scores_list, axis=0)  # 점수 평균
    labels = np.array(labels_list)  # 레이블 리스트 변환
    
    return boxes, scores, labels

# 앙상블 수행 함수
def ensemble_predictions(image_path, iou_thr=0.4, score_thr=0.5):  # ✅ 점수 임계값 높이고 IoU 낮춤
    image_tensor, original_image = preprocess_image(image_path)
    h, w = original_image.shape[:2]

    boxes_list, scores_list, labels_list = [], [], []

    # 각 모델의 예측 수행
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
    indices = non_max_suppression(boxes_list, scores_list, iou_threshold=0.4)  # ✅ IoU 임계값 낮춤
    print(f"After NMS, remaining {len(indices)} boxes")
    
    boxes, scores, labels = ensemble_boxes_mean(boxes_list, scores_list, labels_list)
    
    scores = np.array(scores_list)[indices]
    labels = np.array(labels_list)[indices]

    # 원본 이미지 크기로 복원
    if len(boxes) > 0 and boxes.max() <= 1.0:  # 박스 좌표가 정규화된 경우에만 변환 수행
        boxes = np.round(boxes * [w, h, w, h]).astype(int)  # ✅ 반올림 적용
        boxes = np.clip(boxes, 0, [w-1, h-1, w-1, h-1])  # ✅ 좌표가 이미지 경계를 벗어나지 않도록 보정
    print(f"Final Scaled Boxes (int type): {boxes.astype(int)}")
    
    # 결과 시각화
    for box, score, label in zip(np.array(boxes).reshape(-1, 4), scores, labels):
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 + 1, y2 + 1  # ✅ 바운딩 박스 크기 미세 조정  # ✅ 좌표값을 정수형으로 변환
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_mapping = {1: 'Extruded', 2: 'Crack', 3: 'Cutting', 4: 'Side_stamp'}
        label_text = label_mapping.get(int(label), 'Unknown')
        cv2.putText(original_image, f'{label_text} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 결과 출력
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

import streamlit as st
from PIL import Image

def streamlit_app():
    


    st.title("SSD Ensemble Object Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_path = "temp_uploaded_image.png"
        image.save(image_path)
        
        # 예측 수행
        with st.spinner("Loading models..."):
            models = [load_model(path) for path in model_paths]  # ✅ 업로드 후 모델 로드
        
        with st.spinner("Processing image..."):
            ensemble_predictions(image_path, models)  # ✅ 예측 수행


streamlit_app()
