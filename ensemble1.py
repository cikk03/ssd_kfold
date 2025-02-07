import torch
import torch.nn as nn
import torchvision.models.detection as detection
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# 모델 가중치 경로
model_paths = [
    "best_model_ssd_fold1.pth",
    "best_model_ssd_fold2.pth",
    "best_model_ssd_fold3.pth",
    "best_model_ssd_fold4.pth",
    "best_model_ssd_fold5.pth"
]

def load_model(path):
    model = detection.ssd300_vgg16(pretrained=False)
    num_classes = 5  # 내건 4개+배경1개야
    in_features = model.head.classification.cls_logits.in_features  # 수정된 부분
    model.head.classification.cls_logits = nn.Linear(in_features, num_classes)  # 수정된 부분
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


# 5개의 모델 로드
models = [load_model(path) for path in model_paths]

def ensemble_predict(models, image):
    """ 여러 모델의 출력을 평균내어 앙상블 예측 수행 """
    image_tensor = transform_image(image)
    outputs = [model([image_tensor])[0] for model in models]
    
    # 바운딩 박스 및 점수 앙상블
    boxes = torch.stack([output['boxes'] for output in outputs])
    scores = torch.stack([output['scores'] for output in outputs])
    labels = torch.stack([output['labels'] for output in outputs])
    
    avg_boxes = torch.mean(boxes, dim=0)
    avg_scores = torch.mean(scores, dim=0)
    avg_labels = torch.mode(labels, dim=0)[0]  # 가장 많이 나온 클래스를 선택
    
    return avg_boxes, avg_scores, avg_labels

def transform_image(image):
    """ PIL 이미지를 PyTorch Tensor로 변환 """
    transform = detection.transforms.Compose([
        detection.transforms.ToTensor()
    ])
    return transform(image)

def draw_boxes(image, boxes, scores, labels, threshold=0.5):
    """ 바운딩 박스를 이미지에 그리는 함수 """
    image_np = np.array(image)
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"Class {label}: {score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return Image.fromarray(image_np)

# Streamlit UI
st.title("SSD Ensemble Object Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    boxes, scores, labels = ensemble_predict(models, image)
    result_image = draw_boxes(image, boxes, scores, labels)
    st.image(result_image, caption="Detected Objects", use_column_width=True)

