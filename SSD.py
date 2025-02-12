import torch
import torchvision.ops as ops
from torchvision.models.detection import ssd300_vgg16
import cv2
import numpy as np
import streamlit as st
from io import BytesIO
import json

####################################
# 모델 정의 및 로딩 함수
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
# 이미지 전처리 함수
####################################
def preprocess_image_from_array(image):
    image_resized = cv2.resize(image, (300, 300))
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device)
    return tensor, image

####################################
# 객체 탐지 및 결과 시각화 함수 (NMS 적용 및 원본 이미지 복사 사용)
####################################
def detect_objects(image, model, score_thr=0.5, nms_thr=0.45):
    image_tensor, orig = preprocess_image_from_array(image)
    original_image = orig.copy()
    h, w = original_image.shape[:2]

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    valid_idx = (outputs['scores'] > score_thr) & (outputs['labels'] != 0)
    boxes = outputs['boxes'][valid_idx]
    scores = outputs['scores'][valid_idx]
    labels = outputs['labels'][valid_idx]

    if boxes.numel() > 0 and boxes.max() <= 1.0:
        scale_tensor = torch.tensor([w, h, w, h], device=boxes.device)
        boxes = boxes * scale_tensor
    boxes = boxes.round()

    keep_idx = ops.nms(boxes.float(), scores, nms_thr)
    boxes = boxes[keep_idx].cpu().numpy()
    scores = scores[keep_idx].cpu().numpy()
    labels = labels[keep_idx].cpu().num
