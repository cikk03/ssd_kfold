import torch
from torchvision.models.detection import ssd300_vgg16
import cv2
import numpy as np
import streamlit as st
from io import BytesIO
# matplotlib는 여기서 사용하지 않습니다.
# from matplotlib import pyplot as plt

####################################
# 모델 정의 및 로딩 관련
####################################
def get_ssd_model(num_classes):
    # pretrained 인자는 deprecated 되었다는 경고가 뜨지만, weights=None와 동일합니다.
    model = ssd300_vgg16(pretrained=False)
    model.head.classification_head.num_classes = num_classes
    return model

# 모델 경로 리스트 (Streamlit Cloud에 함께 업로드한 파일 경로)
model_paths = [
    'best_model_ssd_fold1.pth',
    'best_model_ssd_fold2.pth',
    'best_model_ssd_fold3.pth',
    'best_model_ssd_fold4.pth',
    'best_model_ssd_fold5.pth'
]

# GPU 사용 설정 (여기서는 CPU 사용)
device = torch.device('cpu')
st.write(f"Using device: {device}")

def load_model(path, num_classes=6):  # 배경 포함 총 6 클래스 (0: 배경, 1: normal, 2: Extruded, 3: Crack, 4: Cutting, 5: Side_stamp)
    model = get_ssd_model(num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# 모델 로딩 캐싱 (한번만 로드하도록)
@st.cache_resource(show_spinner=False)
def get_models():
    return [load_model(path) for path in model_paths]

####################################
# 이미지 전처리 (이미지 배열 입력)
####################################
def preprocess_image_from_array(image):
    # image는 RGB 형식이라고 가정 (Streamlit에서 변환 후 업로드)
    image_resized = cv2.resize(image, (300, 300))  # SSD300 입력 크기에 맞춤
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device)
    return tensor, image

####################################
# IoU 계산 함수
####################################
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

####################################
# 클러스터링 관련 함수 (가중 다수결)
####################################
def get_cluster_label(cluster):
    label_scores = {}
    for pred in cluster['preds']:
        label = pred['label']
        score = pred['score']
        label_scores[label] = label_scores.get(label, 0) + score
    best_label = max(label_scores, key=label_scores.get)
    return best_label

def cluster_predictions(predictions, iou_threshold=0.5):
    clusters = []
    predictions_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    for pred in predictions_sorted:
        assigned = False
        for cluster in clusters:
            iou_val = compute_iou(pred['box'], cluster['box'])
            if iou_val >= iou_threshold:
                cluster['preds'].append(pred)
                boxes = np.array([p['box'] for p in cluster['preds']])
                cluster['box'] = boxes.mean(axis=0)
                scores = np.array([p['score'] for p in cluster['preds']])
                cluster['score'] = scores.mean()
                cluster['label'] = get_cluster_label(cluster)
                assigned = True
                break
        if not assigned:
            clusters.append({
                'preds': [pred],
                'box': np.array(pred['box']),
                'score': pred['score'],
                'label': pred['label']
            })
    return clusters

####################################
# 앙상블 예측 함수 (이미지 배열, 모델 리스트 인자 사용)
####################################
def ensemble_predictions(image, models, iou_thr=0.5, score_thr=0.5):
    image_tensor, original_image = preprocess_image_from_array(image)
    h, w = original_image.shape[:2]

    predictions = []
    for model in models:
        with torch.no_grad():
            outputs = model(image_tensor)[0]
        # score threshold와 함께 배경(0) 예측 제거
        valid_idx = (outputs['scores'] > score_thr) & (outputs['labels'] != 0)
        boxes = outputs['boxes'][valid_idx].cpu().numpy()
        scores = outputs['scores'][valid_idx].cpu().numpy()
        labels = outputs['labels'][valid_idx].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            predictions.append({
                'box': box,
                'score': score,
                'label': int(label)
            })
    
    # 만약 박스 좌표가 정규화되어 있다면 원본 크기로 변환
    if len(predictions) > 0:
        max_box_val = max([np.max(pred['box']) for pred in predictions])
        if max_box_val <= 1.0:
            for pred in predictions:
                pred['box'] = np.array(pred['box']) * np.array([w, h, w, h])
    
    clusters = cluster_predictions(predictions, iou_threshold=iou_thr)
    st.write(f"최종 검출 객체 수: {len(clusters)}")
    
    # 라벨 매핑 (배경은 제외)
    label_mapping = {
        1: 'normal',
        2: 'Extruded',
        3: 'Crack',
        4: 'Cutting',
        5: 'Side_stamp'
    }
    
    # 원본 이미지에 박스와 라벨 그리기
    for cluster in clusters:
        box = cluster['box']
        score = cluster['score']
        label = cluster['label']
        box = np.round(box).astype(int)
        x1, y1, x2, y2 = box
        cv2.rectangle(original_image, (x1-1, y1-1), (x2+1, y2+1), (0, 255, 0), 2)
        label_text = label_mapping.get(label, 'Unknown')
        cv2.putText(original_image, f'{label_text} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return original_image

####################################
# Streamlit UI
####################################
def main(image=None):
    st.title("🔍 SSD Object Detection Ensemble")
    st.write("💡 SSD300 VGG16 모델 앙상블을 사용한 객체 탐지")
    
    if image is None:
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption="📷 업로드된 이미지", use_container_width=True)
    else:
        st.image(image, caption="불러온 이미지", use_container_width=True)
    
    if image is not None:
        if st.button("🔎 탐지 실행"):
            with st.spinner("모델 실행 중... ⏳"):
                models = get_models()
                result_image = ensemble_predictions(image, models, iou_thr=0.5, score_thr=0.5)
            st.image(result_image, caption="🔍 탐지 결과", use_container_width=True)
            
            # 이미지 다운로드 기능
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

