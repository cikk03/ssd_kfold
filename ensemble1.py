import torch
from torchvision.models.detection import ssd300_vgg16
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ✅ 모델 정의
def get_ssd_model(num_classes):
    model = ssd300_vgg16(pretrained=False)
    model.head.classification_head.num_classes = num_classes
    return model

# 모델 경로 리스트 (Google Drive 경로로 수정)
model_paths = [
    'best_model_ssd_fold1.pth',
    'best_model_ssd_fold2.pth',
    'best_model_ssd_fold3.pth',
    'best_model_ssd_fold4.pth',
    'best_model_ssd_fold5.pth'
]

# GPU 사용 설정 (여기서는 CPU 사용)
device = torch.device('cpu')
print(f"Using device: {device}")

# 모델 로드 함수
def load_model(path, num_classes=6):  # 배경 포함 총 6 클래스
    model = get_ssd_model(num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ✅ 모델 로딩 캐싱 (한번만 로드하도록 캐싱)
@st.cache_resource(show_spinner=False)  # Streamlit 1.18 이상 사용
def get_models():
    return [load_model(path) for path in model_paths]

# 5개의 모델 로드
models = [load_model(path) for path in model_paths]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (300, 300))  # SSD300 입력 크기에 맞춤
    image_normalized = image_resized / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    return torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0).to(device), image

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

# 가중 다수결로 클러스터 내 최종 레이블 결정 함수
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

def ensemble_predictions(image_path, iou_thr=0.5, score_thr=0.5):
    image_tensor, original_image = preprocess_image(image_path)
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
    
    if len(predictions) > 0:
        max_box_val = max([np.max(pred['box']) for pred in predictions])
        if max_box_val <= 1.0:
            for pred in predictions:
                pred['box'] = np.array(pred['box']) * np.array([w, h, w, h])
    
    clusters = cluster_predictions(predictions, iou_threshold=iou_thr)
    print(f"최종 검출 객체 수: {len(clusters)}")
    
    # 수정된 label mapping (배경은 제외)
    label_mapping = {
        1: 'normal',
        2: 'Extruded',
        3: 'Crack',
        4: 'Cutting',
        5: 'Side_stamp'
    }
    
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
    
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.axis('off')
    plt.show()



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
