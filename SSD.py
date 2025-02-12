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
    labels = labels[keep_idx].cpu().numpy()

    # 라벨 텍스트 매핑
    label_mapping = {
        1: 'normal',
        2: 'Extruded',
        3: 'Crack',
        4: 'Cutting',
        5: 'Side_stamp'
    }
    # 라벨별 색상 매핑 (RGB 순서)
    color_mapping = {
        "normal": (0, 255, 0),
        "extruded": (255, 0, 0),
        "crack": (255, 255, 0),
        "cutting": (0, 0, 255),
        "side_stamp": (255, 0, 255)
    }
    
    detection_results = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        label_text = label_mapping.get(int(label), 'Unknown')
        # 라벨명(소문자)로 색상 지정
        color = color_mapping.get(label_text.lower(), (0, 255, 0))
        # 바운딩 박스 그리기
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트와 배경 그리기
        text = f"{label_text} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # 텍스트를 박스 위에 배치 (여유가 없으면 박스 아래쪽에)
        if y1 - text_height - baseline >= 0:
            text_top = y1 - text_height - baseline
            text_bottom = y1
        else:
            text_top = y1
            text_bottom = y1 + text_height + baseline
        # 텍스트 배경 사각형 (채움)
        cv2.rectangle(original_image, (x1, text_top), (x1 + text_width, text_bottom), color, thickness=-1)
        # 라벨 텍스트를 검정색으로 출력
        cv2.putText(original_image, text, (x1, text_bottom - baseline), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
        
        detection_results.append({
            "label": label_text,
            "score": float(score),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return original_image, detection_results

####################################
# Streamlit UI
####################################
def main(image=None):
    st.title("🔍 SSD Object Detection")
    # (기존 설명 문구는 삭제)

    if image is None:
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.success(f"'{uploaded_file.name}' 파일이 저장되었습니다.")
    
    if image is not None:
        if st.button("🔎 탐지 실행"):
            with st.spinner("모델 실행 중..."):
                model = get_model()
                result_image, detection_results = detect_objects(image, model, score_thr=0.5, nms_thr=0.45)
            
            # 불량/정상 판단 (전체를 둘러싼 박스 외에 추가 박스가 있으면 불량)
            h, w = image.shape[:2]
            tol = 0.05  # 5% 오차 허용

            def is_full_box(box):
                x1, y1, x2, y2 = box
                return (x1 <= tol * w and y1 <= tol * h and x2 >= (1 - tol) * w and y2 >= (1 - tol) * h)
            
            if detection_results:
                other_boxes = [d for d in detection_results if not is_full_box(d["box"])]
                if len(other_boxes) > 0:
                    st.markdown("**불량이 검출되었습니다! 🚨**")
                else:
                    st.markdown("**불량이 검출되지 않았습니다! 🎉**")
            else:
                st.markdown("**탐지 결과가 없습니다!**")
                
            st.image(result_image, caption="🔍 탐지 결과", width=550)
            
            # 결과 이미지 다운로드 버튼 (JPG)
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".jpg", result_bgr)
            if is_success:
                st.download_button(
                    label="📥 결과 이미지 다운로드",
                    data=BytesIO(buffer.tobytes()),
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )
            
            # JSON 결과 다운로드 버튼
            json_str = json.dumps(detection_results, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 결과 JSON 다운로드",
                data=json_str,
                file_name="detection_result.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
