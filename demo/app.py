import streamlit as st
import os
import time
import json
import uuid
import requests
import timm
import torch
import cv2
import io
from PIL import Image
from torchvision import transforms as T
from ultralytics import YOLO
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from dotenv import load_dotenv


import numpy as np

from config import *
from blue import get_connector, send_message
import asyncio

load_dotenv()  # 환경 변수 로드

api_url = os.getenv('API_URL')
secret_key = os.getenv('SECRET_KEY')


# 모델과 변환을 로드하는 함수
@st.cache_data
def load_classification_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        'maxxvitv2_rmlp_base_rw_224.sw_in12k',
        num_classes=3,
        checkpoint_path=model_path
    )
    model = model.to(device)
    model.eval()
    return model, device

@st.cache_data
def load_yolo_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # YOLOv8
    model = YOLO(model_path)
    model = model.to(device)
    return model

@st.cache_data
def load_siamese_model(model_path):
    # 파일 경로가 올바른지 확인하고 수정
    model_path = os.path.abspath(model_path)  # 절대 경로로 변환

    # 파일 존재 여부 체크
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일이 존재하지 않습니다. 경로를 확인하세요: {model_path}")
        return None

    K.clear_session()
    model = load_model(model_path, custom_objects=customs_func)
    return model

def process_video(video_file):
    # 임시 파일을 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        temp_video_path = tfile.name


    model = load_yolo_model("./best.pt")

    # 임시 파일 경로를 사용하여 비디오 캡처 객체를 생성
    cap = cv2.VideoCapture(temp_video_path)
    frame_car = 0
    frame_plate = 0
    car_crop = None
    plate_crop = None

    while cap.isOpened() and frame_car < 4 and frame_plate < 2:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        # YOLOv8 Object Detection
        results = model([frame], conf=0.6, vid_stride=True, stream=True, verbose=False)

        # results 처리
        for result in results:
            if 0 in result.boxes.cls and 1 in result.boxes.cls:
                boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
                confidences = result.boxes.conf.to('cpu').numpy().astype(float)
                labels = result.boxes.cls.to('cpu').numpy().astype(int) 

                for box, conf, label in zip(boxes, confidences, labels):
                    if label == 0 and conf >= 0.6: # 번호판 추출 (Confidence Score >= 0.7)
                        print(frame_plate)
                        x_min, y_min, x_max, y_max = box
                        plate_crop = frame[y_min:y_max, x_min:x_max] # plate_crop -> 번호판 이미지 저장 

                        frame_plate += 1
                        
                        # 종료 조건
                        if frame_car == 4 and frame_plate == 2:
                            break
                        
                    elif label == 1 and conf >= 0.65: # 차량 추출 (Confidence Score >= 0.75)
                        x_min, y_min, x_max, y_max = box
                        car_crop = frame[y_min:y_max, x_min:x_max]  # car_crop -> 차량 이미지 저장 
                        frame_car += 1
                        
                        # 종료 조건
                        if frame_car == 4 and frame_plate == 2:
                            break
    
    cap.release()
    os.unlink(temp_video_path)  # 임시 파일 삭제

    return car_crop, plate_crop

# OCR 요청을 수행하는 함수
def ocr_request(api_url, secret_key, image_bytes):
    headers = {'X-OCR-SECRET': secret_key}
    files = [('file', ('image.png', image_bytes, 'image/png'))]
    payload = {'message': json.dumps({
        'images': [{'format': 'png', 'name': 'demo'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }).encode('UTF-8')}
    
    response = requests.post(api_url, headers=headers, files=files, data=payload)
    return response.json()

# JSON 데이터베이스에서 차량 번호에 해당하는 모델 이름을 찾는 함수
def get_vehicle_model_from_db(plate_number, db_path='vehicle_database.json'):
    with open(db_path, 'r', encoding='utf-8') as db_file:  # UTF-8 인코딩으로 파일을 열기
        vehicle_db = json.load(db_file)
        vehicle_info = vehicle_db.get(plate_number)
        return vehicle_info['model_name'] if vehicle_info else None

# JSON 데이터베이스에서 차량 이미지를 찾는 함수
def get_car_image_from_db(plate_number, db_path='vehicle_database.json'):
    with open(db_path, 'r', encoding='utf-8') as db_file:  # UTF-8 인코딩으로 파일을 열기
        vehicle_db = json.load(db_file)
        vehicle_info = vehicle_db.get(plate_number)
        if vehicle_info:
            return vehicle_info.get('car_image'), vehicle_info.get('plate_image')
        return None, None

# two-stream siamese network를 사용하여 두 이미지 간의 유사성 평가
def siamese_predict(model, img1_plate, img2_plate, img1_car, img2_car):
    input1 = (image_size_h_p,image_size_w_p,nchannels)
    input2 = (image_size_h_c,image_size_w_c,nchannels)

    img1 = (process_load(img1_plate, input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img2 = (process_load(img2_plate, input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img3 = (process_load(img1_car, input2)/255.0).reshape(1,input2[0],input2[1],input2[2])
    img4 = (process_load(img2_car, input2)/255.0).reshape(1,input2[0],input2[1],input2[2])

    X = [img1, img2, img3, img4]
    Y_ = model.predict(X)
    result = np.argmax(Y_[0])
    return result

# 결과에 따라 색상을 지정하는 함수
def colored_boolean_text(value, true_color="green", false_color="red"):
    color = true_color if value == True else false_color
    return f'<span style="color: {color}; text-align: center; font-weight: bold; font-size: 18px;">{value}</span>'

def colored_result_text(value, true_color="green", false_color="red"):
    color = true_color if value == True else false_color
    value =  True if value == 1 else False
    return f'<span style="color: {color}; text-align: center; font-weight: bold; font-size: 18px;">{value}</span>'

def colored_pass_text(value, true_color="green", false_color="red"):
    color = true_color if value == "통과" else false_color
    return f'<span style="color: {color}; text-align: center; font-weight: bold; font-size: 18px;">{value}</span>'

def main():
    # Streamlit 인터페이스 구성
    st.title('차량 검증 시스템')

    uploaded_file = st.file_uploader("차량 영상을 업로드하세요.", type=['mp4', 'avi'])

    if uploaded_file is not None:
        # 비디오 처리
        car_crop, plate_crop = process_video(uploaded_file)

        # 결과 표시
        st.success("영상 처리 완료")
        
        # 결과 컬럼 설정
        col1, col2 = st.columns([1, 1])
        
        # 모델 로딩
        model_path = "fine-tuning_cp/epoch16.pth"
        model, device = load_classification_model(model_path)
        
        # 이미지 변환
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with col1:
            # car_crop 이미지 처리 및 모델 추론
            predicted_model = None
            if car_crop is not None:
                image = Image.fromarray(car_crop).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_class = probabilities.argmax().item()
                    predicted_model = ['아반떼', 'K3', 'SM3'][predicted_class]
                    st.image(car_crop, caption='추출된 차량 이미지', use_column_width=True)
                    st.write(f"예측된 차량 모델: {predicted_model}")

            # plate_crop OCR 처리
            plate_text = None
            if plate_crop is not None:
                plate_image = Image.fromarray(plate_crop).convert('RGB')
                buffer = io.BytesIO()
                plate_image.save(buffer, format="JPEG")
                buffer.seek(0)
                ocr_result = ocr_request(api_url, secret_key, buffer.read())
                try:
                    fields = ocr_result['images'][0]['fields']
                    plate_text = ' '.join(field['inferText'] for field in fields if len(field['inferText']) >=3).replace(" ", "")
                    st.write(f"인식된 번호판: {plate_text}")
                except (KeyError, IndexError):
                    st.error("번호판 인식 실패")

        
        # 'queue' 폴더가 없다면 생성
        if not os.path.exists('queue'):
            os.makedirs('queue')

        # car_crop 이미지 저장
        if car_crop is not None:
            car_image_path = os.path.join('queue', 'car_crop.png')
            Image.fromarray(car_crop).save(car_image_path)
            # 이미지 저장 후의 로직...

        # plate_crop 이미지 저장
        if plate_crop is not None:
            plate_image_path = os.path.join('queue', 'plate_crop.png')
            Image.fromarray(plate_crop).save(plate_image_path)

        final_True ="통과"

        with col2:
            # 데이터베이스 확인 및 결과 표시
            if plate_text and predicted_model:
                db_model_name = get_vehicle_model_from_db(plate_text)
                if db_model_name:
                    is_correct_classification = predicted_model == db_model_name
                    st.markdown(f"- 차량 정보 일치 여부:  {colored_boolean_text(is_correct_classification)}", unsafe_allow_html=True)
                    if  is_correct_classification:
                        #st.markdown(f"- 최종 차량 통과 여부:  {colored_pass_text(final_True)}", unsafe_allow_html=True)
                        car_image, plate_image = get_car_image_from_db(plate_text)
                        model_path = "models/model_two_stream_1.tf"
                        siamese_model = load_siamese_model(model_path)
                        # 예를 들어, OCR로 얻은 번호판 이미지와 YOLO로 추출된 차량 이미지를 사용
                        siamese_result = siamese_predict(siamese_model, plate_image, plate_image_path,  car_image, car_image_path)
                        final_decision_text = "통과" if siamese_result == True else "거부"
                        message = "True" if final_decision_text == "통과" else "False"
                        st.markdown(f"- 동일 차량 검증 결과:  {colored_result_text(siamese_result)}", unsafe_allow_html=True)
                        st.markdown(f"- 최종 차량 통과 여부:  {colored_pass_text(final_decision_text)}", unsafe_allow_html=True)

                    
                    # Siamese Network 예측
                    if not is_correct_classification:
                        car_image, plate_image = get_car_image_from_db(plate_text)
                        model_path = "models/model_two_stream_1.tf"
                        siamese_model = load_siamese_model(model_path)
                        # 예를 들어, OCR로 얻은 번호판 이미지와 YOLO로 추출된 차량 이미지를 사용
                        siamese_result = siamese_predict(siamese_model, plate_image, plate_image_path,  car_image, car_image_path)
                        final_decision_text = "통과" if siamese_result == True else "거부"
                        message = "True" if final_decision_text == "통과" else "False"
                        st.markdown(f"- 동일 차량 검증 결과:  {colored_result_text(siamese_result)}", unsafe_allow_html=True)
                        st.markdown(f"- 최종 차량 통과 여부:  {colored_pass_text(final_decision_text)}", unsafe_allow_html=True)
                else:
                    st.error("등록되지 않은 차량입니다.")
        
        if 'bluetooth_connector' in st.session_state:
            send_message(st.session_state.bluetooth_connector, f"P:{plate_text[-4:]}\n")
            time.sleep(0.5)
            send_message(st.session_state.bluetooth_connector, str(message) + "\n")


# 스트림릿 앱 실행
if __name__ == "__main__":
    if 'bluetooth_connector' not in st.session_state:
        st.session_state.bluetooth_connector = get_connector()
        asyncio.run(st.session_state.bluetooth_connector.connect_dev())

    main()