import requests
import uuid
import time
import json
import cv2
import os

"웹캠 프레임 받아와서 추가 추론"

api_url = ''
secret_key = ''

image_file = './input/1.png'
output_file = './output/1.json'

request_json = {
    'images': [
        {
            'format': 'png',
            'name': 'demo'
        }
    ],
    'requestId': str(uuid.uuid4()),
    'version': 'V2',
    'timestamp': int(round(time.time() * 1000))
}
payload = {'message': json.dumps(request_json).encode('UTF-8')}

cap = cv2.VideoCapture(0)           
if cap.isOpened(): 
    while True: 
        ret, img = cap.read() 
        if ret:    
            # cv2.imshow('camera', img)

            tmp_img = "tmp_img.png"
            cv2.imwrite("tmp_img.png", img)

            files = [('file', open(tmp_img,'rb'))]
            headers = {'X-OCR-SECRET': secret_key}
            response = requests.request("POST", api_url, headers=headers, data = payload, files = files)

            res = json.loads(response.text.encode('utf8'))
            # print(res)

            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(res, outfile, indent=4, ensure_ascii=False)

            # 일단 1장만 하고 break             
            break
            if cv2.waitKey(1) == 27:    
                break
        else: 
            print('no frame')     
            break
    
else:
    print("Can't open video.") 

cap.release()
cv2.destroyAllWindows()