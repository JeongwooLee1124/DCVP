import requests
import uuid
import time
import json

api_url = 'https://dilzbvs063.apigw.ntruss.com/custom/v1/24724/083726d2cd84ff03f4f348eb92a208a6c1e5b6c9a05dd0cd40f5ba5988989887/general'
secret_key = 'Y052WmtmdkdtWlFXZXZIZE9nVERjaFRtdklrWEtXTkQ='

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
files = [
  ('file', open(image_file,'rb'))
]
headers = {
  'X-OCR-SECRET': secret_key
}

response = requests.request("POST", api_url, headers=headers, data = payload, files = files)

res = json.loads(response.text.encode('utf8'))
print(res)

with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(res, outfile, indent=4, ensure_ascii=False)