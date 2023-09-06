import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import requests
import uuid
import time
import json
################################################################################
model_path = 'models/RRDB_ESRGAN_x4.pth'
test_img_folder = 'lr/*'
device = torch.device('cuda')
api_url = ''
secret_key = ''

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
##################################################################################
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 1
for path in glob.glob(test_img_folder):
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('sr/{:s}_sr.png'.format(base), output)
    
    print('SR image save to `./sr/{:s}_sr.png`'.format(base))
    
    files = [
        ('file', open('sr/{:s}_sr.png'.format(base),'rb'))
    ]
    headers = {
      'X-OCR-SECRET': secret_key
    }
    
    response = requests.request("POST", api_url, headers=headers, data = payload, files = files)
    res = json.loads(response.text.encode('utf8'))
    
    txt = ''
    for i in range(len(res['images'][0]['fields'])):
        txt += res['images'][0]['fields'][i]['inferText']
        
    print(txt)
    
    with open('ocr/{:s}_ocr.json'.format(base), 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, indent=4, ensure_ascii=False)
        
    print("OCR results save to `./ocr/{:s}.json`".format(base))