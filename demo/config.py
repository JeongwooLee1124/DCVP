import os
import cv2
import numpy as np
from collections import Counter
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import Input, Model, backend as K
from tensorflow.keras.layers import Dense, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing import image
import albumentations as albu


def f1_score(y_true, y_pred):
    # 정밀도 계산
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    
    # 재현율 계산
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    
    # F1 스코어 계산
    f1_score = 2 * ((precision * recall) / (precision + recall + 1e-7))
    
    return f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

kmetrics = {"class_output": ['accuracy', f1_score]}
customs_func = {"f1score": f1_score}
POS = 1  # positive class
NEG = 0  # negative class
batch_size = 128
NUM_EPOCHS = 10
nchannels = 3  # number of channels
image_size_w_p = 96  # image´s width for registration plate
image_size_h_p = 48  # image´s height for registration plate
image_size_w_c = 64  # image´s width for vehicle´s shape
image_size_h_c = 64  # image´s height for vehicle´s shape
path = "data"
folder_cross1 = f'{path}/dataset2/Camera1'
folder_cross2 = f'{path}/dataset2/Camera2'
plt_name = "classes"
car_name = "classes_carros"
plt_folder = "*/classes"
car_folder = "*/classes_carros"
ocr_file = f'{path}/OCR/output.txt'
metadata_length = 35
tam_max = 3
L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))
train_augs = [[],[],[],[],[],[]]
test_augs = [[],[],[],[]]
keys = ['Set01','Set02','Set03','Set04','Set05']

seq_car = albu.Compose(
  [
      albu.CropAndPad(px=(0, 8)),  # IAACropAndPad 대신에 CropAndPad 사용
      albu.Affine(scale=(0.8, 1.2), shear=(-8, 8), cval=0, mode=cv2.BORDER_CONSTANT),  # IAAAffine 대신에 Affine 사용
      albu.ToFloat(max_value=255)
  ], p=0.7
)

seq_car2 = albu.Compose(
  [
      albu.CropAndPad(px=(0, 8)),
      albu.Affine(scale=(0.8, 1.2), shear=(-8, 8), cval=0, mode=cv2.BORDER_CONSTANT),
  ], p=0.7
)

seq_plate = albu.Compose(
  [
      albu.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-5, 5), shear=(-16, 16), cval=0, mode=cv2.BORDER_CONSTANT),
      albu.ToFloat(max_value=255)
  ], p=0.7
)

AUGMENTATIONS_TEST = albu.Compose([
    albu.ToFloat(max_value=255)
])

tam_max = 10  # tam_max 값을 정의
train_augs = [[] for _ in range(6)]  # train_augs 리스트를 초기화
test_augs = [[] for _ in range(4)]  # test_augs 리스트를 초기화

for i in range(tam_max):
  train_augs[0].append(seq_plate)
  train_augs[1].append(seq_plate)
  train_augs[2].append(seq_car)
  train_augs[3].append(seq_car)
  train_augs[4].append(seq_car2)
  train_augs[5].append(seq_car2)
  test_augs[0].append(AUGMENTATIONS_TEST)
  test_augs[1].append(AUGMENTATIONS_TEST)
  test_augs[2].append(AUGMENTATIONS_TEST)
  test_augs[3].append(AUGMENTATIONS_TEST)

#------------------------------------------------------------------------------
def fold(list1, ind, train=False):
    _list1 = list1.copy()
    _list1.pop(ind)
    if train:
        return [_list1[i % len(_list1)] for i in [ind+2,ind+3]]
    else:
        return [_list1[i % len(_list1)] for i in [ind+4,ind+5]]
#------------------------------------------------------------------------------
def calculate_metrics(ytrue1, ypred1):
    conf = metrics.confusion_matrix(ytrue1, ypred1,  labels=[0,1])
    maxres = (conf[1,1],
              conf[0,0],
              conf[0,1],
              conf[1,0],
        metrics.precision_score(ytrue1, ypred1) * 100,
        metrics.recall_score(ytrue1, ypred1) * 100,
        metrics.f1_score(ytrue1, ypred1) * 100,
        metrics.accuracy_score(ytrue1, ypred1) * 100)
    return maxres

#------------------------------------------------------------------------------
def test_report(model_name, model, test_gen):
    print("=== Evaluating model: {:s} ===".format(model_name))
    a = open("%s_inferences_output.txt" % (model_name), "w")
    ytrue, ypred = [], []
    for i in range(len(test_gen)):
      X, Y, paths = test_gen[i]
      Y_ = model.predict(X)
      for y1, yreg, y2, p0, p1 in zip(Y_[0].tolist(), Y_[1].tolist(), Y['class_output'].argmax(axis=-1).tolist(), paths[0], paths[1]):
        y1_class = np.argmax(y1)
        ypred.append(y1_class)
        ytrue.append(y2)
        a.write("%s;%s;%d;%d;%f;%s\n" % (p0, p1, y2, y1_class, yreg[0], str(y1)))

    a.write('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))
    a.close()
#------------------------------------------------------------------------------
def process_load(f1, vec_size):
    _i1 = image.load_img(f1, target_size=vec_size)
    _i1 = image.img_to_array(_i1, dtype='uint8')
    return _i1

def load_img(img, vec_size, vec_size2, metadata_dict):
  iplt0 = process_load(img[0][0], vec_size)
  iplt1 = process_load(img[2][0], vec_size)
  iplt2 = process_load(img[1][0], vec_size2)
  iplt3 = process_load(img[3][0], vec_size2)

  d1 = {"i0":iplt0,
        "i1":iplt1,
        "i2":iplt2,
        "i3":iplt3,
        "l":img[4],
        "p1":img[0][0],
        "p2":img[2][0],
        "c1":img[5]['color'],
        "c2":img[5]['color']
        }
  if metadata_dict is not None:
    diff = abs(np.array(metadata_dict[img[0][0]][:7]) - np.array(metadata_dict[img[2][0]][:7])).tolist()
    diff = [1 if i else 0 for i in diff]
    d1['metadata'] = np.array(metadata_dict[img[0][0]] + metadata_dict[img[2][0]] + diff)
  return d1

#------------------------------------------------------------------------------
class SiameseSequence(Sequence):
    def __init__(self,features, 
                augmentations,
                batch_size=batch_size,
                input1=(image_size_h_p,image_size_w_p,nchannels),
                input2=(image_size_h_c,image_size_w_c,nchannels), 
                type1=None,
                metadata_dict=None, 
                metadata_length=0, 
                with_paths=False):
        self.features = features
        self.batch_size = batch_size
        self.vec_size = input1
        self.vec_size2 = input2
        self.type = type1
        self.metadata_dict = metadata_dict
        self.metadata_length = metadata_length
        self.augment = augmentations
        self.with_paths = with_paths

    def __len__(self):
        return int(np.ceil(len(self.features) / 
            float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch = self.features[start:end]
        futures = []
        _vec_size = (len(batch),) + self.vec_size
        b1 = np.zeros(_vec_size)
        b2 = np.zeros(_vec_size)
        _vec_size2 = (len(batch),) + self.vec_size2
        b3 = np.zeros(_vec_size2)
        b4 = np.zeros(_vec_size2)
        blabels = np.zeros((len(batch)))
        p1 = []
        p2 = []
        c1 = []
        c2 = []
        if self.metadata_length>0:
            metadata = np.zeros((len(batch),self.metadata_length))

        i1 = 0
        for _b in batch:
            res = load_img(_b, self.vec_size, self.vec_size2, self.metadata_dict)
            if self.augment is not None:
                b1[i1,:,:,:] = self.augment[0][0](image=res['i0'])["image"]
                b2[i1,:,:,:] = self.augment[1][0](image=res['i1'])["image"]
                b3[i1,:,:,:] = self.augment[2][0](image=res['i2'])["image"]
                b4[i1,:,:,:] = self.augment[3][0](image=res['i3'])["image"]
            else:
                b1[i1,:,:,:] = res['i0']
                b2[i1,:,:,:] = res['i1']
                b3[i1,:,:,:] = res['i2']
                b4[i1,:,:,:] = res['i3']
            blabels[i1] = res['l']
            p1.append(res['p1'])
            p2.append(res['p2'])
            c1.append(res['c1'])
            c2.append(res['c2'])
            if self.metadata_length>0:
                metadata[i1,:] = res['metadata']
            i1+=1
        blabels2 = np.array(blabels).reshape(-1,1)
        blabels = to_categorical(blabels2, 2)
        y = {"class_output":blabels, "reg_output":blabels2}
        if self.type is None:
            result = [[b1, b2, b3, b4], y]
        elif self.type == 'plate':
            result = [[b1, b2], y]
        elif self.type == 'car':
            result = [[b3, b4], y]
        if self.metadata_length>0:
            result[0].append(metadata)
        if self.with_paths:
            result += [[p1,p2]]

        return result
