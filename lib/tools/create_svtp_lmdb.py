import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import six
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import re
import glob 

import scipy.io as sio


def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  if imageBin == b'':
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)

  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True


def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      try:
        txn.put(k.encode(), v)
      except Exception as e:
        continue
      # txn.put(k.encode(), v)


def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  assert(len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  # print(nSamples)
  env = lmdb.open(outputPath, map_size=1099511627776)
  cache = {}
  cnt = 1
  for i in tqdm(range(nSamples)):
    imagePath = imagePathList[i]
    label = labelList[i]
    if len(label) == 0:
      continue
    if not os.path.exists(imagePath):
      print('%s does not exist' % imagePath)
      continue
    with open(imagePath, 'rb') as f:
      imageBin = f.read()
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue

    imageKey = 'image-%09d' % cnt
    labelKey = 'label-%09d' % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = label.encode()
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = lexiconList[i].encode()
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt-1
  cache['num-samples'] = str(nSamples).encode()
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":
  import scipy.io as sio
  data_dir = "/home/aimenext2/Datasets/"
  data_desc = sio.loadmat(data_dir + "IIIT5K/" + "testdata.mat")
  # print(data_desc['testdata']['mediumLexi'])
  imageNames = data_desc['testdata']['ImgName']
  groundtruth = data_desc['testdata']['GroundTruth']
  smallLexi = data_desc['testdata']['smallLexi']
  mediumLexi = data_desc['testdata']['mediumLexi']
  print(groundtruth[0][0][0])
  data = '/home/aimenext2/manhnh/aster.pytorch/data/'
  lmdb_output_path = '/home/aimenext2/manhnh/aster.pytorch/data/IIIT5K_lexicon50/'
  # import json
  # from json.decoder import JSONDecoder
  # import codecs
  # f = open(data_dir + 'IC03/IC03.json','r')
  # f_read = f.read()
  # decoded_data=codecs.decode(f_read.encode(), 'utf-8-sig')
  # datas = json.loads(decoded_data).get('imagelist').get('image')
  # print(len(datas.get('imagelist').get('image')))

  imagePathList, labelList, lexiconList = [], [], []
  for img, label, lexi50, lexi1000 in zip(imageNames[0], groundtruth[0], smallLexi[0], mediumLexi[0]):
    im_dir = data_dir + "IIIT5K/" + img[0]
    label = label[0]
    # print(im_dir)
    # print(label)
    lexi50_str = ''
    for lexicon in lexi50[0]:
      for word in lexicon:
        print(word)
        lexi50_str += word + ' '
      # print(lexi50[0])
    print(lexi50_str)
    imagePathList.append(im_dir)
    labelList.append(label)
    lexiconList.append(lexi50_str)


  #   imagePathList.append(im_dir)
  #   # print(data['_tag'])
  #   labelList.append(data['_tag'])
    
  createDataset(lmdb_output_path, imagePathList, labelList, lexiconList)
  


  # lmdb_output_path_test = '/home/aimenext2/manhnh/aster.pytorch/data/lmdb_test/'
  # gt_file = '/home/aimenext2/manhnh/svt/gt_text_train.txt'
  # data_dir = '/home/aimenext2/manhnh/svt/'
  # lable_dirs = glob.glob(data_dir + '*.png')
  # set_lable = list(lable_dirs)
  # max_len = 0
  # for i, label in enumerate(set_lable):
  #   imgs = glob.glob(label + '/*')
  #   for img in imgs:
  #     imagePathList.append(img)
  #     labelList.append(label.split('/')[-1])
  #     print(label.split('/')[-1])
  #     print(img)
  # print(len(imagePathList))
  # with open(gt_file, 'r') as f:
  #   lines = [line.strip('\n') for line in f.readlines()]

  # for line in lines:
  #   im_path = line.split(" ")[0]
  #   lable = line.split(" ")[1]
  #   imagePathList.append(im_path)
  #   labelList.append(lable)


  # for i in range(len(lines)):
    # splits = line.split(' ')
    # print(len(lines))
    # image_name = "sentence_" + str(i)+".png"
    # if os.path.exists(os.path.join(image_dir, image_name)):
    #   gt_text = lines[i]
    #   imagePathList.append(os.path.join(image_dir, image_name))
    #   labelList.append(gt_text)
    # gt_text = lines[i+1]
    # # print(image_name)
    # imagePathList.append(os.path.join(image_dir, image_name))
    # labelList.append(gt_text)
  # # print(str(len(imagePathList))+" "+ str(len(labelList)))
  # mark = int(len(imagePathList)*9/10)
  # print(mark)
  # createDataset(lmdb_output_path_test, imagePathList[mark+1:len(imagePathList)], labelList[mark+1:len(labelList)])