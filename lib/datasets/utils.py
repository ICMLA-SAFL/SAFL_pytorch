# from __future__ import absolute_import

import string

import glob

# from . import to_torch, to_numpy

# from __future__ import absolute_import

import torch


def to_numpy(tensor):
  if torch.is_tensor(tensor):
    return tensor.cpu().numpy()
  elif type(tensor).__module__ != 'numpy':
    raise ValueError("Cannot convert {} to numpy array"
                     .format(type(tensor)))
  return tensor


def to_torch(ndarray):
  if type(ndarray).__module__ == 'numpy':
    return torch.from_numpy(ndarray)
  elif not torch.is_tensor(ndarray):
    raise ValueError("Cannot convert {} to torch tensor"
                     .format(type(ndarray)))
  return ndarray

def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
  '''
  voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
  '''
  # datas = glob.glob("../../dataset_OCR/Preprocess/0x*")
  # datas = sorted(datas)
  voc = None
  types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
  if voc_type == 'LOWERCASE':
    voc = list(string.digits + string.ascii_lowercase)
  elif voc_type == 'ALLCASES':
    voc = list(string.digits + string.ascii_letters)
  elif voc_type == 'ALLCASES_SYMBOLS':
    voc = list(string.printable[:-6])
  else:
    raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')
  # for data in datas:
  #   tmp = data.split("/")[-1]
  #   voc.append(tmp.split("x")[-1])
  # update the voc with specifical chars
  voc.append(EOS)
  voc.append(PADDING)
  voc.append(UNKNOWN)
  # voc = []
  # file = open('./lib/datasets/vocab.txt', "r")
  # for line in file:
  #   line = line.split('\n')
  #   voc.append(line[0])

  # # print(voc)
  return voc


## param voc: the list of vocabulary
def char2id(voc):
  return dict(zip(voc, range(len(voc))))

def id2char(voc):
  return dict(zip(range(len(voc)), voc))

def labels2strs(labels, id2char, char2id):
  # labels: batch_size x len_seq
  if labels.ndimension() == 1:
    labels = labels.unsqueeze(0)
  assert labels.dim() == 2
  labels = to_numpy(labels)
  strings = []
  batch_size = labels.shape[0]

  for i in range(batch_size):
    label = labels[i]
    string = []
    for l in label:
      if l == char2id['EOS']:
        break
      else:
        string.append(id2char[l])
    string = ''.join(string)
    strings.append(string)

  return strings