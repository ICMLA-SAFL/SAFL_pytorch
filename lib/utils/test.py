from __future__ import absolute_import

import string

import glob

# from . import to_torch, to_numpy
def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
  '''
  voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
  '''
  datas = glob.glob("./0x*")
  datas = sorted(datas)
  voc = []
  types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
  # if voc_type == 'LOWERCASE':
  #   voc = list(string.digits + string.ascii_lowercase)
  # elif voc_type == 'ALLCASES':
  #   voc = list(string.digits + string.ascii_letters)
  # elif voc_type == 'ALLCASES_SYMBOLS':
  #   voc = list(string.printable[:-6])
  # else:
  #   raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')
  for data in datas:
    tmp = data.split("/")[-1]
    voc.append(tmp.split("x")[-1])
  # update the voc with specifical chars
  voc.append(EOS)
  voc.append(PADDING)
  voc.append(UNKNOWN)

  return voc

voc = get_vocabulary("LOWERCASE","EOS","PADDING","UNKNOWN")
print(voc)