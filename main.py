# from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os.path as osp
import numpy as np
import math
import time

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn import Parameter

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.datasets.concatdataset import ConcatDataset
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from lib.utils.labelmaps import get_vocabulary, labels2strs
from crnn import training_data_generator#tu them vao

global_args = get_args(sys.argv[1:])


def get_data(data_dir, voc_type, max_len, num_samples, height, width, batch_size, workers, is_train, keep_ratio):
  # print(data_dir)
  # print(is_train)
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(LmdbDataset(data_dir_, voc_type, max_len, num_samples))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = LmdbDataset(data_dir, voc_type, max_len, num_samples)
  dataset1 = training_dataset(dataset)
  print('total image: ', len(dataset1))
#  print('dataset: ', type(dataset))

  if is_train:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=True, pin_memory=True, drop_last=True,
      collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  else:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=False, pin_memory=True, drop_last=False,
      collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))

  return dataset, data_loader

def training_dataset(opt):
    return ConcatDataset([training_data_generator.get_training_data_generator(img_h=opt.imgH, img_w=img_w)])


def get_dataset(data_dir, voc_type, max_len, num_samples):
  print("get_dataset..........")
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(LmdbDataset(data_dir_, voc_type, max_len, num_samples))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = LmdbDataset(data_dir, voc_type, max_len, num_samples)
  # print('total image test: ', len(dataset))
  return dataset


def get_dataloader(synthetic_dataset, real_dataset, height, width, batch_size, workers,
                   is_train, keep_ratio):
  num_synthetic_dataset = len(synthetic_dataset)
  num_real_dataset = len(real_dataset)

  synthetic_indices = list(np.random.permutation(num_synthetic_dataset))
  synthetic_indices = synthetic_indices[num_real_dataset:]
  real_indices = list(np.random.permutation(num_real_dataset) + num_synthetic_dataset)
  concated_indices = synthetic_indices + real_indices
  assert len(concated_indices) == num_synthetic_dataset

  sampler = SubsetRandomSampler(concated_indices)
  concated_dataset = ConcatDataset([synthetic_dataset, real_dataset])
  print('total image: ', len(concated_dataset))

  data_loader = DataLoader(concated_dataset, batch_size=batch_size, num_workers=workers,
    shuffle=False, pin_memory=True, drop_last=True, sampler=sampler,
    collate_fn=AlignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  return concated_dataset, data_loader

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  args.cuda = args.cuda and torch.cuda.is_available()
  print(torch.cuda.is_available())
  if args.cuda:
    print('using cuda.')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type('torch.FloatTensor')
  # Redirect print to both console and log file
  if not args.evaluate:
    # make symlink
    make_symlink_if_not_exists(osp.join(args.real_logs_dir, args.logs_dir), osp.dirname(osp.normpath(args.logs_dir)))
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    train_tfLogger = TFLogger(osp.join(args.logs_dir, 'train'))
    eval_tfLogger = TFLogger(osp.join(args.logs_dir, 'eval'))

  # Save the args to disk
  if not args.evaluate:
    cfg_save_path = osp.join(args.logs_dir, 'cfg.txt')
    # print()
    cfgs = vars(args)
    with open(cfg_save_path, 'w') as f:
      for k, v in cfgs.items():
        f.write('{}: {}\n'.format(k, v))

  # Create data loaders
  if args.height is None or args.width is None:
    args.height, args.width = (32, 100)

  if not args.evaluate: 
    train_dataset, train_loader = \
      get_data(args.synthetic_train_data_dir, args.voc_type, args.max_len, args.num_train,
               args.height, args.width, args.batch_size, args.workers, True, args.keep_ratio)
    voc = get_vocabulary('ALLCASES_SYMBOLS', EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN')
    id2char = dict(zip(range(len(voc)), voc))
    char2number = dict(zip(voc, [0]*len(voc)))
    # for _, label, _ in train_dataset:
    #   # word = ''
    #   for i in label:
    #     if not id2char[i] in ['EOS','PADDING','UNKNOWN']:
    #       char2number[id2char[i]] += 1
    #       # word += id2char[i]
    # # print(char2number)
    # for key in char2number.keys():
    #   print("{}:{}".format(key, char2number[key]))
      
      

  test_dataset, test_loader = \
    get_data(args.test_data_dir, args.voc_type, args.max_len, args.num_test,
             args.height, args.width, args.batch_size, args.workers, False, args.keep_ratio)
  # print("len(trainset) ", len(train_dataset))

  if args.evaluate:
    max_len = test_dataset.max_len
  else:
    max_len = max(train_dataset.max_len, test_dataset.max_len)
    train_dataset.max_len = test_dataset.max_len = max_len
  # Create model
  

  model = ModelBuilder(arch=args.arch, rec_num_classes=test_dataset.rec_num_classes,
                       sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=max_len,
                       eos=test_dataset.char2id[test_dataset.EOS], STN_ON=args.STN_ON,
                       encoder_block= args.encoder_block, decoder_block= args.decoder_block)

  for param in model.decoder.parameters():
    if isinstance(param, Parameter):
      param.requires_grad = False

  # for param in model.encoder.parameters():
  #   param.requires_grad = False
  # for param in model.stn_head.parameters():
  #   param.requires_grad = False

  # Load from checkpoint
  if args.evaluation_metric == 'accuracy':
    best_res = 0
  elif args.evaluation_metric == 'editdistance':
    best_res = math.inf
  else:
    raise ValueError("Unsupported evaluation metric:", args.evaluation_metric)
  start_epoch = 0
  start_iters = 0
  if args.resume:
    print("args.resume: ",args.resume)
    checkpoint = load_checkpoint(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    # for param in model.stn_head.parameters():
    #   # print(param.data)
    #   param.requires_grad = False
    # for param in model.encoder.parameters():
    #   param.requires_grad = False

    # compatibility with the epoch-wise evaluation version
    if 'epoch' in checkpoint.keys():
      start_epoch = checkpoint['epoch']
    else:
      start_iters = checkpoint['iters']
      start_epoch = int(start_iters // len(train_loader)) if not args.evaluate else 0
    # checkpoint['best_res'] = 0.802
    best_res = checkpoint['best_res']
    print("=> Start iters {}  best res {:.1%}"
          .format(start_iters, best_res))
  
  if args.cuda:
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model)
  # Evaluator
  evaluator = Evaluator(model, args.evaluation_metric, args.cuda)

  if args.evaluate:
    print('Test on {0}:'.format(args.test_data_dir))
    if len(args.vis_dir) > 0:
      vis_dir = osp.join(args.logs_dir, args.vis_dir)
      if not osp.exists(vis_dir):
        os.makedirs(vis_dir)
    else:
      vis_dir = None

    start = time.time()
    # print(test_dataset.lexicons50)
    evaluator.evaluate(test_loader, dataset=test_dataset, vis_dir=vis_dir)
    print('it took {0} s.'.format(time.time() - start))
    return

  # Optimizer
  param_groups = model.parameters()
  # model.stn_head.weight.requires_grad = False
  # model.encoder.weight.requires_grad = False
  param_groups = filter(lambda p: p.requires_grad, param_groups)
  # optimizer = optim.Adadelta(param_groups, lr=args.lr, weight_decay=args.weight_decay)
  optimizer = optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay, amsgrad=False)
  # optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9)
  # optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
  # optimizer = optim.ASGD(param_groups, lr=args.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
  # optimizer = optim.Adagrad(param_groups, lr=args.lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

  # Trainer
  loss_weights = {}
  loss_weights['loss_rec'] = 1.
  if args.debug:
    args.print_freq = 1
  trainer = Trainer(model, args.evaluation_metric, args.logs_dir, 
                    iters=start_iters, best_res=best_res, grad_clip=args.grad_clip,
                    use_cuda=args.cuda, loss_weights=loss_weights)

  # Start training
  # evaluator.evaluate(test_loader, step=0, tfLogger=eval_tfLogger, dataset=test_dataset)
  # print("args.epoch: ", args.epochs)
  for epoch in range(start_epoch, args.epochs):
    scheduler.step(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    # current_lr = (1.0/(512.0**0.5))*min(1.0/float(trainer.iters + 1)**0.5, float(trainer.iters+1)*1.0/16000.0**1.5)
    # optimizer.param_groups[0]['lr'] = current_lr 
    trainer.train(epoch, train_loader, optimizer, current_lr,
                  print_freq=args.print_freq,
                  train_tfLogger=train_tfLogger, 
                  is_debug=args.debug,
                  evaluator=evaluator, 
                  test_loader=test_loader, 
                  eval_tfLogger=eval_tfLogger,
                  test_dataset=test_dataset)

  # Final test
  print('Test with best model:')
  checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
  model.load_state_dict(checkpoint['state_dict'])
  # print("naruto")
  evaluator.evaluate(test_loader, dataset=test_dataset)
  # print("sasuke")

  # Close the tensorboard logger
  train_tfLogger.close()
  eval_tfLogger.close()


if __name__ == '__main__':
  # parse the config
  args = get_args(sys.argv[1:])
  main(args)
  
