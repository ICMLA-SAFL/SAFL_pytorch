from __future__ import absolute_import

from PIL import Image
import numpy as np
from collections import OrderedDict
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from . import create
from .decoder_transformer import AttentionRecognitionHead
# from .attention_recognition_head import AttentionRecognitionHead
from ..loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead


from config import get_args
global_args = get_args(sys.argv[1:])


class ModelBuilder(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, eos, STN_ON=False, encoder_block=4, decoder_block=4):
    super(ModelBuilder, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.eos = eos
    self.STN_ON = STN_ON
    self.tps_inputsize = global_args.tps_inputsize

    self.encoder = create(self.arch,
                      with_lstm=global_args.with_lstm,
                      n_group=global_args.n_group)
    encoder_out_planes = self.encoder.out_planes

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels,
                      encoder_block= encoder_block,
                      decoder_block= decoder_block)
    self.rec_crit = SequenceCrossEntropyLoss()
    # self.STN_ON = False

    if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple(global_args.tps_outputsize),
        num_control_points=global_args.num_control_points,
        margins=tuple(global_args.tps_margins))
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=global_args.num_control_points,
        activation=global_args.stn_activation)
      # self.stn_head2 = STNHead(
      #   in_planes=3,
      #   num_ctrlpoints=global_args.num_control_points,
      #   activation=global_args.stn_activation)
      # self.stn_head3 = STNHead(
      #   in_planes=3,
      #   num_ctrlpoints=global_args.num_control_points,
      #   activation=global_args.stn_activation)
      # self.stn_head4 = STNHead(
      #   in_planes=3,
      #   num_ctrlpoints=global_args.num_control_points,
      #   activation=global_args.stn_activation)
      # self.stn_head5 = STNHead(
      #   in_planes=3,
      #   num_ctrlpoints=global_args.num_control_points,
      #   activation=global_args.stn_activation)

  def forward(self, input_dict):
    return_dict = {}
    return_dict['losses'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']
    #print("rec_targets: ", rec_targets.size())
    # rectification
    iterative = 1
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      input_image = x
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      # stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      # stn_img_feat, ctrl_points = self.stn_head2(stn_input)
      # x, _ = self.tps(input_image, ctrl_points)
      # stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      # stn_img_feat, ctrl_points = self.stn_head3(stn_input)
      # x, _ = self.tps(input_image, ctrl_points)
      # stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      # stn_img_feat, ctrl_points = self.stn_head4(stn_input)
      # x, _ = self.tps(input_image, ctrl_points)
      # stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      # stn_img_feat, ctrl_points = self.stn_head5(stn_input)
      # x, _ = self.tps(input_image, ctrl_points)

      # for i in range(iterative):
      #   x, _ = self.tps(input_image, ctrl_points)
      #   stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      #   stn_img_feat, ctrl_points = self.stn_head(stn_input)
      
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    # print("x size: ", x.size())
    # x = torch.reshape(x,())

    encoder_feats = self.encoder(x)
    # print("X:", x.size())
    # print("encoder output: ",encoder_feats.size())
    encoder_feats = encoder_feats.contiguous()
    #print("rec_targets: ", rec_targets.size())
    if self.training:
      #print("rec_targets: ", rec_targets.size())
      rec_pred = self.decoder([encoder_feats, rec_targets, rec_lengths])
      #print("rec_targets: ", rec_targets.size())
      #print("rec_lengths: ", rec_lengths)
      # print("decoder output:", rec_pred.size()) 
      loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
      #print("Loss: ", loss_rec.size())
      return_dict['losses']['loss_rec'] = loss_rec
    else:
      # rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, global_args.beam_width, self.eos)
      rec_pred, rec_pred_scores = self.decoder.sample([encoder_feats, rec_targets, rec_lengths])
      rec_pred_ = self.decoder([encoder_feats, rec_targets, rec_lengths])
      loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)
      return_dict['losses']['loss_rec'] = loss_rec
      return_dict['output']['pred_rec'] = rec_pred
      return_dict['output']['pred_rec_score'] = rec_pred_scores

    # pytorch0.4 bug on gathering scalar(0-dim) tensors
    for k, v in return_dict['losses'].items():
      return_dict['losses'][k] = v.unsqueeze(0)

    return return_dict
