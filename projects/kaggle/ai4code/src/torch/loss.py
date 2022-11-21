#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2022-05-11 11:13:01.390146
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from torch import nn
from src.preprocess import get_tokenizer

def calc_loss(res, y, x, step=None, epoch=None, training=None):
  scalars = {}
        
  y_ = res[FLAGS.pred_key]
  reduction = 'mean'

  loss = 0.
  odim = 1
  
  if FLAGS.n_markdowns:
    y_ = res['preds'].squeeze(-1)
    y = torch.cat([y.unsqueeze(-1), x['markdown_labels']], -1)
    if 'cls_preds' in res:
      cls_pred = res['cls_preds']
      try:
        cls_label = torch.cat([x['cls_label'].unsqueeze(-1), x['markdown_cls_labels']], -1)
      except Exception:
        ic(x['cls_label'].unsqueeze(-1).shape, x['markdown_cls_labels'].shape, training)
    if 'global_preds' in res:
      global_pred = res['global_preds']
      try:
        global_label = torch.cat([x['global_label'].unsqueeze(-1), x['markdown_global_labels']], -1)
      except Exception:
        ic(x['global_label'].unsqueeze(-1).shape, x['markdown_global_labels'].shape, training)
    if 'local_preds' in res:
      local_pred = res['local_preds']
      try:
        local_label = torch.cat([x['local_label'].unsqueeze(-1), x['markdown_local_labels']], -1)
      except Exception:
        ic(x['local_label'].unsqueeze(-1).shape, x['markdown_global_labels'].shape, training)
    reduction = 'none'
    if FLAGS.rank_loss_rate > 0:
      loss_obj = lele.losses.TauLoss()
      mask = (y == -100).int()
      y2 = y_ - mask * 10000
      rank_loss = loss_obj(y2, y) 
      # mask = (y != -100).int()
      # rank_loss = loss_obj(y_, y, mask) 
      scalars['loss/rank'] = rank_loss.item()
      loss += rank_loss * FLAGS.rank_loss_rate
  else:
    if 'cls_pred' in res:
      cls_pred = res['cls_pred']
      cls_label = x['cls_label']
    if 'global_pred' in res:
      global_pred = res['global_pred']
      global_label = x['global_label']  
    if 'local_pred' in res:
      global_pred = res['local_pred']
      local_label = x['lobal_label']
    
  y = y.float()
  y_ = y_.float()

  if FLAGS.lm_train:
    loss_obj = nn.CrossEntropyLoss(label_smoothing=FLAGS.label_smoothing, reduction=reduction)
    # loss_obj = nn.CrossEntropyLoss(label_smoothing=FLAGS.label_smoothing, reduction='none')
    # ic(res['pred'].shape, res['label'].shape, get_tokenizer(FLAGS.backbone).vocab_size, len(get_tokenizer(FLAGS.backbone)))
    lm_loss = loss_obj(res['pred'].contiguous().view(-1, len(get_tokenizer(FLAGS.backbone))), res['label'].contiguous().view(-1))
    # lm_loss = lele.masked_mean(lm_loss, (res['label'] != -100).int())
    scalars['loss/lm'] = lm_loss.item()
    loss += lm_loss
    return loss
  
  if FLAGS.base_loss_rate > 0:
    if FLAGS.loss_method == 'mse':
      loss_obj = nn.MSELoss(reduction=reduction)
      base_loss = loss_obj(y_.view(-1), y.view(-1))
    if FLAGS.loss_method == 'mae':
      loss_obj = nn.L1Loss(reduction=reduction)
      base_loss = loss_obj(y_.view(-1), y.view(-1))
    elif FLAGS.loss_method == 'soft_binary':
      loss_obj = nn.BCEWithLogitsLoss(reduction=reduction)
      base_loss = loss_obj(y_.view(-1), y.view(-1))
    elif FLAGS.loss_method == 'softmax':
      loss_obj = nn.CrossEntropyLoss(reduction=reduction)  
      base_loss = loss_obj(y_, y.long())
    
    if reduction == 'none':
      base_loss = lele.masked_mean(base_loss, (y != -100).float())
    
    scalars['loss/base'] = base_loss.item()
    base_loss *= FLAGS.base_loss_rate
    loss += base_loss
    
  if 'markdown_pred' in res:
    loss_obj = nn.BCEWithLogitsLoss(reduction='none')
    markdown_loss = loss_obj(res['markdown_pred'].view(-1), x['markdowns_label'].float().view(-1))
    markdown_loss = lele.masked_mean(markdown_loss, x['markdowns_label_mask'])
    scalars['loss/markdown'] = markdown_loss.item()
    markdown_loss *= FLAGS.markdown_loss_rate
    loss += markdown_loss
  
  # not used
  if FLAGS.context_match_loss_rate > 0:
    loss_obj = nn.BCEWithLogitsLoss(reduction=reduction)
    match_loss = loss_obj(res['context_match_pred'].view(-1), x['context_match'].float().view(-1))
    scalars['loss/match'] = match_loss.item()
    match_loss *= FLAGS.context_match_loss_rate
    loss += match_loss

  if FLAGS.cls_loss_rate > 0:
    loss_obj = nn.CrossEntropyLoss(reduction='mean')  
    # ic(res['cls_pred'].shape, x['cls_label'].shape)
    cls_loss = loss_obj(cls_pred.view(-1, FLAGS.num_classes), cls_label.long().view(-1))
    # if scalars['loss/cls'] = cls_loss, scalars['loss/cls'] will also change when cls_loss *= FLAGS.cls_loss_rate
    scalars['loss/cls'] = cls_loss.item()
    cls_loss *= FLAGS.cls_loss_rate
    loss += cls_loss
    
  if FLAGS.global_loss_rate > 0:
    loss_obj = nn.L1Loss(reduction=reduction) 
    global_loss = loss_obj(global_pred.view(-1), global_label.view(-1))
    if reduction == 'none':
      global_loss = lele.masked_mean(global_loss, (global_label != -100).float())
    scalars['loss/global'] = global_loss.item()
    global_loss *= FLAGS.global_loss_rate
    loss += global_loss
    
  if FLAGS.local_loss_rate > 0:
    loss_obj = nn.L1Loss(reduction=reduction) 
    local_loss = loss_obj(local_pred.view(-1), local_label.view(-1))
    if reduction == 'none':
      local_loss = lele.masked_mean(local_loss, (local_label != -100).float())
    scalars['loss/local'] = local_loss.item()
    local_loss *= FLAGS.local_loss_rate
    loss += local_loss
    
  loss *= FLAGS.loss_scale
  
  if FLAGS.rdrop_rate > 0:
    ## FIXME TODO for electra not work so [electra/roberta] why? not just use awp no rdrop, deberta/bart all ok
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.LongTensor [1, 229]] is at version 3;
    # expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
    def rdrop_loss(p, q):
      rloss = 0.
      rloss += lele.losses.compute_kl_loss(p['pred'], q['pred'])
      return rloss
    gezi.set('rdrop_loss_fn', lambda p, q: rdrop_loss(p, q))
          
  lele.update_scalars(scalars, decay=FLAGS.loss_decay, training=training)
  
  return loss
  
