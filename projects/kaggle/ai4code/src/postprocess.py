#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   postprocess.py
#        \author   chenghuige  
#          \date   2022-05-11 11:17:37.027553
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *
from src.preprocess import *

def pairwise_cat2pred(x, infer=False):
  idxes = (-x['pred']).argsort(-1)
  
  # ic(len(cell_infos))
  # probs = cell_infos[x['cid']].copy()
  probs = get_cell_info(x['cid']).copy()
  max_len = min(1 + FLAGS.num_negs, len(probs))
  
  top_probs = gezi.softmax(x['pred'][:max_len])
  
  top_prob = 0
  for idx in x['code_idxes']:
    top_prob += probs[idx]
    
  for i in range(max_len):
    probs[x['code_idxes'][i]] = top_probs[i] * top_prob
  cls_pred = probs2pred(probs, x['n_code_cell'])
    
  idx = idxes[0]
  pred = (x['code_idxes'][idx] + 0.5) / (x['n_code_cell'] + 1)
    
  res = {
    'cls_pred_ori': x['pred'],
    'pred': pred,
    'cls_pred': cls_pred,
    'probs': probs,
  }
  
  if not infer:
    match = False
    for i, idx in enumerate(idxes):
      if x['code_idxes'][i] == x['match_code']:
        x['match_rank'] = i
        match = True
    if not match:
      x['match_rank'] = x['n_code_cell']
    res['match_rank'] = x['match_rank']
    
  return res

def cls2pred(predictions, method):
  if FLAGS.cls_span:
    span = 0.5
  else:
    span = 0.
  if method == 'argmax':
    pred = (predictions.argmax(-1) + span) / FLAGS.num_classes 
  else:
    pred = (gezi.softmax(predictions) * (np.arange(FLAGS.num_classes) + span)).sum(-1) / FLAGS.num_classes
  return pred

def cls_match_ranks(predictions, match_codes, n_codes):
  idxes = np.argsort(-predictions, -1)
  match_ranks = []
  if FLAGS.cls_span:
    span = 0.5
  else:
    span = 0.
  for i in range(len(idxes)):
    match = False
    for j, idx in enumerate(idxes[i]):
      pred = (idx + span) / FLAGS.num_classes
      code_pred = int(pred * (n_codes[i] + 1))
      if code_pred == match_codes[i]:
        match_ranks.append(j)
        match = True
        break
    if not match:
      # classfication 100 class for match rank eval is only for reference.. may not match using all 100 postion/classes
      match_ranks.append(len(idxes[i]) - 1)

  return np.asarray(match_ranks)

def pairwise_infers(x):
  rows = []
  id = None
  res = []
  xs = gezi.batch2list(x)
  for row in tqdm(xs, desc='pairwise_infers', leave=True):
    if id and row['id'] != id:
      res.append(pairwise_infer(rows, eval=True))
      rows = []
    rows.append(row)
    id = row['id']
  res.append(pairwise_infer(rows, eval=True))
  x = gezi.merge_array_dicts(res)
  return x

def probs2pred(probs, n_code_cell):
  if FLAGS.pairwise_span:
    span = 0.5
  else:
    span = 0
  return (probs * (np.arange(n_code_cell + 1) + span)).sum(-1) / (n_code_cell + 1)

# TODO get metric recall@1, 3, 5, 10
def pairwise_infer(rows, eval=False):
  code_rel_ranks, markdown_rel_ranks = [], []
  markdowns, codes, markdown_ranks, code_ranks = [], [], [], []
  markdown_embs, code_embs = [], []
  if eval:
    match_codes, match_ranks = [], []
  cids = []
  for row in rows:
    if row['cell_type'] == 'markdown':
      markdowns.append(row['cell_id'])
      markdown_embs.append(row['emb'])
      markdown_ranks.append(row['rank'])
      markdown_rel_ranks.append(row['rel_rank'])
      if eval:
        match_codes.append(int(row['match_code']))
      cids.append(row['cid'])
    else:
      codes.append(row['cell_id'])
      code_embs.append(row['emb'])
      code_ranks.append(row['rank'])
      code_rel_ranks.append(row['rel_rank'])
  markdown_embs = np.asarray(markdown_embs)
  code_embs = np.asanyarray(code_embs)
  
  if FLAGS.embs_dir:
    code_dir = f'{FLAGS.embs_dir}/code'
    gezi.try_mkdir(code_dir)
    markdown_dir = f'{FLAGS.embs_dir}/markdown'
    gezi.try_mkdir(markdown_dir)
    id = rows[0]['id']
    np.save(f'{code_dir}/{id}.npy', code_embs)
    np.save(f'{markdown_dir}/{id}.npy', markdown_embs)
  
  n_code_cell = len(codes) - 1 if FLAGS.add_end_source else len(codes)
  try:
    sims = np.matmul(markdown_embs, code_embs.transpose(1, 0))
  except Exception as e:
    ic(e, markdown_embs.shape, code_embs.shape)
    exit(-1)
  temperature = 1.
  if FLAGS.temperature > 0:
    temperature = FLAGS.temperature
  if FLAGS.dtemperature > 0:
    temperature = row['temperature']
  logits = sims / temperature
  # logits *= ((n_code_cell + 1) / 5.)
  probs = gezi.softmax(logits)
  
  if FLAGS.pairwise_markdowns:
    sims2 = np.matmul(markdown_embs, markdown_embs.transpose(1, 0))
    probs2 = gezi.sigmoid(sims2 / temperature)
  
  # TODO add end code... to handle mardown at last problem
  if eval:
    idxes = np.argsort(-sims, -1)
    for i in range(len(idxes)):
      idxes_= list(idxes[i])
      if not FLAGS.add_end_source:
        idxes_.append(len(codes))
      rank = idxes_.index(match_codes[i])
      match_ranks.append(rank)
 
  #TODO Difficult to ensemble with context model
  if not FLAGS.pairwise_markdowns:
    span = 1. / (n_code_cell + 1) / 2.
    # span = 1. / (n_code_cell + 1) / 5
    markdown_rel_ranks = [(code_rel_ranks[sims[i].argmax(-1)] - span) for i in range(len(markdowns))]
  else:
    spans = (probs2 > 0.5).astype(int).sum(-1) 
    spans = (spans + 1) * 0.001
    markdown_rel_ranks = [(code_rel_ranks[sims[i].argmax(-1)] - spans[i]) for i in range(len(markdowns))]
    
  max_sims = [sims[i].max() for i in range(len(markdowns))]
  max_probs = [probs[i].max() for i in range(len(markdowns))]

  ids = [row['id']] * len(markdowns)
  res = {'id': ids, 'cell_id': markdowns, 'cid': cids, 
         'pred': markdown_rel_ranks, 'max_prob': max_probs, 
         'max_sim': max_sims}
  res['cls_pred'] = probs2pred(probs, n_code_cell)
  res['n_words'] = [row['n_words']] * len(markdowns)
  res['n_code_cell'] = [n_code_cell] * len(markdowns)
  res['n_cell'] = [row['n_cell']] * len(markdowns)
  if eval:
    res['match_rank'] = match_ranks
    res['match_code'] = match_codes
  if FLAGS.save_probs:
    res['probs'] = probs
    res['sims'] = sims
  return res

## TODO 注意当前没有做特殊处理需要DP模式valid 而不能DDP
rows = []
id = None
res = []
def out_hook(other, x):
  assert FLAGS.out_hook_per_batch
  assert not FLAGS.distributed, 'use DP not DDP for pairwise_eval'
  global rows, id, res
  other_ = other.copy()
  other_.update(x)
  xs = gezi.batch2list(other_)
  for row in xs:
    if id and row['id'] != id:
      res.append(pairwise_infer(rows))
      rows = []
    rows.append(row)
    id = row['id']
  other.clear()
  x.clear()

def out_hook_finalize():
  global rows, res
  key = 'out_hook:x'
  if not gezi.get(key):
    if rows:
      res.append(pairwise_infer(rows))
      rows = []
    x = gezi.merge_array_dicts(res)
    res = []
    gezi.set(key, x)
  else:
    x = gezi.get(key)
  return x

def clear_infer():
  global rows, id, res
  rows, res = [], []
  id = None
  
def flatten(x, pred_id):
  l = []
  cid = x['cid']
  cids = [cid, *get_cid_group(cid)]
  idxes = list(x['preds'].argsort())
  # cls_idxes = list(x['cls_preds'].argsort())
  group_mean = x['preds'].mean()
  group_min = x['preds'].min()
  group_max = x['preds'].max()
  group_var = np.var(x['preds'])
  # group_cls_mean = x['cls_preds'].mean()
  # group_cls_min = x['cls_preds'].min()
  # group_cls_max = x['cls_preds'].max()
  # group_cls_var = np.var(x['cls_preds'])
  for i, cid in enumerate(cids):
    x_ = {}
    x_['pred'] = x['preds'][i]
    # if i > 0:
    #   x_['pred'] = 1. - x['preds'] * 100
    x_['cls_pred'] = np.asarray(x['cls_preds'][i])
    # x_['cls_pred'] = np.asarray(x['cls_preds'][i]).argmax() / FLAGS.num_classes
    x_['cid'] = cid
    x_['group_pos'] = i
    x_['group_rank'] = idxes.index(i)
    # x_['group_cls_rank'] = cls_idxes.index(i)
    x_['group_mean'] = x['preds'].mean()
    x_['group_min'] = group_min
    x_['group_max'] = group_max
    x_['group_var'] = group_var
    # x_['group_cls_min'] = group_min
    # x_['group_cls_max'] = group_max
    # x_['group_cls_var'] = group_var
    x_['pred_id'] = pred_id
    cid_info = get_cid_info(cid)
    x_.update(cid_info)
    # ic(x_)
    l.append(x_)
  return l
    
# for list infer
def flattens(x):
  key = 'out_hook:x'
  if not gezi.get(key):
    xs = gezi.batch2list(x)
    l = []
    for i, x in tqdm(enumerate(xs), total=len(xs), desc='list infer flattens'):
      l.extend(flatten(x, i))
    df = pd.DataFrame(l)
    ic(len(df))
    # df = df.groupby('cid').mean().reset_index()
    # df['id'] = df.cid.map(lambda x: get_cid_info(x)['id'])
    # df['cell_id'] = df.cid.map(lambda x: get_cid_info(x)['cell_id'])
    df = df.drop_duplicates(subset=['cid'], keep='first')
    df = df.sort_values(['id'])
    x = df.to_dict('list')
    # ic(df[df.group_pos==0][['pred', 'rel_rank']])
    # ic(df[df.group_pos==1][['pred', 'rel_rank']])
    ic(np.abs(df[df.group_pos==0].pred - df[df.group_pos==0].rel_rank).mean())
    ic(np.abs(df[df.group_pos==1].pred - df[df.group_pos==1].rel_rank).mean())
    ic(np.abs(np.asarray([np.random.rand() for _ in range(len(df[df.group_pos==1]))]) - df[df.group_pos==1].rel_rank).mean())
    ic(np.abs(df[df.group_pos==2].pred - df[df.group_pos==2].rel_rank).mean())
    ic(len(xs), len(df), len(x['cid']))
    gezi.set(key, x)
  else:
    x = gezi.get(key)
  return x
