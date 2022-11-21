#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige
#          \date   2022-05-11 11:12:10.236762
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.postprocess import *
from src.preprocess import *
from bisect import bisect

df_gt_ = None


def calc_metric(x_pred, key=None, df_gt=None):
  global df_gt_
  if df_gt is None:
    df_gt = df_gt_

  if df_gt is None:
    df_gt = pd.read_csv(f'{FLAGS.root}/train_orders.csv')
    df_gt['cell_order'] = df_gt['cell_order'].apply(lambda x: x.split())
    df_gt_ = df_gt

  ids = set(x_pred['id'])
  df_gt = df_gt[df_gt.id.isin(ids)]

  x_pred2 = x_pred.copy()
  if key is not None:
    x_pred2['pred'] = x_pred[key]
  df_pred = to_df(x_pred2)
  return kendall_tau(df_gt.cell_order.values, df_pred.cell_order.values)

def calc_metrics(x_pred, key=None, df_gt=None):
  global df_gt_
  if df_gt is None:
    df_gt = df_gt_

  if df_gt is None:
    df_gt = pd.read_csv(f'{FLAGS.root}/train_orders.csv')
    df_gt['cell_order'] = df_gt['cell_order'].apply(lambda x: x.split())
    df_gt_ = df_gt

  ids = set(x_pred['id'])
  df_gt = df_gt[df_gt.id.isin(ids)]

  x_pred2 = x_pred.copy()
  if key is not None:
    x_pred2['pred'] = x_pred[key]
  df_pred = to_df(x_pred2)
  
  df_pred_g = df_pred.groupby('id')
  df_pred_g = {k: v for k, v in df_pred_g}
  
  df_gt_g = df_gt.groupby('id')
  df_gt_g = {k: v for k, v in df_gt_g}
  
  ids = list(ids)
  scores = []
  for id in tqdm(ids, desc='calc_metrics'):
    score = kendall_tau(df_gt_g[id].cell_order.values, df_pred_g[id].cell_order.values)
    scores.append(score)
  return pd.DataFrame({
    'id': ids,
    'score': scores
  })


# def count_inversions(a):
#   inversions = 0
#   sorted_so_far = []
#   for i, u in enumerate(a):
#     j = bisect(sorted_so_far, u)
#     inversions += i - j
#     sorted_so_far.insert(j, u)
#   return inversions


# https://www.kaggle.com/code/pjmathematician/kendall-tau-correlation-faster-o-nlogn/notebook
def count_inversions(a):
  res = 0
  counts = [0] * (len(a) + 1)
  rank = {v: i + 1 for i, v in enumerate(sorted(a))}
  for x in reversed(a):
    i = rank[x] - 1
    while i:
      res += counts[i]
      i -= i & -i
    i = rank[x]
    while i <= len(a):
      counts[i] += 1
      i += i & -i
  return res


def kendall_tau(ground_truth, predictions):
  total_inversions = 0
  total_2max = 0  # twice the maximum possible inversions across all instances
  for gt, pred in zip(ground_truth, predictions):
    ranks = [gt.index(x) for x in pred
            ]  # rank predicted order in terms of ground truth
    total_inversions += count_inversions(ranks)
    n = len(gt)
    total_2max += n * (n - 1)
  return 1 - 4 * total_inversions / total_2max


def order_df(df, pred_key, mark='train'):
  df.loc[df["cell_type"] == 'code',
         pred_key] = df.loc[df["cell_type"] == 'code', 'rel_rank']
  # add this(to remove train/eval bias) otherwise eval a bit higher then actual due to same value bias (multiple markdown before same code got same pred value)
  # especially for pairwise_eval will be much higher since multiple markdown got to same code to get same value.. about 0.6k
  if mark != 'test':
    df = df.sample(frac=1, random_state=1024)
  df = df.sort_values(['id', pred_key])
  df_pred = df.groupby('id')['cell_id'].apply(list).reset_index(
      name='cell_order')
  return df_pred


def to_df(x, mark='train', groupby=True, key='pred', return_dict=False):
  df = get_df(mark, for_eval=True)
  if FLAGS.add_end_source and FLAGS.pairwise:
    df = df[df.cell_id != 'nan']

  x['pred'] = np.asarray(x['pred'])

  if 'cls_pred' in x:
    x['cls_pred'] = np.asarray(x['cls_pred'])

  if FLAGS.pairwise and (not FLAGS.two_tower):
    xs = gezi.batch2list(x)
    l = []
    for x_ in xs:
      l.append(pairwise_cat2pred(x_))
    x_ = gezi.list2batch(l)
    x.update(x_)
  elif len(x['pred'].shape) > 1:
    x['cls_pred_ori'] = x['pred']
    if 'match_code' in x:
      x['match_rank'] = cls_match_ranks(x['cls_pred_ori'], x['match_code'],
                                        x['n_code_cell'])
    x['cls2_pred'] = cls2pred(x['cls_pred_ori'], method='argmax')
    x['cls_pred'] = cls2pred(x['cls_pred_ori'], method=FLAGS.cls2pred)
    x['pred'] = x['cls_pred']
  elif 'cls_pred' in x and len(x['cls_pred'].shape) > 1:
    x['reg_pred'] = x['pred']
    x['cls_pred_ori'] = x['cls_pred']
    if 'match_code' in x:
      x['match_rank'] = cls_match_ranks(x['cls_pred_ori'], x['match_code'],
                                        x['n_code_cell'])
    x['cls2_pred'] = cls2pred(x['cls_pred_ori'], method='argmax')
    x['cls_pred'] = cls2pred(x['cls_pred_ori'], method=FLAGS.cls2pred)
    # x['pred'] = x['reg_pred'] * (
    #     1 - FLAGS.cls_pred_ratio) + x['cls_pred'] * FLAGS.cls_pred_ratio
    if 'local_pred' in x:
      x['gl_pred'] = x['global_pred'] + x['local_pred'] / (1. + x['n_code_cell'])
      x['pred'] = x['gl_pred'] 
      
  m = {
      'id': x['id'],
      'cell_id': x['cell_id'],
      'pred': x[key],
  }
  if 'n_cell' in x:
    m['n_cell'] = x['n_cell']

  if return_dict:
    m.update({k: v for k, v in x.items() if k.endswith('_pred')})
  df_pred = pd.DataFrame(m)

  ids = set(df_pred.id)
  df = df[df.id.isin(ids)]
  df = df.merge(df_pred, on=(['id', 'cell_id']), how='left')

  if not groupby:
    return df

  if not return_dict:
    df_pred = order_df(df, 'pred', mark=mark)
  else:
    df_pred = {'pred': order_df(df, 'pred', mark=mark)}
    if 'cls_pred' in x:
      df_pred['cls'] = order_df(df, 'cls_pred')
    if 'cls2_pred' in x:
      df_pred['cls2'] = order_df(df, 'cls2_pred')
    if 'reg_pred' in x:
      df_pred['reg'] = order_df(df, 'reg_pred')
    if 'global_pred' in x:
      df_pred['global'] = order_df(df, 'global_pred')
    if 'gl_pred' in x:
      df_pred['gl'] = order_df(df, 'gl_pred')
  return df_pred


def evaluate(y_true, y_pred, x, other, is_last=False):
  res = {}

  eval_dict = gezi.get('eval_dict')
  if eval_dict:
    x.update(eval_dict)

  x.update(other)
  
  if FLAGS.dynamic_codes:
    # TODO a bit complex for dynamic_codes to eval, but possible..
    res['score'] = 0
    gezi.save(x, f'{FLAGS.model_dir}/valid.pkl', verbose=False)
    return res
    
  if 'temperature' in x:
    res['temperature'] = x['temperature'][0]

  if (FLAGS.pairwise and FLAGS.two_tower) and (not FLAGS.pairwise_eval):
    if not 'emb' in x:
      acc = (x['pred'].argmax(-1) == 0).mean()
      res['acc'] = acc
      return res
    else:
      # if FLAGS.save_emb:
      #   gezi.set('eval:embs', x.copy())
      x = pairwise_infers(x)

  if FLAGS.pairwise_eval:
    x = out_hook_finalize()
  if (not FLAGS.pairwise) and FLAGS.list_infer:
    x = flattens(x)

  # TODO for context model with cls_loss also can eval recall@n similar as pairwise model

  # NOTICE x['pred'] might be changed, like for loss_method=softmax
  df_pred = to_df(x, return_dict=True)

  if 'match_rank' in x:
    for i in [1, 2, 3, 4, 5, 10, 30, 40, 50]:
      res[f'r@{i}'] = (x['match_rank'] < i).mean()

  if 'match_code' in x:
    res['r@1'] = np.asarray([float(int(pred * (n_code + 1)) == int(match_code)) \
                  for pred, match_code, n_code in zip(x['pred'], x['match_code'], x['n_code_cell'])]).mean()
    res['mae'] = np.abs(np.asarray([float(int(pred * (n_code + 1)) - int(match_code)) / float(n_code + 1) \
                  for pred, match_code, n_code in zip(x['pred'], x['match_code'], x['n_code_cell'])])).mean()
    for key in df_pred:
      if key != 'pred':
        res[f'r@1/{key}'] = np.asarray([float(int(pred * (n_code + 1)) == int(match_code)) \
                for pred, match_code, n_code in zip(x[f'{key}_pred'], x['match_code'], x['n_code_cell'])]).mean()
        res[f'mae/{key}'] = np.abs(np.asarray([float(int(pred * (n_code + 1)) - int(match_code)) / float(n_code + 1) \
                for pred, match_code, n_code in zip(x[f'{key}_pred'], x['match_code'], x['n_code_cell'])])).mean()

  if 'context_match' in x:
    res['context_match'] = np.asarray(x['context_match']).mean()

  ids = set(df_pred['pred'].id)
  df_gt = pd.read_csv(f'{FLAGS.root}/train_orders.csv')
  df_gt = df_gt[df_gt.id.isin(ids)]
  df_gt['cell_order'] = df_gt['cell_order'].apply(lambda x: x.split())

  score = kendall_tau(df_gt.cell_order.values,
                      df_pred['pred'].cell_order.values)
  for key in df_pred:
    if key != 'pred':
      res[f'score/{key}'] = kendall_tau(df_gt.cell_order.values,
                                        df_pred[key].cell_order.values)

  if 'context_match_pred' in x:
    res['context_match/auc'] = sklearn.metrics.roc_auc_score(
        x['context_match'], x['context_match_pred'])
    res['context_match/f1'] = sklearn.metrics.f1_score(
        x['context_match'], x['context_match_pred'] > 0)
    res['context_match/acc'] = sklearn.metrics.accuracy_score(
        x['context_match'], x['context_match_pred'] > 0)

  # if is_last:
  gezi.set('eval:x', x)
  gezi.save(gezi.get('eval:x'), f'{FLAGS.model_dir}/valid.pkl', verbose=False)

  res['score'] = score
  
  if 'n_cell' in x:
    n_cells = dict(zip(x['id'], x['n_cell']))
    for i in [16, 32, 64, 128, 256]:
      ids = set([id for id in n_cells if n_cells[id] > i])
      res[f'score/{i}'] = kendall_tau(df_gt[df_gt.id.isin(ids)].cell_order.values,
                        df_pred['pred'][df_pred['pred'].id.isin(ids)].cell_order.values)

  return res


def valid_write(x, label, predicts, ofile, others={}):
  ofile = f'{FLAGS.model_dir}/valid.csv'
  write_result(x, predicts, ofile, others, is_infer=False)


def infer_write(x, predicts, ofile, others={}):
  ofile = f'{FLAGS.model_dir}/submission.csv'
  write_result(x, predicts, ofile, others, is_infer=True)


def write_result(x, predicts, ofile, others={}, is_infer=False):
  # if FLAGS.pairwise and (not FLAGS.pairwise_eval):
  #   return

  if is_infer:
    m = gezi.get('infer_dict')
  else:
    m = gezi.get('eval_dict')

  if m:
    x.update(m)
  x.update(others)

  if FLAGS.pairwise_eval:
    x = out_hook_finalize()
  if (not FLAGS.pairwise) and FLAGS.list_infer:
    x = flattens(x)

  if is_infer:
    try:
      if (FLAGS.pairwise and FLAGS.two_tower) and (not FLAGS.pairwise_eval):
        x = pairwise_infers(x)
      if not FLAGS.save_emb:
        if 'emb' in x:
          del x['emb']
      # ic(x)
      gezi.set('test:x', x)
      gezi.save(gezi.get('test:x'), f'{FLAGS.model_dir}/test.pkl', verbose=False)
    except Exception as e:
      ic(e)

  mark = 'train' if not is_infer else 'test'
  df = to_df(x, mark)

  ic(df)
  df['cell_order'] = df['cell_order'].apply(lambda x: ' '.join(x))
  df.to_csv(ofile, index=False)
