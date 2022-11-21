#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:25.201944
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *

def get_callbacks(model):
  callbacks = []
  ic(callbacks)
  return callbacks

def get_left_texts(texts, idx, count):
  start = idx - count
  end = idx
  l = []
  for i in range(start, end):
    if i < 0:
      l.append('before')
    else:
      l.append(texts[i])
  return l

def get_right_texts(texts, idx, count):
  start = idx + 1
  end = idx + 1 + count
  l = []
  for i in range(start, end):
    if i < len(texts):
      l.append(texts[i])
    else:
      l.append('after')
  return l
  
def merge_input_ids(input_ids_list, count):
  parts = int(len(input_ids_list) / count)
  ls = []
  for i in range(count):
    idx = i * parts
    l = [input_ids_list[idx][0]]
    for j in range(parts):
      idx_ = idx + j
      l.extend(input_ids_list[idx_][1:])
    ls.append(l)
  return ls    

def select_negs(n, pos_idx, total, probs=None, rng=None, method='rand', rand_prob=None):
  if (n + 1) >= total:
    return np.asarray([i for i in range(total) if i != pos_idx])
  if rand_prob is None:
    if method == 'rand':
      # ic('rand select')
      return select_negs_rand(n, pos_idx, total, rng)
    elif method == 'greedy':
      # ic('greedy select')
      return select_negs_greedy(n, pos_idx, probs)
    elif method == 'sample':
      # ic('sample select')
      return select_negs_sample(n, pos_idx, probs)
  else:
    if FLAGS.neg_strategy == 'rand-greedy':
      if probs is None or rng.random() < rand_prob:
        # ic('rand select')
        return select_negs_sample(n, pos_idx, probs)
      else:
        # ic('greedy select')
        return select_negs_greedy(n, pos_idx, probs)
    else:
        if probs is None or rng.random() < rand_prob:
          # ic('rand select')
          return select_negs_rand(n, pos_idx, total, rng)
        else:
          # ic('sample select')
          return select_negs_sample(n, pos_idx, probs)

def select_negs_rand(n, pos_idx, total, rng):
  idxes = [i for i in range(total) if i != pos_idx]
  rng.shuffle(idxes)
  return np.asarray(idxes[:n])

def select_negs_greedy(n, pos_idx, probs):
  idxes = (-probs).argsort()
  idxes = [i for i in idxes if i != pos_idx]
  return np.asarray(idxes[:n])

def select_negs_sample(n, pos_idx, probs):
  idxes = np.asarray([i for i in range(len(probs)) if i != pos_idx])
  if n + 1 >= len(probs):
    return idxes
  
  probs = probs[idxes].astype('float64')
  if probs.sum() < 1e-5:
    probs = np.asarray([1./len(probs)]*len(probs))
  else:
    probs /= probs.sum()
  # np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
  selected = np.random.choice(len(probs), n, replace=False, p=probs)
  return idxes[selected]
