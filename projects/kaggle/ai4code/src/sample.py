#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   sample.py
#        \author   chenghuige  
#          \date   2022-07-12 11:57:28.700932
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *

def sample_markdowns(markdown, count, markdowns, code_matches, rel_ranks, rng=None):
  def is_match(code_match1, code_match2, rel_rank1, rel_rank2):
    if code_match1 == code_match2 and rel_rank1 < rel_rank2:
      return 1
    return 0
  idx = markdowns.index(markdown)
  idxes = [i for i in range(len(markdowns)) if markdowns[i] != markdown]
  labels = [is_match(code_matches[idx], code_matches[i], rel_ranks[idx], rel_ranks[i]) for i in idxes]
  res = list(zip(idxes, labels))
  if rng:
    rng.shuffle(res)

  pos_idx = -1
  for i in range(len(res)):
    if res[i][1]:
      pos_idx = i
      break
  if pos_idx >= 0:
    res = [res[pos_idx]] + [res[i] for i in range(len(res)) if i != pos_idx]

  res = res[:count]
  if rng:
    rng.shuffle(res)
  
  if res:
    idxes, labels = list(zip(*res))
  else:
    idxes, labels = [], []
  # ic(idxes, labels)
  return idxes, labels
  
def select_samples(n, probs):
  probs = probs.astype('float64')
  if probs.sum() < 1e-5:
    probs = np.asarray([1./len(probs)]*len(probs))
  else:
    probs /= probs.sum()
  selected = np.random.choice(len(probs), n, replace=False, p=probs)
  return selected

# sample codes for context model
def sample_codes_idxes(total, count, method='even', candidates=None, centor=None, rng=None, probs=None, training=True):
  # candidates = None
  if candidates is not None:
    if FLAGS.sample_by_recall:
      # ic('-------fixed codes')
      if training and (probs is not None) and count < len(candidates):
        candidates = select_samples(count, probs)
      else:
        candidates = candidates[:count]
      candidates.sort()
      return candidates
    candidates_set = set(candidates)
    candidates = list(candidates)
  idxes = np.asarray((range(total)))
  if total <= count:
    return idxes
  else:
    if method == 'even':
      idxes = np.linspace(0, total - 1, count, dtype=int)
      # ic(idxes)
      if FLAGS.context_shift:
        shift = FLAGS.context_shift
        if shift > 0:
          for i in range(0, len(idxes) - 1):
            if idxes[i] + shift < idxes[i + 1]:
              idxes[i] += shift
          # if idxes[-1] + 1 < total:
          #   idxes[-1] += 1
        else:
          for i in reversed(range(1, len(idxes))):
            if idxes[i] + shift > idxes[i - 1]:
              idxes[i] += shift
          # if idxes[0] - 1 >= 0:
          #   idxes[0] -= 1
        # ic(idxes)
      elif FLAGS.context_aug:
        # if candidates is not None:
        #   ic(list(candidates))
        if (FLAGS.context_aug_rate is None or (rng.random() < FLAGS.context_aug_rate and training)) \
            and (FLAGS.context_valid_aug or training):
          idxes_ = idxes.copy()
          for i in range(len(idxes)):
            if i == 0:
              l = idxes[i]
            else:
              l = idxes[i] - int((idxes[i]  - idxes[i - 1]) / 2.)
        
            if i == len(idxes) - 1:
              r = idxes[i] + 1
            else:
              r = idxes[i] + int((idxes[i + 1]  - idxes[i]) / 2.)     
            if r - l > 1:
              match = False
              match_index = 1e10
              if candidates is not None:
                for j in range(l, r):
                  if j in candidates_set:
                    index = candidates.index(j)
                    if index < match_index:
                      match_index = index
                      idxes_[i] = j
                    match = True
                    
              if (not match) and (candidates is None):
                val = rng.integers(l, r)
                idxes_[i] = val 
          idxes = idxes_ 
      # ic(idxes) 
    elif method == 'random':
      if FLAGS.context_valid_aug or training:
        rng.shuffle(idxes)
        idxes = idxes[:count]
        idxes.sort()
      else:
        if candidates is not None:
          candidates = candidates[:count]
          candidates.sort()
          idxes = np.asarray(candidates[:count])
        else:
          idxes = np.linspace(0, total - 1, count, dtype=int)
    else:
      # using default even space selection
      idxes = np.linspace(0, total - 1, count, dtype=int)
      # raise ValueError(method)
    return idxes
  
