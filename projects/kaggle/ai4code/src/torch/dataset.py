#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2022-05-31 03:21:26.236767
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib2to3.pgen2.tokenize import tokenize

from gezi.common import * 
from torch.utils.data import Dataset as TorchDataset
from src.config import *
from src.preprocess import *

class Dataset(TorchDataset):
  def __init__(self, subset='valid'):
    self.subset = subset
    self.mark = 'train' if subset in ['train', 'valid', 'eval'] else 'test'
    if subset == 'test' and FLAGS.external_test:
      self.mark = 'train'
    self.rows = []
    
    df = get_df(self.mark)
    
    # if not 'fold' in df.columns:
    #   gezi.set_fold(df, FLAGS.folds, group_key='ancestor_id', seed=FLAGS.fold_seed)
      
    if subset in ['valid', 'eval']:
      df = df[df.fold==FLAGS.fold]
    elif subset == 'train':
      if not FLAGS.external:
        if not FLAGS.online:
          df = df[df.fold!=FLAGS.fold]
      else:
        df = get_df(FLAGS.external)
    elif subset == 'test' and FLAGS.external_test:
      if not FLAGS.online:
        df = df[df.fold!=FLAGS.fold]
    
    if not FLAGS.lm_train:
      if ((not subset in ['train', 'valid']) and ((FLAGS.dump_emb or FLAGS.pairwise_eval) or (FLAGS.pairwise and FLAGS.two_tower))) \
        or ((subset in ['train', 'valid']) and FLAGS.train_code):
        logger.info(f'{subset}: markdown and codes')
        self.df = df
        # self.df = self.df.head(1000)
      else:
        logger.info(f'{subset}: markdown only')
        self.df = df[df.cell_type=='markdown']
    else:
      df = df.groupby(['id'])['source'].apply(list).reset_index(name='source')
      df['source'] = df['source'].apply(lambda l: ''.join(l))
      self.df = df

    if (not FLAGS.pairwise) and FLAGS.list_infer:
      logger.info('filter cids for list infer')
      cids = get_group_cids()
      ic(len(self.df), len(self.df.id.unique()), len(cids))
      self.df = self.df[self.df.cid.isin(cids)]
      
    ic(subset, len(self.df), len(self.df.id.unique()))
    # ic(df[df.cell_id=='9ebeb0b7'][['rank', 'pct_rank', 'rel_rank']])
  
    ic(FLAGS.aug_seed)  
    # self.rng = np.random.default_rng(FLAGS.aug_seed)
    rng = np.random.default_rng()
    self.rng = rng
    
    # TODO FIXME this is too slow to use 30min...
    if FLAGS.dynamic_codes and subset in ['eval', 'test']:
      for row in tqdm(self.df.itertuples(), total=len(self.df), desc='dynamic_codes'):
        row = row._asdict()
        self.rows.extend(parse_example(row, self.subset, rng))
      ic(len(self.df), len(self.rows))
    
    if FLAGS.pairwise and (not FLAGS.two_tower) and FLAGS.rerank_filter:
      if subset in ['eval', 'test']:
        cids = get_rerank_cids(FLAGS.rerank_min_thre, FLAGS.rerank_max_thre)
        ic(len(self.df), len(cids))
        self.df = self.df[self.df.cid.isin(cids)]
        ic(len(self.df))
    
    self.num_examples = len(self.df)
    # if FLAGS.external and FLAGS.external_parts and subset == 'train':
    #   self.num_examples = int(self.num_examples / FLAGS.external_parts)
    # self.epoch = 0
    
    idx = rng.integers(self.num_examples)
    self.show(idx)
    # if subset == 'eval':
    #   self.show(39957)
    #   self.show(250310)
  
  # def set_epoch(self, epoch):
  #   self.epoch = epoch
    
  def show(self, idx=0):
    fe = self[idx].copy()
    assert 'input_ids' in fe
    tokenizer = get_tokenizer(FLAGS.backbone)
    fe['input_tokens'] = ''.join(tokenizer.convert_ids_to_tokens(fe['input_ids']))
    if 'input_ids0' in fe:
      fe['input_tokens0'] = ''.join(tokenizer.convert_ids_to_tokens(fe['input_ids0']))
    fe['num_input_tokens'] = len(fe['input_ids'])
    if 'codes_input_ids' in fe:
      fe['code_tokens'] = ''.join(tokenizer.convert_ids_to_tokens(fe['codes_input_ids']))
    if 'markdowns_input_ids' in fe:
      fe['markdown_tokens'] = ''.join(tokenizer.convert_ids_to_tokens(fe['markdowns_input_ids']))
    if 'token_type_ids' in fe:
      fe['token_type_ids_str'] = ''.join(map(str, fe['token_type_ids']))
    if 'codes_token_type_ids' in fe:
      fe['codes_token_type_ids_str'] = ''.join(map(str, fe['codes_token_type_ids']))
    keys = ['id', 'cell_id', 'source', 'cell_type',
            'rank', 'code_rank', 'markdown_rank', 'pct_rank', 'rel_rank', 
            'n_cell', 'n_code_cell', 'n_markdown_cell',
            'label', 'cls_label', 'markdowns_label',
            'input_tokens', 'num_input_tokens', 'input_tokens0',
            'code_tokens', 'markdown_tokens',
            'token_type_ids_str', 'codes_token_type_ids_str']
    fe = OrderedDict({k: fe[k] for k in keys if k in fe})
    ic(idx, fe)
      
  def __getitem__(self, idx):
    if self.rows:
      return self.rows[idx]
    # need to regen dataset each epoch
    # if FLAGS.external and FLAGS.external_parts and self.subset == 'train':
    #   idx = (int(self.epoch) % FLAGS.external_parts) * self.num_examples + idx
    #   # ic(self.epoch, int(self.epoch) % FLAGS.external_parts, idx)
    row = dict(self.df.iloc[idx])    
    fe = parse_example(row, self.subset, self.rng)
    return fe
     
  def __len__(self):
    if self.rows:
      return len(self.rows)
    return self.num_examples
  