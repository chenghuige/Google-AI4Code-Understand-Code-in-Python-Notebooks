#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2022-02-10 13:08:57.365906
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi
from gezi.common import *
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')


import melt as mt

from src import config
from src.config import *
from src.preprocess import *

flags.DEFINE_string('mark', 'train', 'train')
flags.DEFINE_integer('buf_size', 10000, '')
flags.DEFINE_integer('num_records', None, '')
flags.DEFINE_integer('seed_', 1024, '')

df = None
records_dir = None
tokenizer = None

def deal(index):
  df_ = df[df['worker'] == index]
  num_insts = len(df_)
  ofile = f'{records_dir}/{index}.tfrec'
  # shuffle = True if FLAGS.mark == 'train' else False
  shuffle = False if FLAGS.cell_type != 'mix' else True
  deal_fn = parse_example 
  with mt.tfrecords.Writer(ofile, buffer_size=FLAGS.buf_size, shuffle=shuffle, seed=1024) as writer:
    for i, row in tqdm(enumerate(df_.itertuples()), total=num_insts, desc=ofile, leave=True):
      row = row._asdict()
      fe =  deal_fn(row, tokenizer)
      if fe:
        writer.write(fe)
      
def main(_):    
  config.init()
  global df, records_dir, tokenizer
  
  records_name = get_records_name()
  records_dir = f'../working/tfrecords/{records_name}/{FLAGS.mark}/{FLAGS.cell_type}'
  ic(records_dir)

  if FLAGS.clear_first:
    command = f'rm -rf {records_dir}'
    gezi.system(command)    
  # else:
  #   assert not glob.glob(f'{records_dir}/*.tfrec'), records_dir    
  
  tokenizer = get_tokenizer(FLAGS.backbone)
  df = get_df(mark=FLAGS.mark)
  
  if FLAGS.cell_type != 'mix':
    df = df[df.cell_type == FLAGS.cell_type]
  else:
    df = df.sample(frac=1, random_state=FLAGS.seed_)

  ic(df)
  if FLAGS.mark == 'train':
    ic(len(df[df.fold==0]), ic(len(df[(df.worker % FLAGS.folds)==0])))
  num_records = FLAGS.num_records or df.worker.max() + 1
  FLAGS.num_records = num_records
  ic(num_records)
  
  gezi.prun_loop(deal, range(num_records), num_records)
  
  ic(FLAGS.mark, records_dir, mt.get_num_records_from_dir(records_dir))


if __name__ == '__main__':
  app.run(main)

