#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   infer.py
#        \author   chenghuige  
#          \date   2022-05-15 07:00:15.332073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
if os.path.exists('/kaggle'):
  sys.path.append('/kaggle/input/pikachu/utils')
  sys.path.append('/kaggle/input/pikachu/third')
  sys.path.append('.')
else:
  sys.path.append('..')
  sys.path.append('../../../../utils')
  sys.path.append('../../../../third')

from gezi.common import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'


from src.config import *
from src.preprocess import *
from src.postprocess import *

flags.DEFINE_string('ofile', '../working/x.pkl', '')

def predict(model):
  test_ds = get_dataloaders(test_only=True)

  if FLAGS.pairwise_eval:
    lele.predict(model, test_ds, 
                 out_keys=['id', 'cell_id', 'cid', 'cell_type', 'rank', 'rel_rank', 'n_words', 'n_cell', 'n_code_cell'], 
                 out_hook=out_hook,
                 amp=FLAGS.amp_infer,
                 fp16=FLAGS.fp16_infer) 
    x = out_hook_finalize()
  else:
    x = lele.predict(model, test_ds, 
                     out_keys=['id', 'cell_id', 'cid', 'code_idxes', 'n_cell', 'n_code_cell'],
                     amp=FLAGS.amp_infer,
                     fp16=FLAGS.fp16_infer)
    if FLAGS.list_infer:
      x = flattens(x)

  if FLAGS.pairwise and (not FLAGS.two_tower):
    xs = gezi.batch2list(x)
    l = []
    for x_ in xs:
      l.append(pairwise_cat2pred(x_, infer=True))
    x_ = gezi.list2batch(l)
    x.update(x_)
  
  return x

def main(argv):
  # gezi.init_flags()
  model_dir = FLAGS.model_dir
  bs = FLAGS.eval_bs
  ic(bs)
  gezi.restore_configs(model_dir, ignores=gezi.get_commandline_flags())
  ic(model_dir, bs, FLAGS.eval_bs, FLAGS.bs)
  FLAGS.train_allnew = False
  FLAGS.grad_acc = 1
  FLAGS.restore_configs = False
  FLAGS.bs, FLAGS.eval_bs = bs, bs
  mt.init()
  FLAGS.model_dir = model_dir
  FLAGS.pymp = False
  FLAGS.num_workers = 1
  FLAGS.pin_memory = True
  FLAGS.persistent_workers = True
  FLAGS.workers = 1
  FLAGS.fold = 0
  FLAGS.clear_dfs = True

  if FLAGS.pairwise:
    if FLAGS.two_tower:
      FLAGS.pairwise_eval = True
    FLAGS.n_markdowns = 0
    FLAGS.list_infer = False
    FLAGS.n_markdowns = 0
  else:
    FLAGS.list_infer = True
    # FLAGS.max_len = 128
    # FLAGS.max_context_len = 25
    assert FLAGS.n_markdowns
  
  FLAGS.pairwise_dir2 = None
  # FLAGS.num_negs = 2 
  FLAGS.hack_infer = FLAGS.num_ids is not None
  
  FLAGS.bs = FLAGS.eval_bs
  FLAGS.grad_acc = 1
  ic(FLAGS.model_dir, FLAGS.ofile, FLAGS.amp_infer, FLAGS.fp16_infer)
  ic(FLAGS.static_inputs_len, FLAGS.hack_infer, 
     FLAGS.num_workers, FLAGS.workers, FLAGS.pin_mem,
     FLAGS.bs, FLAGS.eval_bs)
  show()
  
  if gezi.in_kaggle():
    backbone = FLAGS.backbone.split('/')[-1].replace('_', '-')
    FLAGS.backbone = '../input/' + backbone
  
  FLAGS.mode = 'test'
  FLAGS.work_mode = 'test'
  ic(FLAGS.backbone, model_dir, os.path.exists(f'{model_dir}/model.pt'))
  try:
    display(pd.read_csv(f'{model_dir}/metrics.csv'))
  except Exception as e:
    logger.warning(e)
  from src.torch.model import Model
  model = Model()
  ic(gezi.get_mem_gb())
  gezi.load_weights(model, model_dir)
  ic(gezi.get_mem_gb())
  
  if FLAGS.pairwise and (not FLAGS.two_tower) and FLAGS.rerank_filter:
    xs = []

    ##  8922 <3.5h
    FLAGS.rerank_min_thre = 0
    # FLAGS.rerank_max_thre=0.8
    # FLAGS.num_negs = 4
    xs.append(predict(model))
        
    x = gezi.merge_array_dicts(xs)
  else:
    x = predict(model)
  
  ic(gezi.get_mem_gb())
  gezi.save(x, FLAGS.ofile)
  

if __name__ == '__main__':
  app.run(main)  
  
