#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   cp_model.py
#        \author   chenghuige  
#          \date   2019-10-30 00:20:59.308678
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import melt 

from absl import flags, app
FLAGS = flags.FLAGS

flags.DEFINE_bool('use_step', True, '')
flags.DEFINE_bool('use_pb', False, '')
flags.DEFINE_bool('weights_only', False, '')

def main(_):
  model = melt.latest_checkpoint(sys.argv[1])
  if os.path.exists(sys.argv[2]):
    command = 'rm -rf  %s' % sys.argv[2]
    os.system(command)
  command = 'mkdir -p %s' % sys.argv[2]
  os.system(command)
  
  print('use_step', FLAGS.use_step)
  if FLAGS.use_step:
    command = f'rsync -avP --delete {sys.argv[1]}/eval_step.txt {sys.argv[2]}'
    os.system(command)
  
  if FLAGS.weights_only:
    command = f'rsync -avP --delete {sys.argv[1]}/model.h5 {sys.argv[2]}'
    print(command)
    os.system(command)
  else:
    command = f'rsync -avP --delete {sys.argv[1]}/checkpoint {model}* {sys.argv[2]}'
    print(command)
    os.system(command)
    command = f'cp {sys.argv[1]}/checkpoint {sys.argv[2]}/checkpoint.src'
    os.system(command) 
  
  #command = f'rsync -avP --delete {sys.argv[1]}/model* {model}* {sys.argv[2]}'
  #os.system(command)

 
  print('use_pb', FLAGS.use_pb)
  if FLAGS.use_pb:
    command = f'rsync -avP --delete {sys.argv[1]}/model.pb* {sys.argv[1]}/model.map* {sys.argv[2]}'
    os.system(command)

  with open(f'{sys.argv[2]}/finetune_command.txt', 'w') as out:
    print(' '.join(sys.argv), file=out)

if __name__ == '__main__':
  app.run(main)  
