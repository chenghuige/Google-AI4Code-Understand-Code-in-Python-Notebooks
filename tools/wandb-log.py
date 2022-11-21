#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   wandb.py
#        \author   chenghuige  
#          \date   2022-07-15 11:58:55.776559
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import wandb
from absl.testing import flagsaver
import melt as mt
#try:
from src.config import RUN_VERSION
#except Exception:
#  pass
flags.DEFINE_bool('append', False, '')

# usage wandb-log.py --wg=6/pairwise --mn=ModelA
# wandb-log.py --wg=6/pairwise --mn=*modelA.eval*

def work():
  wandb_group = FLAGS.wandb_group
  run_version = FLAGS.wandb_group.split('/')[0] if FLAGS.wandb_group else RUN_VERSION
  if not FLAGS.model_dir:
    FLAGS.model_dir = f'../working/offline/{RUN_VERSION}/{FLAGS.fold}/{FLAGS.mn}'
  else:
    FLAGS.mn = os.path.basename(FLAGS.model_dir)
  wandb_root = '../working/offline/wandb'
  logging.init(FLAGS.model_dir)
  model_name = FLAGS.mn
  metrics_file = f'{FLAGS.model_dir}/metrics.csv'
  if not os.path.exists(metrics_file):
    logging.warning(f'not fild {metrics_file}')
    return
  df = pd.read_csv(metrics_file)
  # # Modify or remove here
  # if df.score.max() < 0.88:
  #   logger.warning(f'{FLAGS.model_dir} bad score')
  #   ic(df.score.max())
  #   return
  # if not os.path.exists(f'{FLAGS.model_dir}/done.txt'):
  #   logger.warning(f'{FLAGS.model_dir} not finished')
  #   ic(df.step.max())
  #   return
  
  gezi.init_flags()
  ignores = ['wandb_scratch', 'wds', 'wandb_group', 'wg', 'wandb_overwrite', 'wo', 'wandb_entity', 'we']
  gezi.restore_configs(FLAGS.model_dir, ignores=ignores + gezi.get_commandline_flags())
  flag_values = flagsaver.save_flag_values()
  FLAGS.mn = model_name
  #FLAGS.wandb_project = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  
  FLAGS.wandb_project = FLAGS.wandb_project or os.path.basename(os.path.dirname(os.getcwd()))
  FLAGS.wandb = True
  FLAGS.wandb_tb = True
  FLAGS.wandb_group = wandb_group
  FLAGS.wandb_silent = True
  os.environ["WANDB_SILENT"] = "true"
  
  wandb.login(key=FLAGS.wandb_key)
  wandb_dir = FLAGS.wandb_dir
  if not wandb_dir:
    if FLAGS.append:
      wandb_dir = FLAGS.model_dir
    else:
      wandb_dir = f'{wandb_root}/{wandb_group}/{FLAGS.mn}'
  gezi.try_mkdir(wandb_dir)
  ic(run_version, FLAGS.wandb_project, FLAGS.wandb_group, FLAGS.mn, FLAGS.model_dir, wandb_dir)
  wandb_config = gezi.get('wandb_config', FLAGS.flag_values_dict())
  wandb_id = gezi.read_str_from(f'{wandb_dir}/wandb_id.txt')
  if wandb_id:
    if FLAGS.wandb_scratch:
      logger.warning(f'find wandb_id:{wandb_id} will delete it first')
      try:
        gezi.wandb_delete(FLAGS.wandb_project, wandb_id)
      except Exception:
        pass
    elif FLAGS.wandb_overwrite:
      logger.warning(f'find wandb_id:{wandb_id} will overwirte')
    else:
      logger.warning(f'find wandb_id:{wandb_id} will ignore')
      return
  group = FLAGS.wandb_group 
  if not group:
    if not FLAGS.folds:
      group = os.path.basename(os.path.dirname(FLAGS.model_dir))
    else:
      group_ = os.path.basename(os.path.dirname(os.path.dirname(FLAGS.model_dir)))
      wandb_dir_folds = f'{os.path.dirname(os.path.dirname(FLAGS.model_dir))}/{FLAGS.folds}'
      gezi.try_mkdir(wandb_dir_folds)
      group_folds = f'{group_}/{FLAGS.folds}'
      group = f'{group_}/{FLAGS.fold}'
      try:
        if FLAGS.online:
          group =  f'{group_}/online'
      except Exception:
        pass

  if not (FLAGS.folds and FLAGS.folds_metrics):
    FLAGS.wandb_group = group 
    FLAGS.wandb_dir = wandb_dir
    ic(group, FLAGS.wandb_id)

    run = wandb.init(project=FLAGS.wandb_project,
                    group=group,
                    dir=wandb_dir,
                    id=wandb_id,
                    config=wandb_config,
                    name=FLAGS.wandb_name or FLAGS.model_name,
                    notes=FLAGS.wandb_notes,
                    tags=FLAGS.wandb_tags,
                    sync_tensorboard=FLAGS.wandb_tb,
                    magic=FLAGS.wandb_magic)
      
  gezi.set('wandb_run', run)
  wandb_id = wandb.run.id
  gezi.set('wandb_id', wandb_id)
  FLAGS.wandb_id = wandb_id
  ic(run.url, FLAGS.wandb_id)
  gezi.write_to_txt(wandb_id, f'{wandb_dir}/wandb_id.txt')

  cols = df.columns
  # cols = [x for x in cols if x not in ['ntime']]
  df = df[cols]
  gezi.pprint(df)
  rows = df.to_dict('records')
  for row in rows:
    row = gezi.dict_prefix(row, 'Metrics/')
    row['step'] = row['Metrics/step']
    row['epoch'] = row['Metrics/epoch']
    wandb.log(row)
  ic(row)
  wandb.finish()
  
def main(_):
  assert FLAGS.mn or FLAGS.model_dir
  if FLAGS.fold is None:
    FLAGS.fold = 0
  if FLAGS.mn and '*' in FLAGS.mn:
    run_version = FLAGS.wandb_group.split('/')[0] if FLAGS.wandb_group else RUN_VERSION
    root = f'../working/offline/{RUN_VERSION}/{FLAGS.fold}'
    pattern = FLAGS.mn 
    model_dirs = glob.glob(f'{root}/{pattern}')
    for model_dir in tqdm(model_dirs):
      FLAGS.model_dir = model_dir
      ic(FLAGS.model_dir)
      work()
  else:
    work()
  
if __name__ == '__main__':
  app.run(main)  
