#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2022-05-11 11:18:18.068436
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.static_data import SAMPLE_IDS

RUN_VERSION = '7'
SUFFIX = ''
MODEL_NAME = 'ai4code'

flags.DEFINE_bool('tf', False, '')
flags.DEFINE_string('root', '../input/AI4Code', '')
flags.DEFINE_string('hug', 'deberta-v3-small', 'codeberta is faster while deberta-v3-small get better result')
flags.DEFINE_string('backbone', None, '')
flags.DEFINE_string('custom_backbone', None, '')
flags.DEFINE_string('backbone_dir', None, '')
flags.DEFINE_bool('lower', False, '')
flags.DEFINE_integer('max_len', 128, '')
flags.DEFINE_integer('last_tokens', 32, '32 or 1/4 * max_len?')
flags.DEFINE_bool('static_inputs_len', False, '')
flags.DEFINE_integer('n_context', 30, 'at mose n_context codes')
flags.DEFINE_integer('n_context2', 80, 'at mose n_context codes')
flags.DEFINE_integer('max_context_len', 23, '')
flags.DEFINE_integer('max_context2_len', 60, '')
flags.DEFINE_bool('bert_encode', True, '')
flags.DEFINE_bool('use_dot', False, '')
flags.DEFINE_bool('use_din', False, '')
flags.DEFINE_bool('use_sigmoid', False, 'not good')
flags.DEFINE_integer('fold_seed', 1024, '')
flags.DEFINE_integer('aug_seed', None, '')
flags.DEFINE_bool('from_zero', False, 'force to re gen train.fea when training')
flags.DEFINE_bool('train_code', False, '--train_code improved a bit for pointwise model where each code also predict rank')
flags.DEFINE_integer('num_ids', None, '')
flags.DEFINE_bool('clear_dfs', False, '')
flags.DEFINE_bool('disable_token_type_ids', False, '')
flags.DEFINE_alias('disable_ttids', 'disable_token_type_ids')
flags.DEFINE_string('cell_type', 'markdown', '')
flags.DEFINE_bool('use_code', True, '')
flags.DEFINE_bool('use_context', True, '')
flags.DEFINE_bool('use_context2', False, '')
flags.DEFINE_bool('use_markdowns', False, '')
flags.DEFINE_bool('use_embs', False, '')
flags.DEFINE_integer('n_markdowns', 0, 'num additional markdowns')
flags.DEFINE_integer('max_markdown_len', 32, '')
flags.DEFINE_string('context_padding', 'do_not_pad', 'do_not_pad or maxlength(fixed padding) or longest(longgest in batch)')
flags.DEFINE_string('context_sep', None, '')
flags.DEFINE_string('end_source', 'The end of the notebook', '')
flags.DEFINE_bool('add_end_source', False, 'add by default for pairwise')
flags.DEFINE_bool('dynamic_context_len', False, '')
flags.DEFINE_float('dynamic_context_weight', None, '')
flags.DEFINE_integer('dynamic_context_topn', 5, '')
flags.DEFINE_bool('dynamic_codes', False, 'only for infer')
flags.DEFINE_float('rerank_max_thre', 0.9, '')
flags.DEFINE_float('rerank_min_thre', 0., '')
flags.DEFINE_bool('rerank_filter', False, '')

flags.DEFINE_string('external', None, '')
# flags.DEFINE_integer('external_parts', None, '')
flags.DEFINE_integer('external_idx', 0, '')
flags.DEFINE_integer('num_externals', 10, '')
flags.DEFINE_bool('external_test', False, '')

flags.DEFINE_bool('pairwise', False, '')
flags.DEFINE_bool('pairwise_reg', False, '')
flags.DEFINE_bool('pairwise_legacy', False, '')
flags.DEFINE_bool('two_tower', True, '')
flags.DEFINE_integer('num_negs', 4, '')
flags.DEFINE_bool('dump_emb', False, '')
flags.DEFINE_bool('save_emb', False, '')
flags.DEFINE_string('embs_dir', None, '')
flags.DEFINE_bool('save_probs', True, '')
flags.DEFINE_bool('pairwise_eval', False, '')
flags.DEFINE_bool('mlp', False, '')
flags.DEFINE_string('mlp_activation', 'GELU', '')
flags.DEFINE_bool('share_mlp', True, '')
flags.DEFINE_bool('layernorm', False, '')
flags.DEFINE_float('layernorm_eps', 1e-6, '')
flags.DEFINE_bool('l2norm', False, '')
flags.DEFINE_float('temperature', 0., '')
flags.DEFINE_float('dtemperature', 0., '')
flags.DEFINE_bool('share_pooling', True, '')
flags.DEFINE_bool('share_poolings', False, '')
flags.DEFINE_bool('encode_code_info', False, '')
flags.DEFINE_bool('encode_code_info_simple', False, '')
flags.DEFINE_bool('encode_more_info', False, '')
flags.DEFINE_alias('encode_more', 'encode_more_info')
flags.DEFINE_bool('encode_markdown_info', False, '')
flags.DEFINE_bool('encode_all_info', False, '')

flags.DEFINE_integer('n_pairwise_context', 0, '')
flags.DEFINE_string('pairwise_context_method', 'local', 'local or global')
flags.DEFINE_bool('pairwise_probs', False, '')
flags.DEFINE_bool('pairwise_markdowns', False, '')
flags.DEFINE_float('markdown_loss_rate', 1., '')
flags.DEFINE_string('pooling_mask', 'token_type_ids', 'unlike feedback-eff here attention_mask is better')
flags.DEFINE_bool('dynamic_token_types', False, '')
flags.DEFINE_bool('sbert', False, '')
flags.DEFINE_bool('insert_mode', False, 'wether to put markdwon inside for pairwise non two power mode, insert_mode is better with no cost')

flags.DEFINE_integer('max_position_embeddings', None, '')
flags.DEFINE_integer('type_vocab_size', None, '')
flags.DEFINE_bool('position_biased_input', None, '')

flags.DEFINE_integer('method', 2, '1 pointwise, 2 random sample 20 code')
flags.DEFINE_string('label_name', 'rel_rank', 'rel_rank or pct_rank')

flags.DEFINE_bool('use_markdown_frac', True, 'markdown frac helps')
flags.DEFINE_bool('encode_markdown_frac', False, '')
flags.DEFINE_bool('add_info', False, 'add like ncell info before fc')
flags.DEFINE_bool('encode_info', False, 'similar info but add on text input')
flags.DEFINE_bool('encode_rel_rank', False, '')

flags.DEFINE_bool('unk_init', True, '')
flags.DEFINE_bool('freeze_emb', False, '') 
flags.DEFINE_string('pooling', 'latt', 'latt better then cls')
flags.DEFINE_bool('multi_lr', False, '')
flags.DEFINE_bool('continue_pretrain', False, '')
flags.DEFINE_float('base_lr', None, '1e-3 or 5e-4')
flags.DEFINE_bool('weight_decay', True, '')

flags.DEFINE_bool('seq_encoder', True, 'seq encoder helps')
flags.DEFINE_bool('seq_encoder2', False, '')
flags.DEFINE_integer('rnn_layers', 1, '')
flags.DEFINE_bool('rnn_bi', True, '')
flags.DEFINE_float('rnn_dropout', 0.1, '')
flags.DEFINE_string('rnn_type', 'LSTM', '')
flags.DEFINE_bool('rnn_double_dim', True, '')

flags.DEFINE_integer('emb_dim', 128, '')
flags.DEFINE_bool('emb_trainable', True, '')
flags.DEFINE_string('emb_weight', 'w2v/128/tokens.npy', '')

flags.DEFINE_integer('context_shift', 0, '')
flags.DEFINE_bool('context_aug', False, '')
flags.DEFINE_bool('context_valid_aug', True, 'tested False is much better')
flags.DEFINE_float('context_aug_rate', None, '')
flags.DEFINE_string('context_method', 'even', 'even,random,input,shift,rand_shift')

flags.DEFINE_integer('num_classes', 100, '')
flags.DEFINE_string('cls2pred', 'softmax', 'argmax, softmax')

flags.DEFINE_string('loss_method', 'mae', 'mae better then mse due to metric is tau for measuring l1 distance')
flags.DEFINE_float('base_loss_rate', 1., '')
flags.DEFINE_float('global_loss_rate', 0., '')
flags.DEFINE_float('local_loss_rate', 0., '')
flags.DEFINE_float('context_match_loss_rate', 0.0, 'not used')
flags.DEFINE_alias('match_loss_rate', 'context_match_loss_rate')
flags.DEFINE_float('cls_loss_rate', 0.0, 'tested 0.1 much better then 1 especially for last epochs')
flags.DEFINE_float('cls_pred_ratio', 0.5, '')
flags.DEFINE_bool('cls_span', True, '')
flags.DEFINE_bool('pairwise_span', True, '')
flags.DEFINE_float('rank_loss_rate', 0.0, '')

flags.DEFINE_string('pairwise_dir', None, 'pairwise probs valid.pkl dir')
flags.DEFINE_string('pairwise_dir2', None, 'pairwise probs test.pkl dir')
flags.DEFINE_bool('pairwise_oof', False, '')

flags.DEFINE_bool('sample_by_recall', True, '')
flags.DEFINE_bool('pre_seg', False, '')

flags.DEFINE_integer('markdown_token_type_id', 0, '')

flags.DEFINE_bool('list_infer', False, 'A bit complex to support list_infer during training so just use it for test mode, veriyfied similear with or without it')
flags.DEFINE_bool('list_leak', False, '')
flags.DEFINE_bool('list_train_ordered', True, '')
flags.DEFINE_bool('add_probs', False, 'HACK  not used, use add_probs2')
flags.DEFINE_bool('add_probs2', False, 'HACK  not used')

flags.DEFINE_float('neg_rand_prob', None, '')
flags.DEFINE_string('neg_method', 'rand', '')
flags.DEFINE_float('mrand_prob', None, '')
flags.DEFINE_float('crand_prob', None, '')
flags.DEFINE_bool('markdown_sample', False, '')
flags.DEFINE_bool('neg_sample', False, '')
flags.DEFINE_string('neg_strategy', 'rand-sample', 'rand-greedy or rand-sample')

# -----------------lm 
flags.DEFINE_bool('lm_train', False, 'mlm now, TODO rtd then deberta v3')
flags.DEFINE_integer('lm_max_len', 512, '')
flags.DEFINE_integer('lm_stride', 128, '')
flags.DEFINE_bool('lm_header', False, 'custom lm header bad valid result and train loss higher at begging, because it is not BertOnlyHeader.. TODO')
flags.DEFINE_bool('lm_line_by_line', False, '')
flags.DEFINE_bool('dynamic_line_by_line', False, '')
flags.DEFINE_integer('lm_max_lines', 0, '')
flags.DEFINE_bool('lm_custom_dataset', False, '')
flags.DEFINE_bool('lm_balance', False, '')
flags.DEFINE_bool('lm_hug_inputs', False, '')
flags.DEFINE_bool('lm_combine', False, '')

#Unicode编号	 U+02B6
BR = 'ʶ'

EVAL_KEYS = [
            'id', 'cell_id', 'cid', 'cell_type', 'rank', 'rel_rank', 
            'context_match', 'match_code', 'n_code_cell', 
            'n_markdown_cell', 'n_cell', 'code_idxes', 'n_words',
          ]
# for lst models, cd pikachu/third/convert_checkpoint_to_lsg
# python modify/convert_bert_checkpoint.py  --initial_model /work/data/huggingface/huggingface/CodeBERTa-small-v1 --model_name /work/data/huggingface/lsg-codeberta  --max_sequence_length 4096
#  python modify/convert_bert_checkpoint.py  --initial_model /work/data/huggingface/microsoft/xtremedistil-l6-h256-uncased --model_name /work/data/huggingface/lsg-xdistill-small --max_sequence_length 4096
# python convert_bert_checkpoint.py  --initial_model /work/data/huggingface/microsoft/MiniLM-L12-H384-uncased --model_name /work/data/huggingface/lsg-minilm-12  --max_sequence_length 4096
# python modify/convert_roberta_checkpoint.py  --initial_model /work/data/huggingface/microsoft/graphcodebert-base --model_name /work/data/huggingface/lsg-graphcodebert-base  --max_sequence_length 4096
# python convert_roberta_checkpoint.py  --initial_model /work/data/huggingface/microsoft/graphcodebert-base --model_name /work/data/huggingface/lsg-graphcodebert-base  --max_sequence_length 4096
# robert need disable token type ids, like flag.sh flags/context3 --hug=lgraphcodebert --n_context=30 --disable_token_type_ids
# python convert_bert_checkpoint.py --initial_model ~/working/ai4code/offline/7/0/paraphrase-multilingual-MiniLM-L12-v2.flag-emlm-mlm-mmnilm  --model_name ~/working/ai4code/offline/7/0/paraphrase-multilingual-MiniLM-L12-v2.flag-emlm-mlm-mmnilm.lsg  --max_sequence_length 4096
hugs = {
  'deberta': 'microsoft/deberta-large',
  'deberta-base': 'microsoft/deberta-base',
  'deberta-xl': 'microsoft/deberta-xlarge',
  'deberta-xlarge': 'microsoft/deberta-xlarge',
  'deberta-v2': 'microsoft/deberta-v2-xlarge', # v2 v3 all has problem... of tokenizer no fast/non python version (hack now)
  'deberta-v2-xlarge': 'microsoft/deberta-v2-xlarge', # v2 v3 all has problem... of tokenizer no fast/non python version (hack now)
  'deberta-v2-xxlarge': 'microsoft/deberta-v2-xxlarge',
  'deberta-v3': 'microsoft/deberta-v3-large', 
  'deberta-v3-base': 'microsoft/deberta-v3-base',
  'deberta-v3-small': 'microsoft/deberta-v3-small',
  'deberta-v3-xsmall': 'microsoft/deberta-v3-xsmall',
  
  'codeberta': 'huggingface/CodeBERTa-small-v1',
  'lcodeberta': 'lsg-codeberta',
  'codebert': 'microsoft/codebert-base',
  'lcodebert': 'lsg-codebert-base',
  'graphcodebert': 'microsoft/graphcodebert-base',
  'lgraphcodebert': 'lsg-graphcodebert-base',
  'funnel-small': 'funnel-transformer/small',
  'xdistill-small': 'microsoft/xtremedistil-l6-h256-uncased',
  'lxdistill-small': 'lsg-xdistill-small',
  'xdistill-base': 'microsoft/xtremedistil-l6-h384-uncased',
  'xdistill-large': 'microsoft/xtremedistil-l12-h384-uncased',
  'lxdistill-base': 'lsg-xdistill-base',
  'lxdistill-large': 'lsg-xdistill-large',
  'electra-small': 'google/electra-small-discriminator',
  'mpnet': 'microsoft/mpnet-base',
  
  'ernie-tiny': 'nghuyong/ernie-tiny',
  'minilm-l12': 'microsoft/MiniLM-L12-H384-uncased',
  'mminilm-l12': 'microsoft/Multilingual-MiniLM-L12-H384',
  
  'lsg-ernie-tiny': 'lsg-ernie-tiny',
  'lsg-minilm-l12': 'lsg-minilm-12',
  'lsg-mminilm-l12': 'lsg-mminilm-12',
  'lsg-mminilm': 'lsg-mminilm-12',
  'lsg-mpnet': 'microsoft/mpnet-base',
  #sentence transformers
  # https://www.sbert.net/docs/pretrained_models.html
  'all-MiniLM-L12-v2': 'sentence-transformers/all-MiniLM-L12-v2',
  'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',

  'paraphrase-MiniLM-L3-v2': 'sentence-transformers/paraphrase-MiniLM-L3-v2',
  'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
  'multi-qa-MiniLM-L6-cos-v1': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
  'paraphrase-multilingual-MiniLM-L12-v2': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
  'paraphrase-multilingual-mpnet-base-v2': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
  'pmminilm': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
  
  'multi-qa-mpnet-base-dot-v1': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
  'all-distilroberta-v1': 'sentence-transformers/all-distilroberta-v1',
  'multi-qa-distilbert-cos-v1': 'sentence-transformers/multi-qa-distilbert-cos-v1',
  'multi-qa-MiniLM-L6-cos-v1': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
  'distiluse-base-multilingual-cased-v1': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
  'distiluse-base-multilingual-cased-v2': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
  
  #cross encoder
  'cross-MiniLM-L12-v2': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
  'cross-deberta-v3-small': 'cross-encoder/nli-deberta-v3-small',
  'cross-deberta-v3-xsmall': 'cross-encoder/nli-deberta-v3-xsmall',
}

def get_backbone(backbone, hug):
  backbone = backbone or hugs.get(hug, hug)
  backbone_ = backbone.split('/')[-1]
  if FLAGS.continue_pretrain:
    backbone_path = f'{FLAGS.root}/pretrain/{backbone_}'
    if os.path.exists(f'{backbone_path}/config.json'):
      backbone = backbone_path
      return backbone
  backbone_path = f'{FLAGS.root}/{backbone_}'
  if os.path.exists(f'{backbone_path}/config.json'):
    backbone = backbone_path
    return backbone
  backbone_path = f'../input/{backbone_}'
  if os.path.exists(f'{backbone_path}/config.json'):
    backbone = backbone_path
    return backbone
    
  return backbone

def get_records_name():
  records_name = FLAGS.backbone.split('/')[-1]
  return records_name

def config_train():
  lr = 5e-5
  bs = 128
  # FLAGS.lr_decay_power = 2
  # FLAGS.lr_decay_power = 0.5
  grad_acc = 1
  if FLAGS.n_context > 20:
    grad_acc = 2
  # if FLAGS.n_context > 30:
  #   grad_acc = 4
  
  if 'xlarge' in FLAGS.hug:
    lr = 1e-5
    bs = int(bs / 2)
    FLAGS.lr_decay_power = 2
    # FLAGS.grad_acc = 2
    
  if 'electra' in FLAGS.hug:
    lr = 1e-5
    
  # if 'deberta-v2-xlarge' in FLAGS.hug:
  #   lr = 1e-5
  #   bs = 64
  #   # FLAGS.lr_decay_power = 2
  
  FLAGS.lr = FLAGS.lr or lr
  if FLAGS.grad_acc == 0:
    FLAGS.grad_acc = 1
  elif FLAGS.grad_acc == 1:
    FLAGS.grad_acc = grad_acc
  FLAGS.clip_gradients = 1.
  
  # versy sensitive to lr ie, for roformer v2 large, 5e-5 + 5e-4 will not converge but 5e-5 + 1e-4 will
  # also TODO with 1e-4 + 1e-3 lr, opt_fused very interesting download with roformer v2 large layer norm .. random init due to key miss in checkpoint will converge
  # but if save_pretrained then reload will not, why ?
  if FLAGS.multi_lr:
    # base_lr = FLAGS.lr * 10.
    # base_lr = 1e-3
    base_lr = 5e-4
    FLAGS.base_lr = FLAGS.base_lr or base_lr
    
  # FLAGS.loss_scale = 100
  FLAGS.bs = FLAGS.bs or bs
  FLAGS.eval_bs = FLAGS.eval_bs or FLAGS.bs * 2
  
  # for ep = 5, cosine scheduler + lr 5e-5 best(cosine means large lr at begining then linear), for ep 9 might be lr 3e-5 
  # for n_context=20 currentl best offline 8615 online 8501 with lr 3e-5, ep=10, linear scheduler
  ep = 9
  # pointwise mode
  if not FLAGS.use_context: 
    ep = 4
  
  # change from 7 to 6 epoch for simple and faster training, notice current best flags/pairwise5 with acc 7802 was trained using 7 epochs, it got best at epoch 6 with 7805
  if FLAGS.pairwise:
    ep = 6
  # if FLAGS.method > 1:
  #   ep = 1
    
  FLAGS.ep = FLAGS.ep or ep
  
  # change from adamw back to adam
  optimizer = 'adamw' 
  FLAGS.optimizer = FLAGS.optimizer or optimizer
  FLAGS.opt_eps = 1e-7

  # cosine does worse for 5e-5 with large epochs like 9, also for epochs like 5 tought better then linear the gain is small 
  # so for convince just set to linear for safe
  # scheduler = 'cosine' 
  # if FLAGS.ep > 5:
  #   scheduler = 'linear'
  scheduler = 'linear'
  FLAGS.scheduler = FLAGS.scheduler or scheduler
  
  # awp_train = True
  # if FLAGS.awp_train is not None:
  #   awp_train = FLAGS.awp_train
  
  if FLAGS.pretrain and (not FLAGS.pretrain_restart):
    FLAGS.adv_start_epoch = 0
  else:
    adv_epochs = 3
    if FLAGS.adv_epochs is None:
      FLAGS.adv_epochs = FLAGS.adv_epochs or adv_epochs 
    if FLAGS.adv_epochs:
      FLAGS.adv_start_epoch = FLAGS.ep - FLAGS.adv_epochs
    
  ic(FLAGS.awp_train, FLAGS.adv_epochs, FLAGS.adv_start_epoch)

  # FLAGS.tf_dataset = True
  if FLAGS.tf_dataset:
    records_pattern = f'../working/{FLAGS.records_name}/{get_records_name()}/train/markdown/*.tfrec'
    ic(records_pattern)
    files = gezi.list_files(records_pattern) 
    FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
    
    if FLAGS.use_code:
      records_pattern = f'../working/{FLAGS.records_name}/{get_records_name()}/train/mix/*.tfrec'
    else:
      records_pattern = f'../working/{FLAGS.records_name}/{get_records_name()}/train/markdown/*.tfrec'
    files = gezi.list_files(records_pattern) 
    if FLAGS.online:
      FLAGS.train_files = files
    else:
      FLAGS.train_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]
    # np.random.shuffle(FLAGS.train_files)
    
    if FLAGS.hack_infer:
      FLAGS.test_files = FLAGS.valid_files
    else:
      records_pattern = f'../working/{FLAGS.records_name}/{get_records_name()}/test/markdown/*.tfrec'
      FLAGS.test_files = gezi.list_files(records_pattern) 
      
    if not FLAGS.train_files:
      if FLAGS.test_files:
        FLAGS.train_files = FLAGS.test_files
        FLAGS.valid_files = FLAGS.test_files
    
    ic(FLAGS.train_files[:2], FLAGS.valid_files[:2], FLAGS.test_files[:2])
    # assert FLAGS.train_files
    
    FLAGS.num_workers = int(FLAGS.num_workers / 2)
  
def config_model():
  # --sample means sample by default with ratio 0.01, is proper for fast protype with 0.01 num_ids selected for train/eval
  sample_frac = None
  if FLAGS.speed_level > 0:
    # FLAGS.hug = 'xdistill-small'
    sample_frac = 0.2
    if FLAGS.speed_level > 1:
      sample_frac = 0.1
    if FLAGS.speed_level > 2:
      sample_frac = 0.01
    if FLAGS.speed_level > 3:
      sample_frac = 0.005
  if FLAGS.sample:
    sample_frac = 0.01
  FLAGS.sample_frac = FLAGS.sample_frac or sample_frac
  
  if FLAGS.sample_frac:
    total_ids =  len(glob.glob(f'{FLAGS.root}/train/*.json'))
    FLAGS.num_ids = int(total_ids * FLAGS.sample_frac)
    ic(total_ids, FLAGS.sample_frac, FLAGS.num_ids)
  
  FLAGS.backbone = get_backbone(FLAGS.backbone, FLAGS.hug)
 
  if any(x in FLAGS.hug for x in ['deberta']):
    if not FLAGS.pairwise:
      FLAGS.find_unused_parameters = False
  
  if FLAGS.use_markdowns:
    FLAGS.type_vocab_size = 3  
  
  if FLAGS.hack_infer:
    FLAGS.num_ids = 512
    
  if FLAGS.mode is None:
    FLAGS.save_final = True
  # if FLAGS.pretrained:
  #   FLAGS.save_final = True
  assert not (FLAGS.pairwise_eval and FLAGS.dump_emb)

  # if FLAGS.pairwise_eval or FLAGS.dump_emb:
  #   FLAGS.save_final = True
  
  if FLAGS.temperature > 0 or FLAGS.dtemperature > 0:
    assert FLAGS.l2norm
  assert not (FLAGS.temperature > 0 and FLAGS.dtemperature > 0)
  
  ic(FLAGS.embs_dir)
  if FLAGS.embs_dir and (not FLAGS.use_embs):
    FLAGS.pairwise_eval = True
  ic(FLAGS.pairwise_eval)
  if FLAGS.pairwise_eval:
    assert not FLAGS.distributed, 'use DP not DDP for pairwise_eval'
  
def show():
  ic(FLAGS.backbone,
     FLAGS.method,
     FLAGS.loss_method,
     FLAGS.max_markdown_len,
     FLAGS.last_tokens,
     FLAGS.max_len,
     FLAGS.max_context_len,
     FLAGS.dynamic_context_len,
     FLAGS.n_context,
     FLAGS.pairwise,
     FLAGS.two_tower,
     FLAGS.dump_emb,
     FLAGS.pairwise_eval, 
     FLAGS.seq_encoder,
     FLAGS.rnn_bi,
     FLAGS.rnn_double_dim,
     FLAGS.context_padding,
     FLAGS.context_valid_aug,
     FLAGS.l2norm,
     FLAGS.temperature,
     FLAGS.dtemperature,
     FLAGS.markdown_token_type_id,
     FLAGS.pooling_mask,
     FLAGS.n_pairwise_context,
     FLAGS.sbert,
     FLAGS.pairwise_dir,
     FLAGS.pairwise_dir2,
     FLAGS.share_poolings,
     FLAGS.num_negs,
     FLAGS.n_markdowns,
     FLAGS.list_infer,
     FLAGS.neg_sample,
     FLAGS.crand_prob,
     FLAGS.mrand_prob,
     )

def init():
  config_model()
  folds = 5
  FLAGS.folds = FLAGS.folds or folds
  FLAGS.fold = FLAGS.fold or 0

  FLAGS.buffer_size = 20000
  FLAGS.static_input = True
  FLAGS.cache_valid = True
  FLAGS.async_eval = True
  FLAGS.async_eval_last = True if not FLAGS.pymp else False
  FLAGS.async_valid = False
  
  # 模型经常NCL pin memory报错 重启设置pin_memory=0 没再出现 但是不确定是否和这里num_workers设置比较大也有关系 也许8？
  # 实测一般设置pin_memory=0并没有减慢程序速度 当然如果不会崩溃的 还是可以继续使用默认的pin_memory=True
  # File "/work/envs/pku/lib/python3.7/site-packages/torch/utils/data/_utils/pin_memory.py", line 28 in _pin_memory_loop
  # FLAGS.num_workers = 16 
  # FLAGS.pin_memory = False
  
  # FLAGS.find_unused_parameters=True
  
  if not FLAGS.tf:
    # make torch by default
    FLAGS.torch = True
  else:
    FLAGS.torch = False
  
  FLAGS.run_version = FLAGS.run_version or RUN_VERSION
  if FLAGS.sample_frac and FLAGS.sample_frac < 1:
    if not FLAGS.sample:
      FLAGS.run_version += f'.sample-{FLAGS.sample_frac}'
    else:
      FLAGS.run_version += '.sample'
      
  if FLAGS.online:
    FLAGS.allow_train_valid = True
    FLAGS.nvs = 1
    # assert FLAGS.fold == 0
    if FLAGS.fold != 0:
      ic(FLAGS.fold)
      exit(0)
     
  if FLAGS.log_all_folds or FLAGS.fold == 0:
    wandb = True
    if FLAGS.wandb is None:
      FLAGS.wandb = wandb
    FLAGS.wandb_project = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # FLAGS.wandb_entity = 'pikachu'
  FLAGS.write_summary = True
  
  FLAGS.run_version += f'/{FLAGS.fold}'
  
  pres = ['offline', 'online']
  pre = pres[int(FLAGS.online)]
  model_dir = f'../working/{pre}/{FLAGS.run_version}/model'  
  FLAGS.model_dir = FLAGS.model_dir or model_dir
  if FLAGS.mn == 'model':
    FLAGS.mn = ''
  if not FLAGS.mn:  
    if FLAGS.hug:
      model_name = FLAGS.hug
    FLAGS.mn = model_name
    if FLAGS.tf:
      FLAGS.mn = f'tf.{FLAGS.mn}'
      
    mt.model_name_from_args(ignores=[
      'tf', 'hug', 'test_file', 'static_inputs_len', 'from_zero', 
      'prepare', 'num_ids', 'backbone_dir', 'external', 'external_idx'
      ])
    FLAGS.mn += SUFFIX  
  
  config_train()
  
  if not FLAGS.online:
    # nvs = max(FLAGS.ep, 3)
    nvs = FLAGS.ep
    # nvs *= 2
    FLAGS.nvs = FLAGS.nvs or nvs
    # FLAGS.vie = 1
    # FLAGS.first_interval_epoch = 0.1
    
  FLAGS.write_valid_final = True
  FLAGS.save_model = False
  
  # by default save after each epoch training finish
  sie = 1
  if FLAGS.sie is None:
    FLAGS.sie = sie
  
  # FLAGS.sie = FLAGS.ep
  if FLAGS.sample_frac and FLAGS.sample_frac < 0.1:
    pass
    # FLAGS.sie = 1e10 
    # FLAGS.last_model = 3 
  else:
    # FLAGS.last_model = 3
    pass
  
  if FLAGS.lm_train:
    FLAGS.seq_encoder = False
    FLAGS.sie = 1
    FLAGS.do_valid = False
    FLAGS.do_test = False
    FLAGS.awp_train = False
    FLAGS.rdrop_rate = 0.
    FLAGS.find_unused_parameters = True
  
  if FLAGS.external:
    FLAGS.save_final = False
    # FLAGS.external_parts = FLAGS.num_externals
    FLAGS.num_decay_epochs = FLAGS.num_externals
    
  # if FLAGS.global_loss_rate > 0:
  #   FLAGS.find_unused_parameters = True
  
    