#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige
#          \date   2022-05-11 11:12:36.045278
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
from src.util import *
from src.sample import *

tokenizers = {}
def get_tokenizer(backbone=None):
  backbone = backbone or FLAGS.backbone
  assert backbone
  
  if backbone in tokenizers:
    return tokenizers[backbone]
  
  ic(backbone)
  from transformers import AutoTokenizer
  if not 'deberta-v' in backbone:
    try:
      tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_dir, use_fast=True)
    except Exception:
      tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
  else:
    # cd pikachu/projects/feedback/tools; sh update.sh
    try:
      from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
      tokenizer = DebertaV2TokenizerFast.from_pretrained(backbone, use_fast=True)
    except Exception as e:
      ic(e)
      tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    
  if tokenizer.convert_tokens_to_ids(BR) == tokenizer.unk_token_id:
    assert len(BR) == 1
    tokenizer.add_tokens([BR], special_tokens=False)
  
  ic(tokenizer.is_fast)
  tokenizers[backbone] = tokenizer
  return tokenizer

def preprocess(text):
  if FLAGS.lower:
    text = text.lower()

  return text  

def encode(text, tokenizer, max_length, last_tokens, padding='longest', token_type_id=0):
  text = preprocess(text)
  # res = tokenizer(text,
  #                 truncation=True,
  #                 max_length=FLAGS.max_len,
  #                 padding=padding,
  #                 return_offsets_mapping=False)
  res = {}
  ori_input_ids = tokenizer.encode(text)
  # ic('|'.join(tokenizer.convert_ids_to_tokens(ori_input_ids)), len(ori_input_ids))
  ori_attention_mask = [1] * len(ori_input_ids)
  input_ids = gezi.trunct(ori_input_ids, max_length, last_tokens=last_tokens)
  attention_mask = [1] * len(input_ids)
  token_type_ids = [token_type_id] * len(input_ids)
  ori_token_type_ids = [token_type_id] * len(input_ids)
  res = {
    'ori_input_ids': ori_input_ids,
    'ori_attention_mask': ori_attention_mask,
    'ori_token_type_ids': ori_token_type_ids,
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids,
  }
  return res

def encode_markdown(text, tokenizer, token_type_id=0):
  res = {}
  ori_input_ids = tokenizer.encode(text)
  input_ids = gezi.trunct(ori_input_ids, FLAGS.max_len, last_tokens=FLAGS.last_tokens)
  attention_mask = [1] * len(input_ids)
  token_type_ids = [token_type_id] * len(input_ids)
  res = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids,
  }
  return res

def encode_code(text, tokenizer, max_length, token_type_id=1):
  res = tokenizer(text,
                  truncation=True,
                  max_length=max_length,
                  padding='longest',
                  return_offsets_mapping=False)
  res['token_type_ids'] = [token_type_id] * len(res['input_ids'])
  return res

# 0626 change from padding max_length to logest
def encodes(texts, tokenizer, max_length, padding=False, token_type_id=1):
  texts = list(texts)
  res = tokenizer.batch_encode_plus(
          texts,
          add_special_tokens=True,
          max_length=max_length,
          padding=padding,
          truncation=True
      )
  if not 'token_type_ids' in res:
    res['token_type_ids'] = []
    for input_ids in res['input_ids']:
      res['token_type_ids'].append([token_type_id] * len(input_ids))
  return res

def normalize(text):
  text = text.replace(BR, '') \
             .replace('[SEP]', '<SEP>') \
             .replace('[CLS]', '<CLS>') \
             .replace('[MASK]', '<MASK>') \
             .replace('[PAD]', '<PAD>') \
             .replace('[UNK]', '<UNK>') \
             .replace('<s>', '[s]') \
             .replace('<\s>', '[\s]') \
             .replace('\n', BR)
  return text

def create_df(folder, workers=80):
  def _create_df(fpath):
    df = pd.read_json(fpath, dtype={'cell_type': 'category', 'source': 'str'}).reset_index().rename({"index":"cell_id"}, axis=1)
    df['id'] = fpath.rsplit('.', 1)[0].rsplit("/", 1)[-1]
    return df
  files = glob.glob(f'{folder}/*.json')
  if FLAGS.num_ids:
    files = files[:FLAGS.num_ids]
  elif FLAGS.num_ids == 0:
    files = [f'{folder}/{a}.json' for a in SAMPLE_IDS]
    ic(files[:2])
  dfs = gezi.prun(_create_df, files, workers, desc='read_jsons', leave=True)
  df = pd.concat(dfs)
  return df

def to_rank(df_orders, cell_dict):
  ids = []
  cell_ids = []
  ranks = []
  code_ranks = []
  markdown_ranks = []
  n_cells = []
  n_code_cells = []
  n_markdown_cells = []
  markdown_fracs = []
  rel_ranks = []
  global_ranks, local_ranks = [], []
  for row in tqdm(df_orders.itertuples(), total=len(df_orders), desc='to_rank'):
    cells = row.cell_order.split() 
    ncell = len(cells)
    n_cells.extend([ncell] * ncell)
    ids.extend([row.id] * ncell)
    cell_ids.extend(cells)
    ranks.extend(list(range(ncell)))
    code_ranks_ = [-1] * ncell
    markdown_ranks_ = [-1] * ncell
    code_rank, markdown_rank = 0, 0
    
    for i, cell in enumerate(cells):
      if cell_dict[f'{row.id}\t{cell}'] == 'code':
        code_ranks_[i] = code_rank
        code_rank += 1
      else:
        markdown_ranks_[i] = markdown_rank
        markdown_rank += 1
    code_ranks.extend(code_ranks_)
    markdown_ranks.extend(markdown_ranks_)
    n_code_cells.extend([code_rank] * ncell)
    n_markdown_cells.extend([markdown_rank] * ncell)
    markdown_fracs.extend([markdown_rank / (code_rank + markdown_rank)] * ncell)
    
    ncode, n_markdown = code_rank, markdown_rank
    code_rank, markdown_rank = 0, 0
    rel_ranks_ = [-1] * ncell
    global_ranks_, local_ranks_ = [-1] * ncell, [-1] * ncell
    for i, cell in enumerate(cells):
      if cell_dict[f'{row.id}\t{cell}'] == 'code':
        prev = code_rank * (1 / (ncode + 1))
        code_rank += 1
        rel_ranks_[i] = code_rank * (1 / (ncode + 1))
        local_ranks_[i] = markdown_rank
        
        j = i - 1
        while j >= 0 and rel_ranks_[j] >= 1:
          global_ranks_[j] = prev
          local_ranks_[j] = rel_ranks_[j] / (markdown_rank + 1)
          rel_ranks_[j] = prev + rel_ranks_[j] * ((1 / (ncode + 1)) / (markdown_rank + 1))
          j -= 1
        markdown_rank = 0
      else:
        markdown_rank += 1
        rel_ranks_[i] = markdown_rank
        
    j = i 
    prev = code_rank * (1 / (ncode + 1))
    while j >= 0 and rel_ranks_[j] >= 1:
      global_ranks_[j] = prev
      local_ranks_[j] = rel_ranks_[j] / (markdown_rank + 1)
      rel_ranks_[j] = prev + rel_ranks_[j] * ((1 / (ncode + 1)) / (markdown_rank + 1))
      j -= 1
    rel_ranks.extend(rel_ranks_)
    global_ranks.extend(global_ranks_)
    local_ranks.extend(local_ranks_)
    
  df_rank = pd.DataFrame({
    'id': ids,
    'cell_id': cell_ids,
    'n_cell': n_cells,
    'n_code_cell': n_code_cells,
    'n_markdown_cell': n_markdown_cells,
    'markdown_frac': markdown_fracs,
    'rank': ranks,
    'code_rank': code_ranks,
    'markdown_rank': markdown_ranks,
    'rel_rank': rel_ranks,
    'global_rank': global_ranks,
    'local_rank': local_ranks,
  })
  return df_rank

def set_fold_worker(df, mark='train'):
  if mark == 'train':
    gezi.set_fold_worker(df, FLAGS.folds, workers=FLAGS.workers, group_key='ancestor_id', seed=FLAGS.fold_seed)
  else:
    gezi.set_worker(df, workers=FLAGS.workers, seed=FLAGS.fold_seed)
    
def match_code_index(markdown_rank, code_ranks):
  for i in range(len(code_ranks)):
    if markdown_rank < code_ranks[i]:
      return i
  return len(code_ranks)

def add_end_sources(df):
  # slow
  df_last_codes = df.loc[df.groupby('id')['code_rank'].idxmax()]
  # df_last_codes = df[df.cell_type=='code'].drop_duplicates(subset=['id'], keep='last')
  df_last_codes['cell_id'] = 'nan'
  df_last_codes['source'] = FLAGS.end_source
  df_last_codes['rel_rank'] = 1.
  df_last_codes['code_rank'] += 1
  df_last_codes['rank'] = df_last_codes['n_cell']
  df = pd.concat([df, df_last_codes], axis=0)
  return df

dfs = {}
infos = {}
cell_infos = {}
cell_infos2 = {}
cid_groups = {}
c2c_probs = {}  # markdown to markdown sim prob
cid_infos = {}

def get_cell_info(cid):
  return cell_infos[cid]

def set_cell_info(cid, probs):
  cell_infos2[cid] = probs

def get_rerank_cids(min_thre, max_thre):
  cids = set()
  for cid, probs in cell_infos.items():
    max_prob = probs.max()
    if max_prob >= min_thre and max_prob < max_thre:
      cids.add(cid)
  return cids

def get_group_cids():
  return set(cid_groups.keys())

def get_cid_group(cid):
  return cid_groups[cid]

def get_cid_info(cid):
  return cid_infos[cid]

def get_c2c_prob(cid):
  return c2c_probs[cid]

# TODO 训练第二阶段考虑不是完全固定 而是sampling？
def gen_cid_groups(cids, x, count):
  # 按照cid对应的文本word数目从小到大排序 似乎不如按预测结果排 也就是优先头部markdown 组队的时候放在前面
  def _deal(cids, probs, preds, cid_groups):
    # ic(len(cids), len(probs))
    # ic(np.asarray([len(x) for x in probs]).min(), np.asarray([len(x) for x in probs]).max())
    ## 训练过程完全按顺序 似乎效果差不多 稍差一丢丢 但是按照preds排序的预测效果明显更好，因此采用训练预测都按preds排序
    ## extpred的反而要比oof pred好一丢丢 为什么？
    if FLAGS.work_mode != 'train' or ((not FLAGS.list_leak) and (not FLAGS.list_train_ordered)):
      ## do not need since using cls_pred not pred
      # preds = [x + (1. + np.random.random()) / 1e6 for x in preds]
      idxes = np.asarray(preds).argsort()
      # # random sort so that will the same as test if not sort over estimated as we do not konw markdown rank at test but know at train
      # # 训练的时候仍然按照有序？ 预测时按照粗排预测结果顺序
      # idxes = np.asarray(range(len(cids)))
      # np.random.shuffle(idxes)
      cids = np.asarray(cids)[idxes]
      probs = np.asarray(probs)[idxes]
      # ic(len(cids), probs.shape)
    else:
      cids, probs = np.asarray(cids), np.asarray(probs)
    
    if FLAGS.markdown_sample and (not FLAGS.list_infer):
      probs = gezi.normalize_vec(probs)
    sims = np.matmul(probs, probs.transpose(1, 0)) 

    # 只在训练过程中需要 infer过程不需要
    if FLAGS.markdown_sample and (FLAGS.work_mode == 'train'):
      temperature = 0.02
      for i, cid in enumerate(cids):
        sims[i][i] = -1e10
        if not FLAGS.list_train_ordered:
          probs_ = gezi.softmax(sims[i] / temperature)
          probs = np.zeros_like(probs_)
          for j, p in enumerate(probs_):
            probs[idxes[j]] = p
        else:
          probs = gezi.softmax(sims[i] / temperature)
        c2c_probs[cid] = probs

    # ic(sims.shape)
    used = set()
    for i in range(len(cids)):
      if FLAGS.list_infer:
        if i in used:
          continue
      cid_groups[cids[i]] = []
      used.add(i)
      idxes = (-sims[i]).argsort()
      # np.random.shuffle(idxes)
      # idxes = (sims[i]).argsort()
      bakups = []
      probs_ = probs[i]
      for idx in idxes:
        if idx != i:
          if FLAGS.list_infer:
            if idx not in used:
              cid_groups[cids[i]].append(cids[idx])
              probs_ += probs[idx]
              used.add(idx) 
            else:
              bakups.append(cids[idx])
          else:
            cid_groups[cids[i]].append(cids[idx])
            probs_ += probs[idx]
        if FLAGS.list_infer:
          if len(cid_groups[cids[i]]) > 1:
            probs_ /= len(cid_groups[cids[i]])
            set_cell_info(cids[i], probs_)
        if len(cid_groups[cids[i]]) == count:
          break
      if FLAGS.list_infer and bakups and len(cid_groups[cids[i]]) < count:
        cid_groups[cids[i]] = [*cid_groups[cids[i]], *bakups][:count]
      
  cid_groups = {}
  total = len(cids)
  ids = x['id']
  id_ = None
  cids_, probs_, preds_ = [], [], []
  for i in tqdm(range(total), desc='gen_cid_groups', leave=True):
    id = ids[i]
    if id_ and id != id_:
      _deal(cids_, probs_, preds_, cid_groups)
      cids_, probs_, preds_ = [], [], []
    id_ = id
    cids_.append(cids[i])
    probs_.append(x['probs'][i])
    preds_.append(x['cls_pred'][i])
  _deal(cids_, probs_, preds_, cid_groups)
  return cid_groups

def get_dynamic_codes(row):
  cell_id = f"{row['id']}\t{row['cell_id']}"
  probs = cell_infos[cell_id]
  idxes = (-probs).argsort()
  count = 0
  if probs[idxes[0]] > 0.9:
    count = 0
  elif len(idxes) > 1 and (probs[idxes[0]] + probs[idxes[1]] > 0.8):
    count = 2
  elif len(idxes) > 2 and (probs[idxes[0]] + probs[idxes[1]] + probs[idxes[2]]) > 0.5:
    count = 3
  elif len(idxes) > 3:
    count = 4
  return idxes[:count]


def get_df(mark='train', for_eval=False):
  global cell_infos
  global cid_groups
  if not cell_infos:
    x = gezi.get('pairwise:x', None)
    if x is None:
      if FLAGS.pairwise_dir:
        logger.info('loading pairwise infer result for eval/infer')
        xs = []
        if FLAGS.work_mode == 'train' and FLAGS.pairwise_oof:
          root = os.path.dirname(os.path.dirname(FLAGS.model_dir))
          root = root.replace('online', 'offline')
          base = FLAGS.pairwise_dir
          for i in range(FLAGS.folds):
            path = f'{root}/{i}/{base}/valid.pkl'
            xs.append(gezi.load(path))
          x = gezi.merge_array_dicts(xs)
        else:
          if os.path.isfile(FLAGS.pairwise_dir):
            path = FLAGS.pairwise_dir
          else:
            pairwise_dir = f'{os.path.dirname(FLAGS.model_dir)}/{FLAGS.pairwise_dir}'
            path = f'{pairwise_dir}/valid.pkl'
          ic(path)
          #assert os.path.exists(path)
          if os.path.exists(path):
            x = gezi.load(path)
            if 'probs' not in x:
              x = None
              logger.warning('probs not in x')
          else:
            logger.warning(f'{path} not exists')
    if x is not None:
      cell_ids = np.asarray([f'{a}\t{b}' for a, b in zip(x['id'], x['cell_id'])])
      cell_infos = dict(zip(cell_ids, x['probs']))
      ic(len(cell_infos), list(cell_infos.keys())[-1])
      if FLAGS.n_markdowns:
        cid_groups_ = gen_cid_groups(cell_ids, x, FLAGS.n_markdowns)
        cid_groups.update(cid_groups_)
        ic(len(cid_groups))
        
    if FLAGS.pairwise_dir2:
      logger.info('loading pairwise infer result for train')
      
      if os.path.isfile(FLAGS.pairwise_dir2):
        path = FLAGS.pairwise_dir2
      else:
        pairwise_dir = f'{os.path.dirname(FLAGS.model_dir)}/{FLAGS.pairwise_dir2}'
        path = f'{pairwise_dir}/test.pkl'
      ic(path)
      #assert os.path.exists(path)
      x = None
      if os.path.exists(path):
        x = gezi.load(path)
        if 'probs' not in x:
          x = None
          logger.warning('probs not in x')
      else:
        x = None
        logger.warning(f'{path} not exists')
      if x is not None:
        cell_ids = np.asarray([f'{a}\t{b}' for a, b in zip(x['id'], x['cell_id'])])
        cell_infos.update(dict(zip(cell_ids, x['probs'])))
        ic(len(cell_infos), list(cell_infos.keys())[-1])
        if FLAGS.n_markdowns:
          cid_groups_ = gen_cid_groups(cell_ids, x, FLAGS.n_markdowns)
          cid_groups.update(cid_groups_)
          ic(len(cid_groups))
      
  if mark in dfs:
    return dfs[mark]
  
  workers = FLAGS.workers 
  
  if FLAGS.hack_infer:
    mark = 'train'    
  # if os.path.exists(train_file) and (not FLAGS.from_zero):
  #   df = pd.read_feather(train_file)
  #   dfs[mark] = df 
  #   return df
  
  train_file = f'../working/{mark}.fea'
  if FLAGS.sample and mark == 'train':
    # generated in jupyter/sample.ipynb
    train_file = f'../working/train_sample.fea'
  logger.info(f'read {train_file}')
  if os.path.exists(train_file):
    df = pd.read_feather(train_file)
  else:
    if mark != FLAGS.external:
      df = create_df(f'{FLAGS.root}/{mark}', workers)
    else: 
      ext_file = f'{FLAGS.root}/{mark}.fea'
      ic(ext_file)
      df = pd.read_feather(ext_file)
      df['cell_id'] = df.cell_id.astype(str)
  
    if FLAGS.lm_train:
      df['source'] = df.source.apply(normalize)
      return df
    ## num_ids sample already done in creat_df
    # if FLAGS.num_ids > 0 and mark == 'train':
    #   ids = gezi.unique_list(df.id)
    #   ids = set(ids[:FLAGS.num_ids])
    #   df = df[df.id.isin(ids)]
    
    df['n_words'] = df.source.apply(lambda x: len(x.split()))
    if 'cid' not in df.columns:
      df['cid'] = df['id'] + '\t' + df['cell_id']
      
    # ic(df)
    ids = set(df.id)
    
    if not 'rank' in df.columns:
      logger.info('to rank')
      # https://www.kaggle.com/competitions/AI4Code/discussion/335958 cell_id not unique so need to use id + cell_id
      cell_dict = dict(zip(df.cid.values, df.cell_type.values))
      if mark == 'train':
        df_ancestors = pd.read_csv(f'{FLAGS.root}/train_ancestors.csv')
        df = df.merge(df_ancestors, on=['id'])
        df_orders = pd.read_csv(f'{FLAGS.root}/train_orders.csv')
      else:
        df_orders = df.groupby('id')['cell_id'].apply(list).reset_index(name='cell_order')
        df_orders['cell_order'] = df_orders.cell_order.apply(lambda x: ' '.join(x))
      
      if FLAGS.num_ids is not None and mark == 'train':
        df_orders = df_orders[df_orders.id.isin(ids)]
      
      df_rank = to_rank(df_orders, cell_dict)
      df_rank['pct_rank'] = (1. / (df_rank['n_cell'] - 1)) * df_rank['rank']  
      
      df = df.merge(df_rank, on=['id', 'cell_id'])
    
    logger.info('sort')
    df = df.sort_values(['id', 'rank'])
    logger.info('set fold worker')
    set_fold_worker(df, mark=mark)
    if mark == 'train':
      ic(len(df[df.fold==0]))
    ic(train_file, os.path.exists(train_file))
    if gezi.get('RANK', 0) == 0:
      if (not os.path.exists(train_file)):
        with gezi.Timer(f'{train_file} to feather'):
          df.reset_index(drop=True).to_feather(train_file) 
    if FLAGS.distributed:
      torch.distributed.barrier()
    if FLAGS.prepare:
      exit(0)

  if FLAGS.list_infer:
    if df.cid.values[0] not in cid_infos:
      for row in tqdm(df.itertuples(), total=len(df), desc='setup cid_infos'):
        row = row._asdict()
        cid = row['cid']
        if cid not in cid_infos:
          cid_infos[cid] = {}
          for key in EVAL_KEYS:
            if key != 'cid' and key in row:
              cid_infos[cid][key] = row[key]
  
  if for_eval:
    df = gezi.reduce_mem(df)
    dfs[mark] = df 
    return df
  
  df['source'] = df.source.apply(normalize)
  df['n_tokens'] = df.n_words.apply(lambda x: min(FLAGS.max_len, x))
  
  if FLAGS.pairwise and FLAGS.add_end_source:
    logger.info('add end sources')
    df = add_end_sources(df)
  
  if FLAGS.encode_code_info:
    # TODO change to lambda x: str(int(x * 1000 + 0.5)) ? or will it better int(x * 100 + 0.5)?
    if not FLAGS.encode_code_info_simple:
      df['code_info'] = df['n_code_cell'].astype(str) + '[SEP]' + df['code_rank'].astype(str) + '[SEP]' + df['rel_rank'].apply(lambda x: str(int(x * 1000)))
    else:
      df['code_info'] = df['rel_rank'].apply(lambda x: str(int(x * 1000)))
    if FLAGS.encode_all_info:
      df['code_info'] = df['n_cell'].astype(str) + '[SEP]' + df['code_info']
    # ic(df[df.cell_id=='127c3517'][['n_cell', 'n_code_cell', 'code_rank', 'rel_rank', 'code_info']])
  if FLAGS.encode_more_info:
    df['more_info'] = df['n_words'].astype(str) \
      + '[SEP]' + df['source'].apply(lambda x: str(len(x.split(BR)))) \
      + '[SEP]' + df['source'].apply(lambda x: str(len([y for y in x.split(BR) if y.lstrip().startswith('#')])))
    df['code_info'] = df['code_info'] + '[SEP]' + df['more_info']
  
  if FLAGS.use_context or FLAGS.use_context2:
    dfg_codes = None
    df_codes = df[df.cell_type=='code']
    logger.info('get_code_sources')
    logger.info('groupby code_sources')
    dfg_codes = df_codes.groupby('id')['source'].apply(list).reset_index(name='code_sources')
    if not 'code_sources' in infos:
      infos['code_sources'] = {}
    infos['code_sources'].update(dict(zip(dfg_codes.id, dfg_codes.code_sources)))

    logger.info('groupby code_ranks')
    dfg_code_ranks = df_codes.groupby('id')['rank'].apply(list).reset_index(name='code_ranks')
    if not 'code_ranks' in infos:
      infos['code_ranks'] = {}
    infos['code_ranks'].update(dict(zip(dfg_code_ranks.id, dfg_code_ranks.code_ranks)))

    if FLAGS.encode_code_info:
      logger.info('groupby code_infos')
      dfg_code_infos = df_codes.groupby('id')['code_info'].apply(list).reset_index(name='code_infos')
      if not 'code_infos' in infos:
        infos['code_infos'] = {}
      infos['code_infos'].update(dict(zip(dfg_code_infos.id, dfg_code_infos.code_infos)))
 
    if FLAGS.encode_rel_rank:
      logger.info('groupby rel_ranks')
      dfg_code_rel_ranks = df_codes.groupby('id')['rel_rank'].apply(list).reset_index(name='code_rel_ranks')
      if not 'code_rel_ranks' in infos:
        infos['code_rel_ranks'] = {}
      infos['code_rel_ranks'].update(dict(zip(dfg_code_rel_ranks.id, dfg_code_rel_ranks.code_rel_ranks)))
    df_markdowns = df[df.cell_type=='markdown']
    df_markdowns['match_code'] = [match_code_index(row.rank, infos['code_ranks'][row.id]) for row in df_markdowns.itertuples()]
    
    if FLAGS.n_markdowns:
      logger.info('groupby markdown_sources')
      dfg = df_markdowns.groupby('id')
      dfg_markdowns = dfg['source'].apply(list).reset_index(name='markdown_sources')
      if not 'markdown_sources' in infos:
        infos['markdown_sources'] = {}
      infos['markdown_sources'].update(dict(zip(dfg_markdowns.id, dfg_markdowns.markdown_sources)))
      dfg_markdown_ids = dfg['cell_id'].apply(list).reset_index(name='markdown_ids')
      if not 'markdown_ids' in infos:
        infos['markdown_ids'] = {}
      infos['markdown_ids'].update(dict(zip(dfg_markdowns.id, dfg_markdown_ids.markdown_ids)))
      dfg_markdown_rel_ranks = dfg['rel_rank'].apply(list).reset_index(name='markdown_rel_ranks')
      if not 'markdown_rel_ranks' in infos:
        infos['markdown_rel_ranks'] = {}
      infos['markdown_rel_ranks'].update(dict(zip(dfg_markdown_rel_ranks.id, dfg_markdown_rel_ranks.markdown_rel_ranks)))
      dfg_markdown_global_ranks = dfg['global_rank'].apply(list).reset_index(name='markdown_global_ranks')
      if not 'markdown_global_ranks' in infos:
        infos['markdown_global_ranks'] = {}
      infos['markdown_global_ranks'].update(dict(zip(dfg_markdown_global_ranks.id, dfg_markdown_global_ranks.markdown_global_ranks)))
      dfg_markdown_local_ranks = dfg['local_rank'].apply(list).reset_index(name='markdown_local_ranks')
      if not 'markdown_local_ranks' in infos:
        infos['markdown_local_ranks'] = {}
      infos['markdown_local_ranks'].update(dict(zip(dfg_markdown_local_ranks.id, dfg_markdown_local_ranks.markdown_local_ranks)))
      dfg_markdown_matches = dfg['match_code'].apply(list).reset_index(name='markdown_matches')
      if not 'markdown_matches' in infos:
        infos['markdown_matches'] = {}
      infos['markdown_matches'].update(dict(zip(dfg_markdown_matches.id, dfg_markdown_matches.markdown_matches)))
 
    df = pd.concat([df_codes, df_markdowns])  

  ## already sampled by FLAGS.num_ids when reading json file in create_df
  # if FLAGS.sample_frac and FLAGS.sample_frac < 1 and mark != 'test':
  #   ids = gezi.random_sample(gezi.unique_list(df.id), FLAGS.sample_frac, seed=1024)    
  #   df = df[df.id.isin(ids)]
    
  # TODO for for pairwise
  # add for each id with 1 additional end line 
  if mark == 'test' and not (FLAGS.dump_emb or FLAGS.pairwise_eval):
    # logger.info('calc n_tokens')
    df['n_tokens'] = df['n_tokens'] + df['n_code_cell'] * FLAGS.max_context_len
    # df['n_tokens'] = df['n_tokens'] + df['code_sources'].progress_apply(
    #   lambda l: int(np.asarray([min(FLAGS.max_context_len, len(x.split())) for x in l[:3]]).mean() * min(len(l), FLAGS.n_context)) if gezi.iterable(l) else 0)
    df = df.sort_values('n_tokens', ascending=False)
  else:
    logger.info('sort df values id,rank')
    df = df.sort_values(['id', 'rank'])

  # logging.info(f'dump df to {train_file}')   
  # df.reset_index(drop=True).to_feather(train_file)  
  # ic(df)
  ic(gezi.get_mem_gb())
  ## will cause 0.99885 to 1.
  # df = gezi.reduce_mem(df)
  # ic(gezi.get_mem_gb())

  dfs[mark] = df 
  return df

def clear_dfs():
  keys = list(dfs.keys())
  for key in keys:
    del dfs[key]

def to_pairwise(df):
  code_ids, markdown_ids = [], []
  codes, markdowns, matches = [], [], []
  for row in tqdm(df.itertuples(), total=len(df)):
    codes_ = row['code_sources'].copy()
    code_ranks_ = row['code_ranks'].copy()
    code_ids_ = row['code_ids'].copy()
    codes_.append('nan')
    code_ranks_.append(1e10)
    
    for i in range(len(codes_)):
      markdown_ids.append(row['cell_id'])
      code_ids.append(code_ids_[i])
      markdowns.append(row['source'])
      

def parse_example(row, subset='train', rng=None, tokenizer=None):
  if tokenizer is None:
    tokenizer = get_tokenizer(FLAGS.backbone)
  fe = row.copy()
  
  if FLAGS.lm_train:
    res = tokenizer(row['source'])
    if len(res['input_ids']) > FLAGS.lm_max_len:
      last_pos = len(res['input_ids']) - FLAGS.lm_max_len
      idx = rng.integers(last_pos)
      res['input_ids'] = res['input_ids'][idx:idx + FLAGS.lm_max_len]
      res['attention_mask'] = res['attention_mask'][idx:idx + FLAGS.lm_max_len]
      if 'token_type_ids' in res:
        res['token_type_ids'] = res['token_type_ids'][idx:idx + FLAGS.lm_max_len]
    fe.update(res)
    fe['label'] = 0
    return fe
    
  fe['label'] = row[FLAGS.label_name] 
  mark = 'train' if subset != 'test' else 'test'
  if FLAGS.hack_infer:
    mark = 'train'
  if mark == 'train' and FLAGS.external:
    mark = FLAGS.external
  padding = 'max_length' if FLAGS.static_inputs_len else 'do_not_pad'
  text = row['source']
  markdown_frac = int(row['markdown_frac'] * 100)
  
  if FLAGS.dump_emb or FLAGS.pairwise_eval or (FLAGS.pairwise and FLAGS.two_tower and subset in ['eval', 'test']):
    encode_fn = encode_markdown if row['cell_type'] == 'markdown' else encode_code
    if row['cell_type'] == 'code':
      if FLAGS.encode_code_info:
        code_info = row['code_info']
        text = f'{code_info}[SEP]{text}'
        if FLAGS.n_pairwise_context > 0:
          code_idx = row['code_rank']
          try:
            code_sources = infos['code_sources'][row['id']]
          except Exception as e:
            ic(mark, e)
            exit(-1)
          # res = encode_code(text, tokenizer, FLAGS.max_len)
          res = encode_code(text, tokenizer, FLAGS.max_len - (FLAGS.max_context_len - 1) * 2 * FLAGS.n_pairwise_context)
          if FLAGS.n_pairwise_context == 1:
          # if FLAGS.n_pairwise_context == None:
            left_text = code_sources[code_idx - 1] if code_idx > 1 else 'before'
            right_text = code_sources[code_idx + 1] if code_idx + 1 < len(code_sources) else 'after'
            left_res = encode_code(left_text, tokenizer, FLAGS.max_context_len, token_type_id=0)
            right_res = encode_code(right_text, tokenizer, FLAGS.max_context_len, token_type_id=0)
            fe['input_ids'] = left_res['input_ids'] + res['input_ids'][1:] + right_res['input_ids'][1:]
            fe['attention_mask'] = left_res['attention_mask'] + res['attention_mask'][1:] + right_res['attention_mask'][1:]
            fe['token_type_ids'] = left_res['token_type_ids'][:-1] + res['token_type_ids'] + right_res['token_type_ids'][1:]
          else:
            left_texts = get_left_texts(code_sources, code_idx, FLAGS.n_pairwise_context)
            left_res = encodes(left_texts, tokenizer, FLAGS.max_context_len, padding='do_not_pad')
            right_texts = get_right_texts(code_sources, code_idx, FLAGS.n_pairwise_context)
            right_res = encodes(right_texts, tokenizer, FLAGS.max_context_len, padding='do_not_pad')
            fe['input_ids'] = [left_res['input_ids'][0][0]]
            fe['token_type_ids'] = []
            for input_ids in left_res['input_ids']:
              fe['input_ids'].extend(input_ids[1:])
            fe['token_type_ids'] = [0] * (len(fe['input_ids']) - 1)
            fe['input_ids'].extend(res['input_ids'][1:])
            fe['token_type_ids'].extend([1] * len(res['input_ids']))
            for input_ids in right_res['input_ids']:
              fe['input_ids'].extend(input_ids[1:])
            fe['attention_mask'] = [1] * len(fe['input_ids'])
            fe['token_type_ids'].extend([0] * (len(fe['input_ids']) - len(fe['token_type_ids'])))    
      else:
        fe.update(encode_code(text, tokenizer, FLAGS.max_len))
    else:
      if FLAGS.encode_markdown_info:
        n_markdown = row['n_markdown_cell']
        text = f'[SEP]{n_markdown}[SEP]{markdown_frac}[SEP]{text}'
      if FLAGS.encode_all_info:
        n_cell = row['n_cell']
        text = f'{n_cell}[SEP]{text}'
      if FLAGS.encode_more_info: # not used
        more_info = row['more_info']
        text = f'{text}[SEP]{more_info}'
      fe.update(encode_markdown(text, tokenizer, token_type_id=FLAGS.markdown_token_type_id))
  else:
    # here is markdown
    if FLAGS.dynamic_token_types:
      text = ''
    if FLAGS.encode_markdown_frac:
      text = f'{markdown_frac}[SEP]{text}'
    if FLAGS.encode_info:
      n_cell, n_code, n_markdown = row['n_cell'], row['n_code_cell'], row['n_markdown_cell']
      text = f'{n_cell}[SEP]{n_code}[SEP]{n_markdown}[SEP]{text}'
    if FLAGS.encode_markdown_info:
      n_markdown = row['n_markdown_cell']
      text = f'{n_markdown}[SEP]{markdown_frac}[SEP]{text}'
    if FLAGS.encode_all_info:
      n_cell = row['n_cell']
      text = f'{n_cell}[SEP]{text}'
    if FLAGS.encode_more_info: # not used
      more_info = row['more_info']
      text = f'{text}[SEP]{more_info}'
    max_len = FLAGS.max_len 
    if FLAGS.pairwise and (not FLAGS.two_tower):
      max_len = FLAGS.max_markdown_len
    res = encode(text, tokenizer, max_len, FLAGS.last_tokens, padding, token_type_id=FLAGS.markdown_token_type_id)
    if FLAGS.dynamic_token_types:
      # 0 global info, 1 code, 2 3 ...
      res_ = encode_markdown(row['source'], tokenizer, token_type_id=2)
      res['input_ids'].extend(res_['input_ids'][1:])
      res['attention_mask'].extend(res_['attention_mask'][1:])
      res['token_type_ids'].extend(res_['token_type_ids'][1:])
    if FLAGS.n_markdowns:
      markdown_sources_ = infos['markdown_sources'][row['id']]
      markdown_ids_ = infos['markdown_ids'][row['id']]
      markdown_labels_ = infos['markdown_rel_ranks'][row['id']]
      markdown_global_labels_ = infos['markdown_global_ranks'][row['id']]
      markdown_local_labels_ = infos['markdown_local_ranks'][row['id']]
      markdown_sources, markdown_ids, markdown_labels, markdown_cls_labels, markdown_global_labels, markdown_local_labels = [], [], [], [], [], []
      if (subset not in ['eval', 'test']) or (row['cid'] not in cid_groups):
        for id, source, label, global_label, local_label in zip(markdown_ids_, markdown_sources_, markdown_labels_, markdown_global_labels_, markdown_local_labels_):
          if id != row['cell_id']:
            markdown_sources.append(source)
            markdown_ids.append(id)
            markdown_labels.append(label)
            markdown_global_labels.append(global_label)
            markdown_local_labels.append(local_label)
            markdown_cls_labels.append(int(label * FLAGS.num_classes))
        
        if len(markdown_sources):
          if FLAGS.mrand_prob is None or row['cid'] not in cid_groups or rng.random() < FLAGS.mrand_prob:
            idxes = np.asarray(range(len(markdown_sources)))
            np.random.shuffle(idxes)
            # ic(idxes)
            # ic(subset, 'random markdown idxes')
            markdown_sources, markdown_ids, markdown_labels, markdown_cls_labels = np.asarray(markdown_sources), np.asarray(markdown_ids), np.asarray(markdown_labels), np.asarray(markdown_cls_labels)
            markdown_sources, markdown_ids, markdown_labels, markdown_cls_labels = markdown_sources[idxes], markdown_ids[idxes], markdown_labels[idxes], markdown_cls_labels[idxes]
            # ic(markdown_labels, markdown_cls_labels)
            markdown_global_labels, markdown_local_labels = np.asarray(markdown_global_labels), np.asarray(markdown_local_labels)
            markdown_global_labels, markdown_local_labels = markdown_global_labels[idxes], markdown_local_labels[idxes]
          else:
            # ic(subset, 'fixed markdown idxes')
            if not FLAGS.markdown_sample:
              markdown_ids = cid_groups[row['cid']]
              markdown_ids = [x.split('\t')[1] for x in markdown_ids]
              idxes = [markdown_ids_.index(markdown_id) for markdown_id in markdown_ids]
            else:
              pos_idx = markdown_ids_.index(row['cid'].split('\t')[1])
              idxes = select_negs_sample(FLAGS.n_markdowns, pos_idx, c2c_probs[row['cid']])
              probs = c2c_probs[row['cid']][idxes]
              idxes_ = (-probs).argsort()
              idxes = idxes[idxes_]
              markdown_ids = np.asarray(markdown_ids_)[idxes]
              # ic(cid_groups[row['cid']], markdown_ids, c2c_probs[row['cid']][idxes], 
              #    len(c2c_probs[row['cid']]), c2c_probs[row['cid']].max(), c2c_probs[row['cid']].mean())

            markdown_sources, markdown_labels, markdown_cls_labels = [], [], []
            markdown_global_labels, markdown_local_labels = [], []
            for i in idxes:
              markdown_sources.append(markdown_sources_[i])
              markdown_labels.append(markdown_labels_[i])
              markdown_global_labels.append(markdown_global_labels_[i])
              markdown_local_labels.append(markdown_local_labels_[i])
              markdown_cls_labels.append(int(markdown_labels_[i] * FLAGS.num_classes))
            markdown_sources, markdown_ids, markdown_labels, markdown_cls_labels = np.asarray(markdown_sources), np.asarray(markdown_ids), np.asarray(markdown_labels), np.asarray(markdown_cls_labels)
            # ic(markdown_labels, markdown_cls_labels)
            markdown_global_labels, markdown_local_labels = np.asarray(markdown_global_labels), np.asarray(markdown_local_labels)
      else:
        # ic(subset, 'fixed markdown idxes')
        markdown_ids = cid_groups[row['cid']]
        markdown_ids = [x.split('\t')[1] for x in markdown_ids]
        markdown_labels = [-100] * len(markdown_ids)
        markdown_cls_labels = [-100] * len(markdown_labels)
        markdown_global_labels = [-100] * len(markdown_ids)
        markdown_local_labels = [-100] * len(markdown_ids)
        markdown_sources = []
        for markdown_id in markdown_ids:
          i = markdown_ids_.index(markdown_id)
          markdown_sources.append(markdown_sources_[i])
        # ic(markdown_labels, markdown_cls_labels)
        
      if len(markdown_sources) < FLAGS.n_markdowns:
        for i in range(FLAGS.n_markdowns - len(markdown_sources)):
          markdown_sources = np.append(markdown_sources, 'Nothing')
          markdown_ids = np.append(markdown_ids, 'nan')
          markdown_labels = np.append(markdown_labels, -100)
          markdown_global_labels = np.append(markdown_global_labels, -100)
          markdown_local_labels = np.append(markdown_local_labels, -100)
          markdown_cls_labels = np.append(markdown_cls_labels, -100)
      #   ic(markdown_labels, markdown_cls_labels)
      # ic(markdown_labels, markdown_cls_labels)
      fe['markdown_labels'] = markdown_labels[:FLAGS.n_markdowns]
      fe['markdown_global_labels'] = markdown_global_labels[:FLAGS.n_markdowns]
      fe['markdown_local_labels'] = markdown_local_labels[:FLAGS.n_markdowns]
      fe['markdown_cls_labels'] = markdown_cls_labels[:FLAGS.n_markdowns]
      # ic(fe['markdown_labels'], fe['markdown_cls_labels'])

      # fe['markdown_ids'] = markdown_ids[:FLAGS.n_markdowns]
      # markdown_sources = markdown_sources[:FLAGS.n_markdowns]
      # markdown_ids = markdown_ids[:FLAGS.n_markdowns]
      # TODO 注意特别infer的时候可以排序让短文本放在前面
      # ic(len(markdown_sources), FLAGS.n_markdowns)
      assert len(markdown_sources) 
      for i in range(FLAGS.n_markdowns):
        source = markdown_sources[i]
        max_len_ = max_len * (i + 2) - len(res['input_ids'])
        markdown_token_type_id = FLAGS.markdown_token_type_id if not FLAGS.dynamic_token_types else 3 + i
        res_ = encode(source, tokenizer, max_len_, FLAGS.last_tokens, padding, token_type_id=markdown_token_type_id)
        # res_ = {}
        if res_:
          res['input_ids'].extend(res_['input_ids'][1:])
          res['attention_mask'].extend(res_['attention_mask'][1:])
          if 'token_type_ids' in res:
            res['token_type_ids'].extend(res_['token_type_ids'][1:])
    fe.update(res)
    # ic(''.join(tokenizer.convert_ids_to_tokens(fe['input_ids'])))
    if not FLAGS.pairwise:
      cls_label = int(row[FLAGS.label_name] * FLAGS.num_classes)
      if cls_label < 0 or cls_label > FLAGS.num_classes - 1:
        logging.error(f'wrong label: {cls_label}', row[FLAGS.label_name], row['id'], row['cell_id'])
        cls_label = max(cls_label, 0)
        cls_label = min(cls_label, FLAGS.num_classes - 1)
      if FLAGS.loss_method == 'softmax':
        fe['label'] = cls_label
      else:
        fe['cls_label'] = cls_label
      fe['global_label'] = row['global_rank']
      fe['local_label'] = row['local_rank']
      if FLAGS.use_context:
        code_sources = np.array(infos['code_sources'][row['id']])
        # if FLAGS.context_method == 'input':
        prob_idxes = None
        probs = None
        if subset not in ['eval', 'test'] and (FLAGS.crand_prob is None or rng.random() < FLAGS.crand_prob):
          # ic(subset, 'random codes')
          idxes = None
        else:
          if row['n_code_cell'] > FLAGS.n_context or FLAGS.add_probs2 or FLAGS.dynamic_context_weight:
            cell_id = f"{row['id']}\t{row['cell_id']}"
            if cell_id in cell_infos:
              if cell_id in cell_infos2:
                probs = cell_infos2[cell_id][:-1]
              else:
                probs = cell_infos[cell_id][:-1]
              prob_idxes = (-probs).argsort(-1)

        probs_ = probs if FLAGS.neg_sample else None
        idxes = sample_codes_idxes(len(code_sources), FLAGS.n_context, 
                                   method=FLAGS.context_method, candidates=prob_idxes,
                                   rng=rng, probs=probs_, training=(subset=='train'))
        
        fe['context_match'] = int((row['match_code'] in idxes) or row['match_code'] == row['n_code_cell'])
        code_sources = list(code_sources[idxes])
        if FLAGS.encode_rel_rank:
          code_rel_ranks = np.array(infos['code_rel_ranks'][row['id']])
          code_rel_ranks = code_rel_ranks[idxes]
          if not FLAGS.add_probs2:
            code_sources = [f'{int(code_rel_rank * 1000)}[SEP]{code_source}' for code_rel_rank, code_source in zip(code_rel_ranks, code_sources)]
          else:
            assert probs is not None
            probs = probs[idxes]
            code_sources = [f'{int(code_rel_rank * 1000)}[SEP]{int(prob * 1000)}[SEP]{code_source}' for code_rel_rank, prob, code_source in zip(code_rel_ranks, probs, code_sources)]
        if not FLAGS.dynamic_context_len:
          max_context_len = FLAGS.max_context_len
        else:
          total_len = (FLAGS.max_context_len - 1) * FLAGS.n_context + FLAGS.max_len * (1 + FLAGS.n_markdowns)
          max_context_len = int((total_len - len(fe['input_ids'])) / min(FLAGS.n_context, row['n_code_cell'])) + 1
        # res = encodes(code_sources, tokenizer, max_context_len, padding=FLAGS.context_padding)
        # for input_ids in res['input_ids']:
        #   fe['input_ids'].extend(input_ids[1:])
        # fe['attention_mask'] = [1] * len(fe['input_ids'])
        # fe['token_type_ids'] = gezi.pad(fe['token_type_ids'], len(fe['input_ids']), 1)
        if FLAGS.dynamic_context_weight is None:
          res = encodes(code_sources, tokenizer, max_context_len, padding=FLAGS.context_padding)
          for input_ids in res['input_ids']:
            fe['input_ids'].extend(input_ids[1:])
        else:
          prob_idxes = list(prob_idxes)
          res = encodes(code_sources, tokenizer, max_context_len + int(FLAGS.max_context_len * FLAGS.dynamic_context_weight), padding=FLAGS.context_padding)
          for i, input_ids in enumerate(res['input_ids']):
            if prob_idxes is not None and prob_idxes.index(idxes[i]) >= FLAGS.dynamic_context_topn:
              input_ids = gezi.trunct(input_ids, max_context_len, last_tokens=1)
            fe['input_ids'].extend(input_ids[1:])    
        fe['attention_mask'] = [1] * len(fe['input_ids'])
        fe['token_type_ids'] = gezi.pad(fe['token_type_ids'], len(fe['input_ids']), 1)
            
      if FLAGS.use_markdowns:
        markdown_sources = infos['markdown_sources'][row['id']].copy()
        markdown_sources = [x for x in markdowns if x != row['source']]
        if markdown_sources:
          rng.shuffle(markdown_sources)
          res = encodes(markdown_sources[:FLAGS.n_markdowns], tokenizer, FLAGS.max_markdown_len, padding='longest')
          for input_ids in res['input_ids']:
            fe['input_ids'].extend(input_ids[1:])
          fe['attention_mask'] = [1] * len(fe['input_ids'])
          fe['token_type_ids'] = gezi.pad(fe['token_type_ids'], len(fe['input_ids']), 2)
      
      # TODO not work now
      if FLAGS.use_context2:
        fe['markdown_input_ids'] = gezi.trunct(fe['ori_input_ids'], FLAGS.max_len * 2, last_tokens=FLAGS.last_tokens * 2)
        fe['markdown_attention_mask'] = [1] * len(fe['markdown_input_ids'])
        fe['input_ids2'] = []
        fe['attention_mask2'] = []
        res = encodes(row['code_sources2'], tokenizer, FLAGS.max_context2_len, padding=FLAGS.context_padding)
        for input_ids in res['input_ids']:
          fe['input_ids2'].extend(input_ids)
        for attention_mask in res['attention_mask']:
          fe['attention_mask2'].extend(attention_mask)
        if FLAGS.use_dot:
          fe['input_ids2'] = gezi.pad(fe['input_ids2'], FLAGS.max_context2_len * FLAGS.n_context2)
          fe['attention_mask2'] = gezi.pad(fe['attention_mask2'], FLAGS.max_context2_len * FLAGS.n_context2)
    else:
      if subset in ['train', 'valid'] or (not FLAGS.two_tower):
        ori_code_sources = infos['code_sources'][row['id']]
        code_sources = ori_code_sources.copy()
        # now use add_end_source
        if not FLAGS.add_end_source:
          code_sources.append(FLAGS.end_source)
  
        if FLAGS.encode_code_info:
          code_infos = infos['code_infos'][row['id']].copy()
          if not FLAGS.add_end_source:
            assert not FLAGS.encode_code_info_simple
            assert FLAGS.encode_all_info
            end_info = str(row['n_cell']) + '[SEP]' + str(row['n_code_cell']) + '[SEP]' + str(len(code_sources)) + '[SEP]' + '1000'
            code_infos.append(end_info)
          code_sources = [f'{code_info}[SEP]{code_source}' for code_info, code_source in zip(code_infos, code_sources)]

        code_sources = np.asarray(code_sources)
        
        n_neg_codes = len(code_sources) - 1
        n_neg_codes = min(n_neg_codes, FLAGS.num_negs)
        label_mask = [1] * (1 + n_neg_codes)
        label_mask = gezi.pad(label_mask, 1 + FLAGS.num_negs, 0)
        fe['label_mask'] = label_mask
        fe['label'] = 0
        
        # ic(subset, 'select codes')
        if (not FLAGS.two_tower) and FLAGS.pairwise_dir and subset in ['eval', 'test']:
          cell_id = f"{row['id']}\t{row['cell_id']}"
          # ic('greedy select')
          if cell_id in cell_infos:
            if not FLAGS.dynamic_codes:
              code_idxes = (-cell_infos[cell_id]).argsort(-1)[:(1 + FLAGS.num_negs)]
            else:
              code_idxes = get_dynamic_codes(row)
              if not len(code_idxes):
                return []
          else:
            if not FLAGS.dynamic_codes:
              # logger.warning('not find cell_infos, normal if running locally')
              code_idxes = np.asarray(range(min(len(code_sources), 1 + FLAGS.num_negs)))
            else:
              return []
        else:
          match_code = int(row['match_code'])
          probs = None
          if row['cid'] in cell_infos:
            probs = cell_infos[row['cid']]
          neg_code_idxes = select_negs(n_neg_codes, match_code, len(code_sources), 
                                       probs=probs, rng=rng, method=FLAGS.neg_method, 
                                       rand_prob=FLAGS.neg_rand_prob)
          # ic((-probs).argsort()[:10], match_code, neg_code_idxes)
          code_idxes = np.asarray([match_code, *neg_code_idxes])

        
        fe['code_idxes'] = code_idxes
        codes = code_sources[code_idxes]
        
        if FLAGS.n_pairwise_context > 0:
          assert FLAGS.add_end_source
          if FLAGS.n_pairwise_context == 1:
            left_codes = [ori_code_sources[i - 1] if i > 1 else 'before' for i in code_idxes]
            right_codes = [ori_code_sources[i + 1] if i < len(ori_code_sources) - 1 else 'after' for i in code_idxes]       
          else:
            left_codes = [get_left_texts(ori_code_sources, i, FLAGS.n_pairwise_context) for i in code_idxes]
            right_codes = [get_right_texts(ori_code_sources, i, FLAGS.n_pairwise_context) for i in code_idxes]
            left_codes = list(itertools.chain(*left_codes))
            right_codes = list(itertools.chain(*right_codes))
          
        if FLAGS.two_tower:
          fe['codes_input_ids'] = []
          fe['codes_attention_mask'] = []
          fe['codes_token_type_ids'] = []
        if FLAGS.n_pairwise_context == 0:
          res = encodes(codes, tokenizer, FLAGS.max_len, padding='max_length')
          for input_ids in res['input_ids']:
            fe['codes_input_ids'].extend(input_ids)
          for attention_mask in res['attention_mask']:
            fe['codes_attention_mask'].extend(attention_mask)
        else:
          max_len = FLAGS.max_len - (FLAGS.max_context_len - 1) * 2 * FLAGS.n_pairwise_context
          res = encodes(codes, tokenizer, max_len, padding='do_not_pad')
          if not FLAGS.dynamic_context_len:
            max_context_len = FLAGS.max_context_len
          else:
            used_len = len(res['input_ids'])
            max_context_len = int((FLAGS.max_len - used_len) / (2 * FLAGS.n_pairwise_context)) + 1
          left_res = encodes(left_codes, tokenizer, max_context_len, padding='do_not_pad')
          # so for 196 23 now need 196 + 22 * 2 = 240
          right_res = encodes(right_codes, tokenizer, max_context_len, padding='do_not_pad')
          if FLAGS.n_pairwise_context > 1:  
            for key in left_res:
              left_res[key] = merge_input_ids(left_res[key], len(code_idxes))
            for key in right_res:
              right_res[key] = merge_input_ids(right_res[key], len(code_idxes))
          for i, (l, m, r) in enumerate(zip(left_res['input_ids'], res['input_ids'], right_res['input_ids'])):
            input_ids = l + m[1:] + r[1:] 
            if FLAGS.two_tower:
              input_ids = gezi.pad(input_ids, FLAGS.max_len)
              fe['codes_input_ids'].extend(input_ids)
            else:
              if not FLAGS.insert_mode:
                fe[f'input_ids{i}'] = fe['input_ids'] + input_ids[1:]
              else:
                fe[f'input_ids{i}'] = l + fe['input_ids'][1:] + m[1:] + r[1:]
          for i, (l, m, r) in enumerate(zip(left_res['attention_mask'], res['attention_mask'], right_res['attention_mask'])):
            attention_mask = l + m[1:] + r[1:]
            if FLAGS.two_tower:
              attention_mask = gezi.pad(attention_mask, FLAGS.max_len)
              fe['codes_attention_mask'].extend(attention_mask)
            else:
              if not FLAGS.insert_mode:
                fe[f'attention_mask{i}'] = fe['attention_mask'] + attention_mask[1:]
              else:
                fe[f'attention_mask{i}'] = l + fe['attention_mask'][1:] + m[1:] + r[1:]
          for i, (l, m, r) in enumerate(zip(left_res['token_type_ids'], res['token_type_ids'], right_res['token_type_ids'])):
            token_type_ids =  [0] * (len(l) - 1) + [1] * len(m) + [0] * (len(r) - 1)
            if FLAGS.two_tower:
              token_type_ids = gezi.pad(token_type_ids, FLAGS.max_len)
              fe['codes_token_type_ids'].extend(token_type_ids)
            else:
              if not FLAGS.insert_mode:
                fe[f'token_type_ids{i}'] = [0] * len(fe['input_ids']) + [1] * (len(token_type_ids) - 1)
              else:
                fe[f'token_type_ids{i}'] = [0] * (len(l) - 1) + [1] * (len(fe['input_ids']) - 1) + [1] * len(m) + [0] * (len(r) - 1)
        
        if FLAGS.two_tower:
          # all padding to fixed length
          fe['codes_input_ids'] = gezi.pad(fe['codes_input_ids'], FLAGS.max_len * (1 + FLAGS.num_negs))
          fe['codes_attention_mask'] = gezi.pad(fe['codes_attention_mask'], len(fe['codes_input_ids']))
          # code token_type_ids set as 1 while markdown set as 0
          if FLAGS.n_pairwise_context == 0:
            fe['codes_token_type_ids'] = [1] * len(fe['codes_input_ids'])
          else:
            fe['codes_token_type_ids'] = gezi.pad(fe['codes_token_type_ids'], len(fe['codes_input_ids']))
        else:
          if not FLAGS.dynamic_codes:
            for i in range(1 + FLAGS.num_negs):
              if not f'input_ids{i}' in fe:
                fe[f'input_ids{i}'] = fe[f'input_ids{i - 1}']
                fe[f'attention_mask{i}'] = fe[f'attention_mask{i - 1}']
                fe[f'token_type_ids{i}'] = fe[f'token_type_ids{i - 1}']
          else:
            res = []
            for i in range(len(code_idxes)):
              fe_ = {
                'id': row['id'], 
                'cell_id': row['cell_id'], 
                'cid': row['cid'],
                'input_ids': fe[f'input_ids{i}'],
                'attention_mask': fe[f'attention_mask{i}'],
                'token_type_ids': fe[f'token_type_ids{i}'],
                'code_id': code_idxes[i],
                'code_idxes': code_idxes.copy(),
                'label': 0,
                }
              res.append(fe_)
            return res
              
        if FLAGS.pairwise_markdowns:
          markdowns = infos['markdown_sources'][row['id']].copy()
          code_matches = infos['markdown_matches'][row['id']].copy()
          rel_ranks = infos['markdown_rel_ranks'][row['id']].copy()
          idxes, labels = sample_markdowns(row['source'], FLAGS.n_markdowns, 
                                           markdowns, code_matches, rel_ranks, rng=rng)
          # TODO encodes?
          fe['markdowns_input_ids'] = []
          fe['markdowns_attention_mask'] = []
          for idx in idxes:
            text = markdowns[idx]
            res = encode(text, tokenizer, FLAGS.max_len, FLAGS.last_tokens, padding='max_length')
            fe['markdowns_input_ids'].extend(res['input_ids'])
            fe['markdowns_attention_mask'].extend(res['attention_mask'])
          fe['markdowns_input_ids'] = gezi.pad(fe['markdowns_input_ids'], FLAGS.max_len * FLAGS.n_markdowns)
          fe['markdowns_attention_mask'] = gezi.pad(fe['markdowns_attention_mask'], len(fe['markdowns_input_ids']))
          fe['markdowns_label_mask'] = gezi.pad([1] * len(labels), FLAGS.n_markdowns, 0)
          fe['markdowns_label'] = gezi.pad(list(labels), FLAGS.n_markdowns, 0)
          fe['markdowns_token_type_ids'] = [0] * len(fe['markdowns_input_ids'])
    
  if FLAGS.static_inputs_len:
    max_len = FLAGS.max_len
    if FLAGS.use_context and (not FLAGS.pairwise):
      max_len = FLAGS.max_len + (FLAGS.max_context_len - 1) * FLAGS.n_context
    fe['input_ids'] = gezi.pad(fe['input_ids'], max_len, FLAGS.padding_val)
    fe['attention_mask'] = gezi.pad(fe['attention_mask'], max_len, FLAGS.padding_val)
  
  if FLAGS.tf_dataset:
    fe['source'] = 'nan'
  
  # only tf dataset need this [,] -> [,1]
  # unsqueeze_keys = ['markdown_frac', 'n_cell', 'n_code_cell', 'n_markdown_cell']
  # gezi.unsqueeze(fe, unsqueeze_keys)
  
  if FLAGS.use_embs:
    id = row['id']
    fe['code_embs'] = np.load(f'{FLAGS.embs_dir}/code/{id}.npy')
    # TODO remove current markdown
    fe['markdown_embs'] = np.load(f'{FLAGS.embs_dir}/markdown/{id}.npy')
    if row['cell_type'] == 'markdown':
      markdown_ids = infos['markdown_ids'][id]
      idx = markdown_ids.index(row['cell_id'])
      try:
        markdown_emb = fe['markdown_embs'][idx]
      except Exception as e:
        ic(e, row['id'], row['cell_id'])
        exit(0)
      markdown_embs = np.concatenate([fe['markdown_embs'][:idx], fe['markdown_embs'][idx+1:]], axis=0)
      markdown_embs = list(markdown_embs)
      rng.shuffle(markdown_embs)
      fe['markdown_embs'] = np.stack([markdown_emb, *markdown_embs], axis=0)
    
    # ic(fe['code_embs'].shape, fe['markdown_embs'].shape, row['n_code_cell'], row['n_markdown_cell'])
    # TODO for code might still use sample strategy
    fe['codes_mask'] = [1] * len(fe['code_embs'])
    fe['markdowns_mask'] = [1] * len(fe['markdown_embs'])
    
    # 128 32 on a100 * 4 OOM
    fe['code_embs'] = fe['code_embs'][:128].reshape(-1)
    fe['markdown_embs'] = fe['markdown_embs'][:32].reshape(-1)
    
    fe['codes_mask'] = fe['codes_mask'][:128]
    fe['markdowns_mask'] = fe['markdowns_mask'][:32]
    
  del_keys = [
              'parent_id', 'source', 
              'code_sources', 'code_sources2',
              'markdown_sources', 'code_ranks', 
              'code_ids', 'code_infos',
             ]
  for key in del_keys:
    if key in fe:
      del fe[key]
  return fe

def get_datasets(valid=True, infer=False):
  return
  from datasets import Dataset
  mark = 'train'
  if infer:
    mark = 'test'
  df = get_df(mark)

  ic(FLAGS.backbone)
  tokenizer = get_tokenizer(FLAGS.backbone)
  ic(tokenizer)

# num_proc = cpu_count() if FLAGS.pymp else 1
  num_proc = 20 if FLAGS.pymp else 1
    
  inputs = [row._asdict() for row in df.itertuples()]
  res = gezi.prun(lambda x: parse_example(x, tokenizer), inputs, num_proc)
  df = pd.DataFrame(res)
  
  ds = Dataset.from_pandas(df)

  # gezi.try_mkdir(f'{FLAGS.root}/cache')
  records_name = get_records_name()
  
  # ds = ds.map(
  #     lambda example: parse_example(example, tokenizer=tokenizer),
  #     remove_columns=ds.column_names,
  #     batched=False,
  #     num_proc=num_proc,
  #     # cache_file_name=f'{FLAGS.root}/cache/{records_name}.infer{int(infer)}.arrow' if not infer else None
  # )
  
  ic(ds)
  
  ignore_feats = [key for key in ds.features if ds.features[key].dtype == 'string' or (ds.features[key].dtype == 'list' and ds.features[key].feature.dtype == 'string')]
  ic(ignore_feats)
  
  if infer:
    m = {}
    for key in ignore_feats:
      m[key] = ds[key]
    gezi.set('infer_dict', m)
    ds = ds.remove_columns(ignore_feats)
    return ds

  if not FLAGS.online:
    train_ds = ds.filter(lambda x: x['fold'] != FLAGS.fold, num_proc=num_proc)
  else:
    train_ds = ds
  eval_ds = ds.filter(lambda x: x['fold'] == FLAGS.fold, num_proc=num_proc)

  m = {}
  for key in ignore_feats:
    m[key] = eval_ds[key]
  gezi.set('eval_dict', m)

  # also ok if not remove here
  train_ds = train_ds.remove_columns(ignore_feats)
  eval_ds = eval_ds.remove_columns(ignore_feats)
  ic(train_ds, eval_ds)
  if valid:
    valid_ds = ds.filter(lambda x: x['fold'] == FLAGS.fold, num_proc=num_proc)
    valid_ds = valid_ds.remove_columns(ignore_feats)
    return train_ds, eval_ds, valid_ds
  else:
    return train_ds, eval_ds
  
def get_dataloader(epoch):
  from src.torch.dataset import Dataset
  collate_fn = gezi.NpDictPadCollate()
  kwargs = {
      'num_workers': FLAGS.num_workers,
      'pin_memory': FLAGS.pin_memory,
      'persistent_workers': FLAGS.persistent_workers,
      'collate_fn': collate_fn,
  }   
  train_ds = Dataset('train')
  train_ds.set_epoch(epoch)
  ic(len(train_ds))
  sampler = lele.get_sampler(train_ds, shuffle=True)
  # melt.batch_size 全局总batch大小，FLAGS.batch_size 单个gpu的batch大小，gezi.batch_size做batch的时候考虑兼容distributed情况下的batch_size
  train_dl = torch.utils.data.DataLoader(train_ds,
                                        batch_size=gezi.batch_size(),
                                        sampler=sampler,
                                        drop_last=FLAGS.drop_last,
                                        **kwargs)
  return train_dl

def get_dataloaders(valid=True, test_only=False):
  from src.torch.dataset import Dataset 

  collate_fn = gezi.NpDictPadCollate()
  kwargs = {
      'num_workers': FLAGS.num_workers,
      'pin_memory': FLAGS.pin_memory,
      'persistent_workers': FLAGS.persistent_workers,
      'collate_fn': collate_fn,
  }  
  
  if test_only or FLAGS.mode == 'test':
    test_ds = Dataset('test')
    ic(len(test_ds))

    sampler_test = lele.get_sampler(test_ds, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=gezi.eval_batch_size(),
                                          sampler=sampler_test,
                                          **kwargs)
  else:
    test_dl = None
    
  if test_only:
    return test_dl

  if FLAGS.mode == 'test':
    return None, None, None, test_dl
  
  if FLAGS.mode != 'valid':
    train_ds = Dataset('train')
    ic(len(train_ds))
    
  eval_ds = Dataset(FLAGS.eval_name)
  ic(len(eval_ds))
  if not FLAGS.online:
    if FLAGS.mode != 'valid':
      assert len(set(train_ds.df.id) & set(eval_ds.df.id)) == 0
      if not (FLAGS.lm_train or FLAGS.external):
        assert len(set(train_ds.df.ancestor_id) & set(eval_ds.df.ancestor_id)) == 0
  if FLAGS.prepare:
    exit(0)
    
  if FLAGS.mode != 'valid':
    if valid:
      valid_ds = Dataset('valid')
  
  # if valid:
  #   train_ds, eval_ds, valid_ds = get_datasets(valid=True)
  # else:
  #   train_ds, eval_ds = get_datasets(valid=False)
  
  if FLAGS.mode != 'valid':
    sampler = lele.get_sampler(train_ds, shuffle=True)
    # melt.batch_size 全局总batch大小，FLAGS.batch_size 单个gpu的batch大小，gezi.batch_size做batch的时候考虑兼容distributed情况下的batch_size
    train_dl = torch.utils.data.DataLoader(train_ds,
                                          batch_size=gezi.batch_size(),
                                          sampler=sampler,
                                          drop_last=FLAGS.drop_last,
                                          **kwargs)
  else:
    train_dl = None
  
  sampler_eval = lele.get_sampler(eval_ds, shuffle=False)
  eval_dl = torch.utils.data.DataLoader(eval_ds,
                                        batch_size=gezi.eval_batch_size(),
                                        sampler=sampler_eval,
                                        **kwargs)
  
  if valid:
    if FLAGS.mode != 'valid':
      sampler_valid = lele.get_sampler(valid_ds, shuffle=False)
      valid_dl = torch.utils.data.DataLoader(valid_ds,
                                            batch_size=gezi.eval_batch_size(),
                                            sampler=sampler_valid,
                                            **kwargs)
    else:
      valid_dl = None
    return train_dl, eval_dl, valid_dl, test_dl
  else:
    return train_dl, eval_dl, test_dl

def get_tf_datasets(valid=True):
  if valid:
    train_ds, eval_ds, valid_ds = get_datasets(valid=True)
  else:
    train_ds, eval_ds = get_datasets(valid=False)
  collate_fn = gezi.DictPadCollate(return_tensors='tf')
  train_ds = train_ds.to_tf_dataset(
      columns=train_ds.columns,
      label_cols=["labels"],
      shuffle=True,
      collate_fn=collate_fn,
      batch_size=gezi.batch_size(),
  )
  eval_ds = eval_ds.to_tf_dataset(
      columns=eval_ds.columns,
      label_cols=["labels"],
      shuffle=False,
      collate_fn=collate_fn,
      batch_size=gezi.eval_batch_size(),
  )
  if valid:
    valid_ds = valid_ds.to_tf_dataset(
        columns=valid_ds.columns,
        label_cols=["labels"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=gezi.eval_batch_size(),
    )
    return train_ds, eval_ds, valid_ds
  else:
    return train_ds, eval_ds


def create_datasets(valid=True):
  if FLAGS.torch:
    return get_dataloaders(valid)
  else:
    return get_tf_datasets(valid)
  
