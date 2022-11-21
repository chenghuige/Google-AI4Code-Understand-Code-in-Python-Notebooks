#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
from gezi.common import *
from src.config import *
from src.preprocess import *
from src.eval import *
gezi.init_flags()
gezi.set_pandas()


# In[2]:


v = 4


# In[4]:


V = 7
root = f'../working/offline/{V}/0'
# pairwise two tower, recall model
# pt_model = 'all-mpnet-base-v2.flag-pairwise14-2-pre_ext_emlm_mlm.ep1'
# p2t_model = 'all-mpnet-base-v2.flag-pairwise14-2'
pt_model = 'all-mpnet-base-v2.flag-pairwise14-2-pre_mlm3'
# pt_model = 'all-mpnet-base-v2.flag-pairwise14-2'
# pt_model2 = 'pmminilm.flag-pairwise14-2-pre_emlm_mlm-mmnilm'
# pt_model2 = 'pmminilm.flag-pairwise14-2-pre_mlm3-pmminilm'
# pt_model2 = 'pmminilm.flag-pairwise14-2'
# pairwise concat, rank model
# pc_model = 'deberta-v3-small.flag-pairwise14-4-cat-insert-extpred-ft.neg_rand_prob-0.neg_strategy-rand-sample'
pc_model = 'deberta-v3-small.flag-pairwise14-4-cat-insert-oof-ft'

# context model
# c_model = 'deberta-v3-small.flag-context4-2-d'
# c_model2 = 'lsg-mminilm.flag-context4-3-d-s-mminilm'
# c_model = 'deberta-v3-small.flag-context4-2-d'
# c_model = 'deberta-v3-small.flag-context4-2-d-oof-ft'
c_model = 'deberta-v3-small.flag-context4-2-d-oof-ft.list_train_ordered'
# c_model = 'deberta-v3-small.flag-context4-2-d-oof-ft.add_probs'
# c_model = 'deberta-v3-small.flag-context4-2-d-extpred-ft.0804'
# c_model = 'deberta-v3-small.flag-context4-2-d-oof-ft.list_train_ordered'
# c_model2 = 'lsg-mminilm.flag-context4-3-d-s-oof-ft-mminilm'


# In[5]:


df_train = pd.read_feather('../working/train.fea')
df_train = df_train[df_train.fold==0]


# In[6]:


xp = gezi.load(f'{root}/{pt_model}.eval/valid.pkl')


# In[7]:


xp.keys()


# In[8]:


#xp2 = gezi.load(f'{root}/{pt_model2}.eval/valid.pkl')


# In[9]:


xc = gezi.load(f'{root}/{c_model}.eval/valid.pkl')


# In[24]:


xpc = gezi.load(f'{root}/{pc_model}.eval/valid.pkl')


# In[170]:


# xc2 = gezi.load(f'{root}/{c_model2}.eval/valid.pkl')


# In[172]:


# gezi.save(xp, '../working/xs/xp.pkl')
# gezi.save(xp2, '../working/xs/xp2.pkl')
# gezi.save(xc, '../working/xs/xc.pkl')
# gezi.save(xpc, '../working/xs/xpc.pkl')
# gezi.save(xc2, '../working/xs/xc2.pkl')


# In[11]:


xpc.keys()


# In[14]:


calc_metric(xc, 'reg_pred')


# In[15]:


calc_metric(xp, 'cls_pred')


# In[16]:


# In[71]:


# x = xp2.copy()
# x['cls_pred'] = xp['cls_pred'] * 0.9 + xp2['cls_pred'] * 0.1
# calc_metric(x, 'cls_pred')


# In[17]:


calc_metric(xpc, 'cls_pred')


# In[18]:


xc.keys()


# In[19]:


# d = pd.DataFrame(gezi.batch2list(xc))
# dg = d.groupby('pred_id')
# max_preds = dg['pred'].max()
# min_preds = dg['pred'].min()
# max_cls_preds = dg['cls_pred'].max()
# min_cls_preds = dg['cls_pred'].min()


# In[20]:


gezi.sort_dict_byid_(xp, 'cid')
gezi.sort_dict_byid_(xc, 'cid')
gezi.sort_dict_byid_(xpc, 'cid')
# gezi.sort_dict_byid_(xc2, 'cid')


# In[53]:


xpc = gezi.load(f'{root}/{pc_model}.eval/valid.pkl')
pc_preds = dict(zip(xpc['cid'], xpc['pred']))
pc_cls_preds = dict(zip(xpc['cid'], xpc['cls_pred']))
pc_probs = dict(zip(xpc['cid'], xpc['probs']))


# In[68]:


preds = []
cls_preds = []
probs = []
marks = []
for cid, pred, cls_pred, prob in zip(xp['cid'], xp['pred'], xp['cls_pred'], xp['probs']):
  preds.append(pc_preds.get(cid, pred))
  if prob.max() < 0.7:
    marks.append(1)
    probs.append(pc_probs.get(cid, prob))
    cls_preds.append(pc_cls_preds.get(cid, cls_pred))
  else:
    probs.append(prob)
    marks.append(0)
    cls_preds.append(cls_pred)

xpc = xp.copy()
xpc['pred'] = np.asarray(preds)
xpc['cls_pred'] = np.asarray(cls_preds)
xpc['probs'] = np.asarray(probs)
xpc['mark'] = np.asarray(marks)
ic(xpc['mark'].mean())


# In[69]:


calc_metric(xpc, 'cls_pred')


# In[76]:


LABEL_COL = 'rel_rank'
def diff_feats(x, l,r, name):
  if isinstance(l, str):
    x[f'{name}_diff'] = x[l] - x[r]
  else:
    x[f'{name}_diff'] = l - r
  x[f'abs_{name}_diff'] = abs(x[f'{name}_diff'])

def gen_pfeat(x_, topn=10):
  x = {}
  keys = ['pred', 'cls_pred']
  x = {k: x_[k] for k in keys}
  # x['rank_pred'] = x['pred'] * (1 + x_['n_code_cell']) - 0.5
  # x['cls_rank_pred'] = x['cls_pred'] * (1 + x_['n_code_cell']) - 0.5
  x['min_prob'] = x_['probs'].min()
  x['var_prob'] = np.var(x_['probs'])
  if 'sims' in x_:
    x['min_sim'] = x_['sims'].min()
    x['var_sim'] = np.var(x_['sims'])
  
  idxes = (-x_['probs']).argsort()
  for i in range(topn):
    if i < len(idxes):
      if i > 0:
        x[f'top_pred_{i}'] = (idxes[i] + 0.5) / (x_['n_code_cell'] + 1)
        # x[f'top_rank_{i}'] = idxes[i]
      x[f'top_prob_{i}'] = x_['probs'][idxes[i]]
      if 'sims' in x_:
        x[f'top_sim_{i}'] = x_['sims'][idxes[i]]
    else:
      x[f'top_pred_{i}'] = -1
      # x[f'top_rank_{i}'] = -1
      x[f'top_prob_{i}'] = -1
      if 'sims' in x_:
        x[f'top_sim_{i}'] = -1
  
  diff_feats(x, 'cls_pred', 'pred', 'cls')
  # diff_feats(x, 'cls_rank_pred', 'rank_pred', 'cls_rank')
  return x

def gen_cfeat(x_):
  topn = 4
  x = {}
  keys = [
    # 'pred', 
    'cls_pred', 
    'reg_pred', 
    'cls2_pred', 
    'group_pos',
    'group_rank',
    ]
  x = {k: x_[k] for k in keys}
  # x['cls_rank_pred'] = x['cls_pred'] * FLAGS.num_classes - 0.5
  # x['cls2_rank_pred'] = x['cls2_pred'] * FLAGS.num_classes - 0.5
  # x['reg_rank_pred'] = x['reg_pred'] * FLAGS.num_classes - 0.5
  preds = x_['cls_pred_ori']
  probs = gezi.softmax(preds)
  x['min_prob'] = probs.min()
  x['var_prob'] = np.var(probs)
  idxes = (-probs).argsort()

  for i in range(topn):
    if i < len(idxes):
      if i > 0:
        x[f'top_pred_{i}'] = (idxes[i] + 0.5) / FLAGS.num_classes
        # x[f'tpop_rank_{i}']  = idxes[i] 
      x[f'top_prob_{i}'] = probs[idxes[i]]
    else:
      x[f'top_pred_{i}'] = -1
      # x[f'top_rank_{i}'] = -1
      x[f'top_prob_{i}'] = -1
  diff_feats(x, 'cls_pred', 'reg_pred', 'cls_reg')
  diff_feats(x, 'cls_pred', 'cls2_pred', 'cls_cls2')
  diff_feats(x, 'cls2_pred', 'reg_pred', 'cls2_reg')
  # diff_feats(x, 'cls_rank_pred', 'reg_rank_pred', 'cls_reg_rank')
  # diff_feats(x, 'cls_rank_pred', 'cls2_rank_pred', 'cls_cls2_rank')
  # diff_feats(x, 'cls2_rank_pred', 'reg_rank_pred', 'cls2_reg_rank')
  # group_id = x_['pred_id']
  # x['max_pred'] = max_preds[group_id]
  # x['min_pred'] = min_preds[group_id]
  # x['pred_ratio'] = (x['pred'] - x['min_pred']) / (x['max_pred'] - x['min_pred']) if (x['max_pred'] - x['min_pred']) else 1
  # x['max_cls_pred'] = max_cls_preds[group_id]
  # x['min_cls_pred'] = min_cls_preds[group_id]
  # x['cls_pred_ratio'] = (x['cls_pred'] - x['min_cls_pred']) / (x['max_cls_pred'] - x['min_cls_pred']) if (x['max_cls_pred'] - x['min_cls_pred']) else 1
  
  return x

def gen_feats():
  xs = gezi.batch2list(xp)
  p_feats = [gen_pfeat(x) for x in tqdm(xs, desc=f'gen_pfeats:xp')]
  xs = gezi.batch2list(xc)
  c_feats = [gen_cfeat(x) for x in tqdm(xs, desc=f'gen_cfeats:xc')]
  xs = gezi.batch2list(xpc)
  pc_feats = [gen_pfeat(x, topn=5) for x in tqdm(xs, desc=f'gen_pfeats:xpc')]

  feats = []
  for i in tqdm(range(len(xs)), desc='merge_feats'):
    fe1 = p_feats[i]
    fec = c_feats[i]
    fepc = pc_feats[i]
    fepc['mark'] = xpc['mark'][i]
    
    fe = {}
    fe['code_ratio'] = xp['n_code_cell'][i] / xp['n_cell'][i]
    diff_feats(fe, fe1['pred'], fec['reg_pred'], 'pc_reg')
    diff_feats(fe, fe1['pred'], fec['cls_pred'], 'pc_cls')
    diff_feats(fe, fe1['pred'], fec['cls2_pred'], 'pc_cls2')
    diff_feats(fe, fe1['cls_pred'], fec['reg_pred'], 'pc_cls_reg')
    diff_feats(fe, fe1['cls_pred'], fec['cls_pred'], 'pc_cls_cls')
    diff_feats(fe, fe1['cls_pred'], fec['cls2_pred'], 'pc_cls_cls2')
    
    diff_feats(fe, fepc['pred'], fec['reg_pred'], 'pcc_reg')
    diff_feats(fe, fepc['pred'], fec['cls_pred'], 'pcc_cls')
    diff_feats(fe, fepc['pred'], fec['cls2_pred'], 'pcc_cls2')
    diff_feats(fe, fepc['cls_pred'], fec['reg_pred'], 'pcc_cls_reg')
    diff_feats(fe, fepc['cls_pred'], fec['cls_pred'], 'pcc_cls_cls')
    diff_feats(fe, fepc['cls_pred'], fec['cls2_pred'], 'pcc_cls_cls2')
    
    # diff_feats(fe, fe1['rank_pred'], fec['reg_rank_pred'], 'pc_cls_rank')
    # diff_feats(fe, fe1['cls_rank_pred'], fec['reg_rank_pred'], 'pc_cls_reg_rank')
    # diff_feats(fe, fepc['rank_pred'], fec['reg_rank_pred'], 'pcc_cls_rank')
    # diff_feats(fe, fepc['cls_rank_pred'], fec['reg_rank_pred'], 'pcc_cls_reg_rank')

    fe1 = gezi.dict_prefix(fe1, 'xp_')
    fe.update(fe1)
    fec = gezi.dict_prefix(fec, 'xc_')
    fe.update(fec)
    fepc = gezi.dict_prefix(fepc, 'xpc_')
    fe.update(fepc)

    keys = [
     'id', 'cell_id', 'cid',
     'n_words', 'n_code_cell', 'n_cell'
    ]
    for key in keys:
      fe[key] = xp[key][i]
      
    #  ic(feat)
    feats.append(fe)
  #  break
  dfeats = pd.DataFrame(feats)
  return dfeats


# In[77]:


dfeats = gen_feats()


# In[78]:


df = dfeats.merge(df_train[['cid', 'ancestor_id', LABEL_COL]], on='cid')


# In[79]:


gezi.set_fold(df, 5, 'ancestor_id')


# In[80]:


reg_cols = [x for x in dfeats.columns if x not in ['id', 'cell_id', 'cid', LABEL_COL]]
cat_cols = []
cols = reg_cols + cat_cols
len(cols)


# In[81]:


fold = 0
dvalid = df[df.fold==fold]
dtrain = df[df.fold!=fold]


# In[82]:


X_train = dtrain[cols]
y_train = dtrain[LABEL_COL]
X_valid = dvalid[cols]
y_valid = dvalid[LABEL_COL]


# In[87]:


import xgboost as xgb
xgb_params = {
      'reg_lambda': 7.960622217848342e-07, 
                      'subsample': 0.7422597612762745,
                      # 'bagging_temperature': 0.2,
                      'max_depth': 10, 
                      # 'early_stopping_rounds': 500,
                      # 'n_estimators': 10000,
                      # 'cat_features': [],
                      # 'loss_function': 'MAE',
                      'min_child_samples': 100,
                      # eval_metric='MAE',
}
xgb_model = xgb.XGBRanker(booster="gbtree", 
                      # objective="rank:ndcg",
                      objective='rank:pairwise',
                      tree_method="gpu_hist", 
                      sampling_method="gradient_based",
                      n_estimators=1000,
                      learning_rate=0.03,
                      **xgb_params
                      )
l = []
for fold in tqdm(range(5)):
  dvalid = df[df.fold==fold]
  dtrain = df[df.fold!=fold]
  X_train = dtrain[cols]
  y_train = dtrain[LABEL_COL]
  X_valid = dvalid[cols]
  y_valid = dvalid[LABEL_COL]
  train_groups = dtrain.groupby('id').size().to_numpy()
  eval_groups = dvalid.groupby('id').size().to_numpy()
  xgb_model.fit(X_train, 
            y_train, 
            group=train_groups, 
            eval_set=[(X_train, y_train),(X_valid, y_valid)],
            eval_group=[train_groups, eval_groups],
            verbose=100)
  dvalid['xgb_pred'] = xgb_model.predict(dvalid[cols])
  dvalid['xgb_pred'] = [gezi.sigmoid(x) for x in dvalid.xgb_pred.values]
  xgb_score = calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.xgb_pred.values})
  ic(xgb_score)
  d_ = dvalid[['cid', 'xgb_pred']]
  l.append(d_)
xgb_oof = pd.concat(l)


# In[ ]:


xgb_oof.to_csv('../working/xgb_oof.csv', index=False)


# In[ ]:


stop


# In[83]:


from catboost import CatBoostRegressor, CatBoostRanker, CatBoostClassifier
cbt_params = {
              # 'bootstrap_type': 'Poisson',
              # 'learning_rate': 0.02,
              'learning_rate': 0.03,
              'reg_lambda': 7.960622217848342e-07, 
              'subsample': 0.7422597612762745,
              # 'bagging_temperature': 0.2,
              'max_depth': 10, 
              'early_stopping_rounds': 500,
              'n_estimators': 10000,
              'cat_features': [],
              'loss_function': 'MAE',
              'min_child_samples': 100,
              # 'task_type': 'GPU',
              # 'devices': '0',
              }


# In[84]:


cbt_model = CatBoostRegressor(**cbt_params)
cbt_model.fit(X_train, y_train,
      eval_set=[(X_train, y_train), (X_valid, y_valid)],
              verbose=100,
              )  
dvalid['cb_pred'] = cbt_model.predict(dvalid[cols])
cbt_score = calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.cb_pred.values})
ic(cbt_score)


# In[85]:


gezi.plot.feature_importance(cbt_model, topn=20)


# In[189]:


import lightgbm as lgb
params = {
          'boosting': 'gbdt',
          'objective': 'regression_l1',
          'metric': {'l1'},
          'num_leaves': 32,
          # 'num_leaves': 16,
          'min_data_in_leaf': 300,
          # 'min_data_in_leaf': 500,
          # 'max_depth': 5,
          'learning_rate': 0.03,
          # "feature_fraction": 0.8,
          # "bagging_fraction": 0.75,
          # "feature_fraction": 0.9,
          # "bagging_fraction": 0.8,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.7,
          'min_data_in_bin':100,
          "lambda_l1": 1e-7,
          'lambda_l2': 1e-7,
          "random_state": 1024,
          "num_threads": 12,
          }

d_train = lgb.Dataset(X_train, y_train)
d_valid = lgb.Dataset(X_valid, y_valid, reference=d_train)

lgb_model = lgb.train(params,
                d_train,
                10000,
                valid_sets=[d_train, d_valid],
                verbose_eval=100,
                early_stopping_rounds=100)
dvalid['lgb_pred'] = lgb_model.predict(dvalid[cols])
lgb_score = calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.lgb_pred.values})
ic(lgb_score)


# In[190]:


# model = lgb.LGBMRanker(
#     objective="lambdarank",
#     metric="ndcg",
#     boosting_type="dart",
#     n_estimators=100,
#     importance_type='gain',
#     verbose=10,
#     random_state = 17
# )

# qids_train = dtrain.groupby(['id'])['cell_id'].count().values
# # y_train2 = [int(x * 1000) for x in y_train]

# model.fit(
#     X=X_train,
#     y=y_train2,
#     group=qids_train,
# )


# In[194]:


gezi.plot.feature_importance(lgb_model, topn=20)


# In[195]:


FOLDS = 5
cbt_scores, lgb_scores = [], []
scores = []
gezi.try_mkdir(f'../working/trees{v}')
for fold in tqdm(range(FOLDS)):
  dvalid = df[df.fold==fold]
  dtrain = df[df.fold!=fold]
  X_train = dtrain[cols]
  y_train = dtrain[LABEL_COL]
  X_valid = dvalid[cols]
  y_valid = dvalid[LABEL_COL]
  if fold > 0:
    cbt_model = CatBoostRegressor(**cbt_params)
    cbt_model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  verbose=500,
                  )  
  cbt_model.save_model(f'../working/trees{v}/{fold}.cbt')
  dvalid['cb_pred'] = cbt_model.predict(dvalid[cols])
  cbt_score = calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.cb_pred.values})
  ic(cbt_score)
  
  d_train = lgb.Dataset(X_train, y_train)
  d_valid = lgb.Dataset(X_valid, y_valid, reference=d_train)
  if fold > 0:
    lgb_model = lgb.train(params,
                    d_train,
                    10000,
                    valid_sets=[d_train, d_valid],
                    verbose_eval=500,
                    early_stopping_rounds=100)
  lgb_model.save_model(f'../working/trees{v}/{fold}.lgb')
  dvalid['lgb_pred'] = lgb_model.predict(dvalid[cols])
  
  lgb_score = calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.lgb_pred.values})
  ic(lgb_score)
  score = calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': (dvalid.cb_pred.values * 0.3 + dvalid.lgb_pred.values * 0.7)})
  ic(score)
  
  cbt_scores.append(cbt_score)
  lgb_scores.append(lgb_score)
  scores.append(score)
  ic(fold, 
     np.asarray(cbt_scores).mean(), 
     np.asarray(lgb_scores).mean(),
     np.asarray(scores).mean(),
     )


# In[196]:


# fold: 0
#                     np.asarray(cbt_scores).mean(): 0.9181025338378637
#                     np.asarray(lgb_scores).mean(): 0.9193848140698876
#                     np.asarray(scores).mean(): 0.9195238841373282

#  fold: 4
#                     np.asarray(cbt_scores).mean(): 0.9189608630607822
#                     np.asarray(lgb_scores).mean(): 0.9196959050732708
#                     np.asarray(scores).mean(): 0.9199400102856614


# In[ ]:




