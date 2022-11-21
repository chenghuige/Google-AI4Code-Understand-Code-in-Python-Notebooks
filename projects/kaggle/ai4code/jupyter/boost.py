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


V = 7
root = f'../working/offline/{V}/0'
# pairwise two tower, recall model
# p2t_model = 'all-mpnet-base-v2.flag-pairwise14-2-pre_ext_emlm_mlm.ep1.eval'
# p2t_model = 'all-mpnet-base-v2.flag-pairwise14-2'
pt_model = 'all-mpnet-base-v2.flag-pairwise14-2-pre_mlm3'
pt_model2 = 'pmminilm.flag-pairwise14-2-pre_emlm_mlm-mmnilm'
pt_model2 = 'pmminilm.flag-pairwise14-2'
# pairwise concat, rank model
pc_model = 'deberta-v3-small.flag-pairwise14-4-cat-insert-extpred-ft.neg_rand_prob-0.neg_strategy-rand-sample'
# context model
c_model = 'deberta-v3-small.flag-context4-2-d'
c_model2 = 'lsg-mminilm.flag-context4-3-d-s-mminilm'


# In[3]:


LABEL_COL = 'rel_rank'
df_train = pd.read_feather('../working/train.fea')
df_train = df_train[df_train.fold==0]


# In[4]:


xp = gezi.load(f'{root}/{pt_model}.eval/valid.pkl')


# In[5]:


xp2 = gezi.load(f'{root}/{pt_model2}.eval/valid.pkl')


# In[6]:


x = xp.copy()
rate = 0.9
x['pred'] = rate * xp['cls_pred'] + (1 - rate) * xp2['cls_pred'] 


# In[7]:


calc_metric(x)


# In[8]:


calc_metric(xp)


# In[9]:


calc_metric(xp2)


# In[10]:


xp.keys()


# In[23]:


def gen_feat(x_):
  topn = 10
  x = {}
  keys = ['pred', 'cls_pred']
  x = {k: x_[k] for k in keys}
  x['rank_pred'] = x['pred'] * (1 + x_['n_code_cell']) - 0.5
  x['cls_rank_pred'] = x['cls_pred'] * (1 + x_['n_code_cell']) - 0.5
  x['min_prob'] = x_['probs'].min()
  x['min_sim'] = x_['sims'].min()
  x['var_prob'] = np.var(x_['probs'])
  x['var_sim'] = np.var(x_['sims'])
  idxes = (-x_['probs']).argsort()
  for i in range(topn):
    if i < len(idxes):
      if i > 0:
        x[f'top_pred_{i}'] = (idxes[i] + 0.5) / (x_['n_code_cell'] + 1)
      x[f'top_prob_{i}'] = x_['probs'][idxes[i]]
      x[f'top_sim_{i}'] = x_['sims'][idxes[i]]
    else:
      x[f'top_pred_{i}'] = -1
      x[f'top_prob_{i}'] = -1
      x[f'top_sim_{i}'] = -1
  x['cls_diff'] = x['cls_pred'] - x['pred']
  x['abs_cls_diff'] = abs(x['cls_diff'])
  return x


# In[30]:


def gen_feats():
  xs = gezi.batch2list(xp)
  p_feats = [gen_feat(x) for x in tqdm(xs, desc=f'gen_feats:xp')]
  # p_feats = gezi.prun_list(gen_feat, xs, 2, desc=f'gen_feats:xp')
  xs = gezi.batch2list(xp2)
  p2_feats = [gen_feat(x) for x in tqdm(xs, desc=f'gen_feats:xp2')]
  # p2_feats = gezi.prun_list(gen_feat, xs, 2, desc=f'gen_feats:xp2')
  feats = []
  for i in range(len(xs)):
    fe = p_feats[i]
    fe2= p2_feats[i]
    
    fe['code_ratio'] = xp['n_code_cell'] / xp['n_cell']
    fe['ps_pred_diff'] = fe['pred'] - fe2['pred']
    fe['abs_ps_pred_diff'] = abs(fe['ps_pred_diff'])
    fe['ps_cls_pred_diff'] = fe['cls_pred'] - fe2['cls_pred']
    fe['abs_ps_cls_pred_diff'] = abs(fe['ps_cls_pred_diff'])
    
    fe = gezi.dict_prefix(fe, 'p_')
    fe2 = gezi.dict_prefix(fe2, 'p2_')
    fe.update(fe2)
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


# In[31]:


dfeats = gen_feats()


# In[ ]:


dfeats


# In[ ]:


df = dfeats.merge(df_train[['cid', 'ancestor_id', LABEL_COL]], on='cid')


# In[ ]:


gezi.set_fold(df, 5, 'ancestor_id')


# In[ ]:


reg_cols = [x for x in dfeats.columns if x not in ['id', 'cell_id', 'cid', LABEL_COL]]
cat_cols = []
cols = reg_cols + cat_cols
len(cols)


# In[ ]:


fold = 1
dvalid = df[df.fold==fold]
dtrain = df[df.fold!=fold]


# In[ ]:


X_train = dtrain[cols]
y_train = dtrain[LABEL_COL]
X_valid = dvalid[cols]
y_valid = dvalid[LABEL_COL]


# In[ ]:


from catboost import CatBoostRegressor, CatBoostRanker
cbt_params = {
              # 'learning_rate': 0.02,
              'learning_rate': 0.03,
              'reg_lambda': 7.960622217848342e-07, 
              'subsample': 0.7422597612762745,
              # 'bagging_temperature': 0.2,
              'max_depth': 10, 
              'early_stopping_rounds': 100,
              'n_estimators': 10000,
              'cat_features': [],
              'loss_function': 'MAE',
              'min_child_samples': 300,
              }

xgb_params2 = {'learning_rate': 0.09827605967564293,'tree_method':'gpu_hist', 'gpu_id':0,
               'early_stopping_rounds': 50,
               'n_estimators': 10000, }


# In[ ]:


model = CatBoostRegressor(**cbt_params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid, y_valid)],
          verbose=100,
          )  
dvalid['cb_pred'] = model.predict(dvalid[cols])
ic(calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.cb_pred.values}))


# In[ ]:


gezi.plot.feature_importance(model, topn=20)


# In[ ]:





# In[ ]:




