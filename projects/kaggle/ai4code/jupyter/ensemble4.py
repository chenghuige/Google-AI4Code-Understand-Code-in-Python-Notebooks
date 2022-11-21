#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gezi.common import *
sys.path.append('..')
from src.config import *
from src.preprocess import *
from src.eval import *
gezi.init_flags()


# In[2]:


root = '../working/offline/6/0'
# context_model_name = 'deberta-v3-small.flag-context2-aug.n_context-40.cls_loss_rate-0.1.eval.p9'
# pairwise_model_name = 'deberta-v3-small.flag-pairwise9'
context_model_name = 'deberta-v3-small.flag-context2-aug.n_context-40.cls_loss_rate-0.1.eval.mpnet13-2'
pairwise_model_name = 'all-mpnet-base-v2.flag-pairwise13-2'


# In[3]:


xc = gezi.load(f'{root}/{context_model_name}/valid.pkl')


# In[4]:


xp = gezi.load(f'{root}/{pairwise_model_name}/valid.pkl')


# In[5]:


gezi.sort_dict_byid_(xc, 'cid')
gezi.sort_dict_byid_(xp, 'cid')


# In[6]:


ids = set(xc['id'])


# In[7]:


df_gt = pd.read_csv(f'{FLAGS.root}/train_orders.csv')
df_gt = df_gt[df_gt.id.isin(ids)]
df_gt['cell_order'] = df_gt['cell_order'].apply(lambda x: x.split())
df_gt.head()


# In[8]:


calc_metric(xc, 'reg_pred')


# In[9]:


calc_metric(xc, 'pred')


# In[10]:


calc_metric(xc, 'cls_pred')


# In[11]:


calc_metric(xp, 'cls_pred')


# In[12]:


x = xc.copy()
x['pred'] = xc['reg_pred'] * 0.5 + xp['cls_pred'] * 0.5
calc_metric(x)


# In[28]:


x = xc.copy()
x['pred'] = xc['cls_pred'] * 0.5 + xp['cls_pred'] * 0.5
calc_metric(x)


# In[29]:


x = xc.copy()
x['pred'] = xc['reg_pred'] * 0.5 + xp['pred'] * 0.5
calc_metric(x)


# In[13]:


def merge(x, y, prob):
  # return y
  # return x
  if prob > 0.9:
    return x * (1 - 0.0001) + y * 0.0001
  elif abs(y - x) < 0.1:
    return x * (1 - 0.0001) + y * 0.0001
  elif abs(y - x) < 0.2:
    return x * 0.95 * prob + y * (1 - 0.95 * prob)
  elif abs(y - x) < 0.3:
    return x * 0.85 * prob + y * (1 - 0.85 * prob)
  elif abs(y - x) < 0.4:
    return x * 0.5 * prob + y * (1 - 0.5 * prob)
  else:
    return y


# In[14]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['pred'], xc['pred'], xp['max_prob'])]
calc_metric(x)


# In[15]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['pred'], xc['reg_pred'], xp['max_prob'])]
calc_metric(x)


# In[16]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['pred'], xc['cls_pred'], xp['max_prob'])]
calc_metric(x)


# In[17]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['cls_pred'], xc['cls_pred'], xp['max_prob'])]
calc_metric(x)


# In[18]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['cls_pred'], xc['reg_pred'], xp['max_prob'])]
calc_metric(x)


# In[19]:


df_p = pd.DataFrame(xp)


# In[20]:


df_c = pd.DataFrame(gezi.batch2list(xc))


# In[23]:


df_p.columns


# In[24]:


df = df_c.merge(df_p[['cid', 'pred', 'cls_pred', 'max_prob', 'max_sim', 'probs', 'sims']], on='cid', suffixes=['_c', '_p'])


# In[26]:


df_train = pd.read_feather('../working/train.fea')


# In[27]:


df_train = df_train[df_train.id.isin(ids)]


# In[30]:


df = df.merge(df_train[['cid', 'ancestor_id', 'n_words', 'source']], on='cid')


# In[31]:


gezi.set_fold(df, 5, 'ancestor_id')


# In[32]:


df.head()


# In[92]:


df['pred_diff0'] = abs(df['pred_c'] - df['pred_p'])
df['pred_diff1'] = abs(df['reg_pred'] - df['pred_p'])
df['pred_diff2'] = abs(df['cls_pred_c'] - df['pred_p'])
df['pred_diff3'] = abs(df['cls2_pred'] - df['pred_p'])
df['markdown_frac'] = df['n_markdown_cell'] / df['n_cell']
df['span'] = 1 / (df['n_code_cell'] + 1)
top2, top3, top4, top5 = [], [], [], []
top2_prob, top3_prob, top4_prob, top5_prob = [], [], [], []
top2_sim, top3_sim, top4_sim, top5_sim = [], [], [], []
for i in tqdm(range(len(df)), desc='top'):
  # cls_preds = df['cls_pred_ori'].values[i]
  n_code = df['n_code_cell'].values[i]
  probs = df['probs'].values[i]
  sims = df['sims'].values[i]
  idxes = (-probs).argsort()
  if len(idxes) > 1:
    top2.append((idxes[1] + 0.5) / (n_code + 1))
    top2_prob.append(probs[idxes[1]])
    top2_sim.append(sims[idxes[1]])
  else:
    top2.append(-1)
    top2_prob.append(-1)
    top2_sim.append(-1)
  if len(idxes) > 2:
    top3.append((idxes[2] + 0.5) / (n_code + 1))
    top3_prob.append(probs[idxes[2]])
    top3_sim.append(sims[idxes[2]])
  else:
    top3.append(-1)
    top3_prob.append(-1)
    top3_sim.append(-1)
  if len(idxes) > 3:
    top4.append((idxes[3] + 0.5) / (n_code + 1))
    top4_prob.append(probs[idxes[3]])
    top4_sim.append(sims[idxes[3]])
  else:
    top4.append(-1)
    top4_prob.append(-1)
    top4_sim.append(-1)
  if len(idxes) > 4:
    top5.append((idxes[4] + 0.5) / (n_code + 1))
    top5_prob.append(probs[idxes[4]])
    top5_sim.append(sims[idxes[4]])
  else:
    top5.append(-1)
    top5_prob.append(-1)
    top5_sim.append(-1)
ctop_prob, ctop2, ctop3, ctop4, ctop2_prob, ctop3_prob, ctop4_prob = [], [], [], [], [], [], []
for i in tqdm(range(len(df)), desc='ctop'):
  preds = df['cls_pred_ori'].values[i]
  probs = gezi.softmax(preds)
  idxes = (-probs).argsort()
  ctop_prob.append(probs[idxes[0]])
  ctop2.append((idxes[1] + 0.5) / FLAGS.num_classes)
  ctop2_prob.append(probs[idxes[1]])
  ctop3.append((idxes[2] + 0.5) / FLAGS.num_classes)
  ctop3_prob.append(probs[idxes[2]])
  ctop4.append((idxes[3] + 0.5) / FLAGS.num_classes)
  ctop4_prob.append(probs[idxes[3]])
# for i in range(FLAGS.num_classes):
#   df[f'cls_pred{i}'] = df['cls_pred_ori'].apply(lambda x: x[i])
df['top2'] = top2
df['top2_prob'] = top2_prob
df['top2_sim'] = top2_sim
df['top3'] = top3
df['top3_prob'] = top3_prob
df['top3_sim'] = top3_sim
df['top4'] = top4
df['top4_prob'] = top4_prob
df['top4_sim'] = top4_sim
df['top5'] = top5
df['top5_prob'] = top5_prob
df['top5_sim'] = top5_sim
df['ctop_prob'] = ctop_prob
df['ctop2'] = ctop2
df['ctop2_prob'] = ctop2_prob
df['ctop3'] = ctop3
df['ctop3_prob'] = ctop3_prob
df['ctop4'] = ctop4
df['ctop4_prob'] = ctop4_prob
df['pred_diff4'] = abs(df['pred_c'] - df['top2'])
df['pred_diff5'] = abs(df['reg_pred'] - df['top2'])
df['pred_diff6'] = abs(df['cls_pred_c'] - df['top2'])
df['pred_diff7'] = abs(df['cls2_pred'] - df['top2'])
df['pred_diff8'] = abs(df['pred_p'] - df['top2'])
df['pred_diff9'] = abs(df['cls_pred_c'] - df['top3'])
df['pred_diff10'] = abs(df['pred_p'] - df['top3'])
df['rule_pred'] = [merge(x, y, prob) for x, y, prob in tqdm(zip(df.pred_p.values, df.pred_c.values, df.max_prob.values), total=len(df), desc='rule')]
reg_pred_probs = []
for row in tqdm(df.itertuples(), total=len(df), desc='cls_pred_p_prob'):
  probs = row.probs
  n_code = row.n_code_cell
  reg_pred = row.reg_pred
  try:
    reg_pred_probs.append(probs[int(min(reg_pred, 1) * (n_code + 1) - 0.5)])
  except Exception:
    ic(n_code, len(probs), reg_pred)
    break
df['reg_pred_prob'] = reg_pred_probs


# In[111]:


fold = 0
dvalid = df[df.fold==fold]
dtrain = df[df.fold!=fold]


# In[112]:


df.columns


# In[113]:


df


# In[135]:


reg_cols =  [
          'n_code_cell',
          'n_markdown_cell',
          'n_cell',
          'cls_pred_c',
          'pred_c',
          'reg_pred',
          'cls2_pred',
          'pred_p',
          'cls_pred_p',
          'rule_pred',
          'pred_diff0',
          'pred_diff1',
          'pred_diff2',
          'pred_diff3',
          'pred_diff4',
          'pred_diff5',
          'pred_diff6',
          'pred_diff7',
          'pred_diff8',
          'pred_diff9',
          'pred_diff10',
          'max_sim',
          'max_prob',
          'markdown_frac',
          'span',
          'top2',
          'top2_prob',
          'top2_sim',
          'top3',
          'top3_prob',
          'top3_sim',
          'top4',
          'top4_prob',
          'top4_sim',
          'top5',
          'top5_prob',
          'top5_sim',
          'ctop_prob', 
          'ctop2', 
          'ctop3', 
          'ctop4', 
          'ctop2_prob', 
          'ctop3_prob', 
          'ctop4_prob',
          # 'cls_pred_ori',
          # 'cls2_pred_top2',
          # 'cls2_pred_top3',
          # 'sims',
          # 'probs',
          # 'cls_pred_p_prob',
          # 'reg_pred_prob',
        ]
# for i in range(FLAGS.num_classes):
#   reg_cols += [f'cls_pred{i}']
cat_cols = [
          
            ]
label_col = 'rel_rank'
cols = reg_cols + cat_cols
X_train = dtrain[cols]
y_train = dtrain[label_col]
X_valid = dvalid[cols]
y_valid = dvalid[label_col]


# In[136]:


params = {
          'boosting': 'gbdt',
          'objective': 'regression_l1',
          'metric': {'l1'},
          'num_leaves': 8,
          'min_data_in_leaf': 30,
          'max_depth': 6,
          'learning_rate': 0.01,
          "feature_fraction": 0.9,
          "bagging_fraction": 0.75,
          'min_data_in_bin':15,
          "lambda_l1": 5,
          'lambda_l2': 5,
          "random_state": 1024,
          "num_threads": 12,
          }


# In[137]:


# import lightgbm as lgb
# d_train = lgb.Dataset(X_train, y_train)
# d_valid = lgb.Dataset(X_valid, y_valid, reference=d_train)

# bst = lgb.train(params,
#                 d_train,
#                 10000,
#                 valid_sets=[d_train, d_valid],
#                 verbose_eval=10,
#                 early_stopping_rounds=100)


# In[138]:


# abs(dvalid['pred_c'] - dvalid['rel_rank']).mean()


# In[139]:


# dvalid['lgb_pred'] = bst.predict(dvalid[cols])


# In[140]:


# calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.lgb_pred.values})


# In[141]:


from catboost import CatBoostRegressor, Pool
xgb_params = {'learning_rate': 0.02,
              'reg_lambda': 7.960622217848342e-07, 
              'subsample': 0.7422597612762745,
              'max_depth': 10, 
              'early_stopping_rounds': 500,
              'n_estimators': 10000,
              'cat_features': [],
              'loss_function': 'MAE',
              }

xgb_params2 = {'learning_rate': 0.09827605967564293,'tree_method':'gpu_hist', 'gpu_id':0,
               'early_stopping_rounds': 50,
               'n_estimators': 10000, }


# In[142]:


model = CatBoostRegressor(**xgb_params)
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
                verbose=10,
                )  


# In[143]:


# model = CatBoostRegressor()
# model.load_model('../working/cbt/cbt.bin')


# In[144]:


dvalid['cb_pred'] = model.predict(dvalid[cols])


# In[145]:


x = {'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values}
x['pred'] = [merge(x, y, prob) for x, y, prob in zip(dvalid.pred_p.values, dvalid.pred_c.values, dvalid.max_prob.values)]
score = calc_metric(x, 'pred', df_gt)
ic(score)
ic(abs(x['pred'] - dvalid['rel_rank']).mean())


# In[146]:


ic(calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.cb_pred.values}))


# In[147]:


model.save_model('../working/cbt.bin')


# In[148]:


gezi.plot.feature_importance(model, topn=20)


# In[149]:


gezi.plot.feature_importance(model, topn=-20)


# In[128]:


stop


# In[ ]:


dvalid[cols + ['cb_pred']]


# In[ ]:


calc_metric({'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values, 'pred': dvalid.reg_pred.values})


# In[ ]:


import optuna
from optuna import Trial


# In[ ]:


P = {}
def merge(x, y, prob):
  if prob > P['prob']:
    return x * (1 - 0.0001) + y * 0.0001
  elif abs(y - x) < P['diff']:
    return x * (1 - 0.0001) + y * 0.0001
  elif abs(y - x) < 0.2:
    return x * P['rate0'] * prob + y * (1 - P['rate0'] * prob)
  elif abs(y - x) < 0.3:
    return x * P['rate1'] * prob + y * (1 - P['rate1'] * prob)
  elif abs(y - x) < 0.4:
    return x * P['rate2'] * prob + y * (1 - P['rate2'] * prob)
  elif abs(y - x) < 0.5:
    return x * P['rate3'] * prob + y * (1 - P['rate3'] * prob)
  else:
    return x * P['rate4'] * prob + y * (1 - P['rate4'] * prob)


# In[ ]:


df_gt = df_gt[df_gt.id.isin(ids)]
x = {'id': dtrain.id.values, 'cell_id': dtrain.cell_id.values}


# In[ ]:


# def objective(trial):
#   suggest = trial.suggest_float 
#   P['prob'] = suggest('prob', 0., 1.)
#   P['diff'] = suggest('diff', 0., 1.)
#   P['rate0'] = suggest('rate0', 0., 1.)
#   P['rate1'] = suggest('rate1', 0., 1.)
#   P['rate2'] = suggest('rate2', 0., 1.)
#   P['rate3'] = suggest('rate3', 0., 1.)
#   P['rate4'] = suggest('rate4', 0., 1.)
          
#   x['pred'] = [merge(x, y, prob) for x, y, prob in zip(dtrain.pred_p.values, dtrain.pred_c.values, dtrain.max_prob.values)]
#   score = calc_metric(x, 'pred', df_gt)
#   return score
  
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)
# ic(study.best_value, study.best_params)


# In[ ]:


# x = {'id': dvalid.id.values, 'cell_id': dvalid.cell_id.values}
# x['pred'] = [merge(x, y, prob) for x, y, prob in zip(dvalid.pred_p.values, dvalid.pred_c.values, dvalid.max_prob.values)]
# score = calc_metric(x, 'pred', df_gt)
# ic(score)


# In[ ]:





# In[ ]:





# In[ ]:




