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
context_model_name = 'deberta-v3-small.flag-context2-aug.n_context-40.cls_loss_rate-0.1.eval.p9'
pairwise_model_name = 'deberta-v3-small.flag-pairwise9'


# In[3]:


xc = gezi.load(f'{root}/{context_model_name}/valid.pkl')


# In[4]:


xp = gezi.load(f'{root}/{pairwise_model_name}/valid.pkl')


# In[5]:


ids = set(xc['id'])


# In[6]:


df_gt = pd.read_csv(f'{FLAGS.root}/train_orders.csv')
df_gt = df_gt[df_gt.id.isin(ids)]
df_gt['cell_order'] = df_gt['cell_order'].apply(lambda x: x.split())
df_gt.head()


# In[7]:


calc_metric(df_gt, xc, 'reg_pred')


# In[8]:


calc_metric(df_gt, xp, 'cls_pred')


# In[9]:


x = xc.copy()
x['pred'] = xc['reg_pred'] * 0.5 + xp['cls_pred'] * 0.5
calc_metric(df_gt, x)


# In[10]:


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
calc_metric(df_gt, x)


# In[13]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['pred'], xc['reg_pred'], xp['max_prob'])]
calc_metric(df_gt, x)


# In[15]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['pred'], xc['cls_pred'], xp['max_prob'])]
calc_metric(df_gt, x)


# In[16]:


x['pred'] = [merge(x, y, prob) for x, y, prob in zip(xp['cls_pred'], xc['cls_pred'], xp['max_prob'])]
calc_metric(df_gt, x)


# In[ ]:




