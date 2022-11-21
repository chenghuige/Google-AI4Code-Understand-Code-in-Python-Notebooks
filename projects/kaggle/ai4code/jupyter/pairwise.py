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


# root = '../working/offline/5/0'
# context_model_name = 'deberta-v3-small.loss_method-mae.0705'
# # context_model_name = 'deberta-v3-small.n_context-40.loss_method-mae.0705'
# pairwise_model_name = 'deberta-v3-small.flag-pairwise5.encode_code_info.encode_markdown_info.encode_all_info.0706.dump_emb'
root = '../working/offline/6/0'
context_model_name = 'deberta-v3-small.flag-context2.cls_loss_rate-0.1.awp_train.adv_epochs-4'
pairwise_model_name = 'deberta-v3-small.flag-pairwise7.dump_emb'


# In[3]:


# offline  0.833608772799882 acc 764
# x = gezi.load('../working/offline/5/0/debug2/valid.pkl')
# offline 8389 acc 775
x = gezi.load(f'{root}/{pairwise_model_name}/embs.pkl')


# In[4]:


x


# In[5]:


x['emb'].shape


# In[6]:


df = get_df('train', for_eval=True)


# In[7]:


df


# In[8]:


x['emb'] = list(x['emb'])


# In[9]:


df_pred = pd.DataFrame(x)


# In[10]:


df_pred


# In[11]:


# offline 8730 online 8630 here offline 8725 check why lower
# x_context = gezi.load('../working/offline/5/0/deberta-v3-small.n_context-30.grad_acc-2.scheduler-linear/valid.pkl')
# offline 8749
x_context = gezi.load(f'{root}/{context_model_name}/valid.pkl')


# In[12]:


def to_df(x, mark='train'):
  df = get_df(mark, for_eval=True)
  df_pred = pd.DataFrame({
      'id': x['id'],
      'cell_id': x['cell_id'],
      'pred': x['pred'],
  })

  ids = set(df_pred.id)
  df = df[df.id.isin(ids)]
  df = df.merge(df_pred, on=(['id', 'cell_id']), how='left')

  df.loc[df["cell_type"] == 'code', 'pred'] = df.loc[df["cell_type"] == 'code',
                                                     'rel_rank']
  df = df.sample(frac=1)
  df = df.sort_values(['id', 'pred'])
  df_pred = df.groupby('id')['cell_id'].apply(list).reset_index(
      name='cell_order')

  return df_pred


# In[13]:


df_pred_context = to_df(x_context)


# In[14]:


context_model = dict(zip(x_context['cell_id'], x_context['pred']))


# In[15]:


ids = gezi.unique_list(df_pred.id)


# In[16]:


len(ids)


# In[17]:


df_gt = pd.read_csv(f'{FLAGS.root}/train_orders.csv')
df_gt = df_gt[df_gt.id.isin(ids)]
df_gt['cell_order'] = df_gt['cell_order'].apply(lambda x: x.split())
df_gt


# In[18]:


ic(kendall_tau(df_gt.cell_order.values, df_pred_context.cell_order.values))


# In[19]:


# TODO code embs add nan as end code 这个向量可以预先单独做好 一个即可。。
# df添加一个。。 但是最好后续改一下 避免冲突 独立表示最后位置的code + 1
def deal(rows):
  # ic(len(rows))
  # cell_orders = []
  code_rel_ranks, markdown_rel_ranks = [], []
  markdowns, codes, markdown_ranks, code_ranks = [], [], [], []
  markdown_embs, code_embs = [], []
  for row in rows:
    if row['cell_type'] == 'markdown':
      markdowns.append(row['cell_id'])
      markdown_embs.append(row['emb'])
      markdown_ranks.append(row['rank'])
      markdown_rel_ranks.append(row['rel_rank'])
    else:
      codes.append(row['cell_id'])
      code_embs.append(row['emb'])
      code_ranks.append(row['rank'])
      code_rel_ranks.append(row['rel_rank'])
  markdown_embs = np.asarray(markdown_embs)
  code_embs = np.asanyarray(code_embs)
  # ic(markdown_embs.shape, code_embs.shape)
  sims = np.matmul(markdown_embs, code_embs.transpose(1, 0))
  sims = gezi.softmax(sims)
  # ic(sims, sims.shape)
  # ic(sims.argmax(-1))
  # for i in range(len(markdowns)):
  #   ic(i, markdowns[i], markdown_ranks[i], code_ranks[sims[i].argmax(-1)], 
  #      code_ranks[(-sims[i]).argsort()[1]], 
  #      sims[i].max(), 
  #      markdown_rel_ranks[i],
  #      code_rel_ranks[sims[i].argmax(-1)],
  #      context_model[markdowns[i]])
  # sims = gezi.softmax(sims)
  # code_rel_ranks = np.asarray(code_rel_ranks)
  # def get_rel_rank(rel_ranks, probs):
  #   return (rel_ranks * probs).sum()
  # weight = 0.5
  # weight = 0.75
  weight = 0.6 # can get  0.8820775771987659 from 8725
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
  # weight = 1.
  # markdown_rel_ranks = [weight * sims[i].max() * (code_rel_ranks[sims[i].argmax(-1)] - 0.0001) + (1 - weight * sims[i].max()) * context_model[markdowns[i]] for i in range(len(markdowns))]
  span = 1. / (len(codes) + 1) / 2.
  markdown_rel_ranks = [merge((code_rel_ranks[sims[i].argmax(-1)] - span), context_model[markdowns[i]], sims[i].max()) for i in range(len(markdowns))]

  # weight = 0
  # markdown_rel_ranks = [weight * pointwise_model[markdowns[i]] + (1 - weight) * context_model[markdowns[i]] for i in range(len(markdowns))]
  # markdown_rel_ranks = [get_rel_rank(code_rel_ranks, sims[i]) for i in range(len(markdowns))]
  cells = np.asarray([*markdowns, *codes])
  rel_ranks = np.asarray([*markdown_rel_ranks, *code_rel_ranks])
  idxes = rel_ranks.argsort()
  return {'id': row['id'], 'cell_order': cells[idxes]}


# In[20]:


rows = []
id = None
count = 0
res = []
for row in tqdm(df_pred.itertuples(), total=len(df_pred)):
  row = row._asdict()
  if id and row['id'] != id:
    res.append(deal(rows))
    count += 1
    # if count == 10:
    #   break
    rows = []
  rows.append(row)
  id = row['id']
res.append(deal(rows))


# In[ ]:


# rows = []
# id = None
# count = 0
# res = []
# rows_list = []
# with gezi.Timer('calc'):
#   for row in tqdm(df_pred.itertuples(), total=len(df_pred)):
#     row = row._asdict()
#     if id and row['id'] != id:
#       rows_list.append(rows)
#       count += 1
#       # if count == 10:
#       #   break
#       rows = []
#     rows.append(row)
#     id = row['id']
#   rows_list.append(rows)
#   res = gezi.prun_list(deal, rows_list, 64)


# In[ ]:


df_pred2 = pd.DataFrame(res)
df_pred2


# In[ ]:


df_gt = df_gt.sort_values('id')
df_pred2 = df_pred2.sort_values('id')
kendall_tau(df_gt.cell_order.values, df_pred2.cell_order.values)


# In[ ]:


ic(kendall_tau(df_gt.cell_order.values, df_pred2.cell_order.values))


# In[ ]:





# In[ ]:





# In[ ]:




