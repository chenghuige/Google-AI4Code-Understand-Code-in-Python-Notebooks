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


notebooks_list = gezi.load(f'{FLAGS.root}/ntbs_list.json')
random.seed(1024)
random.shuffle(notebooks_list)


# In[3]:


NUM_NOTEBOOKS = 100


# In[4]:


def get_df(count):
  dfs = []
  i = 0
  pbar = tqdm(notebooks_list)
  for notebook in pbar:
    id = notebook.split('.')[0]
    csv_file = f'{FLAGS.root}/dfs/{id}.csv'
    if os.path.exists(csv_file):
      try:
        df = pd.read_csv(csv_file)
        if len(df[df.source.isnull()]) == 0:
          dfs.append(df)
          # ic(len(dfs), count)
          if len(dfs) == count:
            ofile = f'{FLAGS.root}/ext_100000_{i}.fea'
            if not os.path.exists(ofile):
              df = pd.concat(dfs)
              ic(ofile, len(df.id.unique()), len(df))
              df.reset_index().to_feather(ofile)
            i += 1
            if i == 100:
              return df
            dfs = []
      except Exception:
        pass

    pbar.update(1)
    pbar.set_postfix({'count': len(dfs)})
  
  return df


# In[5]:

count = 100000
df = get_df(count)


# In[ ]:


ic(df)


# In[ ]:




