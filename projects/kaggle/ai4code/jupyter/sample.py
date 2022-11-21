#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gezi.common import *
gezi.set_pd_widder()
sys.path.append('..')
from src import config
from src.preprocess import *
gezi.init_flags()


# In[2]:


df = pd.read_feather('../working/train.fea')


# In[3]:


df


# In[4]:


all_ids = df.id.unique()
len(all_ids)


# In[5]:


FLAGS.sample = True
config.init()
FLAGS.num_ids


# In[6]:


sampled_ids = gezi.random_sample(all_ids, FLAGS.num_ids, seed=1024)


# In[7]:


df = df[df.id.isin(sampled_ids)]


# In[8]:


df


# In[9]:


df[df.fold==0]


# In[10]:


df.reset_index(drop=True).to_feather('../working/train_sample.fea')


# In[11]:


df[df.source.str.contains('\[SEP\]')]


# In[ ]:




