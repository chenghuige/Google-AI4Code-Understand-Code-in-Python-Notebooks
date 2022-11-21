#!/usr/bin/env python
# coding: utf-8

# In[9]:


from gezi.common import *
sys.path.append('..')
gezi.set_pandas()
# gezi.set_pandas_widder()
from src.config import *
gezi.init_flags()


# In[10]:


FLAGS.root


# In[11]:


# !wget -P {FLAGS.root}  https://github-notebooks-samples.s3-eu-west-1.amazonaws.com/ntbs_list.json 


# In[12]:


gezi.system('wc -l {FLAGS.root}/ntbs_list.json')


# In[14]:


gezi.try_mkdir(f'{FLAGS.root}/ntbs')


# In[18]:


NUM_NOTEBOOKS = 9941039


# In[21]:


from gezi.common import *
from urllib.request import urlretrieve
import random
import json

def read_json(filepath):  
  return json.load(open(filepath))

def read_ipynb(file): 
  with open(file, mode= 'r', encoding= 'utf-8') as f:
    return json.loads(f.read())

notebooks_list = read_json(f'{FLAGS.root}/ntbs_list.json')
random.seed(1024)
random.shuffle(notebooks_list)

def download(ipynb_file):
  ipynb_save_path =  f'{FLAGS.root}/ntbs/{ipynb_file}'
  id = ipynb_file.split('.')[0]
  csv_file = f'{FLAGS.root}/dfs/{id}.csv'
  if (not os.path.exists(ipynb_save_path)) and (not os.path.exists(csv_file)):
    try:
      urlretrieve(f'https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/{ipynb_file}', ipynb_save_path)
    except Exception as e:
      ic(e)


# In[22]:


gezi.prun(download, notebooks_list, 100)


# In[ ]:




