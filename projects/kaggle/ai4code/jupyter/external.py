#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def read_json(filepath):  
  return json.load(open(filepath))

def read_ipynb(file): 
  with open(file, mode= 'r', encoding= 'utf-8') as f:
    return json.loads(f.read())


notebooks_list = read_json(f'{FLAGS.root}/ntbs_list.json')
import random
random.seed(1024)
random.shuffle(notebooks_list)

def is_ok(x):
  if not 'cells' in x:
    return False
  
  c_count = 0
  m_count = 0
  for item in x['cells']:
    if item['cell_type'] == 'code':
      c_count += 1
      if m_count:
        return True
    else:
      m_count += 1
      if c_count:
        return True
  return False


def to_df(x, id):
  l = []
  for i, cell in enumerate(x['cells']):
    m = {
      'id': id,
      'cell_id': str(i),
      'cid': f'{id}\t{i}',
      'cell_type': cell['cell_type'],
      'source': ''.join(cell['source'])
    }
    l.append(m)
  return pd.DataFrame(l)


def deal(notebook):
  find = False
  id = notebook.split('.')[0]
  ofile = f'{FLAGS.root}/dfs/{id}.csv'
  file = f'{FLAGS.root}/ntbs/{notebook}'
  if os.path.exists(ofile):
    find = True
  if os.path.exists(file):
    if find:
      os.remove(file)
    else:
      try:
        x = json.load(open(file))
        if is_ok(x):
          df = to_df(x, id)
          df.to_csv(ofile, index=False)
        os.remove(file)
      except Exception:
        os.remove(file)

gezi.prun(deal, notebooks_list, 100)


