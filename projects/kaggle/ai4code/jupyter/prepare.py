#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gezi.common import *


# In[ ]:


root = '../input/AI4Code'
data_dir = Path(root)


# In[ ]:


def create_df(folder, workers=80):
  def _create_df(fpath):
    df = pd.read_json(fpath, dtype={'cell_type': 'category', 'source': 'str'}).reset_index().rename({"index":"cell_id"}, axis=1)
    df["id"] = fpath.rsplit(".", 1)[0].rsplit("/", 1)[-1]
    return df
  dfs = gezi.prun(_create_df, glob.glob(f'{folder}/*.json'), workers)
  df = pd.concat(dfs)
  return df


# In[ ]:


train_file = f'{root}/train.fea'
if os.path.exists(train_file):
  df = pd.read_feather(train_file)
else:
  df = create_df(f'{root}/train')
  df.reset_index().to_feather(train_file)


# In[ ]:


df['source_len'] = df['source'].apply(lambda x: len(x))
df['source_words'] = df['source'].apply(lambda x: len(x.split()))


# In[ ]:


df.describe()


# In[ ]:


df


# In[ ]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')


# In[ ]:


input_ids_list = gezi.prun(lambda x: tokenizer(x).input_ids, df.source.values, 80)
# for source in tqdm(df.source.values, total=len(df)):
#   input_ids_list.append(tokenizer(source).input_ids)


# In[ ]:


df['input_ids'] = input_ids_list


# In[ ]:


df.reset_index().to_feather(train_file)

