#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gezi.common import *
sys.path.append('..')
gezi.set_pandas()
# gezi.set_pandas_widder()
from src.config import *
gezi.init_flags()


# In[3]:


root = FLAGS.root


# In[4]:


def create_df(folder, workers=80):
  def _create_df(fpath):
    df = pd.read_json(fpath, dtype={'cell_type': 'category', 'source': 'str'}).reset_index().rename({"index":"cell_id"}, axis=1)
    df["id"] = fpath.rsplit(".", 1)[0].rsplit("/", 1)[-1]
    return df
  dfs = gezi.prun(_create_df, glob.glob(f'{folder}/*.json'), workers)
  df = pd.concat(dfs)
  df['source'] = df.source.apply(lambda x: x.replace('\n', BR))
  return df


# In[5]:


workers = 80
train_file = f'{FLAGS.root}/train2.fea'
if os.path.exists(train_file):
  df = pd.read_feather(train_file)
else:
  df = create_df(f'{FLAGS.root}/train', workers)


# In[6]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
if not 'input_ids' in df.columns:
  if tokenizer.convert_tokens_to_ids(BR) == tokenizer.unk_token_id:
    assert len(BR) == 1
    tokenizer.add_tokens([BR], special_tokens=False)
  input_ids_list = gezi.prun(lambda x: tokenizer(x).input_ids, df.source.values, 80, desc='tokenize')
  df['input_ids'] = input_ids_list
  df['tokens'] = gezi.prun(tokenizer.convert_ids_to_tokens, df.input_ids.values, 80, desc='convert_ids_to_tokens')
  df.reset_index().to_feather(train_file)


# In[7]:


df


# In[9]:


tokenizer.vocab_size


# In[8]:


from gensim.models import Word2Vec


# In[16]:


# 10w using 3.14min 1 epoch for emb 256
# for emb 128 only 0.3min
def gen_w2v(df, tokenizer, name='tokens', window=16, min_count=5, emb_dim=256, limit=0):
  sentences = df[name].values
  ic(len(sentences))

  if limit:
    sentences = sentences[:limit]
    ic(len(sentences))
    name = name + f'.limit{limit}'
  monitor = gezi.MonitorCallback(name) 
  w2v = Word2Vec(sentences, vector_size=emb_dim, window=window, min_count=min_count, sg=1, workers=cpu_count(), epochs=10, callbacks=[monitor])
  
  root = f'../input/w2v/{emb_dim}'
  ofile = f'{root}/{name}.pkl'
  gezi.try_mkdir(os.path.dirname(ofile))
  gezi.save(w2v, ofile)
  emb = np.random.uniform(-0.05, 0.05,(tokenizer.vocab_size, emb_dim))
  for i in range(tokenizer.vocab_size):
    token = tokenizer.convert_ids_to_tokens(i)
    if token in w2v.wv:
      emb[i] = w2v.wv[token]
  ofile = f'{root}/{name}.npy'
  np.save(ofile, emb)
  return w2v


# In[17]:


gen_w2v(df, tokenizer, emb_dim=128, limit=10000)



