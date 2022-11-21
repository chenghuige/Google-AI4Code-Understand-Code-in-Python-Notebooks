v1 = 0
mns1 = [

]

v2 = 0
mns2 = [    
]

v3 = 6
mns3 = [    
  # 'deberta-v3-small.flag-context2-aug.n_context-40.cls_loss_rate-0.1',
  # 'deberta-v3-small.flag-context2-rand.n_context-40.cls_loss_rate-0.1',
  
  # 'deberta-v3-small.flag-context2-aug.n_context-40.cls_loss_rate-0.1.eval.p9.d001',
  # 'deberta-v3-small.flag-context2-rand.n_context-40.cls_loss_rate-0.1.eval.p9',
  
  # 'deberta-v3-small.flag-context2-aug.n_context-40.cls_loss_rate-0.1.eval.p9.d001',
  
  'all-mpnet-base-v2.flag-pairwise13-2',
  'all-MiniLM-L12-v2.flag-pairwise13-3',
  
  #'deberta-v3-small.flag-context2-aug.n_context-40.cls_loss_rate-0.1.eval.mpnet.p12',
  #'deberta-v3-small.flag-context2-rand.n_context-40.cls_loss_rate-0.1.eval.mpnet.p12',
]


mns = mns1 + mns2 + mns3
v = v3

weights_dict = {}

# weights = [1] * len(mns)
def get_weight(x):
  # return weights_dict.get(x, 0)
  # return weights_dict[x]
  return 1

if all(x in weights_dict for x in mns):
  weights = [weights_dict[x] for x in mns]
else:
  weights = [get_weight(x) for x in mns]
weights.extend([1] * 100)
ic(list(zip(mns, weights)), len(mns))

SAVE_PRED = 0
