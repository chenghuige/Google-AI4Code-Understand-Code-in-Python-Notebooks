#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:57.220407
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.dummy import active_children
from secrets import token_bytes

from gezi.common import * 

from transformers import AutoModel, AutoTokenizer, AutoConfig
# TODO FIXME is not BertOnlyMLMHead for deberta or mpnet ..
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from lele.nlp.pretrain.masklm import MaskLM
from transformers import (
  AutoModelForMaskedLM,
  DataCollatorForLanguageModeling,
)

from src.config import *
from src.preprocess import *

class Model(nn.Module):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)  
            
    self.backbone, self.tokenizer = self.init_backbone(FLAGS.backbone)
    
    config = self.backbone.config
    dim = config.hidden_size    
    Linear = nn.Linear
    
    if FLAGS.seq_encoder:
      RNN = getattr(nn, FLAGS.rnn_type)
      if not FLAGS.rnn_bi:
        self.seq_encoder = RNN(dim, dim, FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=False, batch_first=True)
      else:
        if not FLAGS.rnn_double_dim:
          self.seq_encoder = RNN(dim, int(dim / 2), FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=True, batch_first=True)
        else:
          self.seq_encoder = RNN(dim, dim, FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=True, batch_first=True)
          dim *= 2
      if FLAGS.use_embs:
        if FLAGS.seq_encoder2:
          self.seq_encoder2 = RNN(dim, int(dim / 2), FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=True, batch_first=True)
    
    self.pooling = lele.layers.Pooling(FLAGS.pooling, dim)
    self.pooling2 = self.pooling if FLAGS.share_pooling else lele.layers.Pooling(FLAGS.pooling, dim)
    if FLAGS.n_markdowns > 0 and (not FLAGS.share_poolings):
      self.poolings = nn.ModuleList([lele.layers.Pooling(FLAGS.pooling, dim) for _ in range(FLAGS.n_markdowns)])
    dim = self.pooling.output_dim
    
    if FLAGS.mlp:
      self.mlp = lele.layers.MLP(dim, [dim], activation=FLAGS.mlp_activation)
      self.mlp2 = self.mlp if FLAGS.share_mlp else lele.layers.MLP(dim, [dim], activation=FLAGS.mlp_activation)
    
    if FLAGS.layernorm:
      self.layer_norm = torch.nn.LayerNorm(dim, eps=FLAGS.layernorm_eps)
    
    if FLAGS.use_context2:
      dim2 = FLAGS.emb_dim
      self.word_emb = lele.get_embedding(self.tokenizer.vocab_size + 10, FLAGS.emb_dim, embedding_weight=FLAGS.emb_weight, padding_idx=0)
      if not FLAGS.use_din:
        self.code_pooling = lele.layers.Pooling('latt', dim2)
      else:
        import deepctr_torch
        DinAtt = deepctr_torch.layers.sequence.AttentionSequencePoolingLayer
        self.code_pooling = DinAtt()
      self.markdown_pooling = lele.layers.Pooling('latt', dim2)
      self.seqs_encoder = lele.layers.TimeDistributed(self.code_pooling)
      self.codes_pooling = lele.layers.Pooling('latt', dim2)
      
      # markdown_encode, codes_encode
      dim2 *= 2
      dim = dim + dim2 if FLAGS.bert_encode else dim2
      if FLAGS.use_dot:
        dim += FLAGS.n_context2
  
    if FLAGS.use_markdown_frac:
      dim += 1
    if FLAGS.add_info:
      dim += 3
      
    odim = 1 if FLAGS.loss_method != 'softmax' else FLAGS.num_classes
    if FLAGS.sbert:
      dim = self.pooling.output_dim * 3
      odim = 1
    if FLAGS.pairwise and (not FLAGS.two_tower):
      dim = self.pooling.output_dim
      odim = 1
      
    self.dense = Linear(dim, odim)
    if FLAGS.context_match_loss_rate > 0:
      self.context_match_dense = Linear(dim, odim)
    if FLAGS.cls_loss_rate > 0:
      self.cls_dense = Linear(dim, FLAGS.num_classes)
    if FLAGS.global_loss_rate > 0:
      self.global_dense = Linear(dim, odim)
    if FLAGS.local_loss_rate > 0:
      self.local_dense = Linear(dim, odim)
      
    if FLAGS.dtemperature > 0:
      self.temperature = nn.Parameter(torch.tensor(FLAGS.dtemperature))

    if FLAGS.lm_train:
      self.lm = MaskLM(tokenizer=self.tokenizer)
      # self.lm = DataCollatorForLanguageModeling(self.tokenizer)
      config.vocab_size = len(self.tokenizer)
      if FLAGS.lm_header:
        self.lm_header = BertOnlyMLMHead(config)
      
    if FLAGS.opt_fused and FLAGS.fused_layernorm:
      lele.replace_with_fused_layernorm(self)
  
  def init_backbone(self, backbone_name, model_dir=None, load_weights=False):
    backbone_dir = f'{os.path.dirname(FLAGS.model_dir)}/{FLAGS.backbone_dir}' if FLAGS.backbone_dir is not None else None
    model_dir = model_dir or backbone_dir or FLAGS.model_dir
    
    try:
      config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
      # logger.warning(e)
      try:
        config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
      except Exception:
        config = AutoConfig.from_pretrained(backbone_name.lower(), trust_remote_code=True)
  
    if FLAGS.type_vocab_size:
      config.update({'type_vocab_size': FLAGS.type_vocab_size})    
  
    self.config = config
    
    ic(model_dir, backbone_name, os.path.exists(f'{model_dir}/model.pt'), os.path.exists(f'{model_dir}/pytorch_model.bin'))
    
    if FLAGS.lm_train and (not FLAGS.lm_header):
      backbone = AutoModelForMaskedLM.from_pretrained(FLAGS.backbone, trust_remote_code=True)
    else:
      if os.path.exists(f'{model_dir}/model.pt') and (not os.path.exists(f'{model_dir}/pytorch_model.bin')):
        try:
          backbone = AutoModel.from_config(config, trust_remote_code=True)
          logger.info(f'backbone init from config')
        except Exception as e:
          # logger.warning(e)
          backbone = AutoModel.from_pretrained(backbone_name, config=config, trust_remote_code=True)
          logger.info(f'backbone init from {backbone_name}')
      else:
        try:
          backbone = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)
          logger.info(f'backbone init from {model_dir}')
        except Exception as e:
          # logger.warning(e)
          backbone = AutoModel.from_pretrained(backbone_name, config=config, trust_remote_code=True)
          logger.info(f'backbone init from {backbone_name}')
 
    if os.path.exists(f'{model_dir}/config.json'):
      try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
      except Exception:
        tokenizer = get_tokenizer(backbone_name)
    else:
      tokenizer = get_tokenizer(backbone_name)
      
    backbone.resize_token_embeddings(len(tokenizer)) 
    
    if FLAGS.unk_init:
      try:
        # TODO bart not ok
        unk_id = tokenizer.unk_token_id
        with torch.no_grad(): 
          word_embeddings = lele.get_word_embeddings(backbone)
          br_id = tokenizer.convert_tokens_to_ids(BR)
          word_embeddings.weight[br_id, :] = word_embeddings.weight[unk_id, :]  
      except Exception as e:
        ic(e)            
        
    if FLAGS.freeze_emb:
      lele.freeze(lele.get_word_embeddings(backbone))    

    if FLAGS.gradient_checkpointing:
      backbone.gradient_checkpointing_enable()
    return backbone, tokenizer 
  
  def encode(self, inputs, idx=None):
    backbone, tokenizer = self.backbone, self.tokenizer
    
    m = {
      'input_ids': inputs['input_ids'],
      'attention_mask': inputs['attention_mask'],
    } 
    
    if not FLAGS.disable_token_type_ids:
      if 'token_type_ids' in inputs:
        m['token_type_ids'] = inputs['token_type_ids']
    
    x = backbone(**m)[0]
    if FLAGS.seq_encoder:
      x, _ = self.seq_encoder(x)
    
    if FLAGS.use_embs:
      if not FLAGS.seq_encoder2:
        x = torch.cat([x, 
                      inputs['code_embs'].view(x.shape[0], -1, x.shape[-1]),
                      inputs['markdown_embs'].view(x.shape[0], -1, x.shape[-1]),
                      ], 1)  
      else:
        # x2, _ = self.seq_encoder2(torch.cat([inputs['code_embs'], inputs['markdown_embs']], 1))
        # x = torch.cat([x, x2], 1)
        x = torch.cat([x, 
                      inputs['code_embs'].view(x.shape[0], -1, x.shape[-1]),
                      inputs['markdown_embs'].view(x.shape[0], -1, x.shape[-1]),
                      ], 1)  
        x, _ = self.seq_encoder2(x)
    return x
  
  def fake_pred(self, inputs, requires_grad=False):
    input_ids =  inputs['input_ids'] if not 0 in inputs else inputs[0]['input_ids']
    bs = input_ids.shape[0] 
    return torch.rand([bs, 1], device=input_ids.device, requires_grad=requires_grad)
    
  def forward(self, inputs):
    if FLAGS.fake_infer:
      return {
        'pred': self.fake_pred(inputs, requires_grad=self.training),
      }
    
    res = {}
    
    if FLAGS.lm_train:
      if not FLAGS.lm_hug_inputs:
        input_ids, lm_label = self.lm.mask_tokens(inputs['input_ids'])
      else:
        input_ids, lm_label = inputs['input_ids'], inputs['labels']
      # input_ids, lm_label = self.lm.torch_mask_tokens(inputs['input_ids'])
      # ic(input_ids, lm_label)
      # ic(((input_ids == 0).float() * (lm_label != -100).float()).mean())
      # res['label'] = inputs['attention_mask'] * lm_label + (1 - inputs['attention_mask']) * -100
      # lm_label[input_ids != self.tokenizer.mask_token_id] = -100 
      res['label'] = lm_label
      # inputs['input_ids'] = input_ids
      m = {
        'input_ids': input_ids,
        'attention_mask': inputs['attention_mask'],
      } 
      if 'token_type_ids' in inputs:
        m['token_type_ids'] = inputs['token_type_ids']
      
      if FLAGS.lm_header:
        res['pred'] = self.lm_header(self.encode(m)) 
      else:
        res['pred'] = self.encode(m)
      return res  
    
    if FLAGS.dtemperature > 0:
      bs = inputs['input_ids'].shape[0]
      res['temperature'] = torch.ones([bs, 1], device=inputs['input_ids'].device)
      res['temperature'] *= self.temperature
    
    pooling_mask_key = 'attention_mask' 
    if FLAGS.pairwise and FLAGS.n_pairwise_context > 0:
      pooling_mask_key = FLAGS.pooling_mask
    if FLAGS.bert_encode and not (FLAGS.pairwise and (not FLAGS.two_tower) and (not FLAGS.dynamic_codes)):
      xs = self.encode(inputs)
      if FLAGS.dynamic_token_types:
        pooling_mask = (inputs['token_type_ids'] < 3).int()
      else:
        pooling_mask = inputs[pooling_mask_key]
      if FLAGS.use_embs:
        pooling_mask = torch.cat([pooling_mask, 
                                  inputs['codes_mask'],
                                  inputs['markdowns_mask'],
                                  ], -1)
      x = self.pooling(xs, pooling_mask)
      if FLAGS.n_markdowns > 0:
        if not FLAGS.dynamic_token_types:
          if not FLAGS.share_poolings:
            xs_ = [self.poolings[i](xs, pooling_mask) for i in range(FLAGS.n_markdowns)]
          else:
            xs_ = [self.pooling(xs, pooling_mask) for i in range(FLAGS.n_markdowns)]
        else:
          xs_ = []
          pooling_mask_ = (inputs['token_type_ids'] < 2).int()
          for i in range(FLAGS.n_markdowns):
            pooling_mask = (inputs['token_type_ids'] == 3 + i).int() + pooling_mask_
            if FLAGS.use_embs:
              pooling_mask = torch.cat([pooling_mask, 
                                        inputs['codes_mask'],
                                        inputs['markdowns_mask'],
                                        ], -1)
            if not FLAGS.share_poolings:
              xs_.append(self.poolings[i](xs, pooling_mask))
            else:
              xs_.append(self.pooling(xs, pooling_mask))
          
      if FLAGS.mlp:
        x = self.mlp(x)
      if FLAGS.layernorm:
        x = self.layer_norm(x)
      if FLAGS.l2norm:
        x = F.normalize(x, p=2, dim=1)

    if FLAGS.dump_emb or FLAGS.pairwise_eval or (FLAGS.pairwise and not 'codes_input_ids' in inputs and FLAGS.two_tower):
      res['pred'] = self.fake_pred(inputs)
      res['emb'] = x
      return res
    
    if FLAGS.pairwise and (not FLAGS.dynamic_codes):
      if not FLAGS.two_tower:
        logits = []
        for i in range(1 + FLAGS.num_negs):
          m = {
            'input_ids': inputs[f'input_ids{i}'],
            'attention_mask': inputs[f'attention_mask{i}'],
            'token_type_ids': inputs[f'token_type_ids{i}']
          }
          x = self.encode(m)
          x = self.pooling(x, m[pooling_mask_key])
          logit = self.dense(x)
          logits.append(logit)
        res['pred'] = torch.cat(logits, -1)
        lele.inf_mask_(res['pred'], inputs['label_mask'])   
        return res
      else:
        bs = inputs['codes_input_ids'].shape[0]
        m = {
          'input_ids': inputs['codes_input_ids'].view(bs * (1 + FLAGS.num_negs), -1),
          'attention_mask': inputs['codes_attention_mask'].view(bs * (1 + FLAGS.num_negs), -1),
          'token_type_ids': inputs['codes_token_type_ids'].view(bs * (1 + FLAGS.num_negs), -1),
        }
        x_ = self.encode(m)
        x_ = self.pooling2(x_, m[pooling_mask_key])
        if FLAGS.mlp:
          x_ = self.mlp2(x_)
        if FLAGS.layernorm:
          x_ = self.layer_norm(x_)
        if FLAGS.l2norm:
          x_ = F.normalize(x_, p=2, dim=1)
        x_ = x_.view(bs, 1 + FLAGS.num_negs, -1)
        if FLAGS.sbert:
          xs_ = torch.split(x_, 1, dim=1)  
          logits = []
          for x_ in xs_:
            x_ = x_.squeeze(1)
            logit = self.dense(torch.cat([x, x_, (x - x_).abs()], -1))
            logits.append(logit)
          res['pred'] = torch.cat(logits, -1)
        else:
          res['pred'] = x.unsqueeze(1).bmm(x_.transpose(2, 1)).squeeze(1)
          if FLAGS.temperature > 0:
            res['pred'] /= FLAGS.temperature
          elif FLAGS.dtemperature > 0:
            res['pred'] /= self.temperature
        lele.inf_mask_(res['pred'], inputs['label_mask'])   
        
        if FLAGS.pairwise_markdowns:
          m = {
            'input_ids': inputs['markdowns_input_ids'].view(bs * FLAGS.n_markdowns, -1),
            'attention_mask': inputs['markdowns_attention_mask'].view(bs * FLAGS.n_markdowns, -1),
            'token_type_ids': inputs['markdowns_token_type_ids'].view(bs * FLAGS.n_markdowns, -1),
          }
          
          x_ = self.encode(m)
          x_ = self.pooling(x_, m[pooling_mask])
          if FLAGS.mlp:
            x_ = self.mlp(x_)
          if FLAGS.layernorm:
            x_ = self.layer_norm(x_)
          if FLAGS.l2norm:
            x_ = F.normalize(x_, p=2, dim=1)
          x_ = x_.view(bs, FLAGS.n_markdowns, -1)
          res['markdown_pred'] = x.unsqueeze(1).bmm(x_.transpose(2, 1)).squeeze(1)
          if FLAGS.temperature > 0:
            res['markdown_pred'] /= FLAGS.temperature
          elif FLAGS.dtemperature > 0:
            res['markdown_pred'] /= self.temperature
          ## -inf will cause nan
          # lele.inf_mask_(res['markdown_pred'], inputs['markdowns_label_mask'])      
        
        return res
    else:
      if FLAGS.dynamic_codes:
        res['pred']  = self.dense(x)
        return res
      
      if FLAGS.use_context2:
        # ic(inputs['input_ids2'].max(), inputs['markdown_input_ids'].max())
        x_codes = self.word_emb(inputs['input_ids2'])
        x_codes = x_codes.view(x_codes.shape[0], -1, FLAGS.max_context2_len, x_codes.shape[-1])
        # bs, n_code_cells, out_dim
        x_codes = self.seqs_encoder(x_codes)
        x_code = self.codes_pooling(x_codes)
        
        x_markdowns = self.word_emb(inputs['markdown_input_ids'])
        x_markdown = self.markdown_pooling(x_markdowns, inputs['markdown_attention_mask'])
        
        # TODO add dot of markdown and code
        if FLAGS.bert_encode:
          x = torch.cat([x, x_markdown, x_code], -1)
        else:
          x = torch.cat([x_markdown, x_code], -1)
        
        if FLAGS.use_dot:
          if FLAGS.l2_norm:
            x_markdown = torch.nn.functional.normalize(x_markdown, dim=-1)
            x_codes = torch.nn.functional.normalize(x_codes, dim=-1)
          intersect = x_markdown.unsqueeze(1).bmm(x_codes.transpose(-2, -1)).squeeze(1)
          x = torch.cat([x, intersect], -1)
      
      def post_processing(x):
        if FLAGS.use_markdown_frac:
          x = torch.cat([x, inputs['markdown_frac'].unsqueeze(-1).float()], -1)
        
        if FLAGS.add_info:
          x = torch.cat([
                          x, 
                          1. / (inputs['n_cell'].unsqueeze(-1).float() + 1.), 
                          1. / (inputs['n_code_cell'].unsqueeze(-1).float() + 1.), 
                          1. / (inputs['n_markdown_cell'].unsqueeze(-1).float() + 1.)
                        ], 
                        -1)
        return x

      x = post_processing(x)
      res['pred'] = self.dense(x)
      if FLAGS.context_match_loss_rate > 0:
        res['context_match_pred'] = self.context_match_dense(x)
      if FLAGS.cls_loss_rate > 0:
        res['cls_pred'] = self.cls_dense(x)
      if FLAGS.global_loss_rate > 0:
        res['global_pred'] = self.global_dense(x)
      if FLAGS.local_loss_rate > 0:
        res['local_pred'] = self.local_dense(x)
      
      if FLAGS.n_markdowns:
        xs_ = [post_processing(x) for x in xs_]
        x = torch.stack(xs_, 1)
        res['preds'] = self.dense(x)
        res['preds'] = torch.cat([res['pred'].unsqueeze(1), res['preds']], 1)
        if FLAGS.cls_loss_rate > 0:
          res['cls_preds'] = self.cls_dense(x)
          # ic(res['cls_pred'].shape, res['cls_preds'].shape)
          res['cls_preds'] = torch.cat([res['cls_pred'].unsqueeze(1), res['cls_preds']], 1)
          # ic(res['cls_preds'].shape)
        if FLAGS.global_loss_rate > 0:
          res['global_preds'] = self.global_dense(x)
          res['global_preds'] = torch.cat([res['global_pred'].unsqueeze(1), res['global_preds']], 1)
        if FLAGS.local_loss_rate > 0:
          res['local_preds'] = self.local_dense(x)
          res['local_preds'] = torch.cat([res['local_pred'].unsqueeze(1), res['local_preds']], 1)
      if FLAGS.use_sigmoid: # worse
        res['pred'] = torch.sigmoid(res['pred'])

      if not 'pred' in res:
        res['pred'] = self.fake_pred(inputs)
      return res

  def get_loss_fn(self):
    from src.torch.loss import calc_loss
    return calc_loss

  def get_valid_fn(self):
    valid_fn = None
    if not FLAGS.lm_train:
      if FLAGS.pairwise:
        def valid_fn(y_, y, x):
          res = {}
          pred = y_['pred']
          acc = (pred.argmax(-1) == y).float().mean()
          res['acc/code'] = acc
          
          if 'markdown_pred' in y_:
            pred = (torch.sigmoid(y_['markdown_pred']) > 0.5).int()
            acc = ((pred == x['markdowns_label']) * x['markdowns_label_mask']).sum() / x['markdowns_label_mask'].sum()
            res['acc/markdown'] = acc
          
          if FLAGS.dtemperature > 0:
            res['temperature'] = y_['temperature'].mean()
          
          return res
      else:
        # FIXME seems wrong result but could not find why comparing eval.py which is ok..
        # above pairwise acc is ok
        def valid_fn(y_, y, x):
          res = {}
          pred = y_['pred']
          # if 'cls_pred' in y_:
          #   pred = (y_['pred'] + y_['cls_pred']) / 2.
          if 'local_pred' in y_:
            # NOTICE [32,1], [32,1], [32] -> [32,32]...
            pred = y_['global_pred'] + y_['local_pred'] / (x['n_code_cell'].unsqueeze(-1).int() + 1).float()
  
          if pred.shape[1] > 1:
            span = 0. if not FLAGS.cls_span else 0.5
            pred = ((torch.arange(FLAGS.num_classes, dtype=pred.dtype, device=pred.device) + span) * F.softmax(pred, -1)).sum(-1) / FLAGS.num_classes
          
          n_code_cell = x['n_code_cell'].unsqueeze(-1).int()
          match_code = x['match_code'].unsqueeze(-1).int()
          distance = ((pred * (n_code_cell + 1).float()).int() - match_code).float().abs()
          res['mae'] = (distance / (n_code_cell + 1)).mean()
          return res
        
    return valid_fn
