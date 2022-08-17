# DialogPrompt
# Copyright 2022-present NAVER Corp.
# BSD 3-clause

import torch
import torch.nn as nn
import torch.optim as optim

import os
from os import path as path
import numpy as np
import random
import sys
from tqdm import trange
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules

from transformers import (GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config)
from modules import MaskableLSTM

class PromptEncoder(torch.nn.Module):
    '''
    Conditional Prompt Encoder which Generates Prompt Encodings conditioned on Utterance (using Encoder-Decoder)
    '''
    def __init__(self, num_trigs, base_model_name, init_embedding):
        super().__init__()
        self.num_trigs = num_trigs
        self.seq_indices = torch.arange(num_trigs).long() 
        self.config = GPT2Config.from_pretrained(base_model_name, cache_dir='./cache/')
        self.config.vocab_size = num_trigs
        self.config.n_positions = num_trigs
        self.config.n_ctx = num_trigs
        self.config.add_cross_attention=True
        #self.config = GPT2Config(vocab_size=num_trigs, n_positions=num_trigs, add_cross_attention=True)
        self.transformer = GPT2Model(self.config)
        with torch.no_grad():
            self.transformer.wte.weight[:num_trigs,:] = init_embedding.weight[5:num_trigs+5,:].data
            
    def forward(self, context_hiddens, context_attn_mask):
        device = context_attn_mask.device
        batch_size, maxlen = context_attn_mask.size()
        prompt_tokens = self.seq_indices[None, :].expand(batch_size, -1).to(device)
        output = self.transformer(input_ids = prompt_tokens, encoder_hidden_states=context_hiddens, encoder_attention_mask=context_attn_mask)
        # a list of num_layers tensors, with each of a size [2, batch_size, num_heads, seq_len, embed_size_per_head]
        return output.past_key_values 
    
class PromptEncoderLite(torch.nn.Module):
    '''
    Conditional Prompt Encoder which Generates Prompt Encodings conditioned on Utterance (using Encoder-Decoder)
    '''
    def __init__(self, num_trigs, base_model_name, init_embedding):
        super().__init__()
        self.num_trigs = num_trigs
        self.seq_indices = torch.arange(num_trigs).long() 
        self.config = GPT2Config.from_pretrained(base_model_name, cache_dir='./cache/')
        self.config.vocab_size = num_trigs
        self.config.n_positions = num_trigs
        self.config.n_ctx = num_trigs
        self.config.add_cross_attention=True
        self.config.n_heads = 1
        self.config.n_layers = 4
        
        #self.config = GPT2Config(vocab_size=num_trigs, n_positions=num_trigs, add_cross_attention=True)
        self.transformer = GPT2Model(self.config)
        with torch.no_grad():
            self.transformer.wte.weight[:num_trigs,:] = init_embedding.weight[5:num_trigs+5,:].data
            
        self.mlp_head = nn.Sequential(nn.Linear(self.config.n_embd, self.config.n_embd),
                                      nn.Tanh(),
                                      nn.Linear(self.config.n_embd, self.config.n_layer * 2 * self.config.n_embd)
                                     )
            
    def forward(self, context_hiddens, context_attn_mask):
        device = context_attn_mask.device
        batch_size, maxlen = context_attn_mask.size()
        prompt_tokens = self.seq_indices[None, :].expand(batch_size, -1).to(device)
        output = self.transformer(input_ids = prompt_tokens, encoder_hidden_states = context_hiddens, encoder_attention_mask = context_attn_mask)
        # a list of num_layers tensors, with each of a size [2, batch_size, num_heads, seq_len, embed_size_per_head]
#        prompt_encoding =   # [batch_size x n_trigs x n_embd]
        
        
        past_key_values = self.mlp_head(prompt_encoding)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.config.n_layer * 2, self.config.n_head,
                                               self.config.n_embd//self.config.n_head)
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # a list of num_layers tensors, with each of a size [2, batch_size, num_heads, seq_len, embed_size_per_head]
        return past_key_values 
    
class PromptEncoderPast(torch.nn.Module):
    '''
    Conditional Prompt Encoder which Generates Prompt Encodings conditioned on Utterance (using Past key values)
    '''
    def __init__(self, num_trigs, base_model_name, init_embedding):
        super().__init__()
        self.num_trigs = num_trigs
        self.seq_indices = torch.arange(num_trigs).long() 
        self.config = GPT2Config.from_pretrained(base_model_name, cache_dir='./cache/')
        self.config.vocab_size = num_trigs
        self.transformer = GPT2Model(self.config)
        with torch.no_grad():
            self.transformer.wte.weight = init_embedding.weight[5:num_trigs+5,:].data

    def forward(self, context_past, context_attn_mask):
        device = context_attn_mask.device
        batch_size, maxlen = context_attn_mask.size()
        prompt_tokens = self.seq_indices[None, :].expand(batch_size, -1).to(device)
        prefix_attn = torch.ones((batch_size, self.num_trigs)).long().to(device)
        attn_mask = torch.cat((context_attn_mask, prefix_attn), 1)
        
        output = self.transformer(input_ids = prompt_tokens, past_key_values=context_past, attention_mask=attn_mask)
        # a list of num_layers tensors, with each of a size [2, batch_size, num_heads, seq_len, embed_size_per_head]
        past_key_values = output.past_key_values 
        past_key_values_prompt = tuple([tuple([pkvv[:,:,-self.num_trigs:,:] for pkvv in pkv]) for pkv in past_key_values])
        return past_key_values_prompt
    
class PromptEncoderLSTM(torch.nn.Module):
    '''
    Conditional Prompt Encoder which Generates Prompt Encodings conditioned on Utterance (using LSTM)
    '''
    def __init__(self, num_trigs, base_model_name, init_embedding, mid_dim=512):
        super().__init__()
        self.num_trigs = num_trigs
        self.config = GPT2Config.from_pretrained(base_model_name, cache_dir='./cache/')
        self.seq_indices = torch.arange(num_trigs).long()
        self.embedding = torch.nn.Embedding(num_trigs, self.config.n_embd)
        with torch.no_grad():
            self.embedding.weight[:num_trigs+1,:] = init_embedding.weight[5:num_trigs+5,:].data
            
        self.lstm_head = MaskableLSTM(input_size=self.config.n_embd,
                                       hidden_size=self.config.n_embd // 2,
                                       num_layers=2,
                                       dropout=0.1,
                                       bidirectional=True,
                                       batch_first=True) 
        self.mlp_head = nn.Sequential(nn.Linear(self.config.n_embd, mid_dim),
                                      nn.Tanh(),
                                      nn.Linear(mid_dim, self.config.n_layer * 2 * self.config.n_embd)
                                     )
        
    def forward(self, context_hids, attn_mask):
        device = context_hids.device
        batch_size, maxlen, _ = context_hids.size()
        prompt_tokens = self.seq_indices.unsqueeze(0).expand(batch_size, -1).to(device)
        prompt_embds = self.embedding(prompt_tokens) # [batch x n_trigs x dim]
        lstm_inputs = torch.cat((prompt_embds, context_hids), dim=1) # [batch_size x (n_trigs + ctx_len) x dim]
        prefix_attn = torch.ones(batch_size, self.num_trigs).long().to(attn_mask.device)
        attn_mask = torch.cat((prefix_attn, attn_mask), 1)
        #############
        # NOTE: 1. padding tokens for lstm, 2. only backward when putting prompts as prefix 
        lstm_out,_ = self.lstm_head(lstm_inputs, attn_mask) # [batch_size x (n_trigs+ctx_len) x dim]
        
        prompt_encoding = lstm_out[:,:self.num_trigs,:] # [batch_size x n_trigs x n_embd]
        past_key_values = self.mlp_head(prompt_encoding)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.config.n_layer * 2, self.config.n_head,
                                               self.config.n_embd//self.config.n_head)
        #past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # a list of num_layers tensors, with each of a size [2, batch_size, num_heads, seq_len, embed_size_per_head]
        return past_key_values 
    

class PromptEncoderMLP(torch.nn.Module):
    '''
    A simple extension of Prefix-tuning, a conditional mlp
    '''
    def __init__(self, num_trigs, base_model_name, init_embedding, mid_dim=512):
        super().__init__()
        self.num_trigs = num_trigs
        self.config = GPT2Config.from_pretrained(base_model_name, cache_dir='./cache/')
        self.seq_indices = torch.arange(num_trigs).long()
        self.embedding = torch.nn.Embedding(num_trigs, self.config.n_embd)
        with torch.no_grad():
            self.embedding.weight[:num_trigs,:] = init_embedding.weight[5:num_trigs+5,:].data
        #self.embedding = nn.Embedding.from_pretrained(init_embedding.weight.data, freeze=False)
        
        self.control_trans = nn.Sequential(
                nn.Linear(self.config.n_embd, mid_dim),
                nn.Tanh(),
                nn.Linear(mid_dim, self.config.n_layer * 2 * self.config.n_embd)
        )
        
    def forward(self, context_hids, attn_mask):
        device = self.embedding.weight.device
        batch_size, ctx_len, _ = context_hids.size()
        context_encoding = context_hids.mean(1) # mean pooling
        context_encoding_expand = context_encoding[:,None,:].expand(-1, self.num_trigs, -1)
        input_tokens = self.seq_indices.unsqueeze(0).expand(batch_size, -1).to(device)
        input_embedding = self.embedding(input_tokens)
        
        past_key_values = self.control_trans(input_embedding+context_encoding_expand) #bsz, seqlen, layer*2*n_head*emb
        bsz, prompt_len, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, prompt_len, self.config.n_layer * 2, self.config.n_head,
                                               self.config.n_embd//self.config.n_head)
        #past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # a list of num_layers tensors, with each of a size [2, batch_size, num_heads, seq_len, embed_size_per_head]
        return past_key_values     
    

class DialogPrompt(nn.Module):
    '''
    Conditional Promt Tuning for Dialogue.
    Adopted from PADA: https://arxiv.org/pdf/2102.12206.pdf
    Issue1: padding tokens in LSTM input?
    Issue2: target length is set to 20 at most, but the generation step can generate 30 token at most.
    '''
    def __init__(self, args):
        super(DialogPrompt, self).__init__()   
        base_model_name='gpt2-medium' if args.model_size =='medium' else 'gpt2'
        base_model_name ='gpt2-large' if args.model_size == 'large' else base_model_name
        base_model_name ='gpt2-xl' if args.model_size == 'xlarge' else base_model_name
        
        self.num_trigs = args.num_trig_tokens
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            base_model_name if args.language =='english' else 'gpt2-chinese', cache_dir='./cache/')
        
        self.gpt = GPT2LMHeadModel.from_pretrained(base_model_name, cache_dir='./cache/')
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        # freeze gpt2's parameters, only update prompt encoder
        for param in self.gpt.parameters():
            param.requires_grad = False
        print(f"number of basic parameters: {sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)}")
        self.prompt_encoder = PromptEncoder(args.num_trig_tokens, base_model_name, self.gpt.get_input_embeddings())
        print(f"number of additional parameters: {sum(p.numel() for p in self.prompt_encoder.parameters() if p.requires_grad)}")
        
    def init_weights(self, m):# Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.08, 0.08)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)  
    
    @classmethod        
    def from_pretrained(self, model_dir):
        self.tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=args.do_lower_case)
        self.prompt_encoder = prompt_encoder.from_pretrained(model_dir)
        self.gpt = GPTForPreTraining.from_pretrained(model_dir)
        
    def save_pretrained(self, model_dir):   
        def save_module(model, save_path):       
            torch.save(model_to_save.state_dict(), save_path)
        def make_list_dirs(dir_list):
            for dir_ in dir_list: os.makedirs(dir_, exist_ok=True)
        make_list_dirs([path.join(model_dir, name) for name in ['tokenizer', 'gpt']])        
        model_to_save = self.module if hasattr(self, 'module') else self
        model_to_save.tokenizer.save_pretrained(path.join(model_dir,'tokenizer'))
        model_to_save.gpt.save_pretrained(path.join(model_dir, 'gpt'))

    def forward(self, input_ids, attn_mask, lm_labels):
        self.train()
        batch_size, max_seq_len = input_ids.size() #context: [batch_size x seq_len]  
        context_attn_mask = attn_mask.clone() 
        context_attn_mask[lm_labels>0]=0 # avoid attending to responses
        context_encoding = self.gpt.transformer(input_ids, None, context_attn_mask)
        past_key_values_prompt = self.prompt_encoder(context_encoding[0], context_attn_mask)
        prefix_attn = torch.ones(batch_size, self.num_trigs).long().to(attn_mask.device)
        attn_mask = torch.cat((prefix_attn, attn_mask), 1)
        output = self.gpt(
            input_ids=input_ids, 
            past_key_values=past_key_values_prompt, 
            attention_mask=attn_mask,
            labels=lm_labels
        )
        loss, logits = output.loss, output.logits
        
        return {'loss':loss}
    
    def validate(self, input_ids, attn_mask, lm_labels):
        self.eval()
        batch_size, max_seq_len = input_ids.size() #context: [batch_size x seq_len]  
        context_attn_mask = attn_mask.clone() 
        context_attn_mask[lm_labels>0]=0 # avoid attending to responses
        context_encoding = self.gpt.transformer(input_ids, None, context_attn_mask)
        past_key_values_prompt = self.prompt_encoder(context_encoding[0], context_attn_mask)
        prefix_attn = torch.ones(batch_size, self.num_trigs).long().to(attn_mask.device)
        attn_mask = torch.cat((prefix_attn, attn_mask), 1)
        output = self.gpt(
            input_ids=input_ids, 
            past_key_values=past_key_values_prompt, 
            attention_mask=attn_mask,
            labels=lm_labels
        )
        loss, logits = output.loss, output.logits
        return loss.item()

    def generate(self, input_batch, max_len=30, num_samples=1, mode='sample'): 
        """
        Using huggineface's default generator. 
        
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids, attn_mask = [t.to(device) for t in input_batch[:2]]   
        batch_size, ctx_len = input_ids.size()
        lm_labels = input_batch[2]
        lm_labels[attn_mask==0]=0 # remove -100's in padding positions
        context, context_attn_mask = input_ids[lm_labels==-100][None, :-1], attn_mask[lm_labels==-100][None,:-1]
        ground_truth = lm_labels[lm_labels>0][None,:].numpy()
        
        context = context.repeat(num_samples, 1)
        context_attn_mask = context_attn_mask.repeat(num_samples, 1)
        context_encoding = self.gpt.transformer(context, None, context_attn_mask)
        
        past_key_values_prompt = self.prompt_encoder(context_encoding[0], context_attn_mask)
        prefix_attn = torch.ones((batch_size, self.num_trigs), device=device).long()
        context_attn_mask = torch.cat((prefix_attn, context_attn_mask), 1)
        context_encoding = self.gpt(context, past_key_values = past_key_values_prompt, attention_mask=context_attn_mask)
        predictions = self.gpt.generate( # [(batch_size*num_samples) x seq_len]
            bos_token_id = self.tokenizer.eos_token_id,
            past = context_encoding.past_key_values,
            attention_mask= torch.cat((context_attn_mask, torch.ones((batch_size, 1), device=device).long()),1),
            max_length= max_len, 
            do_sample = True, 
            early_stopping=True,
            num_return_sequences=num_samples,
            pad_token_id = 0,
        )
        
        # to numpy
        sample_words = predictions.data.cpu().numpy()
        sample_lens = np.array([predictions.size(1)]) 
        context = context.data.cpu().numpy()
        return sample_words, sample_lens, context, ground_truth # nparray: [repeat x seq_len]
    

