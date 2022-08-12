# DialogBERT
# Copyright 2021-present NAVER Corp.
# BSD 3-clause

import os
import random
from copy import deepcopy
import numpy as np
import tables
import json
import itertools
from tqdm import tqdm
import torch
import torch.utils.data as data
import logging
logger = logging.getLogger(__name__)

class DialogTransformerDataset(data.Dataset):
    """
    A base class for Transformer dataset
    """
    def __init__(self, file_path, tokenizer, 
                 min_num_utts=1, max_num_utts=7, max_utt_len=30, 
                 block_size=256, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.min_num_utts = min_num_utts #if not context_shuf and not context_masklm else 3
        self.max_num_utts = max_num_utts
        self.max_utt_len =max_utt_len
        self.block_size = block_size # segment size to train BERT. when set -1 by default, use indivicual sentences(responses) as BERT inputs.
                            # Otherwise, clip a block from the context.
        
        self.utt_masklm = utt_masklm
        self.utt_sop =utt_sop
        self.context_shuf =context_shuf
        self.context_masklm =context_masklm
        
        self.rand_utt = [tokenizer.mask_token_id]*(max_utt_len-1) + [tokenizer.sep_token_id] # update during loading
        
        # a cache to store context and response that are longer than min_num_utts
        self.cache = [[tokenizer.mask_token_id]*max_utt_len]*max_num_utts, [tokenizer.mask_token_id]*max_utt_len
        
        print("loading data...")
        table = tables.open_file(file_path)
        self.contexts = table.get_node('/sentences')[:].astype(np.long)
        #self.knowlege = table.get_node('/knowledge')[:].astype(np.long)
        self.index = table.get_node('/indices')[:]
        self.data_len = self.index.shape[0]
        
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        index = self.index[offset]
        pos_utt, ctx_len, res_len,  = index['pos_utt'], index['ctx_len'], index['res_len']
        #pos_knowl, knowl_len = index['pos_knowl'], index['knowl_len']
        
        ctx_len = min(ctx_len, self.block_size) if self.block_size>-1 else ctx_len# trunck too long context
        
        ctx_arr=self.contexts[pos_utt-ctx_len:pos_utt].tolist()
        res_arr=self.contexts[pos_utt:pos_utt+res_len].tolist()
        #knowl_arr = self.knowledge[pos_knowl:pos_knowl+knowl_len].tolist()
        
        ## split context array into utterances        
        context = []
        tmp_utt = []
        for i, tok in enumerate(ctx_arr):
            tmp_utt.append(ctx_arr[i])
            if tok == self.tokenizer.sep_token_id:
                floor = tmp_utt[0]
                tmp_utt = tmp_utt[1:] 
                utt_len = min(len(tmp_utt), self.max_utt_len) # floor is not counted in the utt length
                utt = tmp_utt[:utt_len]            
                context.append(utt)  # append utt to context          
                tmp_utt=[]  # reset tmp utt
        response = res_arr[1:] # ignore cls token at the begining            
        res_len = min(len(response),self.max_utt_len)
        response = response[:res_len-1] + [self.tokenizer.sep_token_id] 
        
        num_utts = min(len(context), self.max_num_utts)
        context = context[-num_utts:]
        
        return context, response 
    
    def list2array(self, L, d1_len, d2_len=0, d3_len=0, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''            
        def list_dim(a):
            if type(a)!=list: return 0
            elif len(a)==0: return 1
            else: return list_dim(a[0])+1
        
        if type(L) is not list:
            print("requires a (nested) list as input")
            return None
        
        if list_dim(L)==0: return L
        elif list_dim(L) == 1:
            arr = np.zeros(d1_len, dtype=dtype)+pad_idx
            for i, v in enumerate(L): arr[i] = v
            return arr
        elif list_dim(L) == 2:
            arr = np.zeros((d2_len, d1_len), dtype=dtype)+pad_idx
            for i, row in enumerate(L):
                for j, v in enumerate(row):
                    arr[i][j] = v
            return arr
        elif list_dim(L) == 3:
            arr = np.zeros((d3_len, d2_len, d1_len), dtype=dtype)+pad_idx
            for k, group in enumerate(L):
                for i, row in enumerate(group):
                    for j, v in enumerate(row):
                        arr[k][i][j] = v
            return arr
        else:
            print('error: the list to be converted cannot have a dimenson exceeding 3')
    
    def mask_words(self, utt):
        output_label = []
        tokens = [tok for tok in utt]
        for i, token in enumerate(utt):
            prob = random.random()
            if prob < 0.15 and not token in [self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]:
                prob /= 0.15                
                if prob < 0.8: 
                    tokens[i] = self.tokenizer.mask_token_id   # 80% randomly change token to mask token                
                elif prob < 0.9: 
                    tokens[i] = random.randint(5, len(self.tokenizer)-5)# 10% randomly change token to random token            
                output_label.append(token)
            else:
                output_label.append(-100)
        return tokens, output_label
                    

    def __len__(self):
        return self.data_len    
    
 
class BartDataset(DialogTransformerDataset):
    """
    
    """
    def __init__(self, file_path, tokenizer, 
                 min_num_utts=1, max_num_utts=7, max_utt_len=30, 
                 block_size=200, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False):
        # 1. Initialize file path or list of file names.
        super(BartDataset, self).__init__(
            file_path, tokenizer, min_num_utts, max_num_utts, max_utt_len, 
            block_size, utt_masklm, utt_sop, context_shuf, context_masklm
        )

    def __getitem__(self, offset):
        context, response = super().__getitem__(offset)
        context_flt, context_attn_mask= [], []
        for i, utt in enumerate(context): 
            if utt[0]==self.tokenizer.cls_token_id: utt=utt[1:] #ignore [cls] token for each utterance
            context_flt.extend(utt)
            context_attn_mask.extend([1]*len(utt))
        context = self.list2array(context_flt, self.block_size, pad_idx=self.tokenizer.pad_token_id) 
        context_attn_mask = self.list2array(context_attn_mask, self.block_size) 
        response = self.list2array(response, self.max_utt_len, pad_idx=self.tokenizer.pad_token_id)# for decoder training

        return context, context_attn_mask, response

    
    
class DialogGPTDataset(DialogTransformerDataset):
    """
    """
    def __init__(self, file_path, tokenizer, 
                 max_num_utts=4, max_utt_len=20,
                 fewshot=-1):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.max_num_utts = max_num_utts
        self.max_utt_len=max_utt_len
        
        #assert self.tokenizer.eos_token_id is not None, "must specify the eos token id of the tokenizer"
        #assert self.tokenizer.pad_token_id is not None, "must specify the pad token id of the tokenizer"
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = 0
        self.mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
        print("loading data...")
        table = tables.open_file(file_path)
        self.data = table.get_node('/sentences')[:].astype(np.long)
        self.index = table.get_node('/indices')[:]
        if fewshot>-1:
            self.index = self.index[:fewshot]
        self.data_len = self.index.shape[0]
        
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        pos_utt, ctx_len, res_len = self.index[offset]['pos_utt'], self.index[offset]['ctx_len'], self.index[offset]['res_len']
        ctx_arr=self.data[pos_utt-ctx_len:pos_utt].tolist()
        res_arr=self.data[pos_utt:pos_utt+res_len].tolist()
        
        #ctx_arr, res_arr = ctx_arr[:-1], res_arr[:-1] # remove default [eos] token
        
        ## split context array into utterances        
        context = [tok for tok in ctx_arr if tok>-1]
        context_len= min(len(context), self.max_num_utts*self.max_utt_len-1)
        context = context[-context_len:]
          
        res_len = min(len(res_arr), self.max_utt_len)
        res = res_arr[:res_len]  
        
        input_ids = context + res
        attn_mask = [1]*len(input_ids)
        lm_labels = [-100]*len(context) + res
              # NOTE: no need to shift labels since GPT will automatically do that
        input_ids = self.list2array(input_ids, self.max_utt_len*(self.max_num_utts+1), pad_idx=self.pad_token_id)   
        attn_mask = self.list2array(attn_mask, self.max_utt_len*(self.max_num_utts+1))
        lm_labels = self.list2array(lm_labels, self.max_utt_len*(self.max_num_utts+1), pad_idx=-100)
        
        return input_ids, attn_mask, lm_labels
    
   
class DialogPromptDataset(DialogTransformerDataset):
    """
    A variant of DialoGPT dataset for Prompt-based GPT2(https://arxiv.org/abs/1911.00536). 
    """
    def __init__(self, file_path, tokenizer, 
                 max_num_utts=4, max_utt_len=20,
                 prefix=True, num_trig_tok=-1):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.max_num_utts = max_num_utts
        self.max_utt_len=max_utt_len
        self.prefix = prefix # whether prepending (prefix) or appending (postfix) prompts to utterances
        self.num_trig_tok = num_trig_tok
        
        assert self.tokenizer.eos_token_id is not None, "must specify the eos token id of the tokenizer"
        #assert self.tokenizer.pad_token_id is not None, "must specify the pad token id of the tokenizer"
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        print("loading data...")
        table = tables.open_file(file_path)
        self.data = table.get_node('/sentences')[:].astype(np.long)
        self.index = table.get_node('/indices')[:]
        self.data_len = self.index.shape[0]
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        pos_utt, ctx_len, res_len = self.index[offset]['pos_utt'], self.index[offset]['ctx_len'], self.index[offset]['res_len']
        ctx_arr=self.data[pos_utt-ctx_len:pos_utt].tolist()
        res_arr=self.data[pos_utt:pos_utt+res_len].tolist()

        #ctx_arr, res_arr = ctx_arr[:-1], res_arr[:-1] # remove default [eos] tokens
        
        ## split context array into utterances           
        context, prompts = [], []
        tmp_utt = []
        for i, tok in enumerate(ctx_arr):
            tmp_utt.append(ctx_arr[i])
            if tok == self.tokenizer.eos_token_id:
                #tmp_utt = tmp_utt[2:-1] 
                utt_len = min(len(tmp_utt), self.max_utt_len) # floor is not counted in the utt length
                if self.prefix:
                    utt = [self.pad_token_id]*self.num_trig_tok + tmp_utt[:utt_len]
                else:
                    utt = tmp_utt[:utt_len] + [self.pad_token_id]*self.num_trig_tok  
                context.append(utt)  # append utt to context     
                if self.prefix:
                    prompt = [t+1 for t in range(self.num_trig_tok)] + [self.pad_token_id]*utt_len
                else:
                    prompt = [self.pad_token_id]*utt_len + [t+1 for t in range(self.num_trig_tok)]
                prompts.append(prompt)
                tmp_utt=[]  # reset tmp utt
        
        num_utts = min(len(context), self.max_num_utts)
        context = context[-num_utts:]
        prompts = prompts[-num_utts:]

        context_flt, prompts_flt = [], []
        for i, utt in enumerate(context): 
            context_flt.extend(utt)
            prompts_flt.extend(prompts[i])
        
        res_len = min(len(res_arr), self.max_utt_len)
        res = res_arr[:res_len]  
        input_ids = context_flt  + res
        prompt_ids = prompts_flt + [self.pad_token_id]*len(res)
        attn_mask = [1]*len(input_ids)
        lm_labels = [-100]*len(context_flt) + res 
               # NOTE: no need to shift labels since GPT will automatically do that
        maxlen = (self.max_utt_len+self.num_trig_tok)*self.max_num_utts + self.max_utt_len
        input_ids = self.list2array(input_ids, maxlen, pad_idx=self.pad_token_id)        
        attn_mask = self.list2array(attn_mask, maxlen)
        prompts = self.list2array(prompt_ids, maxlen, pad_idx=self.pad_token_id)
        lm_labels = self.list2array(lm_labels, maxlen, pad_idx=-100)
        return input_ids, attn_mask, prompts, lm_labels
          
    
def load_dict(filename):
    return json.loads(open(filename, "r").readline())

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs

def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()
    

if __name__ == '__main__':
    
    input_dir='./data/reddit/'
    VALID_FILE=input_dir+'train.h5'
    task = 'test_ctx'#'test_utt' # 'test_ctx'
    
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if task == 'test_utt':
        dataset=DialogTransformerDataset(VALID_FILE, tokenizer, utt_masklm=True, utt_sop=True)
    elif task == 'test_ctx':
        dataset=DialogTransformerDataset(VALID_FILE, tokenizer, context_shuf=True, context_masklm=False)
    else:
        dataset=DialogTransformerDataset(VALID_FILE, tokenizer)
    data_loader=torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
        
    if task == 'test_utt':
        k=0
        for batch in data_loader:
            response, res_bert_input, res_attn_mask, res_segment_ids, res_lm_labels, res_sop_label = batch[9:]
            k+=1
            if k>3: break
            print('response:', tokenizer.decode(response[0].numpy().tolist()))
            print(f'response:\n {response[0]}')
            print('res_bert_input:', tokenizer.decode(res_bert_input[0].numpy().tolist()))
            print(f'res_bert_input\n {res_bert_input[0]}')
            print(f'attn_mask:\n {res_attn_mask[0]}')
            print(f'segment_ids:\n {res_segment_ids[0]}')
            print(f'lm_labels:\n {res_lm_labels[0]}')
            print(f'sop_label:\n {res_sop_label[0]}')
            
    elif task == 'test_ctx':
        k=0
        for batch in data_loader:
            context, context_attn_mask, context_seg_ids, \
            context_mlm_labels, context_position_perm_id, response = batch
            
            k+=1
            if k>10: break

  #          print(f'context:\n {context}')
  #          print('context_str:', tokenizer.decode(context[0].numpy().tolist()))
  #          print(f'context_attn_mask:\n {context_attn_mask}')
  #          print(f'context_segment_ids:\n {context_seg_ids}')
  #          print(f'context_lm_labels:\n {context_mlm_labels}')
  #          print(f'context_position_perm_id:\n {context_position_perm_id}')
            #print(f'utts_segment_ids:\n {utts_segment_ids}')
            #print(f'utts_lm_labels:\n {utts_lm_labels}')
            #print(f'utts_sop_labels:\n {utts_sop_labels}')
   #         print('response:', tokenizer.decode(response[0].numpy().tolist()))