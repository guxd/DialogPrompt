# DialogPrompt
# Copyright 2022-present NAVER Corp.
# BSD 3-clause

import random
import numpy as np
import argparse
import json
import tables
import os
import re
from tqdm import tqdm
import pickle as pkl
from transformers import GPT2Tokenizer

from data_loader import load_dict, save_vecs

class Index(tables.IsDescription):
    pos_utt = tables.Int32Col() # start offset of an utterance
    res_len = tables.Int32Col() # number of tokens till the end of response
    ctx_len = tables.Int32Col() # number of tokens from the start of dialog 
    #pos_knowl = tables.Int32Col() #start offset of knowlege in the knowledge array
    #knowl_len = tables.Int32Col() # number of tokens till the end of knowledge in the knowledge array
def binarize(dialogs, tokenizer, output_path):
    """binarize data and save the processed data into a hdf5 file
       :param dialogs: an array of dialogs, 
        each element is a list of <caller, utt, feature> where caller is a string of "A" or "B",
        utt is a sentence, feature is an 2D numpy array 
    """
    f = tables.open_file(output_path, 'w')
    filters = tables.Filters(complib='blosc', complevel=5)
    arr_contexts = f.create_earray(f.root, 'sentences', tables.Int32Atom(),shape=(0,),filters=filters)
    #arr_knowledge = f.create_earray(f.root, 'knowledge', tables.Int16Atom(),shape=(0,),filters=filters)
    indices = f.create_table("/", 'indices', Index, "a table of indices and lengths")
    pos_utt = 0
    for i, dialog in enumerate(tqdm(dialogs)):
        
        n_tokens=0
        ctx_len=0
        for k, (caller, utt, feature) in enumerate(dialog['utts']):
            #floor = -1 if caller == 'A' else -2
            idx_utt = tokenizer.encode(utt)
            #if idx_utt[0]!=tokenizer.cls_token_id: idx_utt = [tokenizer.cls_token_id] + idx_utt
            if idx_utt[-1]!=tokenizer.eos_token_id: idx_utt = idx_utt + [tokenizer.eos_token_id]
            #arr_contexts.append([floor])
            arr_contexts.append(idx_utt)
            n_tokens+=len(idx_utt)#+1
            if k>0: # ignore the first utterance which has no context
                ind = indices.row
                ind['pos_utt'] = pos_utt
                ind['res_len'] = len(idx_utt)#+1
                ind['ctx_len'] = ctx_len   
                ind.append()
            ctx_len+=len(idx_utt)#+1
            pos_utt += len(idx_utt)#+1
      
        ctx_len=0
    f.close()
    
    


def get_dailydial_data(data_path):
    dialogs = []
    dials = open(data_path, 'r', encoding='utf-8').readlines()
    for dial in dials:
        utts = []
        for i, utt in enumerate(dial.rsplit(' __eou__ ')):
            caller = 'A' if i % 2 == 0 else 'B'
            utts.append((caller, utt, np.zeros((1, 1))))
        dialog = {'knowledge': '', 'utts': utts}
        dialogs.append(dialog)
    return dialogs


def load_data(data_path, data_name):
    data={'train':[],'valid':[], 'test':[]}
    if data_name=='SWDA':
        data = pkl.load(open(data_path+'full_swda_clean_42da_sentiment_dialog_corpus.p', "rb"))
    elif args.data_set=='dailydial':
        data['train'] = get_dailydial_data(data_path+'train.utts.txt')
        data['valid'] = get_dailydial_data(data_path+'valid.utts.txt')
        data['test'] = get_dailydial_data(data_path+'test.utts.txt')
    return data



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_set", default='multiwoz', help='dailydial')
    parser.add_argument('-m', "--model_name", required=True, help='gpt2, gpt2-medium, gpt2-large, gpt2-xl, bert-base-chinese, blenderbot3b')
    return parser.parse_args()
 
if __name__ == "__main__":
    args=get_args()
    
    #home = os.path.expanduser("~/workspace/dialogGAN/")
    work_dir = "./data/"
    data_dir = work_dir + args.data_set+'/'
    
    print("loading data...")
    data = load_data(data_dir, args.data_set)
        
    train_data=data["train"]
    valid_data=data["valid"]
    test_data=data["test"]    
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, do_lower_case=True)
    
    print('binarizing training data')
    train_out_path = os.path.join(data_dir, "train.h5")
    binarize(train_data, tokenizer, train_out_path)
                            
    print('binarizing validation data')
    dev_out_path = os.path.join(data_dir, "valid.h5")
    dev_data_binary = binarize(valid_data, tokenizer, dev_out_path) 
    
    print('binarizing test data')
    test_out_path = os.path.join(data_dir, "test.h5")
    test_data_binary = binarize(test_data, tokenizer, test_out_path) 
    
    ### test binarized by visualization
 #   dialog=train_data[0]
 #   for caller, utt, feature in dialog['utts']:
 #       print(caller+':'+utt.lower())
            
    table = tables.open_file(train_out_path)
    data = table.get_node('/sentences')
    index = table.get_node('/indices')
    for offset in range(2000,2010):
        pos_utt, ctx_len, res_len = index[offset]['pos_utt'], index[offset]['ctx_len'], index[offset]['res_len']
        print('pos_utt:{}, ctx_len:{}, res_len:{}'.format(pos_utt, ctx_len, res_len))
        print(data[pos_utt-ctx_len:pos_utt])
        print('context:'+ tokenizer.decode(data[pos_utt-ctx_len: pos_utt]))
        print('response:'+ tokenizer.decode(data[pos_utt:pos_utt+res_len]))
