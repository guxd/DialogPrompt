# DialogPrompt
# Copyright 2022-present NAVER Corp.
# BSD 3-clause

import argparse
import numpy as np
import random
import json
from tqdm import tqdm, trange
import logging
from collections import Counter
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
import models, data_loader
from data_loader import DialogTransformerDataset, load_vecs

#import rouge # pip install py-rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
try: 
    meteor_score(["hello world"], "hi world")
except LookupError: 
    nltk.download('wordnet')

        
class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()
        '''
        self.rouge_evaluator = rouge.Rouge(metrics=['rouge-l'],
                           max_n=4,
                           limit_length=True,
                           length_limit=200,
                           length_limit_type='words',
                           apply_avg=True,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        '''
    @classmethod
    def sim_bleu(self, hyps, ref):
        """
        :param ref - a list of tokens of the reference
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./4, 1./4, 1./4, 1./4]))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)
    
    @classmethod
    def sim_meteor(self, hyps, ref):
        """
        :param refs - a list of strings representing references
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            #try:
            scores.append(meteor_score([ref], hyp))
            #except:
            #    scores.append(0.0)
        return np.max(scores), np.mean(scores)
    
    @classmethod
    def sim_nist(self, hyps, ref):
        """
        :param refs - a list of strings representing references
        :param hyps - a list of tokens of the hypothesis
        :return maxnist - recall nist
        :return avgnist - precision nist
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_nist([ref], hyp))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)
    
    @classmethod
    def sim_rougeL(self, hyps, ref):
        """
        Compute ROUGE-L score given a list of candidates and a reference
        :param hyps: list : candidate sentences to be evaluated
        :param ref: list: reference sentence to be evaluated
        :returns score: float (ROUGE-L score for the candidate evaluated against references)
        This class is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
        """
        def lcs(string, sub):
            """
            Calculates longest common subsequence for a pair of tokenized strings
            :param string : list : tokens from a string split using whitespace
            :param sub : list: shorter string, also split using whitespace
            :returns: length (list of int): length of the longest common subsequence between the two strings
            Note: only gives length of the longest common subsequence, not the actual LCS
            This function is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
            """
            if len(string) < len(sub): sub, string = string, sub
            lengths = [[0 for i in range(0, len(sub)+1)] for j in range(0,len(string)+1)]
            for j in range(1, len(sub) + 1):
                for i in range(1, len(string) + 1):
                    if string[i-1] == sub[j-1]:
                        lengths[i][j] = lengths[i-1][j-1] + 1
                    else:
                        lengths[i][j] = max(lengths[i-1][j], lengths[i][j-1])
            return lengths[len(string)][len(sub)]
        def rougeL(hyp, refs):
            assert len(refs)>0 and type(refs[0]) is list, "number of references should >0 for rouge"
            beta=1.2
            prec, rec = [], []
            for ref in refs:
                _lcs = lcs(ref, hyp)# compute the longest common subsequence
                prec.append(_lcs/float(len(hyp)))
                rec.append(_lcs/float(len(ref)))
            prec_max, rec_max = max(prec), max(rec)

            if prec_max!=0 and rec_max!=0:
                score = ((1+beta**2)*prec_max*rec_max)/float(rec_max+beta**2*prec_max)
            else:
                score = 0.0
            return score
        
        scores = []
        for hyp in hyps:
            try:
                scores.append(rougeL(hyp, [ref]))
            except:
                print('exception in RougeL')
                scores.append(0.0)
        return np.max(scores), np.mean(scores)
    
    
    @classmethod
    def div_distinct(self, seqs, seq_lens):
        """
        distinct-1 distinct-2 metrics for diversity measure proposed 
        by Li et al. "A Diversity-Promoting Objective Function for Neural Conversation Models"
        we counted numbers of distinct unigrams and bigrams in the generated responses 
        and divide the numbers by total number of unigrams and bigrams. 
        The two metrics measure how informative and diverse the generated responses are. 
        High numbers and high ratios mean that there is much content in the generated responses, 
        and high numbers further indicate that the generated responses are long
        """
        batch_size = seqs.shape[0]
        intra_dist1, intra_dist2=np.zeros(batch_size), np.zeros(batch_size)
        
        n_unigrams, n_bigrams, n_unigrams_total , n_bigrams_total = 0. ,0., 0., 0.
        unigrams_all, bigrams_all = Counter(), Counter()
        for b in range(batch_size):
            unigrams= Counter([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams = Counter([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
            intra_dist1[b]=(len(unigrams.items())+1e-12)/(seq_lens[b]+1e-5)
            intra_dist2[b]=(len(bigrams.items())+1e-12)/(max(0, seq_lens[b]-1)+1e-5)
            
            unigrams_all.update([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams_all.update([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
            n_unigrams_total += seq_lens[b]
            n_bigrams_total += max(0, seq_lens[b]-1)
        inter_dist1 = (len(unigrams_all.items())+1e-12)/(n_unigrams_total+1e-5)
        inter_dist2 = (len(bigrams_all.items())+1e-12)/(n_bigrams_total+1e-5)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2
    
    @classmethod
    def tok_f1(self, predictions, pred_lens, targets, target_lens):
        batch_size = predictions.shape[0]        
        f1s = []
        for b in range(batch_size):
            pred = predictions[b][:pred_lens[b]]
            target = targets[b][:target_lens[b]]
            common = Counter(target) & Counter(pred)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.
            precision = 1. * num_same / pred_lens[b]
            recall = 1. * num_same / target_lens[b]
            f1= (2. * recall * precision) / (precision + recall)
            f1s.append(f1)
        return np.mean(f1)

logger = logging.getLogger(__name__)    
        
class Learner(object):
    
    def run_train(self, args, model, train_set, optim_params, entry='forward', max_steps = -1, 
                  valid_set=None,  test_set=None):         
        tb_writer=None
        if args.local_rank in [-1, 0]: 
            tb_writer = SummaryWriter(f"./output/{args.model}/logs/")# the first process create tensorboard writer
        result_dir = f"./output/{args.model}/"
        if args.local_rank in [-1, 0]: os.makedirs(result_dir, exist_ok=True)
        checkpoint_dir = f"./output/{args.model}/models/"
        if args.local_rank in [-1, 0]: os.makedirs(checkpoint_dir, exist_ok=True)
            
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"number of training parameters: {num_params}")
            
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        data_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
        dataloader = DataLoader(train_set, sampler=data_sampler, batch_size=args.train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            args.n_epochs = max_steps // (len(dataloader) // args.grad_accum_steps) + 1
        else:
            t_total = len(dataloader) // args.grad_accum_steps * args.n_epochs

        optimizer = AdamW(optim_params, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1: model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        # Train!
        #global global_step
        global_step = 0
        train_loss, prev_train_loss = 0.0, 0.0
        best_valid_loss, best_step, no_improv_steps, early_stop = 1e10, 0, 0, False # keep the optimal step
        model.zero_grad()
        train_iterator = trange(int(args.n_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])        
        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                batch = [t.to(args.device) for t in batch]
                model.train()
                model1 = model.module if hasattr(model, 'module') else model
                results = getattr(model1, entry)(*batch)    
                
                if args.n_gpu > 1:
                    results = {name:loss.mean() for name, loss in results.items()}# mean() to average on multi-gpu parallel training
                if args.grad_accum_steps > 1:
                    results = {name:loss/args.grad_accum_steps for name, loss in results.items()}
                loss = results['loss']
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                train_loss += loss.item()
                if (step + 1) % args.grad_accum_steps == 0:
                    if args.fp16:
                        nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logmsg = {'lr': scheduler.get_lr()[0], 'train_loss': (train_loss - prev_train_loss)/args.logging_steps}
                        logmsg.update({f"train_{name}":loss.item() for name, loss in results.items()})
                        self.report(logmsg, global_step, tb_writer)
                        prev_train_loss = train_loss

                    if args.local_rank ==-1 and args.do_validate and global_step>=args.start_eval and global_step % args.validating_steps == 0:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        assert valid_set is not None, "validation set is not provided"
                        valid_loss, results = self.run_valid(args, model, valid_set)
                        #valid_loss, results, generated_text = self.run_eval(args, model, valid_set)
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            no_improv_steps = 0
                            early_stop = False
                        else:
                            no_improv_steps += 1
                            if no_improv_steps > 5: early_stop = True
                        
                        results={f"valid_{key}":val for key, val in results.items()}                       
                        self.report(results, global_step, tb_writer)
                            
                    if args.local_rank ==-1 and args.do_test and global_step>=args.start_eval and global_step%args.validating_steps == 0 and no_improv_steps == 0:
                        assert test_set is not None, "test set is not provided"
                        test_loss, results, generated_text = self.run_eval(args, model, test_set)
                        results={f"test_{key}":val for key, val in results.items()}
                        with open(os.path.join(result_dir, f"test_results_{global_step}.txt"), 'w') as f_test:
                            f_test.write(str(results)+'\n')
                            f_test.write(generated_text+'\n')
                        
                    #if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        checkpoint_name = f'checkpoint-optimal'# only save the optimal checkpoint # f'checkpoint-{global_step}'
                        self.save(args, model, checkpoint_dir, checkpoint_name) # Save model checkpoint

                if max_steps > 0 and global_step > max_steps or early_stop:
                    epoch_iterator.close()
                    break
            if max_steps > 0 and global_step > max_steps or early_stop:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0] and tb_writer is not None: tb_writer.close()

        return global_step, best_valid_loss
    
    def run_valid(self, args, model, dataset):
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        model1 = model.module if hasattr(model, 'module') else model       

        eval_batch_size = 32 #args.per_gpu_eval_batch_size * max(1, args.n_gpu)        
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)       

        device = next(model1.parameters()).device
        tokenizer = model1.tokenizer

        losses = []
        dlg_id = 0
        for batch in tqdm(dataloader): 

            if random.random() > args.fast_eval_ratio: continue

            batch_gpu = [t.to(device) for t in batch]
            with torch.no_grad():
                loss = model1.validate(*batch_gpu)
            losses.append(loss)

        loss = float(np.mean(losses))
        perplexity = torch.exp(torch.tensor(loss)).item()
        result = {'perplexity': perplexity}
            
        return loss, result
                
    def run_eval(self, args, model, dataset, num_samples=1, decode_mode='sample'):
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        model1 = model.module if hasattr(model, 'module') else model       

        eval_batch_size = 1 #args.per_gpu_eval_batch_size * max(1, args.n_gpu)        
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)       

        device = next(model1.parameters()).device
        tokenizer = model1.tokenizer

        bleus, meteors, nists, rougeLs, avg_lens = [], [], [], [], []
        losses = []
        generated_text = []
        dlg_id = 0
        for batch in tqdm(dataloader): 

            if random.random() > args.fast_eval_ratio: continue

            batch_gpu = [t.to(device) for t in batch]
            with torch.no_grad():
                loss = model1.validate(*batch_gpu)
            losses.append(loss)
            
            with torch.no_grad():
                sample_words, sample_lens, context, gt_response = model1.generate(batch)# nparray: [repeat x seq_len] 
                                                                            
            pred_sents = [tokenizer.decode(sample_words[i].tolist(), skip_special_tokens=True) for i in range(num_samples)] 
            pred_tokens = [sent.split(' ') for sent in pred_sents]   
            ref_str = tokenizer.decode(gt_response[0].tolist(), skip_special_tokens=True)#.encode('utf-8')

            max_bleu, avg_bleu = Metrics.sim_bleu(pred_tokens, ref_str.split(' '))
            bleus.append(max_bleu)
            max_meteor, avg_meteor = Metrics.sim_meteor(pred_sents, ref_str)
            meteors.append(max_meteor)
            max_nist, avg_nist = Metrics.sim_nist(pred_tokens, ref_str.split(' '))
            nists.append(max_nist)
            max_rougeL, avg_rougeL = Metrics.sim_rougeL(pred_tokens, ref_str.split(' '))
            rougeLs.append(max_rougeL)
            avg_lens.append(np.mean(sample_lens))
            
            ## Write concrete results to a text file
            dlg_id += 1 
            generated_text.append("Batch {:d} \n".format(dlg_id))
            # print the context
            if context.ndim<3: context = np.expand_dims(context, axis=1) # in case context is flattened
            batch_size, ctx_len, max_utt_len = context.shape
            start = np.maximum(0, ctx_len-8)
            for t_id in range(start, ctx_len, 1):
                context_str = tokenizer.decode(context[0, t_id].tolist(), skip_special_tokens=False)
                if context_str.strip() == '': continue
                generated_text.append(f"Context {t_id}: {context_str}\n")
            #print the ground truth response    
            generated_text.append(f"Target >> {ref_str}\n")
            for res_id, pred_sent in enumerate(pred_sents):
                generated_text.append("Sample {:d} >> {}\n".format(res_id, pred_sent.replace(" ' ", "'")))
            generated_text.append("\n\n")
        loss = float(np.mean(losses))
        perplexity = torch.exp(torch.tensor(loss)).item()
        bleu= float(np.mean(bleus))
        meteor = float(np.mean(meteors))
        nist = float(np.mean(nists))
        rougeL = float(np.mean(rougeLs))
        result = {'perplexity': perplexity, 
                  'avg_len':float(np.mean(avg_lens)), 'bleu': bleu, 
                  'meteor': meteor, 'nist': nist, 'rouge-L': rougeL
                 }
            
        logger.info("***** Validation results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            generated_text.append("%s = %s\n" % (key, str(result[key])))
        generated_text = ''.join(generated_text)
        #print(generated_text)
        return loss, result, generated_text


    def report(self, results, step, tb_writer):
        if tb_writer is not None:
            for key, value in results.items():
                tb_writer.add_scalar(key, value, step)
    
    def save(self, args, model, output_dir, checkpoint_name):    
        output_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))# save arguments together with the model
                

