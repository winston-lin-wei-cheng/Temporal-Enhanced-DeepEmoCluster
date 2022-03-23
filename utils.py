#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import pandas as pd
import numpy as np
import os
import pickle
import torch
from torch.utils.data.sampler import Sampler
from models import vgg16_GRU, vgg16_CNN, vgg16_Trans, vgg16_Triplet


def getPaths_attri(path_label, split_set, emo_attr):
    """
    This function is for filtering data by different constraints of the emotions and split sets
    Args:
        path_label$ (str): path of label
        split_set$ (str): 'Train', 'Validation' or 'Test1' 
        emo_attr$ (str): 'Act', 'Dom' or 'Val'
    """
    label_table = pd.read_csv(path_label)
    whole_fnames = (label_table['FileName'].values).astype('str')
    split_sets = (label_table['Split_Set'].values).astype('str')
    emo_act = label_table['EmoAct'].values
    emo_dom = label_table['EmoDom'].values
    emo_val = label_table['EmoVal'].values
    _paths = []
    _label_act = []
    _label_dom = []
    _label_val = []
    for i in range(len(whole_fnames)):
        # Constrain with Split Sets      
        if split_sets[i]==split_set:
            # Constrain with Emotional Labels
            _paths.append(whole_fnames[i])
            _label_act.append(emo_act[i])
            _label_dom.append(emo_dom[i])
            _label_val.append(emo_val[i])
        else:
            pass
    if emo_attr == 'Act':
        return np.array(_paths), np.array(_label_act)
    elif emo_attr == 'Dom':
        return np.array(_paths), np.array(_label_dom)
    elif emo_attr == 'Val':
        return np.array(_paths), np.array(_label_val)
    
def CombineListToMatrix(Data):
    length_all = []
    for i in range(len(Data)):
        length_all.append(len(Data[i])) 
    feat_num = len(Data[0].T)
    Data_All = np.zeros((sum(length_all),feat_num))
    idx = 0
    Idx = []
    for i in range(len(length_all)):
        idx = idx+length_all[i]
        Idx.append(idx)        
    for i in range(len(Idx)):
        if i==0:    
            start = 0
            end = Idx[i]
            Data_All[start:end]=Data[i]
        else:
            start = Idx[i-1]
            end = Idx[i]
            Data_All[start:end]=Data[i]
    return Data_All  

# split original batch data into batch small-chunks data with
# proposed dynamic window step size which depends on the sentence duration 
def DynamicChunkSplitData(Online_data, m, C, n):
    """
    Note! This function can't process sequence length which less than given m=62
    (e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
    Please make sure all your input data's length are greater then given m.
    
    Args:
         Online_data$ (list): list of data array for a single sentence
                   m$ (int) : chunk window length (i.e., number of frames within a chunk)
                   C$ (int) : number of chunks splitted for a sentence
                   n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n*C-1  # Tmax = 11sec (for the MSP-Podcast corpus), 
                        # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
    Split_Data = []
    for i in range(len(Online_data)):
        data = Online_data[i]
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size = int(int(len(data)-m)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )    
    return np.array(Split_Data)

# split original batch data into batch small-chunks data with
# proposed dynamic window step size which depends on the sentence duration 
def DynamicChunkSplitEmoData(Batch_data, Batch_label, m, C, n):
    """
    Note! This function can't process sequence length which less than given m=62
    (e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
    Please make sure all your input data's length are greater then given m.
    
    Args:
         Batch_data$ (list): list of data arrays for a single batch.
        Batch_label$ (list): list of training targets for a single batch.
                  m$ (int) : chunk window length (i.e., number of frames within a chunk)
                  C$ (int) : number of chunks splitted for a sentence
                  n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n*C-1  # Tmax = 11sec (for the MSP-Podcast corpus), 
                        # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
    Split_Data = []
    Split_Label = np.array([])
    for i in range(len(Batch_data)):
        data = Batch_data[i]
        label = Batch_label[i]
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size = int(int(len(data)-m)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )    
        # Output Split Label
        split_label = np.repeat( label,len(start_idx) )
        Split_Label = np.concatenate((Split_Label,split_label))
    return np.array(Split_Data), Split_Label

def cc_coef(output, target):
    mu_y_true = torch.mean(target)
    mu_y_pred = torch.mean(output)                                                                                                                                                                                              
    return 1 - 2 * torch.mean((target - mu_y_true) * (output - mu_y_pred)) / (torch.var(target) + torch.var(output) + torch.mean((mu_y_pred - mu_y_true)**2))

def evaluation_metrics(true_value,predicted_value):
    corr_coeff = np.corrcoef(true_value,predicted_value)
    ccc = 2*predicted_value.std()*true_value.std()*corr_coeff[0,1]/(predicted_value.var() + true_value.var() + (predicted_value.mean() - true_value.mean())**2)
    return(ccc,corr_coeff)

class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

class SentUnifSampler(Sampler):
    """ Sentence-level Uniform Sampling for the uniform clustering pseudolabels, which also preserves 
        the temporal orders of data chunks in the sentence for additional temporal modeling
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        # index table (emo_cluster_dataset VS original_dataset)
        cluster_assign = []
        new_idx = []
        old_idx = []
        init = 0
        for i in range(len(self.images_lists)):
            for j in range(len(self.images_lists[i])):
                cluster_assign.append(i)
                new_idx.append(init)
                old_idx.append(self.images_lists[i][j])
                init += 1
                
        idx_map_table = {}
        for i in range(len(old_idx)):
            idx_map_table[old_idx[i]]=np.array([cluster_assign[i], new_idx[i]])
    
        chunk_idx = np.arange(0, self.N, 11)
        Sent_Idx = []
        for i in range(len(chunk_idx)):
            try:
                start = chunk_idx[i]
                end = chunk_idx[i+1]
                sent_idx = []
                for key in np.arange(start, end):
                    sent_idx.append(idx_map_table[key])
                Sent_Idx.append(np.array(sent_idx))
    
            except: # special treatment for the last sentence
                start = chunk_idx[i]
                end = self.N
                sent_idx = []
                for key in np.arange(start, end):
                    sent_idx.append(idx_map_table[key])
                Sent_Idx.append(np.array(sent_idx))
            
        # cluster assignment probs
        weights = []
        for i in range(len(self.images_lists)):
            weights.append(len(self.images_lists[i]))
        weights = [1/w for w in weights]              # Invert all weights
        weights = [w/sum(weights) for w in weights]   # Normalize weights (in prob. form)
        
        # sentence-level random selection based on chunks-level cluster prob (uniform cluster dist.)
        rdn_select_prob = []
        for i in range(len(Sent_Idx)):
            sent_w = 0
            for j in range(len(Sent_Idx[i][:,0])):
                sent_w += weights[Sent_Idx[i][j,0]]
            rdn_select_prob.append(sent_w)
        rdn_select_prob = np.array(rdn_select_prob)/sum(rdn_select_prob)    
        rdn_select_idx = np.random.choice(np.arange(len(rdn_select_prob)), size=len(rdn_select_prob) , p=rdn_select_prob) # Sample
        np.random.shuffle(rdn_select_idx)
        
        # aggregate final sampler index & output
        res = []
        for idx in rdn_select_idx:
            res.extend(Sent_Idx[idx][:,1].tolist())
        res = np.array(res)
        
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]           
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr

def load_model(path, num_clusters, model_type):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        
        if model_type=='Temp-GRU':
            model = vgg16_GRU.__dict__[checkpoint['arch']](bn=True, out=num_clusters)
        elif model_type=='Temp-CNN':
            model = vgg16_CNN.__dict__[checkpoint['arch']](bn=True, out=num_clusters)
        elif model_type=='Temp-Trans':
            model = vgg16_Trans.__dict__[checkpoint['arch']](bn=True, out=num_clusters)
        elif model_type=='Temp-Triplet':
            model = vgg16_Triplet.__dict__[checkpoint['arch']](bn=True, out=num_clusters)
        
        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model
