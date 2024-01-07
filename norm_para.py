#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin
"""
import os
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
from utils import CombineListToMatrix


if __name__=='__main__':
    
    # Get label table & data (features)
    label_table = pd.read_csv('/YOUR/ROOT/PATH/TO/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv')
    data_root = '/YOUR/ROOT/PATH/TO/MSP-PODCAST-Publish-1.8/Features/Mel_Spec128/feat_mat/'
    
    # Get needed file info
    whole_fnames = label_table['FileName'].values.astype('str').tolist()
    split_set = label_table['Split_Set'].values.astype('str').tolist()
    emo_act = label_table['EmoAct'].values.tolist()
    emo_dom = label_table['EmoDom'].values.tolist()
    emo_val = label_table['EmoVal'].values.tolist()
    
    # Prepare feature & label norm parameters based on the 'Train' set
    Train_Data = []
    Train_Label_act, Train_Label_dom, Train_Label_val = [], [], []
    for idx in range(len(whole_fnames)):
        if split_set[idx]=='Train':
            data = loadmat(data_root+whole_fnames[idx].replace('.wav','.mat'))['Audio_data']
            Train_Data.append(data)
            Train_Label_act.append(emo_act[idx])
            Train_Label_dom.append(emo_dom[idx])
            Train_Label_val.append(emo_val[idx])
    Train_Data = CombineListToMatrix(Train_Data)
    Train_Label_act = np.array(Train_Label_act)
    Train_Label_dom = np.array(Train_Label_dom)
    Train_Label_val = np.array(Train_Label_val)
    
    # Creating output folder
    if not os.path.isdir('./NormTerm/'):
        os.makedirs('./NormTerm/')     
    
    # Save feature & label normalization parameters
    Feat_mean = np.mean(Train_Data,axis=0)
    Feat_std = np.std(Train_Data,axis=0)       
    savemat('./NormTerm/feat_norm_means.mat', {'normal_para':Feat_mean})
    savemat('./NormTerm/feat_norm_stds.mat', {'normal_para':Feat_std})
    Label_mean_Act = np.mean(Train_Label_act)
    Label_std_Act = np.std(Train_Label_act)
    savemat('./NormTerm/act_norm_means.mat', {'normal_para':Label_mean_Act})
    savemat('./NormTerm/act_norm_stds.mat', {'normal_para':Label_std_Act})    
    Label_mean_Dom = np.mean(Train_Label_dom)
    Label_std_Dom = np.std(Train_Label_dom)    
    savemat('./NormTerm/dom_norm_means.mat', {'normal_para':Label_mean_Dom})
    savemat('./NormTerm/dom_norm_stds.mat', {'normal_para':Label_std_Dom})    
    Label_mean_Val = np.mean(Train_Label_val)
    Label_std_Val = np.std(Train_Label_val)      
    savemat('./NormTerm/val_norm_means.mat', {'normal_para':Label_mean_Val})
    savemat('./NormTerm/val_norm_stds.mat', {'normal_para':Label_std_Val})    
