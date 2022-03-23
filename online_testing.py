#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import torch
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from utils import DynamicChunkSplitData, evaluation_metrics
from utils import getPaths_attri, load_model
import argparse


argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-nc", "--num_clusters", required=True)
argparse.add_argument("-mt", "--model_type", required=True)
args = vars(argparse.parse_args())

# Parameters
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']
num_clusters = int(args['num_clusters'])
model_type = args['model_type']

# loading entire test set
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.8/Features/Mel_Spec128/feat_mat/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv'
_paths, _gt_labels = getPaths_attri(label_dir, split_set='Test1', emo_attr=emo_attr)

# de-normalize parameters
Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
if emo_attr == 'Act':
    Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Dom':
    Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Val':
    Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]    

# load trained model
MODEL_PATH = './trained_models/TempDeepEmoCluster_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'_'+model_type+'.pth.tar'
model = load_model(MODEL_PATH, num_clusters, model_type)
model.cuda()
model.eval()

# testing process
print('Start Online Testing Process')
Pred_Rsl = []
GT_Label = []
for i in tqdm(range(len(_paths))):
    # Loading data
    data = loadmat(root_dir + _paths[i].replace('.wav','.mat'))['Audio_data'] 
    # Z-normalization
    data = (data-Feat_mean)/Feat_std
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3         
    # chunk segmentation
    chunk_data = DynamicChunkSplitData([data], m=62, C=11, n=1)
    # numpy to GPU tensor & reshape input data to feed into model
    chunk_data = torch.from_numpy(chunk_data)
    chunk_data = chunk_data.unsqueeze(1).cuda().float()
    # models flow
    if 'Triplet' in model_type:
        _, _, _, _, pred_rsl = model(chunk_data)
    else:
        _, pred_rsl = model(chunk_data)
    pred_rsl = torch.mean(pred_rsl)
    # output
    GT_Label.append(_gt_labels[i])
    Pred_Rsl.append(pred_rsl.data.cpu().numpy())
GT_Label = np.array(GT_Label)
Pred_Rsl = np.array(Pred_Rsl)

# Regression Task => De-Normalize Target and Prediction
Pred_Rsl = (Label_std*Pred_Rsl)+Label_mean
        
# Output Predict Reulst
pred_CCC_Rsl = evaluation_metrics(GT_Label, Pred_Rsl)[0]
print('Epochs: '+str(epochs))
print('Batch_Size: '+str(batch_size))
print('EmoClusters (#): '+str(num_clusters))
print('Model: Vgg16_DeepEmoCluster')
print('Temporal-Modeling: '+model_type)
print('Test '+emo_attr+'-CCC: '+str(pred_CCC_Rsl))
