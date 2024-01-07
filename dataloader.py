#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin
"""
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import getPaths_attri, getPaths_unlabel
NUM_CHUNKS_PER_SENT = 11


class MspPodcastEmoDataset(Dataset):
    """MSP-Podcast Dataset (labeled data)"""

    def __init__(self, root_dir, label_dir, split_set, emo_attr):
        # Init parameters
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.split_set = split_set
        self.emo_attr = emo_attr

        # Label and data paths
        self._paths, self._labels = getPaths_attri(label_dir, split_set, emo_attr)

        # Norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Dom':
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]
        
        # Each utterance is split into fixed C chunks
        C = NUM_CHUNKS_PER_SENT
        self.imgs = []
        # Every sentence becomes C chunks, so we repeat the same path/label for C times
        repeat_paths = self._paths.tolist()
        repeat_labels = ((self._labels-self.Label_mean)/self.Label_std).tolist()
        for i in range(len(repeat_paths)):
            self.imgs.extend([(root_dir+repeat_paths[i], repeat_labels[i])]*C)
  
    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading acoustic features
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data']
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3
        # Loading Label & Normalization
        label = self._labels[idx]
        label = (label-self.Label_mean)/self.Label_std
        return data, label

class UnlabeledDataset(Dataset):
    """Unlabeled Dataset"""

    def __init__(self, root_dir, size=None):
        # Init parameters
        self.root_dir = root_dir
        self.size = size

        # Data paths
        self._paths = getPaths_unlabel(self.root_dir, sample_num=self.size)
        np.random.shuffle(self._paths)

        # Norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']

        # Each utterance is split into fixed C chunks
        C = NUM_CHUNKS_PER_SENT
        self.imgs = []
        # Every sentence becomes C chunks, so we repeat the same path/label for C times
        repeat_paths = self._paths.tolist()
        for i in range(len(repeat_paths)):
            self.imgs.extend([root_dir+repeat_paths[i]]*C)
        
    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading acoustic features
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data']
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3
        return data
