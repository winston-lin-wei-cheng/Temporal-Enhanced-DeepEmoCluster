#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin
"""
import os
from tqdm import tqdm
from utils import AverageMeter, Logger, SentUnifSampler
from utils import DynamicChunkSplitEmoData, DynamicChunkSplitData, cc_coef
from sklearn.metrics.cluster import normalized_mutual_info_score
import clustering
from dataloader import MspPodcastEmoDataset, UnlabeledDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import numpy as np
import matplotlib.pyplot as plt
from models import vgg16_GRU, vgg16_CNN, vgg16_Trans, vgg16_Triplet
import argparse
NUM_CHUNKS_PER_SENT = 11
NUM_FRAMES_PER_CHUNK = 62
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def collate_fn(batch):
    data, label = zip(*batch)
    chunk_data, chunk_label = DynamicChunkSplitEmoData(data, label, m=NUM_FRAMES_PER_CHUNK, C=NUM_CHUNKS_PER_SENT, n=1)
    chunk_data, chunk_label = torch.from_numpy(chunk_data), torch.from_numpy(chunk_label)
    return chunk_data.unsqueeze(1), chunk_label.unsqueeze(1)

def collate_fn_unlabel(batch):
    chunk_data = DynamicChunkSplitData(batch, m=NUM_FRAMES_PER_CHUNK, C=NUM_CHUNKS_PER_SENT, n=1)
    chunk_data = torch.from_numpy(chunk_data)
    return chunk_data.unsqueeze(1)

def compute_features(dataloader, model, N, dataset_type):
    if verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, batch_data in enumerate(dataloader):
        if dataset_type == 'supervised':
            input_tensor, target = batch_data
        elif dataset_type == 'unsupervised':
            input_tensor = batch_data
        
        input_data = torch.autograd.Variable(input_tensor.to(device))
        input_data = input_data.float()
        
        if model_type=='Temp-Triplet':
            _, _, _, aux, _ = model(input_data)
        else:
            aux, _ = model(input_data)
        aux = aux.data.cpu().numpy()
        
        # each utterance split into C chunks
        C = NUM_CHUNKS_PER_SENT
        if i == 0:
            features = np.zeros((N*C, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch_size * C: (i + 1) * batch_size * C] = aux
        else:
            # special treatment for final batch
            features[i * batch_size * C:] = aux

        # measure elapsed time
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i % 500) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features

def train_joint_Triplet(loader, model,
                        crit_contrast, crit_class, crit_attri,
                        opt_class, opt_attri,
                        epoch, gamma=1):
    """ Joint training of the CNN model for emotional clusters.
        Args:
            loader$ (torch.utils.data.DataLoader): Dataloader
            model$ (nn.Module): VGG16-CNN encoder
            crit_contrast$ (torch.nn): loss of contrast learning (i.e., Triplet loss)
            crit_class$ (torch.nn): loss of cluster classification (i.e., cross entropy)
            crit_attri$ (torch.nn): loss of attribute emotion regression (i.e., CCC)
            opt_class$ (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            opt_attri$ (torch.optim.Adam):optimizer for every parameters with True
                                   requires_grad in model
            epoch$ (int)
            gamma$ (float): weighting factor of Triplet loss
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode
    model.train()
    for param in model.emo_regressor.parameters():
        param.requires_grad = True
    for param in model.top_layer_attri.parameters():
        param.requires_grad = True

    # Create an optimizer for the last fc layer (classification)
    optimizer_tl = torch.optim.SGD(model.top_layer_class.parameters(), lr=0.05)
    
    end = time.time()
    for i, (input_tensor, class_tar, attri_tar) in enumerate(tqdm(loader)):
        data_time.update(time.time() - end)
        
        # input data to GPU tensor
        input_data = torch.autograd.Variable(input_tensor.to(device))
        input_data = input_data.float()
        
        # input labels to GPU tensor
        input_class_tar = torch.autograd.Variable(class_tar.to(device))
        input_attri_tar = torch.autograd.Variable(attri_tar.unsqueeze(1).to(device)) # match shape for the CCC loss calculation
        input_attri_tar = input_attri_tar.float()

        # model flow
        anc, pos, neg, pred_class, pred_attri = model(input_data)
        
        # loss calculation
        loss1 = crit_contrast(anc, pos, neg)
        loss2 = crit_class(pred_class, input_class_tar)
        loss3 = crit_attri(pred_attri, input_attri_tar)
        loss = gamma*loss1 + loss2 + loss3
        
        # record loss
        losses.update(loss.data.item(), input_tensor.size(0))
        
        # compute gradient and do optimizer step
        opt_class.zero_grad()
        opt_attri.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt_class.step()
        opt_attri.step()
        optimizer_tl.step()
            
        # measure elapsed time
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def train_joint_TempNet(loader, model,
                        crit_class, crit_attri,
                        opt_class, opt_attri,
                        epoch):
    """ Joint training of the CNN model for emotional clusters.
        Args:
            loader$ (torch.utils.data.DataLoader): Dataloader
            model$ (nn.Module): VGG16-CNN encoder
            crit_class$ (torch.nn): loss of cluster classification (i.e., cross entropy)
            crit_attri$ (torch.nn): loss of attribute emotion regression (i.e., CCC)
            opt_class$ (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            opt_attri$ (torch.optim.Adam):optimizer for every parameters with True
                                   requires_grad in model
            epoch$ (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode
    model.train()      
    for param in model.emo_regressor.parameters():
        param.requires_grad = True
    for param in model.top_layer_attri.parameters():
        param.requires_grad = True

    # Create an optimizer for the last fc layer (classification)
    optimizer_tl = torch.optim.SGD(model.top_layer_class.parameters(), lr=0.05)
    
    end = time.time()
    for i, (input_tensor, class_tar, attri_tar) in enumerate(tqdm(loader)):
        data_time.update(time.time() - end)
        
        # input data to GPU tensor
        input_data = torch.autograd.Variable(input_tensor.to(device))
        input_data = input_data.float()
        
        # input labels to GPU tensor
        input_class_tar = torch.autograd.Variable(class_tar.to(device))
        input_attri_tar = torch.autograd.Variable(attri_tar.unsqueeze(1).to(device)) # match shape for the CCC loss calculation
        input_attri_tar = input_attri_tar.float()

        # model flow
        pred_class, pred_attri = model(input_data)
        
        # loss calculation
        loss1 = crit_class(pred_class, input_class_tar)
        loss2 = crit_attri(pred_attri, input_attri_tar)
        loss = loss1 + loss2
        
        # record loss
        losses.update(loss.data.item(), input_tensor.size(0))
        
        # compute gradient and do optimizer step
        opt_class.zero_grad()
        opt_attri.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt_class.step()
        opt_attri.step()
        optimizer_tl.step()
            
        # measure elapsed time
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def train_cluster_class(loader, model, crit_contrast, crit_class, opt, epoch):
    """ Train cluster classifier with labeled and unlabeled data.
        Args:
            loader$ (torch.utils.data.DataLoader): Dataloader
            model$ (nn.Module): VGG16-CNN encoder
            crit_contrast$ (torch.nn): loss of contrast learning (i.e., Triplet loss)
            crit_class$ (torch.nn): loss of cluster classification (i.e., cross entropy)
            opt$ (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch$ (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode & freeze emotion regressor weights
    model.train()
    for param in model.cnn_features.parameters():
        param.requires_grad = True
    for param in model.cnn_out.parameters():
        param.requires_grad = True
    if model_type!='Temp-Triplet':
        for param in model.temporal_net.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.top_layer_class.parameters():
        param.requires_grad = True
    for param in model.emo_regressor.parameters():
        param.requires_grad = False
    for param in model.top_layer_attri.parameters():
        param.requires_grad = False
        
    # Create an optimizer for the last fc layer (classification)
    optimizer_tl = torch.optim.SGD(model.top_layer_class.parameters(), lr=0.05)
    
    end = time.time()
    for i, (input_tensor, class_tar) in enumerate(tqdm(loader)):
        data_time.update(time.time() - end)
        
        # input data to GPU tensor
        input_data = torch.autograd.Variable(input_tensor.to(device))
        input_data = input_data.float()
        
        # input labels to GPU tensor
        input_class_tar = torch.autograd.Variable(class_tar.to(device))
        
        # model flow and loss calculation
        if model_type=='Temp-Triplet':
            anc, pos, neg, pred_class, _ = model(input_data)
            loss1 = crit_contrast(anc, pos, neg)
            loss2 = crit_class(pred_class, input_class_tar)
            loss = loss1 + loss2
        else:
            pred_class, _ = model(input_data)
            loss = crit_class(pred_class, input_class_tar)
        
        # record loss
        losses.update(loss.data.item(), input_tensor.size(0))
        
        # compute gradient and do optimizer step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()
        
        # measure elapsed time
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()
        
    return losses.avg

def validation(loader, model, crit):
    """Validation of the model.
        Args:
            loader$ (torch.utils.data.DataLoader): validation dataloader
            model$ (nn.Module): trained CNN model
            crit$ (torch.nn): calculate validation loss
    """
    # validation process
    model.eval()
    batch_loss_valid = []
    with torch.no_grad():
        for i, (input_tensor, target) in enumerate(tqdm(loader)):
            # Input Tensor Data
            input_var = torch.autograd.Variable(input_tensor.to(device))
            input_var = input_var.float()
            # Input Tensor Target
            target_var = torch.autograd.Variable(target.to(device))
            target_var = target_var.float()
            # models flow
            if model_type=='Temp-Triplet':
                _, _, _, _, pred_attri = model(input_var)
            else:
                _, pred_attri = model(input_var)
            # loss calculation
            loss = crit(pred_attri, target_var)  
            batch_loss_valid.append(loss.data.cpu().numpy())       
            torch.cuda.empty_cache()
    return np.mean(batch_loss_valid)
###############################################################################


argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-nc", "--num_clusters", required=True)
argparse.add_argument("-mt", "--model_type", required=True)
argparse.add_argument("-unlabel", "--unlabel_size", required=True)
args = vars(argparse.parse_args())

# Parameters
root_dir = '/YOUR/ROOT/PATH/TO/MSP-PODCAST-Publish-1.8/Features/Mel_Spec128/feat_mat/'
label_dir = '/YOUR/ROOT/PATH/TO/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv'
unlabel_dir = '/YOUR/ROOT/PATH/TO/Unlabeled-Pool/Features/Mel_Spec128/feat_mat/'

batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']
num_clusters = int(args['num_clusters'])
model_type = args['model_type'] # 'Temp-GRU', 'Temp-CNN', 'Temp-Trans', 'Temp-Triplet'
unlabel_size = int(args['unlabel_size'])
arch = 'vgg16'
exp = './trained_models/'
C = NUM_CHUNKS_PER_SENT
reassign = 1
num_workers = 12
verbose = True

# Loading the DeepEmoCluster model with different temporal modeling approaches
if verbose:
    print('Architecture: {}'.format(arch))
    print('Temporal Modeling: '+model_type)
if model_type=='Temp-GRU':
    model = vgg16_GRU.__dict__[arch](bn=True, out=num_clusters)
elif model_type=='Temp-CNN':
    model = vgg16_CNN.__dict__[arch](bn=True, out=num_clusters)
elif model_type=='Temp-Trans':
    model = vgg16_Trans.__dict__[arch](bn=True, out=num_clusters)
elif model_type=='Temp-Triplet':
    model = vgg16_Triplet.__dict__[arch](bn=True, out=num_clusters)
fd = int(model.top_layer_class.weight.size()[1])
model.top_layer_class = None
model.cnn_features = torch.nn.DataParallel(model.cnn_features)
model.cnn_out = torch.nn.DataParallel(model.cnn_out)
model.to(device)
cudnn.benchmark = True

# Define optimizers
optimizer_class = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer_attri = torch.optim.Adam(model.parameters(), lr=0.0005)

# Define the losses
criterion_class = nn.CrossEntropyLoss().to(device)
criterion_Triplet = nn.TripletMarginLoss(margin=1.0, p=2).to(device)

# Creating checkpoint folder
if not os.path.isdir(exp):
    os.makedirs(exp)

# Creating cluster assignments log
cluster_log = Logger(os.path.join(exp, 'clusters'))

# Define datasets
end = time.time()
if unlabel_size!=0:
    unlabel_dataset = UnlabeledDataset(unlabel_dir, size=unlabel_size)
training_dataset = MspPodcastEmoDataset(root_dir, label_dir, split_set='Train', emo_attr=emo_attr)
validation_dataset = MspPodcastEmoDataset(root_dir, label_dir, split_set='Validation', emo_attr=emo_attr)

# Creating data loaders
if unlabel_size!=0:
    unlabel_loader = torch.utils.data.DataLoader(unlabel_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=collate_fn_unlabel)

train_loader = torch.utils.data.DataLoader(training_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn)

valid_sampler = SubsetRandomSampler(list(range(len(validation_dataset))))
valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn)

if verbose:
    print('Load dataset: {0:.2f} s'.format(time.time() - end))

# Clustering algorithm to use
deepcluster = clustering.__dict__['Kmeans'](num_clusters)

# Training DeepTempEmoCluster
NMI = []
Loss_Joint = []
Loss_CCC_Valid = []
loss_valid_best = 0
for epoch in range(epochs):
    end = time.time()
    
    # remove head
    model.top_layer_class = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    
    ###########################################################################
    if unlabel_size==0: # fully supervised learning (DeepEmoCluster-FSL)
        pass
    else:
        # semi-supervised learning (DeepEmoCluster-SSL), stage1: unsupervised part
        features_unlabel = compute_features(unlabel_loader, model, len(unlabel_dataset), dataset_type='unsupervised') 
        clustering_loss_unlabel = deepcluster.cluster(features_unlabel, verbose=verbose)
        cluster_training_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                             unlabel_dataset.imgs,
                                                             dataset_type='unsupervised')
        unlabel_sampler = SentUnifSampler(int(reassign * len(cluster_training_dataset)),
                                          deepcluster.images_lists)
        cluster_dataloader = torch.utils.data.DataLoader(cluster_training_dataset,
                                                         batch_size=batch_size*C,
                                                         num_workers=num_workers,
                                                         sampler=unlabel_sampler,
                                                         pin_memory=True)
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).to(device))
        model.classifier = nn.Sequential(*mlp)
        model.top_layer_class = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer_class.weight.data.normal_(0, 0.01)
        model.top_layer_class.bias.data.zero_()
        model.top_layer_class.to(device)
        loss_class_unlabel = train_cluster_class(cluster_dataloader, model,
                                                 criterion_Triplet, criterion_class,
                                                 optimizer_class, epoch)
        model.top_layer_class = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    ###########################################################################
    
    # get the CNN features for the training set
    features = compute_features(train_loader, model, len(training_dataset), dataset_type='supervised')

    # cluster the features
    if verbose:
        print('Cluster the features')
    clustering_loss = deepcluster.cluster(features, verbose=verbose)
    
    # assign pseudo-labels
    if verbose:
        print('Assign pseudo labels')
    emo_cluster_training_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                             training_dataset.imgs,
                                                             dataset_type='supervised')

    # sentence-level uniform sampling
    sampler = SentUnifSampler(int(reassign * len(emo_cluster_training_dataset)),
                              deepcluster.images_lists)

    emo_cluster_dataloader = torch.utils.data.DataLoader(emo_cluster_training_dataset,
                                                         batch_size=batch_size*C,
                                                         num_workers=num_workers,
                                                         sampler=sampler,
                                                         pin_memory=True)

    # set last fully connected layer
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).to(device))
    model.classifier = nn.Sequential(*mlp)
    model.top_layer_class = nn.Linear(fd, len(deepcluster.images_lists))
    model.top_layer_class.weight.data.normal_(0, 0.01)
    model.top_layer_class.bias.data.zero_()
    model.top_layer_class.to(device)
    
    # joint training for emotional clusters
    end = time.time()
    print('======== Epoch '+str(epoch)+' ========')
    if model_type=='Temp-Triplet':
        loss_joint = train_joint_Triplet(emo_cluster_dataloader, model,
                                         criterion_Triplet, criterion_class, cc_coef,
                                         optimizer_class, optimizer_attri,
                                         epoch)
    else:
        loss_joint = train_joint_TempNet(emo_cluster_dataloader, model,
                                         criterion_class, cc_coef,
                                         optimizer_class, optimizer_attri,
                                         epoch)
    Loss_Joint.append(loss_joint)
    print('Loss Joint: '+str(loss_joint))
    
    try:
        print('=====================================')
        nmi = normalized_mutual_info_score(
            clustering.arrange_clustering(deepcluster.images_lists),
            clustering.arrange_clustering(cluster_log.data[-1])
        )
        NMI.append(nmi)
        print('NMI against previous assignment: {0:.3f}'.format(nmi))
    except IndexError:
        pass
    
    # Validation Stage: considering CCC performance only
    loss_valid = validation(valid_loader, model, cc_coef)
    Loss_CCC_Valid.append(loss_valid)
    cluster_log.log(deepcluster.images_lists)
    print('Loss Validation CCC: '+str(loss_valid))
    
    # save model checkpoint based on best validation CCC performance
    if epoch == 0:
        # initial CCC value
        loss_valid_best = loss_valid
        print("=> Saving the initial best model (Epoch="+str(epoch)+")")
        # save running checkpoint
        torch.save({'arch': arch,
                    'state_dict': model.state_dict()},
                     os.path.join(exp, 'TempDeepEmoCluster_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'_'+model_type+'_unlabel'+str(unlabel_size)+'.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)
        
    else:
        if loss_valid < loss_valid_best:
            print("=> Saving the best model (Epoch="+str(epoch)+")")
            print("=> CCC loss decrease from "+str(loss_valid_best)+" to "+str(loss_valid) )
            # save running checkpoint
            torch.save({'arch': arch,
                        'state_dict': model.state_dict()},
                         os.path.join(exp, 'TempDeepEmoCluster_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'_'+model_type+'_unlabel'+str(unlabel_size)+'.pth.tar'))
            # save cluster assignments
            cluster_log.log(deepcluster.images_lists)
            # update best CCC value
            loss_valid_best = loss_valid
        else:
            print("=> CCC did not improve (Epoch="+str(epoch)+")")
    print('=================================================================')
    
# save nmi/val-ccc trend for epochs
plt.plot(NMI,'bo--')
plt.plot(Loss_CCC_Valid,'ro--')
plt.savefig(exp+'NMI_CCC_trend_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'_'+model_type+'_unlabel'+str(unlabel_size)+'.png')
    
# save train loss trend for epochs
plt.plot(Loss_Joint,'ko--')
plt.savefig(exp+'TrainJoint_Loss_trend_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'_'+model_type+'_unlabel'+str(unlabel_size)+'.png')
