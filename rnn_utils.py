from typing import Callable

import pickle
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence
import torch.nn.functional as F

from sklearn.model_selection import ParameterGrid, ParameterSampler

from tqdm.notebook import tqdm

import warnings

import pandas as pd
import numpy as np
from math import ceil

import json
import re

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, accuracy_score

from tqdm.auto import tqdm

import torchmetrics.functional as f


class ICareDataset(Dataset):
    
    def __init__(self, 
                 diagnoses_file, 
                 universe_grouping, 
                 grouping='ccs', # desired grouping to use (for both input and output currently),
                 train_size:float = 0.70,
                 val_size:float = 0.15,
                 test_size:float = 0.15,
                 shuffle_dataset:bool = True,
                 random_seed :int = 432
                ):
        
        assert train_size+val_size+test_size == 1, 'Oops'

        with open(diagnoses_file,'r') as fp:
            self.raw_data = json.load(fp)

        # list patients
        self.patients = list(self.raw_data.keys())
        
        self.grouping = grouping
        self.universe_grouping=universe_grouping
        
        self.__preprocess()
        
        self.data = {}
        
        print('processing each patient')
        for pat in tqdm(self.raw_data):
            
            history_original = self.raw_data[pat][self.grouping]['history']
            target_original = self.raw_data[pat][self.grouping]['targets']
            
            history_mhot = self.adms2multihot(self.raw_data[pat][self.grouping]['history'])
            target_mhot = self.adms2multihot(self.raw_data[pat][self.grouping]['targets'])
            
            length = len(self.raw_data[pat][self.grouping]['history'])
            
            self.data[pat] = {'history_original': history_original,
                              'target_original': target_original,
                              'history_mhot':history_mhot,
                              'target_mhot':target_mhot,
                              'pid': pat,
                              'length':length
                             }
        
        # train-val-test splits
        dataset_size = len(self.patients)
        indices = list(range(dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            
        train_split = int(np.floor(train_size * dataset_size))
        val_split = int(np.floor(val_size * dataset_size))
        
        self.train_indices = indices[:train_split]
        self.val_indices = indices[train_split:train_split+val_split]
        self.test_indices = indices[(train_split+val_split):]
            
            
    def adms2multihot(self,adms):
        return (torch.stack(
                                [ F.one_hot( # list comprehension
                                    # create a multi-hot of diagnoses of each admission
                                     torch.tensor( 
                                         list(map(lambda code: self.grouping_data[self.grouping]['code2int'][code],
                                             set(admission) # we don't care about repeated codes
                                            ))
                                     ),
                                     num_classes=self.grouping_data[self.grouping]['n_labels']
                                 )
                                 .sum(dim=0)
                                 .float()
                                 if admission 
                                 else
                                 torch.zeros(size=(self.grouping_data[self.grouping]['n_labels'],))
                                 for admission in adms
                                ]
                            )
               )
    def __preprocess(self):
        # necessary data of each code_grouping (eg. ccs, chapters) for posterior padding and one_hot_encoding of batches
        self.grouping_data = {}
        for grouping_code in self.raw_data[list(self.raw_data.keys())[0]].keys():
            self.grouping_data[grouping_code] = {}

            # get all codes of this group
            all_data_grouping = self.universe_grouping

            # store n_labels this group
            self.grouping_data[grouping_code]['n_labels'] = len(set(all_data_grouping))

            # store unique sorted codes from dataset
            self.grouping_data[grouping_code]['sorted'] = sorted(set(all_data_grouping))

            # store code2int & int2code
            int2code = dict(enumerate(self.grouping_data[grouping_code]['sorted']))
            code2int = {ch: ii for ii, ch in int2code.items()}

            self.grouping_data[grouping_code]['int2code'] = int2code
            self.grouping_data[grouping_code]['code2int'] = code2int
            self.grouping_data[grouping_code]['int2code_converter'] = lambda idx: self.grouping_data[grouping_code]['int2code'][idx]

    def __str__(self):
        return 'Available groupings: ' +str(self.data[list(self.data.keys())[0]].keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        gets original converted from int2code
        """
        patient_data = self.data[self.patients[idx]]


        return {'history_original': patient_data['history_original'],
                'target_original': patient_data['target_original'],
                'history_mhot':patient_data['history_mhot'],
                'target_mhot':patient_data['target_mhot'],
                'pid': patient_data['pid'],
                'length':patient_data['length']
               }
    
    
    
class ICareCOLLATE:
    """
    This collate class gets a dataset in the format of:
    [
    {'train':[[[code1,code2],[code3]],[[etc..],[etc...]]]
      'target:':[[[code1],[code2]],[[etc..],[etc...]]]
    },
     {etc..},
     etc..
    ]
    
    And outputs a pack of train and pad of test sequences
    """
    def __init__(self):
        pass
    
    def __call__(self,batch):
        lengths = [pat['length'] for pat in batch]
        pids = [pat['pid'] for pat in batch]
        
        history_original = [pat['history_original'] for pat in batch]
        target_original = [pat['target_original'] for pat in batch]
        
        history_mhot = [pat['history_mhot'] for pat in batch]
        target_mhot = [pat['target_mhot'] for pat in batch]
        
        history_pack = pack_sequence(history_mhot,enforce_sorted=False)
        target_sequence = torch.vstack(target_mhot)
        
        return {'history_original':history_original,
                'target_original':target_original,
                'history_mhot':history_mhot,
                'target_mhot':target_mhot,
                'history_pack':history_pack,
                'target_sequence':target_sequence,
                'pids':pids,
                'lengths':lengths
               }
    

class DiagnosesDataset(Dataset):
    def __init__(self, diagnoses_file,
                 grouping='ccs' # desired grouping to use (for both input and output currently),
                ):
        
        # load admissions data
        with open(diagnoses_file,'r') as fp:
            self.data = json.load(fp)
        
        # list patients
        self.patients = list(self.data['data'].keys())
        
        self.grouping = grouping
        
        self.__preprocess()
            
    def __preprocess(self):
        # necessary data of each code_grouping (eg. ccs, chapters) for posterior padding and one_hot_encoding of batches
        self.grouping_data = {}
        for grouping_code in self.data['metadata']['groupings']:
            self.grouping_data[grouping_code] = {}
            
            # get all codes of this group
            all_data_grouping = [self.data['data'][pat][grouping_code] for pat in self.data['data']]
            
            #flatten list of lists of lists
            all_data_grouping = [item for sublist in all_data_grouping for item in sublist]
            all_data_grouping = [item for sublist in all_data_grouping for item in sublist]
            
            # store n_labels this group
            self.grouping_data[grouping_code]['n_labels'] = len(set(all_data_grouping))
            
            # store unique sorted codes from dataset
            self.grouping_data[grouping_code]['sorted'] = sorted(set(all_data_grouping))
            
            # store code2int & int2code
            int2code = dict(enumerate(self.grouping_data[grouping_code]['sorted']))
            code2int = {ch: ii for ii, ch in int2code.items()}
            
            self.grouping_data[grouping_code]['int2code'] = int2code
            self.grouping_data[grouping_code]['code2int'] = code2int
            self.grouping_data[grouping_code]['int2code_converter'] = lambda idx: self.grouping_data[grouping_code]['int2code'][idx]
        
        
    def __str__(self):
        return 'Available groupings: ' +str(self.data['metadata']['groupings'])

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, idx):
        """
        gets original converted from int2code
        """
        patient_data = self.data['data'][self.patients[idx]][self.grouping]
        
        train = patient_data[:-1]
        target = patient_data[1:]
        
        # remove duplicates (can happen in low granuality codes such as ccs)
        
        train = [list(set(admission)) for admission in train]
        target = [list(set(admission)) for admission in target]
        
        return {'train':train,
                'target':target,
                'pid':self.patients[idx]
               }


class RNN(nn.Module):
    
    def __init__(self,input_size,hidden_size,num_layers,n_labels,model='rnn'):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_labels = n_labels
        
        if model == 'rnn':
            self.model = nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers
                               )
        elif model == 'gru':
            self.model = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers
                             )
        elif model == 'lstm':
            self.model = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers
                             )
        else:
            raise ValueError(f'oops. expecting model in [rnn,lstm,gru], got model={model}')
        
        self.lin = nn.Linear(in_features = hidden_size,
                            out_features=n_labels
                           )
    
    def forward(self, input,ignore=False,take_mc_average=False):
        """
        input: pack_sequence
        
        """
        
        hn,_ = self.model(input)
        
        #linear layers don't support packs
        out = self.lin(pad_packed_sequence(hn,batch_first=True)[0])
        return out
    
    
def gen_non_padded_positions(lengths : list):
    """
    Given the lengths of sequences in a batch. returns all the 
    non-padded positions of all sequences in the batch, assuming 
    the batch is padded and has dimensions: (batch_size * max_seq_len, n_labels).
    
    Parameters
    ---------
    
    lengths : list
        list of lengths of each sequence of a batch
        
        
    Example
    -------
    
    # init
    model = ...
    criterion = ...
    
    # create batch of sequences of random sizes
    batch_size=64
    n_classes = 10
    random_seq_sizes = torch.randint(low=1,high=32,size=(batch_size,))
    # create inputs
    sequences = [torch.rand(size=(size,n_classes)) for size in random_seq_sizes]
    # create targets - in this case its multi-label target, but doesn't have to be.
    targets = torch.vstack([torch.randint(0,2,size=(size,n_classes) for size in random_seq_sizes)])
    
    sequences_packed = pack_sequence(sequences,enforce_sorted=False)
    outs = model(sequences_packed)
    
    non_padded_positions = gen_non_padded_positions(random_seq_sizes) # <-- function call here
    
    non_padded_outs = outs.view(-1,n_classes)[non_padded_positions,:]
    
    loss = criterion(non_padded_outs,targets)
    
    loss.backward()
    
    """
    
    max_seq_length = max(lengths)

    seq_size_per_seq = list(zip(range(0,len(lengths)),lengths))

    # i.e. [ (pos_in_batch,seq_index),(pos_in_batch,seq_index),...]
    # e.g. imagine batch of two seqs. first has size 2, second has size 1. produces: [[(0,0),(0,1)],[(1,0)]]
    real_seq_pos_per_seq = [list(zip([a[0]]*a[1],range(0,a[1]))) for a in seq_size_per_seq]
    
    # just flattens the previous list.
    # i.e. (taking the previous example) produces: [(0,0),(0,1),(1,0)]
    real_seq_pos_per_seq = [item for seq in real_seq_pos_per_seq for item in seq] 
    
    # position of sequence when all sequences are stacked
    non_padded_positions = [i[0] * max_seq_length + i[1] for i in real_seq_pos_per_seq]
    
    return non_padded_positions

    

def eval_model(model, dataloader, decision_thresholds, metrics, only_loss=False,name=None):
    """
    return either the loss or the loss and a metrics dataframe.
    
    
    Parameters:
    -----------
    
    decision_thresholds : pd.DataFrame
        The result of calling the method get_prediction_thresholds
    
    metrics : list
        ['roc,avgprec','acc','recall','precision','f1']
        
        
    Returns
    -------
    loss : torch.tensor
    
    metrics : pd.DataFrame, column_names ~ [roc,avgprec,accuracy,recall,precision,f1],
                index = ['median','mean','std']
        each column is followed by either "_diag" or "_adm"
        
    """
    
    loss = compute_loss(model, dataloader)
    
    if only_loss:
        return loss
    
    predictions = make_predictions(model_outputs,golden,decision_thresholds)
    
    metrics = compute_metrics(model_outputs,predictions,golden,metrics)
    
    if name is not None:
        metrics.name = name
    return loss,metrics


def compute_dataloader_loss(model,dataloader):
    """
    Computes the loss of N2N model on a particular dataloader.
    """
        
    model.eval()
    
    total_loss = 0
    total_sequences = 0
    n_labels = next(iter(dataloader))['history_mhot'][0].shape[-1]
    
    with torch.no_grad():
        print('forward passing each batch to compute the loss')
        for i, batch in tqdm(enumerate(iter(dataloader))):
            
            total_loss += compute_batch_loss(model,batch,reduction='sum')
            total_sequences += sum(batch['lengths'])
        
    loss = total_loss / (total_sequences * n_labels)
    return loss

def compute_batch_loss(model,batch,reduction):
    """
    Computes the loss (sum) of a batch. 
    Ignores padded_positions to obtain a more correct loss.
    
    Parameters
    ----------
    
    reduction : str | either 'mean' or 'sum'
        reduction of loss.
    """
    
    n_labels = batch['history_mhot'][0].shape[-1]
    
    criterion = nn.BCEWithLogitsLoss(reduction=reduction)
    
    outs = model(batch['history_pack'])
    
    non_padded_outs = outs2nonpadded(outs,batch['lengths'])
    
    total_loss = criterion(non_padded_outs,batch['target_sequence'])
    
    return total_loss

def outs2nonpadded(outs,lengths):
    """
    Parameters
    ----------
    
    outs : torch.tensor , shape = (batch_size, max_seq_len, n_classes)
        model outputs
    
    lengths : list
        lenghts of each sequence in a batch
    """
    
    non_padded_positions = gen_non_padded_positions(lengths)
    
    return outs.view(-1,outs.shape[-1])[non_padded_positions,:]


def train_one_batch(model,batch,optimizer):
    """
    Receives a batch of input and labels. trains a model on this data and returns the loss.
    """
    
    # zero the parameter gradients
    model.zero_grad()

    loss = compute_batch_loss(model,batch,reduction='mean')

    loss.backward()

    optimizer.step()
    
    return loss.item()


def train_one_epoch(model, train_loader, optimizer):
    
    model.train()
    
    print('Starting to train each batch')
    losses = list()
    for i, batch in tqdm(enumerate(iter(train_loader))):
        
        batch_loss = train_one_batch(model,batch,optimizer)
        
        losses.append(batch_loss)
        
    # last batch prob has different size so the mean isn't a weighted mean. but shouldn't affect too much
    dataloader_loss = np.mean(losses) 
    return dataloader_loss
        

############################
###### METRICS #############
############################

def compute_metrics(model, dataloader):
    
    metrics = None
    for i,batch in tqdm(enumerate(iter(dataloader))):
        
        batch_metrics = compute_metricsV3_batch(model,batch)
        if metrics is None:
            metrics = dict()
            for key in batch_metrics:
                metrics[key] = [batch_metrics[key]]
        else:
            for key in batch_metrics:
                metrics[key].append(batch_metrics[key])
    
    for key in batch_metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics

def compute_metrics_batch(model,batch):

    outs = model(batch['history_pack'])
    
    non_padded_outs = outs2nonpadded(outs,batch['lengths'])
    targets = batch['target_sequence'].int()
    
    return {'recall@30':f.recall(non_padded_outs,targets,top_k=30,average='samples').item(),
            'precision@30':f.precision(non_padded_outs,targets,top_k=30,average='samples').item(),
            'f1@30':f.f1_score(non_padded_outs,targets,top_k=30,average='samples').item(),
            'recall':f.recall(non_padded_outs,targets,average='samples').item(),
            'precision':f.precision(non_padded_outs,targets,average='samples').item(),
            'f1':f.f1_score(non_padded_outs,targets,average='samples').item()
           }