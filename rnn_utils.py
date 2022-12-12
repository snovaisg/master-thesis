from typing import Callable

import pickle
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch import nn
from torch.nn import ReLU,Linear,Module, Sigmoid
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence
import torch.nn.functional as F

from scipy.stats import skew as compute_skew
from scipy.stats import kurtosis as compute_kurtosis



from sklearn.model_selection import ParameterGrid, ParameterSampler,StratifiedKFold

from tqdm.notebook import tqdm

import warnings

import pandas as pd
import numpy as np
from math import ceil

import json
import re

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, accuracy_score 
from sklearn.base import BaseEstimator, ClassifierMixin

from tqdm.auto import tqdm

import torchmetrics.functional as f

from torchmetrics import Recall, Precision, F1Score, AUROC, AveragePrecision


class ICareDataset(Dataset):
    
    def __init__(self, 
                 diagnoses_file, 
                 universe_grouping,
                 input:str, #either history_cumulative or history
                 target:str, #either target or new_target
                 grouping='ccs', # desired grouping to use (for both input and output currently),
                 train_size:float = 0.70,
                 val_size:float = 0.15,
                 test_size:float = 0.15,
                 opt_size:float=0.15,
                 shuffle_dataset:bool = True,
                 random_seed :int = 432,
                 partial:int=None # number of patients to process. good for debugging
                ):
            
            
        self.input = input
        self.target = target
        
        assert train_size+val_size+test_size == 1, 'Oops'
        assert opt_size < train_size, 'Oops'

        with open(diagnoses_file,'r') as fp:
            self.raw_data = json.load(fp)

        # list patients
        self.patients = list(self.raw_data.keys())
        
        if partial is not None:
            assert type(partial) == int
            self.patients = self.patients[:partial]
        
        self.grouping = grouping
        self.universe_grouping=universe_grouping
        
        self.__preprocess()
        
        self.data = {}
        
        print('processing each patient')
        for pat in tqdm(self.patients):
            pat_data = dict()
            
            history_original = self.raw_data[pat][self.grouping]['history']
            if self.input == 'history_cumulative':
                history_cumulative = [[item for sublist in history_original[:i] for item in sublist] for i in range(1,len(history_original)+1)]
                history_cumulative_mhot = self.adms2multihot(history_cumulative).to(dtype=torch.float)
                history_cumulative_hot = torch.where(history_cumulative_mhot > 0,1,0).to(dtype=torch.float)
                pat_data['history_hot'] = history_cumulative_hot
                pat_data['history_original'] = history_cumulative
            elif self.input == 'history':
                history_mhot = self.adms2multihot(history_original).to(dtype=torch.float)
                history_hot = torch.where(history_mhot > 0,1,0).to(dtype=torch.float)
                pat_data['history_hot'] = history_hot
                pat_data['history_original'] = history_original
            else:
                raise ValueError()
            
                
            target_original = self.raw_data[pat][self.grouping]['targets']
            
            if self.target == 'new_target':
                new_target_original = self.raw_data[pat][self.grouping]['new_targets']
                new_target_mhot = self.adms2multihot(new_target_original).to(dtype=torch.int64)
                new_target_hot = torch.where(new_target_mhot>0,1,0).to(dtype=torch.float)
                pat_data['target_hot'] = new_target_hot
                pat_data['target_original'] = new_target_original
            elif self.target == 'target':
                target_mhot = self.adms2multihot(target_original).to(dtype=torch.int64)
                target_hot = torch.where(target_mhot>0,1,0).to(dtype=torch.float)
                pat_data['target_hot'] = target_hot
                pat_data['target_original'] = target_original
            else:
                raise ValueError()
            
            length = len(self.raw_data[pat][self.grouping]['history'])
            
            pat_data.update({'length':length,'pid':pat})
            self.data[pat] = pat_data
            
        
        # train-val-test splits
        dataset_size = len(self.patients)
        indices = list(range(dataset_size))
        
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            
        train_split = int(np.floor(train_size * dataset_size))
        val_split = int(np.floor(    val_size * dataset_size))
        test_split = int(np.floor(  test_size * dataset_size))
        opt_split = int(np.floor(  opt_size * dataset_size))
        
        self.train_indices = indices[:train_split]
        self.train_no_opt_indices = indices[:train_split-opt_split] # train without opt
        self.opt_indices = indices[train_split-opt_split:train_split]
        self.val_indices = indices[train_split:train_split+val_split]
        self.test_indices = indices[(train_split+val_split):train_split+val_split+test_split]
            
            
    def adms2multihot(self,adms):
        return (torch.stack(
                                [ F.one_hot( # list comprehension
                                    # create a multi-hot of diagnoses of each admission
                                     torch.tensor( 
                                         list(map(lambda code: self.grouping_data[self.grouping]['code2int'][code],
                                             admission 
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
        return patient_data

        #return {'history_original': patient_data['history_original'],
        #        'history_cumulative': patient_data['history_cumulative'],
        #        'target_original': patient_data['target_original'],
        #        'new_target_original':patient_data['new_target_original'],
        #        'history_hot':patient_data['history_hot'],
        #        'history_cumulative_hot':patient_data['history_cumulative_hot'],
        #        'target_hot':patient_data['target_hot'],
        #        'new_target_hot':patient_data['new_target_hot'],
        #        'history_mhot':patient_data['history_mhot'],
        #        'history_cumulative_mhot':patient_data['history_cumulative_mhot'],
        #        'target_mhot':patient_data['target_mhot'],
        #        'new_target_mhot':patient_data['new_target_mhot'],
        #        'pid': patient_data['pid'],
        #        'length':patient_data['length']
        #       }
    
    
    
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
        #self.POSSIBLE_INPUTS = ['history_hot','history_mhot','history_cumulative_hot','history_cumulative_mhot']
        #self.POSSIBLE_OUTPUTS = ['target_hot','new_target_hot']
        
        #assert input in self.POSSIBLE_INPUTS, f"input chosen doesn't exist. choose one of the following {str(self.POSSIBLE_INPUTS)}"
        
        #assert output in self.POSSIBLE_OUTPUTS, f"output chosen doesn't exist. choose one of the following {str(self.POSSIBLE_OUTPUT)}"
        #self.OTIMIZE = ['pid','length','history_original','target_original',input,output]
        
        #self.input = input
        #self.output = output
        #self.optimize = optimize
        pass
    
    def __call__(self,batch):
        
        result = {field:[pat[field] for pat in batch] for field in batch[0].keys()}

        result.update(
            dict(
                input_pack=pack_sequence(result['history_hot'],enforce_sorted=False),
                target_sequence=torch.vstack(result['target_hot'])
            )
        )
        result.pop('history_hot')
        result.pop('target_hot')
        
        return result
    

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

    
#############################################################
########################## MODELS ###########################
#############################################################

class repeatPast(nn.Module):
    """
    This is a baseline model where the output is repeating the past using one of 3 strategies:
    
    1. immediate: uses the last admission as prediction
    2. whole: uses the whole past diagnoses as prediction
    3. top@k: uses the top k most frequent diagnoses of the whole past as prediction
    
    """
    def __init__(self,how:str):
        super(repeatPast,self).__init__()
        self.supported_modes = ['last_admission','whole_history','top@']
        self.how = how
        self._check_inputs() # creates self.topk if no errors are raised
        
    def forward(self,input):
        """
        input : pack-sequence of 3 dimensions (batch_size, max_seq_length,n_labels)
        """
        history,lengths = pad_packed_sequence(input,batch_first=True)
        
        if self.how == 'last_admission':
            logits = history
        elif self.how == 'whole_history':
            logits = torch.cumsum(history,axis=1) #cumumulative sum of history of each sequence but at most, prediction is equal to 1
        elif self.how[:4] == 'top@':
            whole_history = torch.cumsum(history,axis=1)
            
            # obtain topk most frequent of each target
            pos_topk = torch.argsort(whole_history,axis=2,descending=True)

            # create index positions to build logits tensor
            logits_indexes = torch.tensor([(i,j,pos_topk[i,j,k]) for i in range(pos_topk.shape[0]) for j in range(pos_topk.shape[1]) for k in range(self.topk)])
            
            # create preds tensor
            logits = torch.zeros(size=whole_history.shape)
            logits[torch.split(logits_indexes,1,-1)] = 1
            
            # ignore predictions that don't exist in the history while retrieving initial frequencies to be used as logits
            logits = logits * whole_history
        
        return logits
    
    def _check_inputs(self):
        assert any([self.how[:4] in mode for mode in self.supported_modes]),\
               'Oops. Must specify one of the following:' + str(self.supported_modes)
        
        # check if top@k is well formatted with an integer after 'top@'
        if self.how[:4] in 'top@':
            try:
                self.topk = int(self.how[4:])
            except:
                raise ValueError('argument <how> Must contain an integer after top@. Eg.: top@10')


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
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input,ignore_sigmoid=False,**kwargs):
        """
        input: pack_sequence
        
        """
        
        hn,_ = self.model(input)
        
        #linear layers don't support packs
        out = self.lin(pad_packed_sequence(hn,batch_first=True)[0])
        
        if ignore_sigmoid:
            return out
        
        logits = self.sigmoid(out)
        return logits

class MLP(Module):
    """
    Adapted from https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/
    """
    def __init__(self, input_size,n_labels, hidden_sizes = [150]):
        super(MLP, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        
        self.model = nn.Sequential()
        
        self.model.add_module(f"hidden0", nn.Linear(input_size,hidden_sizes[0]))
        self.model.add_module(f"act_hidden0",nn.ReLU())
        
        for idx,hidden_size in enumerate(hidden_sizes[1:]):
            self.model.add_module(f"hidden{idx+1}", nn.Linear(hidden_sizes[idx],hidden_size))
            self.model.add_module(f"act_hidden{idx+1}",nn.ReLU())
        
        self.model.add_module(f"final",nn.Linear(hidden_sizes[-1],n_labels))
        
    def forward(self, input, ignore_sigmoid=False):
        
        
        #linear layers don't support packs
        X = pad_packed_sequence(input,batch_first=True)[0]
        
        outs = self.model(X)

        if ignore_sigmoid:
            return outs
        else:
            return self.sigmoid(outs)
        

class MyDropout(nn.Module):
    """
    Adapted from https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/
    """
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p
        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0-p)
        else:
            self.multiplier_ = 0.0
    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input
        
        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        selected_ = torch.Tensor(input.shape).uniform_(0,1)>self.p
        selected_.requires_grad = False
            
        # Multiply output by multiplier as described in the paper [1]
        return torch.mul(selected_,input) * self.multiplier_

class MLP_MC(Module):
    def __init__(self, input_size,n_labels,hidden_sizes = [150],drop_rates=[0., 0.]):
        """
        Adapted from https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/
        
        drop_rates : list
            First position is dropout applied to the input
            Second position is dropout applied to all the hidden units of the module
        """
        
        super(MLP_MC, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        
        self.model = nn.Sequential()
        
        self.model.add_module("dropout_input",MyDropout(drop_rates[0]))
        self.model.add_module("hidden0", nn.Linear(input_size,hidden_sizes[0]))
        self.model.add_module("act_hidden0",nn.ReLU())
        
        for idx,hidden_size in enumerate(hidden_sizes[1:]):
            
            self.model.add_module(f"dropout_hidden{idx}", MyDropout(drop_rates[1]))
            self.model.add_module(f"hidden{idx+1}",nn.Linear(hidden_sizes[idx],hidden_size))
            self.model.add_module(f"act_hidden{idx+1}",nn.ReLU())
        
        self.model.add_module(f"dropout_final", MyDropout(drop_rates[1]))
        self.model.add_module(f"final",nn.Linear(hidden_sizes[-1],n_labels))
        
    def forward(self, input, ignore_sigmoid=False):
        
        
        #linear layers don't support packs
        X = pad_packed_sequence(input,batch_first=True)[0]
        
        outs = self.model(X)

        if ignore_sigmoid:
            return outs
        else:
            return self.sigmoid(outs)

class MeanPredictive(Module):
    """
    Meant to be used after a dropout model.
    The forward of this class implements several forward passes and returns the mean
    """
    def __init__(self, model,T):
        super(MeanPredictive, self).__init__()
        
        self.model = model
        self.T = T # nº forward passes
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input, ignore_sigmoid=False):
        self.model.train()

        res = list()
        for t in range(self.T):
            logits = self.model(input)
            res.append(logits)
        assert torch.any(res[0] != res[1]), 'oops, dropout is not active'
        res = torch.stack(res).mean(axis=0)
        if ignore_sigmoid:
            return res
        return self.sigmoid(res)
        
class DumbClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass
    def fit(self, X, y):
        return self

    def predict(self, X):
        y = np.zeros(shape=(X.shape[0],),dtype=int)
        return y
    def predict_proba(self,X):
        
        return np.zeros(shape=(X.shape[0],2),dtype=float)

#############################################################
######################### DL UTILS ##########################
#############################################################


class MaskTimesteps():
    """
    creates masks above|below|equal to a position, for stacked sequences with variable sized lengths.
    
    # single sequence
    when length = [5] and level = 3
    
    if mode == 'below' returns [0,1,2]
    if mode == 'above' returns [2,3,4]
    if mode == 'equal' returns [2]
    
    # batch of sequences
    when length = [5,2,3] and level = 3
    
    if mode == 'below' returns [0,1,2,5,6,7,8,9] #note: jumps from 2 to 5 because it moved on to sequence 2 which starts at position 5.
    if mode == 'above' returns [2,3,4,9]
    if mode == 'equal' returns [2,9]
    """
    
    def __init__(self,mode,k=None):
        assert mode in ['above','below','equal','last','all'], 'Oops'
        self.mode = mode
        self.k = k
        
        # if mode != 'last' or 'all', then k has to be specified
        if mode not in ['last','all']:
            assert k is not None, 'Must specify k'
        
        if mode not in ['last','all']:
            self.name = self.mode + '_' + str(self.k)
        else:
            self.name = self.mode
      
    def __call__(self,lengths):
        selection_idx = []
        
        for seq_pos,length in enumerate(lengths):
            start_pos = sum(lengths[:seq_pos])
            if self.mode == 'above':
                for l in range(self.k-1,length):
                    selection_idx.append(start_pos+l)
            elif self.mode == 'below':
                for l in range(min([self.k,length])):
                    selection_idx.append(start_pos+l)
            elif self.mode == 'equal':
                if self.k <= length:
                    selection_idx.append(start_pos+self.k-1)
            elif self.mode == 'last':
                selection_idx.append(start_pos+length-1)
        if self.mode == 'all':
            selection_idx = list(range(sum(lengths)))
        return selection_idx
                
    
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


def compute_loss_dataloader(model, dataloader, criterion,timestep_selector:Callable=None):
    """
    Computes the loss of N2N model on a particular dataloader.
    """
        
    model.eval()
    loss = list()
    all_nseqs = list()
    
    print('Starting to compute the loss on the dataloader')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_loss = compute_loss_batch(model, batch, criterion,timestep_selector)
            if batch_loss is not None:
                loss.append(batch_loss.item())
        
    return np.mean(loss)
                
                
def compute_loss_batch(model,batch,criterion,timestep_selector:Callable=None,ignore_empty_targets=False):
    """
    Computes the loss (sum) of a batch. 
    Ignores padded_positions to obtain a more correct loss.
    timestep_selector may also ignore some sequences.
    
    Parameters
    ----------
    
        
        
    Returns
    ----------
    
    loss : irreductible loss of the model
    
    n_seqs: number of sequences eligible in this batch to compute the loss.
    """
    
    lengths = batch['length']
    targets = batch['target_sequence']
    
    outs = model(batch['input_pack'],ignore_sigmoid=True)
    
    non_padded_outs = outs2nonpadded(outs,batch['length'])
    
    if timestep_selector is not None:
        mask = timestep_selector(lengths)
        nseqs = len(mask)
        if mask:
            non_padded_outs = non_padded_outs[mask,:]
            targets = targets[mask,:]
        else: # no eligible sequences so no loss
            return None
        
    if ignore_empty_targets:
        non_empty_target_sequences = torch.where(targets.sum(axis=1) > 0)[0]
        non_padded_outs = non_padded_outs[non_empty_target_sequences,:]
        targets = targets[non_empty_target_sequences,:]
    
    if torch.numel(non_padded_outs) > 0:
        loss = criterion(non_padded_outs,targets)
        return loss
    return None


def compute_loss_decomposed(model,dataloader,criterion):
    loss_positives = list()
    loss_negatives = list()
    loss = list()
    print('Iterating dataloader to compute decomposed loss')
    for batch in tqdm(dataloader):

        lengths = batch['length']
        input = batch['input_pack']
        targets = batch['target_sequence']

        outs = model(input,ignore_sigmoid=True)
        outs = outs2nonpadded(outs,lengths)

        idx_positives = torch.where(targets.sum(axis=1) > 0)[0]
        outs_positives = outs[idx_positives,:]
        targets_positives = targets[idx_positives,:]
        loss_positives.append(criterion(outs_positives,targets_positives).item())

        idx_negatives = torch.where(targets.sum(axis=1) == 0)[0]
        outs_negatives = outs[idx_negatives,:]
        targets_negatives = targets[idx_negatives,:]
        loss_negatives.append(criterion(outs_negatives,targets_negatives).item())

        loss.append(criterion(outs,targets).item())
    return {'loss':np.nanmean(loss),
            'loss_positives':np.nanmean(loss_positives),
            'loss_negatives':np.nanmean(loss_negatives)
           }

def outs2nonpadded(outs,lengths):
    """
    
    Receives a tensor of a batch of variable-sized sequences. Since most sequences are padded to max_seq_len,
    we want to return only the actual positions that aren't paddings. 
    
    Note: We sacrifice a dimension, so the resulting input goes from (batch_size, max_seq_len, n_classes) to 
    (mult(lengths),n_classes) where mult() is like sum() but for multiplication.
    
    Parameters
    ----------
    
    outs : torch.tensor , shape = (batch_size, max_seq_len, n_classes)
        model outputs
    
    lengths : list
        lengths of each sequence in a batch
        
    Returns
    ------
    
    non_padded_outs : torch.tensor, shape = (mult(lengths), n_classes) where mult is like sum() but for multiplication
    """
    
    non_padded_positions = gen_non_padded_positions(lengths)
    
    return outs.view(-1,outs.shape[-1])[non_padded_positions,:]


def train_model_batch(model,batch,criterion,optimizer,timestep_selector:Callable=None,ignore_empty_targets=False):
    """
    Receives a model and a batch of input and labels.
    Trains a model on this data and returns the loss.
    """
    
    model.train()
    
    # zero the parameter gradients
    model.zero_grad()

    loss = compute_loss_batch(model,batch,criterion,timestep_selector,ignore_empty_targets)
    if loss is not None: # timestep selector may return zero, making it impossible to compute loss on this batch

        loss.backward()

        optimizer.step()
    
        return loss.item()
    
    return None


def train_model_dataloader(model, dataloader, criterion, optimizer,timestep_selector:Callable=None,ignore_empty_targets=False):
    
    model.train()
    
    print('Starting to train each batch')
    loss = list()
    
    for batch in tqdm(dataloader):
        
        batch_loss = train_model_batch(model,batch,criterion,optimizer,timestep_selector,ignore_empty_targets)
        if batch_loss is not None:
            loss.append(batch_loss)
        
    return np.nanmean(loss)


def compute_model_logits_batch(model,batch,timestep_selector:Callable=None):
    
    lengths = batch['length']
    logits = model(batch['input_pack'])
    logits = outs2nonpadded(logits,lengths)
    
    if timestep_selector is not None:
        mask = timestep_selector(lengths)
        logits = logits[mask,:]
    
    return logits

def compute_model_preds_batch(model,batch,thresholds : dict,topk=None,timestep_selector:Callable=None):
    
    lengths = batch['length']
    logits = compute_model_logits_batch(model,batch,timestep_selector)
    
    preds = logits2preds(logits,thresholds,topk)
    
    return preds


def logits2preds(logits,thresholds : dict,topk:int=None):
    """
    If topk is specifided, uses topk logits for predictions.
    Otherwise computes predictions given decision thresholds
    
    Parameters
    logits : torch.tensor, shape= (n_examples, n_labels), or shape=(batch_size,max_seq_length,n_labels)
    
    thresholds : dict, example {0:0.5, 1:0.43, 2:0.57, ...} (one for each diagnostic)
    """
    
    if topk is not None:
        sorted_idx = torch.argsort(logits,descending=True)[:,:topk]
        preds = F.one_hot(sorted_idx,num_classes=logits.shape[-1]).sum(axis=1)
    else:
        assert len(thresholds) == logits.shape[-1], "Last dimension must match. It's supposed to be the universe of diagnostics"

        # create thresholds matrix with the same shape as logits
        ths = torch.tensor([thresholds[diag] for diag in range(len(thresholds))]).expand(logits.shape)

        # computes preds
        preds = torch.where(logits > ths,1,0)
    
    return preds
        
    
##########################################################################
########################## POSTERIOR PREDICTIVE ##########################
##########################################################################
    
    
def compute_model_predictive_samples_batch(model,batch,timestep_selector,T:int=25):
    model.train() #activate dropout
    res = []
    for t in range(T):
        logits = compute_model_logits_batch(model,batch,timestep_selector)
        res.append(logits)
    return torch.stack(res) # first dimension is each forward pass of the batch

def compute_model_predictive_logits_batch(model,batch,timestep_selector,T):
    
    logits_T = compute_model_predictive_samples_batch(model,batch,timestep_selector,T)
    # predictions are the mean of the forward passes
    logits = torch.mean(logits_T,axis=0)
    return logits

def compute_predictive_stats(logits_T):
    """
    logits: shape=(N_passes,sequences_batch,n_labels)
    """
    mean = torch.mean(logits_T,axis=0)
    median = torch.quantile(logits_T,q=0.5,dim=0)
    quart1 = torch.quantile(logits_T,q=0.25,dim=0)
    quart3 = torch.quantile(logits_T,q=0.75,dim=0)
    var = torch.var(logits_T,axis=0)
    std = torch.sqrt(var)
    skew = 3 * (mean - median) / std
    stats = torch.stack([mean,quart1,median,quart3,var,std,skew],dim=1)
    return dict(stats=stats,stats_names=['mean','q25','median','q75','var','std','skew'])

def compute_predictive_stats_V3(logits_T):
    """
    logits: shape=(N_passes,sequences_batch,n_labels)
    """
    
    #std = torch.sqrt(torch.var(logits_T,axis=0))
    mean = torch.mean(logits_T,axis=0)
    
    median = torch.quantile(logits_T,q=0.5,dim=0)
    quant10 = torch.quantile(logits_T,q=0.10,dim=0)
    #quant25 = torch.quantile(logits_T,q=0.25,dim=0)
    #quant75 = torch.quantile(logits_T,q=0.75,dim=0)
    quant90 = torch.quantile(logits_T,q=0.90,dim=0)
    skew = torch.Tensor(compute_skew(logits_T.detach(),axis=0))
    kurtosis = torch.Tensor(compute_kurtosis(logits_T.detach(),axis=0))
    
    #q50_10 = median - quant10
    #q50_25 = median - quant25
    #q50_75 = quant75 - median
    q50_90 = quant90 - median
    #stats = torch.stack([mean,median,std,quant10,quant25,quant75,quant90,skew,kurtosis,q50_10,q50_25,q50_75,q50_90],dim=1)
    stats = torch.stack([mean,skew,kurtosis,q50_90],dim=1)
    return dict(stats=stats,stats_names=['mean','skew','kurtosis','iqd_50_90'])

def compute_predictive_stats_V2(logits_T):
    """
    logits: shape=(N_passes,sequences_batch,n_labels)
    """
    
    std = torch.sqrt(torch.var(logits_T,axis=0))
    mean = torch.mean(logits_T,axis=0)
    
    median = torch.quantile(logits_T,q=0.5,dim=0)
    quant10 = torch.quantile(logits_T,q=0.10,dim=0)
    quant25 = torch.quantile(logits_T,q=0.25,dim=0)
    quant75 = torch.quantile(logits_T,q=0.75,dim=0)
    quant90 = torch.quantile(logits_T,q=0.90,dim=0)
    skew = torch.Tensor(compute_skew(logits_T.detach(),axis=0))
    kurtosis = torch.Tensor(compute_kurtosis(logits_T.detach(),axis=0))
    
    q50_10 = median - quant10
    q50_25 = median - quant25
    q50_75 = quant75 - median
    q50_90 = quant90 - median
    stats = torch.stack([mean,median,std,quant10,quant25,quant75,quant90,skew,kurtosis,q50_10,q50_25,q50_75,q50_90],dim=1)
    return dict(stats=stats,stats_names=['mean','median','std','q10','q25','q75','q90','skew','kurtosis','iqd_50_10','iqd_50_25','iqd_50_75','iqd_50_90'])
    
def compute_model_predictive_stats_dataloader(model,dataloader,timestep_selector,T:int = 25):
    all_stats = list()
    all_targets = list()
    print('Starting to iterate dataloader to compute predictive distribution statistics')
    for batch in tqdm(dataloader):
        logits_T = compute_model_logits_batch_T(model,batch,None,T)
        all_stats.append(compute_predictive_stats(logits_T))
        all_targets.append(batch['target_sequence'])
    stats_names = all_stats[0].keys()
    all_stats = torch.stack([torch.vstack([e[key] for e in all_stats]) for key in all_stats[0].keys()]).detach().numpy()
    all_targets = torch.vstack(all_targets).detach().numpy()
    return dict(stats_names=stats_names,all_stats=all_stats,all_targets=all_targets)

#############################################################
########################## METRICS ##########################
#############################################################


def build_default_metrics(n_labels):
    return {#'recall_micro': Recall(num_classes=n_labels,average='micro',multiclass=False),
           #'recall_macro': Recall(num_classes=n_labels,average='macro',multiclass=False),
           #'precision_micro': Precision(num_classes=n_labels,average='micro',multiclass=False),
           #'precision_macro': Precision(num_classes=n_labels,average='macro',multiclass=False),
           #'f1score_weigthed':F1Score(num_classes=n_labels,average='weighted',multiclass=False),
           #'f1score_macro':F1Score(num_classes=n_labels,average='macro',multiclass=False),
           'auroc_weighted':AUROC(num_classes=n_labels,average=None,multiclass=False),
           'avgprec_weighted':AveragePrecision(num_classes=n_labels,average=None,multiclass=False)
          }

class Metrics:
    """
    Wrapper for torchmetrics that implements the basic functions (update and compute) but for several multilabel metrics (without average!)
    
    Accepts a dictionary like: {metric_name:torch.metric, metric_name:torch.metric, ...} 
    """
    
    def __init__(self,metrics : dict):
        
        self.metrics = metrics
        self.n_labels = None
        self.positives = None
        self.all_positives = None
        
        
    def update(self,logits,preds,target):
        if self.n_labels is None:
            self.n_labels = logits.shape[-1]
            self.positives = {label:0 for label in range(self.n_labels)}
        
        # update positives count
        target_positives = target.sum(axis=0)
        for label in self.positives:
            self.positives[label] += target_positives[label].item()
            self.all_positives = sum(self.positives.values())
        
        #update metrics
        for key in self.metrics:
            if 'roc' in key or 'avgprec' in key:
                self.metrics[key].update(logits,target.int())
            else:
                self.metrics[key].update(preds,target.int())
        
    def compute(self):
        res = {}
        for key in self.metrics:
            metric_result = self.metrics[key].compute()
            metric_result = [0 if torch.isnan(e) else e.item() for e in metric_result]

            res[key] = {'each':{label:value for label,value in enumerate(metric_result)},
                        'weighted': sum([self.positives[label]/self.all_positives*value for label,value in enumerate(metric_result)])
                       }
        return res
    
    
class MetricsV2:
    """
    Wrapper for torchmetrics that implements the basic functions (update and compute) but for several multilabel metrics (without average!)
    
    Accepts a dictionary like: {metric_name:torch.metric, metric_name:torch.metric, ...} 
    """
    
    def __init__(self,metrics_factory : dict, timestep_selectors : list = [MaskTimesteps('all')]):
        
        self.metrics_factory = metrics_factory
        self.timestep_selectors = timestep_selectors
        self.selectors = timestep_selectors
        self.n_labels = None
        
        self.metrics = self._build_metrics()
        self.stats = self._build_stats_for_selectors(self.timestep_selectors)
    
    def reset_metrics(self):
        self.metrics = self._build_metrics()
        self.stats = self._build_stats_for_selectors(self.timestep_selectors)
        
    def update_metrics_dataloader(self,model,dataloader,thresholds):
        print('Starting to iterate dataloader to update metrics')
        for batch in tqdm(dataloader):
            self.update_metrics_batch(model,batch,thresholds)
        return
        
    def update_metrics_batch(self,model,batch,thresholds : dict,topk=None):
        
        self._update_stats_batch(batch)
        
        targets = batch['target_sequence']
        lengths = batch['length']
        
        for selector in self.selectors:
            logits = compute_model_logits_batch(model,batch,selector)
            preds = logits2preds(logits,thresholds,topk)
            
            mask = selector(lengths)
            selected_target = targets[mask,:].int()
            
            for metric_name in self.metrics:
                if 'roc' in metric_name or 'avgprec' in metric_name or "@" in metric_name:
                    self.metrics[metric_name][selector.name]['meta']['metric'].update(logits,selected_target)
                else:
                    print(preds.shape)
                    print(logits.shape)
                    print(preds)
                    print(preds.sum(axis=-1))
                    self.metrics[metric_name][selector.name]['meta']['metric'].update(preds,selected_target)
        return
    
    def compute_metrics(self):
        for metric_name in self.metrics:
            for selector_name in self.metrics[metric_name]:
                result = self.metrics[metric_name][selector_name]['meta']['metric'].compute()
                self.metrics[metric_name][selector_name]['results']['original'] = result
                
                for extra_pool in self.metrics[metric_name][selector_name]['meta']['extra_pooling']:
                    if extra_pool == 'weighted':
                        weighted_result = sum([torch.nan_to_num(value,0)*self.stats[selector_name]['positives'][label]/self.stats[selector_name]['all_positives'] for label,value in enumerate(result)])
                        self.metrics[metric_name][selector_name]['results'][extra_pool] = weighted_result.item()
                    else:
                        print(f"Don't recognize extra_pool: ",extra_pool)
        return self.show_metrics()
    
    def show_metrics(self):
        res = {}
        for metric_name in self.metrics:
            res[metric_name] = dict()
            for selector_name in self.metrics[metric_name]:
                res[metric_name][selector_name] = dict()
                for result_type in self.metrics[metric_name][selector_name]['results']:
                    res[metric_name][selector_name][result_type] = self.metrics[metric_name][selector_name]['results'][result_type]
        return res
        
        
        
    def _build_metrics(self):
        metrics = {}
        for metric in self.metrics_factory:
            func = self.metrics_factory[metric]['func']
            kwargs = self.metrics_factory[metric]['kwargs']
            
            metrics[metric] = dict()
            for selector in self.timestep_selectors:
                metrics[metric][selector.name] = {'meta':dict(metric=func(**kwargs),
                                                               selector=selector,
                                                              extra_pooling=self.metrics_factory[metric]['extra_pooling']
                                                              ),
                                                  'results':dict()
                                                 }
        return metrics
    
    def _build_stats_for_selectors(self,selectors):
        """
        Counting the positives helps perform weight averages
        """
        stats = dict()
        for selector in selectors:
            stats[selector.name] = dict(positives=None,all_positives=0)
        return stats
    
    def _update_stats_batch(self,batch):
        
        for selector in self.selectors:
            positives_each = compute_positives_batch(batch,how='each',timestep_selector=selector)

            if self.n_labels is None:
                self.n_labels = batch['target_sequence'].shape[-1]
            if self.stats[selector.name]['positives'] is None:
                self.stats[selector.name]['positives'] = {label:0 for label in range(self.n_labels)}
            
            for label in positives_each:
                self.stats[selector.name]['positives'][label] += positives_each[label]
            self.stats[selector.name]['all_positives'] = sum(self.stats[selector.name]['positives'].values())
        return 
    

def compute_metrics_dataloader(metrics, model, dataloader, thresholds: dict,topk=None,timestep_selector:Callable=None):
    """
    
    Parameters
    ----------
    thresholds : dict, example {0:0.5, 1:0.43, 2:0.57, ...} (one for each diagnostic)
    
    """
    model.eval()
    
    
    with torch.no_grad():
        print('Starting to iterate the dataloader to update metrics')
        for batch in tqdm(dataloader):

            lengths = batch['length']
            logits = compute_model_logits_batch(model,batch)
            targets = batch['target_sequence']
            
            if timestep_selector is not None:
                eligible_timesteps = timestep_selector(lengths)
                if eligible_timesteps:
                
                    logits = logits[eligible_timesteps,:]
                    targets = targets[eligible_timesteps,:]
                else:
                    continue
            
            preds = logits2preds(logits,thresholds,topk)
            metrics.update(logits,preds,targets)

    print('Now its time to compute metrics. this may take a while')
    return metrics.compute()

def compute_metrics_batch(metrics: dict, model, batch, thresholds : dict,topk=None):
    """
    Returns metrics of a batch. By default returns the sum over records (and you can average it later). But you can set average=True.
    
    Parameters
    ----------
    thresholds : dict, example {0:0.5, 1:0.43, 2:0.57, ...} (one for each diagnostic)
    """
    
    logits = compute_model_logits_batch(model,batch)
    preds = logits2preds(logits,thresholds,topk)
    

    metrics.update(logits=logits,preds=preds,target=batch['target_sequence'])
    
    return metrics.compute()


##################################################################
########################### TPFP dataset ########################
##################################################################

def build_tp_fp_dataset(mc_model,dataloader,thresholds,topk,stats_fun):
    
    n_labels = next(iter(dataloader))['target_sequence'].shape[-1]
    dataset_tp_fp = {diag:{'features':list(),'target':list()} for diag in range(n_labels)}
    feature_names = None
    
    print('Iterating the dataloader to produce the TP/FP dataset')
    for batch in tqdm(dataloader):

        # compute logits and stats of the posterior predictive
        logits_T = compute_model_predictive_samples_batch(mc_model,batch,MaskTimesteps('all'),T=25)
        predictive_stats = stats_fun(logits_T)

        # get the mean of logits
        mean_pos = predictive_stats['stats_names'].index('mean')
        mean_logits = predictive_stats['stats'][:,mean_pos,:]
        
        # save feature names
        if feature_names is None:
            feature_names = predictive_stats['stats_names']

        # get top30 logits as predictions
        preds = logits2preds(mean_logits,thresholds,topk)

        # get labels
        P = preds == 1 # Positives
        TP = P & (batch['target_sequence'] == 1) # True positives (target)

        # save only datapoints from positives
        for diag in range(logits_T.shape[-1]):
            dataset_tp_fp[diag]['features'].append(predictive_stats['stats'][P[:,diag],:,diag].type(torch.float16))
            dataset_tp_fp[diag]['target'].append(TP[P[:,diag],diag].int().view(-1,).type(torch.int8))

    for diag in dataset_tp_fp:
        dataset_tp_fp[diag]['features'] = torch.vstack(dataset_tp_fp[diag]['features']).detach().numpy()
        dataset_tp_fp[diag]['target'] = torch.hstack(dataset_tp_fp[diag]['target']).detach().numpy()
        dataset_tp_fp[diag]['feature_names'] = feature_names
        
    dataset_tp_fp[diag]['positive_class'] = 'True Positives' # for reference
    return dataset_tp_fp


def build_tp_fp_statistics(dataset_tp_fp):
    """
    Given a TPFP dataset, computes the following statistics: 
    size, n_positives, pos_prevalence (positive prevalence).
    Also computes the average precision when using just the mean as the final logit.
    And computes the avg prec in different splits to test stability.
    
    Parameters
    ----------
    dataset_tp_fp : dict, keys= {<diag>:{'features':np.ndarray,'target':np.ndarray}}
    
    Returns
    -------
    None (adds the additional statistics to the dictionary that was passed as input)
    """
    
    n_splits = 3
    splitter = StratifiedKFold(n_splits=n_splits,shuffle=True)
    
    print('Starting to iterate TPFP dataset to compute statistics and original average precision')
    for diag in tqdm(dataset_tp_fp):
        
        dataset_tp_fp[diag]['size'] = dataset_tp_fp[diag]['target'].shape[0]
        dataset_tp_fp[diag]['n_positives'] = (dataset_tp_fp[diag]['target']==1).sum()
        
        

        if dataset_tp_fp[diag]['size'] == 0:
            dataset_tp_fp[diag]['pos_prevalence'] = 0
            dataset_tp_fp[diag]['avgprec'] = dict(original=dict(full=np.nan,mean=np.nan,std=np.nan))
        else:
            dataset_tp_fp[diag]['pos_prevalence'] = round(dataset_tp_fp[diag]['n_positives'] / dataset_tp_fp[diag]['size'],5)
            
            minority_class = 0 if dataset_tp_fp[diag]['pos_prevalence'] > 0.5 else 1
            min_req_4_kfold = dataset_tp_fp[diag]['n_positives'] if minority_class == 1 else dataset_tp_fp[diag]['size'] - dataset_tp_fp[diag]['n_positives']
            
            # copute average precision of original logits (aka: when the mean of the posterior is the final logit)
            full_avgprec = np.nan
            if dataset_tp_fp[diag]['pos_prevalence'] != 0:
                full_avgprec = average_precision_score(dataset_tp_fp[diag]['target'],
                                                       dataset_tp_fp[diag]['features'][:,0]
                                                      )
            if np.isnan(full_avgprec): # sometimes we get a nan that is a np.float and messes up checks later on
                full_avgprec = np.nan
                
            # now compute at different folds to test stability
            scores_original = list()
            if (min_req_4_kfold > n_splits) and full_avgprec is not np.nan:
                for train_idx,val_idx in splitter.split(dataset_tp_fp[diag]['features'],dataset_tp_fp[diag]['target']):
                    scores_original.append(average_precision_score(dataset_tp_fp[diag]['target'][val_idx],
                                                                   dataset_tp_fp[diag]['features'][val_idx,0])
                                          )
            mean_avgprec_original = np.nan
            std_avgprec_original = np.nan
            
            if scores_original:
                mean_avgprec_original = np.mean(scores_original)
                std_avgprec_original = np.std(scores_original)
            
            dataset_tp_fp[diag]['avgprec'] = {'original': dict(full=full_avgprec,
                                                               mean=mean_avgprec_original,
                                                               std=std_avgprec_original
                                                              )
                                             }

def train_tpfp_models(dataset_tp_fp):
    n_splits = 3
    corr_threshold = 0.8
    
    print('Starting to iterate tpfp dataset to train Logistic Regressions')
    for diag in tqdm(dataset_tp_fp):
        dataset_tp_fp[diag]['correlation'] = dict()


        if dataset_tp_fp[diag]['pos_prevalence'] in [np.nan,0,1] or dataset_tp_fp[diag]['size'] < 30:
            # not enough data to create model
            pipeline = np.nan
            mean_avgprec = np.nan
            std_avgprec = np.nan
            full_avgprec = np.nan
        else:
            X = dataset_tp_fp[diag]['features']
            y = dataset_tp_fp[diag]['target']

            linear_correlations = eval_linear_correlations(X,corr_threshold)
            dataset_tp_fp[diag]['correlation']['to_drop'] = [dataset_tp_fp[diag]['feature_names'][i] for i in linear_correlations['to_drop']]
            dataset_tp_fp[diag]['correlation']['to_keep'] = [dataset_tp_fp[diag]['feature_names'][i] for i in linear_correlations['to_keep']]
            dataset_tp_fp[diag]['correlation']['to_keep_idx'] = linear_correlations['to_keep']
            

            # drop linearly correlated features
            features_to_keep = linear_correlations['to_keep']
            X = X[:,features_to_keep]

            pipeline = make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced'))
            scores = cross_val_score(pipeline, X, y, cv=n_splits, scoring='average_precision')
            mean_avgprec = np.mean(scores)
            std_avgprec = np.std(scores)
            
            pipeline.fit(X,y)
            
            logits = pipeline.predict_proba(X)[:,1]
            full_avgprec = average_precision_score(y,logits)
            

        dataset_tp_fp[diag]['pipeline'] = pipeline
        dataset_tp_fp[diag]['avgprec']['model'] = dict(full=full_avgprec,mean=mean_avgprec,std=std_avgprec)
        
def eval_tpfp_models(train_dataset_tp_fp,val_dataset_tp_fp):
    """
    evalues the pipelines in train_dataset_tp_fp on the dataset val_dataset_tp_fp.
    Records results in val_dataset_tp_fp.
    """
    n_splits = 3
    print('Starting to iterate dataset to evaluate the LR models on a validation set')
    for diag in val_dataset_tp_fp:
        if train_dataset_tp_fp[diag]['pipeline'] is np.nan:
            avgprec = np.nan
        else:
            X = val_dataset_tp_fp[diag]['features']
            y = val_dataset_tp_fp[diag]['target']

            # drop linearly correlated features
            features_to_keep = train_dataset_tp_fp[diag]['correlation']['to_keep_idx']
            X = X[:,features_to_keep]

            pipeline = train_dataset_tp_fp[diag]['pipeline']

            logits = pipeline.predict_proba(X)[:,1]
            avgprec = average_precision_score(y,logits)
        val_dataset_tp_fp[diag]['avgprec']['model'] = dict(full=avgprec)
        
def train_tpfp_modelsSearchOverfit(train_dataset_tp_fp, val_dataset_tp_fp):
    n_splits = 3
    corr_threshold = 0.8
    
    Uparams = dict(C=np.linspace(0,1,11)[1:],class_weight=[None,'balanced'])
    params = ParameterGrid(Uparams)
    
    print('Starting to iterate tpfp dataset to train Logistic Regressions')
    for diag in tqdm(train_dataset_tp_fp):
        train_dataset_tp_fp[diag]['correlation'] = dict(to_keep=[],to_drop=[],to_keep_idx=[])


        if train_dataset_tp_fp[diag]['pos_prevalence'] in [np.nan,0,1] or val_dataset_tp_fp[diag]['pos_prevalence'] in [np.nan,0,1]:# or train_dataset_tp_fp[diag]['size'] < 30:
            # not enough data to create model
            pipeline = np.nan
            avgprec_val = np.nan
            avgprec_train = np.nan
            best_param = np.nan
        else:
            X_train = train_dataset_tp_fp[diag]['features']
            y_train = train_dataset_tp_fp[diag]['target']
            X_val = val_dataset_tp_fp[diag]['features']
            y_val = val_dataset_tp_fp[diag]['target']

            linear_correlations = eval_linear_correlations(X_train,corr_threshold)
            train_dataset_tp_fp[diag]['correlation']['to_drop'] = [train_dataset_tp_fp[diag]['feature_names'][i] for i in linear_correlations['to_drop']]
            train_dataset_tp_fp[diag]['correlation']['to_keep'] = [train_dataset_tp_fp[diag]['feature_names'][i] for i in linear_correlations['to_keep']]
            train_dataset_tp_fp[diag]['correlation']['to_keep_idx'] = linear_correlations['to_keep']
            

            # drop linearly correlated features
            features_to_keep = linear_correlations['to_keep']
            X_train = X_train[:,features_to_keep]
            X_val = X_val[:,features_to_keep]
            
            best_param = None
            best_avgprec = None
            for param in params:
                pipeline = make_pipeline(StandardScaler(),LogisticRegression(**param))
                pipeline.fit(X_train,y_train)
                logits = pipeline.predict_proba(X_val)[:,1]
                avgprec = average_precision_score(y_val,logits)
                if best_avgprec is None:
                    best_avgprec = avgprec
                    best_param = param
                elif best_avgprec < avgprec:
                    best_avgprec = avgprec
                    best_param = param
                    
            # retrain the model with best parameters
            # and record train and val metrics
            pipeline = make_pipeline(StandardScaler(),LogisticRegression(**best_param))
            pipeline.fit(X_train,y_train)
            logits = pipeline.predict_proba(X_train)[:,1]
            avgprec_train = average_precision_score(y_train,logits)
            
            logits_val = pipeline.predict_proba(X_val)[:,1]
            avgprec_val = average_precision_score(y_val,logits_val)
            
            

        train_dataset_tp_fp[diag]['grid_search_pipeline'] = pipeline
        train_dataset_tp_fp[diag]['avgprec']['grid_search_model'] = dict(train_score=avgprec_train,
                                                                   val_score=avgprec_val,
                                                                   best_params=best_param
                                                                  )
        val_dataset_tp_fp[diag]['grid_search_pipeline'] = pipeline
        val_dataset_tp_fp[diag]['avgprec']['grid_search_model'] = dict(train_score=avgprec_train,
                                                                   val_score=avgprec_val,
                                                                   best_params=best_param
                                                                  )
        
def eval_linear_correlations(feature_matrix : np.ndarray, threshold=0.8):
    """
    Returns the features with pearson correlation above and below <threshold>
    
    
    Parameters
    ----------
    feature_matrix: np.array, shape=(n_datapoints,n_features)
    
    threshold : float, 
        Pearson correlation threshold
    
    Returns
    --------
    dict -> {'to_drop':<list of indices of features to drop>,
             'to_keep':<list of indices of features to keep> 
            } # yeah a bit redundant but helps later on to do some statistics
    """
    corr_matrix = np.abs(np.corrcoef(feature_matrix,rowvar=False)).round(4)
    corr_matrix_upper = np.triu(corr_matrix)
    corr_matrix_upper = np.where(corr_matrix_upper == 1,0,corr_matrix_upper)
    to_drop = [column for column in range(len(corr_matrix_upper)) if any(corr_matrix_upper[:,column] > threshold)]
    to_keep = [c for c in range(len(corr_matrix_upper)) if c not in to_drop]
    return dict(to_drop=to_drop,to_keep=to_keep)

##################################################################
########################### THRESHOLDS ###########################
##################################################################

def compute_thresholds_dataloader(model,dataloader,timestep_selector:Callable=None):
    """
    Computes the best thresholds for a given based on max f1 score on that dataloader
    """
    # load logits and targets into memory
    print('Starting to compute thresholds')
    
    all_logits = list()
    all_target = list()
    print('Iterating the dataloader to obtain the logits and targets')
    for batch in tqdm(iter(dataloader)):
        lengths = batch['length']
        logits = compute_model_logits_batch(model,batch)
        targets = batch['target_sequence']
        
        if timestep_selector is not None:
            mask = timestep_selector(lengths)
            if not mask:
                continue
            logits = logits[mask,:]
            targets = targets[mask,:]

        all_target.append(targets)
        all_logits.append(logits)

    all_logits = torch.vstack(all_logits)
    all_target = torch.vstack(all_target)
    
    # compute best threshold for each diagnostic, one at a time
    
    def weighted_average(precision_weight,recall_weight):
        assert precision_weight + recall_weight == 1
        return lambda precision,recall: precision_weight * precision + recall_weight *recall
    
    f1score = lambda precision,recall: 2 * precision * recall / (precision + recall)
    
    scores_db = {'f1_score':f1score}#,'prec_focus':weighted_average(0.6,0.4),'recall_focus':weighted_average(0.4,0.6)}
    
    max_scores = {key:list() for key in scores_db}
    best_thresholds = {key:list() for key in scores_db}
    all_prevalence = list()
    all_positives = list()

    num_elements = all_target.shape[0]
    
    
    print(f'Computing the thresholds for each of the {all_target.shape[1]} diagnostics')
    for diag in tqdm(range(all_target.shape[1])):
        precision, recall, thresholds = precision_recall_curve(all_target.numpy()[:,diag],
                                                               all_logits.detach().numpy()[:,diag])
        
        positives = all_target[:,diag].sum().item()
        prevalence = positives / num_elements
        
        for scoring_fun in scores_db:

            scores = scores_db[scoring_fun](precision,recall)

            try:
                max_idx = np.nanargmax(scores)
                best_score = scores[max_idx]
                best_ths = thresholds[max_idx]
            except:
                best_score = np.nan
                best_ths = np.nan

            max_scores[scoring_fun].append(best_score)
            best_thresholds[scoring_fun].append(best_ths)
        all_prevalence.append(prevalence)
        all_positives.append(positives)
        

    return pd.DataFrame(data=[max_scores['f1_score'],best_thresholds['f1_score'],all_prevalence,all_positives],
             index=["f1_score","best_thresholds","all_prevalence","all_positives"]
            ).T

def compute_thresholds_recall_above_topk_dataloader(model,dataloader,topk,timestep_selector:Callable=None):
    """
    Computes the best thresholds for a given based on max f1 score on that dataloader
    """
    # load logits and targets into memory
    print('Starting to compute thresholds')
    
    all_logits = list()
    all_target = list()
    print('Iterating the dataloader to obtain the logits and targets')
    for batch in tqdm(iter(dataloader)):
        lengths = batch['length']
        logits = compute_model_logits_batch(model,batch)
        targets = batch['target_sequence']
        
        if timestep_selector is not None:
            mask = timestep_selector(lengths)
            if not mask:
                continue
            logits = logits[mask,:]
            targets = targets[mask,:]

        all_target.append(targets)
        all_logits.append(logits)

    all_logits = torch.vstack(all_logits)
    all_target = torch.vstack(all_target)
    
    
    f1score = lambda precision,recall: 2 * precision * recall / (precision + recall)
    
    num_elements = all_target.shape[0]
    
    res = {}
    print(f'Computing the thresholds for each of the {all_target.shape[1]} diagnostics')
    for diag in tqdm(range(all_target.shape[1])):
        precision, recall, thresholds = precision_recall_curve(all_target.numpy()[:,diag],
                                                               all_logits.detach().numpy()[:,diag])
        
        #print(str(len(recall)) + " "+str(len(thresholds)))
        #print(len(recall))
        #print(np.median(recall))
        #print(np.quantile(recall,0.75))
        #print('------')
        positives = all_target[:,diag].sum().item()
        prevalence = positives / num_elements
        
        idx_recall_above_topk = np.where(recall<topk)[0] # recall start at 100% with the lowest threshold.
        if not idx_recall_above_topk.size:
            recall_score = np.nan
            precision_score = np.nan
            f1_score = np.nan
            mean_score = np.nan
            threshold = np.nan
        else:
            idx_recall_at_topk = idx_recall_above_topk[0]-1 # get first threshold that enables recall higher than 0.7
            #print(idx_recall_above_70[:3])
            recall_score = recall[idx_recall_at_topk]
            precision_score = precision[idx_recall_at_topk]
            f1_score = f1score(precision_score,recall_score)
            mean_score = (recall_score + precision_score) / 2
            threshold = thresholds[idx_recall_at_topk]
        res[diag] = dict(recall_score=recall_score,precision_score=precision_score,
                         f1_score=f1_score,mean_score=mean_score,
                         threshold=threshold, positives=positives,prevalence=prevalence,
                         all_recalls=recall,
                         all_thresholds=thresholds
                        )

    return pd.DataFrame(data=res).T


def compute_thresholds_posterior_predictive(dataset_dict,models_dict):
    all_logits = list()
    all_targets = list()
    print('iterating dataset to obtain logits and targets')
    for diag in tqdm(dataset_dict):
        X = dataset_dict[diag]['features'].detach().numpy()
        y = dataset_dict[diag]['target']
        logits = models_dict[diag]['pipeline'].predict_proba(X)[:,1]
        all_logits.append(logits)
        all_targets.append(y)


    f1score = lambda precision,recall: 2 * precision * recall / (precision + recall)

    scores_db = {'f1_score':f1score}#,'prec_focus':weighted_average(0.6,0.4),'recall_focus':weighted_average(0.4,0.6)}

    max_scores = {key:list() for key in scores_db}
    best_thresholds = {key:list() for key in scores_db}
    all_prevalence = list()
    all_positives = list()

    num_elements = sum([sum(e) for e in all_targets]) # total number of positives


    print(f'Computing the thresholds for each of the {len(dataset_dict.keys())} diagnostics')
    for diag in tqdm(range(len(dataset_dict))):
        if len(dataset_dict[diag]['target']) > 0:
            precision, recall, thresholds = precision_recall_curve(all_targets[diag],
                                                                   all_logits[diag])

            positives = all_targets[diag].sum().item()
            prevalence = positives / num_elements

            for scoring_fun in scores_db:

                scores = scores_db[scoring_fun](precision,recall)

                try:
                    max_idx = np.nanargmax(scores)
                    best_score = scores[max_idx]
                    best_ths = thresholds[max_idx]
                except:
                    best_score = np.nan
                    best_ths = np.nan

                max_scores[scoring_fun].append(best_score)
                best_thresholds[scoring_fun].append(best_ths)
        else:
            for scoring_fun in scores_db:
                best_score = np.nan
                best_ths = np.nan
                
                max_scores[scoring_fun].append(best_score)
                best_thresholds[scoring_fun].append(best_ths)
        all_prevalence.append(prevalence)
        all_positives.append(positives)


    return pd.DataFrame(data=[max_scores['f1_score'],best_thresholds['f1_score'],all_prevalence,all_positives],
             index=["f1_score","best_thresholds","all_prevalence","all_positives"]
        ).T.astype({'f1_score':float,'best_thresholds':float,'all_prevalence':float,'all_positives':int})

def compute_best_f1_threshold(golden,logits):
    assert len(golden) == len(logits)
    
    f1score = lambda precision,recall: 2 * precision * recall / (precision + recall)
    
    try:
        precision, recall, thresholds = precision_recall_curve(golden,logits)
        scores = [f1score(prec,recall) for prec,recall in zip(precision,recall)]
        max_idx = np.nanargmax(scores)
        best_score = scores[max_idx]
        best_ths = thresholds[max_idx]
        avgprec=average_precision_score(golden,logits)
    except Exception as e:
        best_score = np.nan
        best_ths = np.nan
        avgprec= np.nan
    return dict(threshold=best_ths,f1_score=best_score,avgprec=avgprec)
    

###############################################################
######################## Target statistics ####################
###############################################################

def compute_positives_batch(batch,how : str='all',timestep_selector:Callable=None):
    assert how in ['all','each'], 'Oops'
    
    targets = batch['target_sequence']
    lengths = batch['length']
    if timestep_selector is not None:
        mask = timestep_selector(lengths)
        targets = targets[mask,:]
    
    positives_per_diag = targets.sum(axis=0).tolist()
    
    if how == 'all':
        return sum(positives_per_diag)
    else:
        return {label:value for label,value in enumerate(positives_per_diag)}
    
def compute_positives_dataloader(dataloader,how : str='all',timestep_selector:Callable=None):
    assert how in ['all','each'], 'Oops'
    
    if how == 'all':
        positives = 0
        print('iterating dataloader to compute n_positives')
        for batch in tqdm(dataloader):
            positives += compute_positives_batch(batch,how,timestep_selector)
    else:
        positives = None
        print('iterating dataloader to compute n_positives')
        for batch in tqdm(dataloader):
            positives_batch = compute_positives_batch(batch,how,timestep_selector)
            if positives is None:
                positives = positives_batch
            for k in positives:
                positives[k] += positives_batch[k]
    return positives

def compute_size_batch(batch,how:str='all',timestep_selector:Callable=None):
    assert how in ['all','each'], 'Oops'
    
    lengths = batch['length']
    targets = batch['target_sequence']
    
    if timestep_selector is not None:
        mask = timestep_selector(lengths)
        targets = targets[mask,:]
    
    if how =='all':
        return torch.numel(targets)
    else:
        n_seqs = targets.shape[0]
        n_labels = targets.shape[1]
        return {i:n_seqs for i in range(n_labels)}
   
   
def compute_size_dataloader(dataloader,how='all',timestep_selector:Callable=None):
    
    assert how in ['all','each'], 'Oops'
    
    if how =='all':
        size = 0
        print('iterating dataloader to compute size')
        for batch in tqdm(dataloader):
            size += compute_size_batch(batch,how,timestep_selector)
    else:
        size = None
        print('iterating dataloader to compute size')
        for batch in tqdm(dataloader):
            size_batch = compute_size_batch(batch,how,timestep_selector)
            if size is None:
                size = size_batch
            else:
                for k in size:
                    size[k] += size_batch[k]
    return size
    
def compute_prevalence_batch(batch,how='all',timestep_selector:Callable=None):
    
    assert how in ['all','each'], 'Oops'
    
    positives = compute_positives_batch(batch,how,timestep_selector)
    size = compute_size_batch(batch,how,timestep_selector)
    
    if how == 'all':
        return positives / size
    else:
        return {k:positives[k]/size[k] for k in positives}
    
def compute_prevalence_dataloader(dataloader,how='all',timestep_selector:Callable=None):
    
    assert how in ['all','each'], 'Oops'
    
    positives = size = None
    
    print('iterating dataloader to compute prevalence')
    for batch in tqdm(dataloader):
        positives_batch = compute_positives_batch(batch,how,timestep_selector)
        size_batch = compute_size_batch(batch,how,timestep_selector)
        
        if how == 'all':
            if positives is None:
                positives = positives_batch
                size = size_batch
            else:
                positives += positives_batch
                size += size_batch
        else:
            if positives is None:
                positives = positives_batch
                size = size_batch
            else:
                for k in positives:
                    positives[k] += positives_batch[k]
                    size[k] += size_batch[k]
                    
    if how == 'all':
        return positives / size
    else:
        return {k:positives[k] / size[k] for k in positives}


#############################################################
########################## UTILS ############################
#############################################################

def compute_n_preds_batch(model,batch,thresholds : dict,topk=None,how:str='all',timestep_selector:Callable=None):
    """
    Computes number of predictions
    """
    
    assert how in ['all','each'], 'Oops'
    
    lengths = batch['length']
    logits = model(batch['input_pack'])

    logits = outs2nonpadded(logits,batch['length'])
    
    if timestep_selector is not None:
        mask = timestep_selector(lengths)
        logits = logits[mask,:]

    preds = logits2preds(logits,thresholds,topk)
    
    if how == 'all':
        n_preds = (preds == 1).sum().item()
    else:
        n_preds = (preds == 1).sum(axis=0).tolist()
        n_preds = {label:value for label,value in enumerate(n_preds)}
    return n_preds

def compute_n_preds_dataloader(model,dataloader,thresholds : dict,how:str='all',timestep_selector:Callable=None):
    
    assert how in ['all','each'], 'Oops'
    
    n_preds = None
    print('iterating dataloader to compute n_preds')
    for batch in tqdm(dataloader):
        n_preds_batch = compute_n_preds_batch(model,batch,thresholds,how=how,timestep_selector=timestep_selector)
        if n_preds is None:
            n_preds = n_preds_batch
        else:
            if how == 'all':
                n_preds += n_preds_batch
            else:
                for label in n_preds:
                    n_preds[label] += n_preds_batch[label]
    return n_preds

def code2int_nested(code2int : dict, item):
    if isinstance(item, list):
        return [code2int_nested(code2int,x) for x in item]
    else:
        return code2int[item]

def preds2code(preds):
    """
    Converts predictions to code (index)
    preds = torch.tensor shape=(n_sequences,n_labels)
    """
    preds_positions = list(zip(*torch.where(preds == 1)))
    
    code_list = [ [] for i in range(preds.shape[0]) ]
    for e in a:
        code_list[e[0]] = code_list[e[0]] + [e[1].item()]
    return code_list

def create_random_batch(size:int, dataset : Dataset, collate_fn, manual_seed=232):
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    
    overfit_indices = np.random.randint(0,len(dataset),size=(size,))
    
    small_dataloader = DataLoader(dataset,batch_size=size,collate_fn=collate_fn,sampler=SubsetRandomSampler(overfit_indices))
    batch = next(iter(small_dataloader))
    return batch

    
########### CONFIDENCE AND ECE ###########

def compute_starting_info_for_confidence_analysis_dataloader(model,dataloader,thresholds,topk:None):
    """
    returns the logits, preds and golden of a dataloader.
    if thresholds then topk should be none. If topk is specified, then it has priority over thresholds.
    """
    all_adm_indexes = list()
    all_pids = list()
    all_logits = list()
    all_preds = list()
    all_golden = list()
    for batch in tqdm(dataloader):
        logits = compute_model_logits_batch(model,batch)
        preds = compute_model_preds_batch(model,batch,thresholds,topk)
        target = batch['target_sequence']

        all_logits.append(logits)
        all_preds.append(preds)
        all_golden.append(target)

        # get pids and adm_index of each sequence
        pids = batch['pid']
        lengths = batch['length']
        pids = [[pid]*lengths[idx] for idx,pid in enumerate(pids)]
        adm_index = [idx+1 for sublist in pids for idx,_ in enumerate(sublist)]
        pids = [item for sublist in pids for item in sublist] # flatten

        all_pids.extend(pids)
        all_adm_indexes.extend(adm_index)

    all_logits = torch.vstack(all_logits)
    all_preds = torch.vstack(all_preds)
    all_golden = torch.vstack(all_golden)

    logits_df = pd.DataFrame(all_logits.detach().numpy())
    logits_df.loc[:,'pid'] = all_pids
    logits_df.loc[:,'adm_index'] = all_adm_indexes
    logits_df = logits_df.set_index(['pid','adm_index']).sort_index()

    preds_df = pd.DataFrame(all_preds.detach().numpy())
    preds_df.loc[:,'pid'] = all_pids
    preds_df.loc[:,'adm_index'] = all_adm_indexes
    preds_df = preds_df.set_index(['pid','adm_index']).sort_index()

    golden_df = pd.DataFrame(all_golden.detach().numpy())
    golden_df.loc[:,'pid'] = all_pids
    golden_df.loc[:,'adm_index'] = all_adm_indexes
    golden_df = golden_df.set_index(['pid','adm_index']).sort_index()
    
    return logits_df,preds_df,golden_df

def compute_ece(logits, preds, goldens, nbins:int = 10,use_positives=False):
    """
    works either receiving all as pd.DataFrame (multiple classes) or all as pd.Series (1 class)
    
    dataframes have in each column the label and each row is an admission from a patient
    """

    preds_mask = preds == 1
    if use_positives == False:
        confidences = logits.where(preds_mask)
        accuracies = (preds == goldens).where(preds_mask) # only counting positions where a prediction occurs
    else:
        confidences = logits
        accuracies = goldens

    bins = np.linspace(0,1,nbins+1)
    
    ece = 0

    weights = list()
    accs_in_bin = list()
    avgs_confidences = list()
    bin_ces = list()
    n_bins = list()

    for left,right in zip(bins[:-1],bins[1:]):
        in_bin = ((confidences > left) & (confidences < right))

        # avg acc in bin
        accs_in_bin.append(np.nanmean(accuracies[in_bin]))

        # avg confidence in bin
        avgs_confidences.append(np.nanmean(confidences[in_bin]))
        
        # bin-wise calibration error
        bin_ces.append(abs(accs_in_bin[-1] - avgs_confidences[-1]))

        n = np.nansum(in_bin)
        n_bins.append(n)

        ece += n * bin_ces[-1]
        
    ece /= sum(n_bins)
    
    weights = [e/sum(n_bins) for e in n_bins]
    
    center_of_mass = sum([avgs_confidences[i]*weights[i] for i,_ in enumerate(avgs_confidences)]) / sum(weights)

    return dict(com=center_of_mass,ece=ece,bin_ces=bin_ces,n_bins=n_bins,bin_acc=accs_in_bin,nbins=nbins,bins=bins,avg_confidences=avgs_confidences)