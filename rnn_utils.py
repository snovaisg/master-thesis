from typing import Callable

import pickle
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
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

from torchmetrics import Recall, Precision, F1Score, AUROC, AveragePrecision


class ICareDataset(Dataset):
    
    def __init__(self, 
                 diagnoses_file, 
                 universe_grouping, 
                 grouping='ccs', # desired grouping to use (for both input and output currently),
                 train_size:float = 0.70,
                 val_size:float = 0.15,
                 test_size:float = 0.15,
                 shuffle_dataset:bool = True,
                 random_seed :int = 432,
                 partial:int=None # number of patients to process. good for debugging
                ):
        
        assert train_size+val_size+test_size == 1, 'Oops'

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
            
            history_original = self.raw_data[pat][self.grouping]['history']
            target_original = self.raw_data[pat][self.grouping]['targets']
            new_target_original = self.raw_data[pat][self.grouping]['new_targets']

            history_mhot = self.adms2multihot(history_original).to(dtype=torch.float)
            target_mhot = self.adms2multihot(target_original).to(dtype=torch.int64)
            new_target_mhot = self.adms2multihot(new_target_original).to(dtype=torch.int64)
            
            length = len(self.raw_data[pat][self.grouping]['history'])
            
            self.data[pat] = {'history_original': history_original,
                              'target_original': target_original,
                              'new_target_original':new_target_original,
                              'history_mhot':history_mhot,
                              'target_mhot':target_mhot,
                              'new_target_mhot':new_target_mhot,
                              'history_hot':torch.where(history_mhot > 0,1,0).to(dtype=torch.float),
                              'target_hot':torch.where(target_mhot>0,1,0).to(dtype=torch.float),
                              'new_target_hot':torch.where(new_target_mhot>0,1,0).to(dtype=torch.float),
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


        return {'history_original': patient_data['history_original'],
                'target_original': patient_data['target_original'],
                'new_target_original':patient_data['new_target_original'],
                'history_hot':patient_data['history_hot'],
                'target_hot':patient_data['target_hot'],
                'new_target_hot':patient_data['new_target_hot'],
                'history_mhot':patient_data['history_mhot'],
                'target_mhot':patient_data['target_mhot'],
                'new_target_mhot':patient_data['new_target_mhot'],
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
    
    def __init__(self,input,output):
        self.POSSIBLE_INPUTS = ['history_hot','history_mhot']
        self.POSSIBLE_OUTPUTS = ['target_hot','new_target_hot']
        
        assert input in self.POSSIBLE_INPUTS, f"input chosen doesn't exist. choose one of the following {self.POSSIBLE_INPUTS}"
        
        assert output in self.POSSIBLE_OUTPUTS, f"output chosen doesn't exist. choose one of the following {self.POSSIBLE_OUTPUT}"
        
        self.input = input
        self.output = output
    
    def __call__(self,batch):
        
        result = {field:[pat[field] for pat in batch] for field in batch[0].keys()}
        
        result.update(
            dict(
                input_pack=pack_sequence(result[self.input],enforce_sorted=False),
                target_sequence=torch.vstack(result[self.output])
            )
        )
        
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


#############################################################
######################### DL UTILS ##########################
#############################################################
    
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


def compute_loss_dataloader(model, dataloader, criterion):
    """
    Computes the loss of N2N model on a particular dataloader.
    """
        
    model.eval()
    loss = list()
    
    print('Starting to compute the loss on the dataloader')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            batch_loss = compute_loss_batch(model, batch, criterion).item()
            loss.append(batch_loss)
        
    return np.mean(loss) # not true weighted loss but works
                
                
def compute_loss_batch(model,batch,criterion):
    """
    Computes the loss (sum) of a batch. 
    Ignores padded_positions to obtain a more correct loss.
    
    Parameters
    ----------
    
    reduction : str | either 'mean' or 'sum'
        reduction of loss.
    """
    
    n_labels = batch['history_mhot'][0].shape[-1]
    
    outs = model(batch['input_pack'],ignore_sigmoid=True)
    
    non_padded_outs = outs2nonpadded(outs,batch['length'])
    
    loss = criterion(non_padded_outs,batch['target_sequence'])
    
    return loss

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


def train_model_batch(model,batch,criterion,optimizer):
    """
    Receives a model and a batch of input and labels.
    Trains a model on this data and returns the loss.
    """
    
    model.train()
    
    # zero the parameter gradients
    model.zero_grad()

    loss = compute_loss_batch(model,batch,criterion)

    loss.backward()

    optimizer.step()
    
    return loss.item()


def train_model_dataloader(model, dataloader, criterion, optimizer):
    
    model.train()
    
    print('Starting to train each batch')
    losses = list()
    for i, batch in tqdm(enumerate(dataloader)):
        
        batch_loss = train_model_batch(model,batch,criterion,optimizer)
        
        losses.append(batch_loss)
        
    # last batch prob has different size so the mean isn't true weighted mean. but shouldn't affect too much
    dataloader_loss = np.mean(losses) 
    return dataloader_loss


def compute_model_logits_batch(model,batch):
    
    logits = model(batch['input_pack'])
    
    non_padded_logits = outs2nonpadded(logits,batch['length'])
    
    return non_padded_logits

def compute_model_preds_batch(model,batch,thresholds : dict):
    
    logits = compute_model_logits_batch(model,batch)
    
    preds = logits2preds(logits,thresholds)
    
    return preds

def logits2preds(logits,thresholds : dict):
    """
    computes predictions given some logits and decision thresholds
    
    Parameters
    logits : torch.tensor, shape= (n_examples, n_labels), or shape=(batch_size,max_seq_length,n_labels)
    
    thresholds : dict, example {0:0.5, 1:0.43, 2:0.57, ...} (one for each diagnostic)
    """
    
    assert len(thresholds) == logits.shape[-1], "Last dimension must match. It's supposed to be the universe of diagnostics"
    
    # create thresholds matrix with the same shape as logits
    ths = torch.tensor([thresholds[diag] for diag in range(len(thresholds))]).expand(logits.shape)
    
    # computes preds
    preds = torch.where(logits > ths,1,0)
    
    return preds
        
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
           'auroc_weighted':AUROC(num_classes=n_labels,average='weighted',multiclass=False),
           'avgprec_weighted':AveragePrecision(num_classes=n_labels,average='weighted',multiclass=False)
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
            for e in metric_result:
                continue
            res[key] = {'each':{label:value.item() for label,value in enumerate(metric_result)},
                        'weighted': sum([self.positives[label]/self.all_positives*value.item() for label,value in enumerate(metric_result)])
                       }
        return res
    

def compute_metrics_dataloader(metrics, model, dataloader, thresholds: dict):
    """
    
    Parameters
    ----------
    thresholds : dict, example {0:0.5, 1:0.43, 2:0.57, ...} (one for each diagnostic)
    
    """
    model.eval()
    
    with torch.no_grad():
        print('Starting to iterate the dataloader to update metrics')
        for i,batch in tqdm(enumerate(dataloader)):

            logits = compute_model_logits_batch(model,batch)
            preds = logits2preds(logits,thresholds)

            metrics.update(logits,preds,batch['target_sequence'])

    print('Now its time to compute metrics. this may take a while')
    return metrics.compute()

def compute_metrics_batch(metrics: dict, model, batch, thresholds : dict):
    """
    Returns metrics of a batch. By default returns the sum over records (and you can average it later). But you can set average=True.
    
    Parameters
    ----------
    thresholds : dict, example {0:0.5, 1:0.43, 2:0.57, ...} (one for each diagnostic)
    """
    
    logits = compute_model_logits_batch(model,batch)
    preds = logits2preds(logits,thresholds)
    

    metrics.update(logits=logits,preds=preds,target=batch['target_sequence'])
    
    return metrics.compute()

def compute_thresholds_dataloader(model,dataloader):
    """
    Computes the best thresholds for a given based on max f1 score on that dataloader
    """
    # load logits and targets into memory
    
    all_logits = list()
    all_target = list()
    print('Iterating the dataloader to obtain the logits and targets')
    for batch in tqdm(iter(dataloader)):
        logits = compute_model_logits_batch(model,batch)

        all_target.append(batch['target_sequence'])
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


###############################################################
######################## Target statistics ####################
###############################################################

def compute_positives_batch(batch,how : str='all'):
    assert how in ['all','each'], 'Oops'
    
    if how == 'all':
        return batch['target_sequence'].sum().item()
    else:
        return {label:value for label,value in enumerate(batch['target_sequence'].sum(axis=0).tolist())}
    
def compute_positives_dataloader(dataloader,how : str='all'):
    assert how in ['all','each'], 'Oops'
    
    if how == 'all':
        positives = 0
        print('iterating dataloader to compute n_positives')
        for batch in tqdm(dataloader):
            positives += compute_positives_batch(batch,how)
    else:
        positives = None
        print('iterating dataloader to compute n_positives')
        for batch in tqdm(dataloader):
            positives_batch = compute_positives_batch(batch,how)
            if positives is None:
                positives = positives_batch
            else:
                positives = {k:positives[k]+positives_batch[k] for k in positives}
    return positives

def compute_size_batch(batch,how:str='all'):
    assert how in ['all','each'], 'Oops'
    
    if how =='all':
        return torch.numel(batch['target_sequence'])
    else:
        n_seqs = batch['target_sequence'].shape[0]
        n_labels = batch['target_sequence'].shape[1]
        return {i:n_seqs for i in range(n_labels)}
def compute_size_dataloader(dataloader,how='all'):
    
    assert how in ['all','each'], 'Oops'
    
    if how =='all':
        size = 0
        print('iterating dataloader to compute size')
        for batch in tqdm(dataloader):
            size += compute_size_batch(batch,how)
    else:
        size = None
        print('iterating dataloader to compute size')
        for batch in tqdm(dataloader):
            size_batch = compute_size_batch(batch,how)
            if size is None:
                size = size_batch
            else:
                size = {k:size[k]+size_batch[k] for k in size}
    return size
    
def compute_prevalence_batch(batch,how='all'):
    
    assert how in ['all','each'], 'Oops'
    
    positives = compute_positives_batch(batch,how)
    size = compute_size_batch(batch,how)
    
    if how == 'all':
        return positives / size
    else:
        return {k:positives[k]/size[k] for k in positives}
    
def compute_prevalence_dataloader(dataloader,how='all'):
    
    assert how in ['all','each'], 'Oops'
    
    positives = size = None
    
    print('iterating dataloader to compute prevalence')
    for batch in tqdm(dataloader):
        positives_batch = compute_positives_batch(batch,how)
        size_batch = compute_size_batch(batch,how)
        
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
                positives = {k:positives[k]+positives_batch[k] for k in positives}
                size = {k:size[k]+size_batch[k] for k in size}
    if how == 'all':
        return positives / size
    else:
        return {k:positives[k] / size[k] for k in positives}


#############################################################
########################## UTILS ############################
#############################################################

def compute_n_preds_batch(model,batch,thresholds : dict,how:str='all'):
    """
    Computes number of predictions
    """
    
    assert how in ['all','each'], 'Oops'
    
    
    logits = model(batch['input_pack'])

    logits = outs2nonpadded(logits,batch['length'])

    preds = logits2preds(logits,thresholds)
    
    if how == 'all':
        n_preds = (preds == 1).sum().item()
    else:
        n_preds = (preds == 1).sum(axis=0).tolist()
        n_preds = {label:value for label,value in enumerate(n_preds)}
    return n_preds

def compute_n_preds_dataloader(model,dataloader,thresholds : dict,how:str='all'):
    
    assert how in ['all','each'], 'Oops'
    
    n_preds = None
    print('iterating dataloader to compute n_preds')
    for batch in tqdm(dataloader):
        n_preds_batch = compute_n_preds_batch(model,batch,thresholds,how=how)
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


################################################
############### WORK IN PROGRESS ###############
################################################


from torchmetrics import Metric
class RecordRecall(Metric):
    def __init__(self,average='macro'):
        full_state_update=False
        super().__init__()
        self.add_state("record_recall", default=torch.tensor(0,dtype=float), dist_reduce_fx="sum")
        self.add_state("total_records", default=torch.tensor(0,dtype=float), dist_reduce_fx="sum")
        self.add_state("undefined", default=torch.tensor(0,dtype=float), dist_reduce_fx="sum")
        self.average = average
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds and target must be ones and zeros
        shape of both must be (records, labels)
        
        i think this fails when there are records in target that has no positive labels (divide by zero)
        """
        
        assert preds.shape == target.shape
        assert len(preds.shape) == 2 and len(target.shape) == 2
        assert (preds == 1).sum().sum() + (preds == 0).sum().sum() == preds.shape[0] * preds.shape[1], 'must be only zeros and ones'
        assert (target == 1).sum().sum() + (target == 0).sum().sum() == target.shape[0] * target.shape[1], 'must be only zeros and ones'
        
        
        self.undefined += (torch.sum(target,axis=-1) == 0).sum()
        # non-zero targets
        non_zero_records = torch.where(torch.sum(target,axis=-1) != 0)[0]
        target = target[non_zero_records]
        preds = preds[non_zero_records]
        
        TP = (preds == 1) & (target == 1)
        FN = (preds == 0) & (target == 1)
        P = target == 1
        
        if self.average == 'macro':
            self.record_recall += (TP.sum(axis=1) / (TP.sum(axis=-1) + FN.sum(axis=-1))).sum()
            self.total_records += target.shape[0]
        elif self.average == 'weighted':
            self.record_recall += ((TP.sum(axis=1) / (TP.sum(axis=-1) + FN.sum(axis=-1))) * P.sum(axis=-1)).sum()
            self.total_records += P.sum().sum()
        else:
            raise ValueError('choose an accepted average value')

    def compute(self):
        return {'value':(self.record_recall.float() / self.total_records).item(),
                'undefined':self.undefined.item()
               }