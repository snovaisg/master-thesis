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

import pandas as pd
import numpy as np
from math import ceil

import json

from Metrics import Metrics

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
    
    
def split_dataset(dataset_,test_size_):
    a, b = random_split(dataset_,
                  [
                      ceil(len(dataset_)*(1-test_size_)),
                      len(dataset_) - ceil(len(dataset_)*(1-test_size_))
                  ]
                 )
    return a,b
    
class MYCOLLATE:
    """
    This collate class gets a dataset in the format of:
    [
    {'train':[[code1,code2],[code3,code4,code5],etc..]
      'target:':[[code1,code2],[code3,code4,code5],etc..]
    },
     {etc..},
     etc..
    ]
    
    And outputs a pack of train and pad of test sequences
    """
    def __init__(self,dataset):
        self.dataset = dataset
    
    def __call__(self,batch):
        patients = {'train':{'sequence':[],'original':[],'pids':[]},
                    'target':{'sequence':[],'original':[],'pids':[]}
                   }
        
        grouping_code = self.dataset.grouping
        n_labels = self.dataset.grouping_data[grouping_code]['n_labels']
        code2int = self.dataset.grouping_data[grouping_code]['code2int']
        
        # <Nº admissions - 1> of each patient
        seq_lengths = []
        
        # 1-to-1 correspondence between each admission in {train/target}_admissions_sequenced and the patient's id.
        patients_list = []
        for pat in batch:
            
            pid = pat['pid'] #patient id
            train_admissions_sequenced = []
            target_admissions_sequenced = []
            seq_lengths.append(len(pat))

            # convert each train admission into a multi-hot vector
            for train_admission in pat['train']:
                admission = (F.one_hot(torch.tensor(list(map(lambda code: code2int[code],train_admission))),num_classes=n_labels)
                             .sum(dim=0).float() #one-hot of each diagnose to multi-hot vector of diagnoses
                            )
                train_admissions_sequenced.append(admission)
            
            

            # convert each target admission into a one-hot vector
            for target_admission in pat['target']:
                # convert each admission to multi-hot vector
                admission = (F.one_hot(torch.tensor(list(map(lambda code: code2int[code],target_admission))),num_classes=n_labels)
                             .sum(dim=0).float() #one-hot of each diagnose to multi-hot vector of diagnoses
                            )
                target_admissions_sequenced.append(admission)

            # stack multiple train admissions of a single patient into a single tensor
            if len(train_admissions_sequenced) > 1:
                train_admissions_sequenced = torch.stack(train_admissions_sequenced)
            else:
                train_admissions_sequenced = train_admissions_sequenced[0].view((1,-1))

            # stack multiple target admissions of a single patient into a single tensor
            if len(target_admissions_sequenced) > 1:
                target_admissions_sequenced = torch.stack(target_admissions_sequenced)
            else:
                target_admissions_sequenced = target_admissions_sequenced[0].view((1,-1))

            # store final train and test tensors
            patients['train']['sequence'].append(train_admissions_sequenced)
            patients['target']['sequence'].append(target_admissions_sequenced)
            
            patients['train']['original'].append(pat['train'])
            patients['target']['original'].append(pat['target'])
            
            # repeat pid for each admission they have on target
            pid_train_list = [pid] * len(pat['train'])
            pid_target_list = [pid] * len(pat['target'])
            patients['train']['pids'].extend(pid_train_list)
            patients['target']['pids'].extend(pid_target_list)

        # pad sequences (some patients have more admissions than others)
        patients['train']['sequence'] = pack_sequence(patients['train']['sequence'],enforce_sorted=False)
        patients['target']['sequence'] = pad_sequence(patients['target']['sequence'],batch_first=True)
        
        return {'train_sequences':patients['train'],
                'target_sequences':patients['target'],
                'train_pids':patients['train']['pids'],
                'target_pids':patients['target']['pids']
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
    
    def forward(self, input):
        """
        input: pack_sequence
        
        """
        
        hn,_ = self.model(input)
        
        out = self.lin(pad_packed_sequence(hn,batch_first=True)[0])
        return out
    
    
def convert_batch_out_2_predictions_flattened(out,targets,converter : Callable,threshold = 0.5):
    """Converts the outputs of a model (logits) to list of predictions. Each list is an admission being predicted
    Each admission contains predictions sorted descending by the output of the model
    
    (old version of out2pred)
    
    """
    activations = nn.Sigmoid()(out).detach().numpy()
    
    # sort activations to later order the predictions by highest activation to lowest
    sorted_idx = np.argsort(activations)
    predictions = np.where(activations > threshold)
    # each element is an admission. Admissions are all flattened
    # i.e. instead of [pat1[adm1,adm2],pat2[adm3,adm4]], where each pat1 is a list and each adm is a list, we have
    # [adm1,adm2,adm3,adm4] where each adm1 is a list of codes
    predictions_admission = []
    for idx,patient in enumerate(activations):
        for admission_i in range(targets[idx]):
            
            int_predictions = np.where(activations[idx][admission_i] > 0.5)[0]
            # magic line below
            # int_predictions_sorted contains the predictions (activations above threshold)
            # sorted by the activation value
            int_predictions_sorted = [e for e in sorted_idx[idx][admission_i] if e in int_predictions][::-1]
            
            # convert from int2code
            code_predictions = list(map(converter,int_predictions_sorted))
            
            predictions_admission.append(code_predictions)
    return predictions_admission

def outs2pred(outs, int2code : dict):
    """Converts the outputs of a model (logits) to diagnostic code predictions.
    
    Parameters
    ---------
    
    out: outputs of a model on a batch of variable-sized sequences
    
    int2code: dict mapping idx to diagnostic code
    
    """
    activations = nn.Sigmoid()(outs).detach().numpy()
    
    sorted_idx = np.argsort(activations)
    
    return np.vectorize(int2code.get)(sorted_idx)[:,:,::-1]


def eval_model(model, dataloader, dataset, criterion, epoch, name, only_loss=False,level_interest=None,k_interest=None):
    """
    This functions evaluates and computes metrics of a model checkpoint on a dataloader
    
    criterion must be reduction='none'
    """
    
    model.eval()
    # eg:: ccs, icd9, etc..
    code_type = dataset.grouping
    
    int2code = dataset.grouping_data[code_type]['int2code']
    
    result = {'name':name,
              'epoch':epoch
             }
    
    total_loss = 0
    total_seq = 0 #total sequences
    
    all_metrics = None
    with torch.no_grad():
        for i, batch in enumerate(iter(dataloader)):
            
            # get the inputs; data is a list of [inputs, labels]
            history_sequences, target_sequences = batch['train_sequences'],batch['target_sequences']

            inputs = history_sequences['sequence']
            outs = model(inputs)

            loss = criterion(outs, target_sequences['sequence'])
            
            # zero-out positions of the loss corresponding to padded inputs
            # if a sequence has all zeros it is considered to be a padding.
            # Comment: safer way to do this would be a solution using the lengths...
            sequences,lengths = pad_packed_sequence(inputs,batch_first=True)
            mask = ~sequences.any(dim=2).unsqueeze(2).repeat(1,1,sequences.shape[-1])
            loss.masked_fill_(mask, 0)
        
            loss = loss.sum() / (lengths.sum()*sequences.shape[-1])

            # compute loss
            n = target_sequences['sequence'].size(0)
            total_seq += n
            total_loss += loss.item() * n
            
            # compute other metrics

            _,lengths = pad_packed_sequence(history_sequences['sequence'])
            
            preds = outs2pred(outs,int2code)
            
            if all_metrics is None:
                all_metrics = compute_metrics(preds,target_sequences['original'],level_interest, k_interest)
            else:
                new_metrics = compute_metrics(preds,target_sequences['original'],level_interest, k_interest)
                concat_metrics(all_metrics,new_metrics)

        result['loss'] = total_loss / total_seq
        if only_loss:
            return result
        for level in all_metrics:
            if level not in result:
                result[level] = {}
            for metric in all_metrics[level]:
                if metric not in result[level].keys():
                    result[level][metric] = {}
                result[level][metric] = {'mean':np.mean(all_metrics[level][metric]),
                                         'std':np.std(all_metrics[level][metric]),
                                         'n': len(all_metrics[level][metric])
                                        }
    return result


def train_one_epoch(model, train_loader, epoch, criterion, optimizer):
    """
    Trains one epoch and returns mean loss over training
    """
    model.train()
    
    total_loss = 0
    total_n = 0
    for i, batch in enumerate(iter(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        history_sequences, target_sequences = batch['train_sequences'],batch['target_sequences']

        # zero the parameter gradients
        model.zero_grad()

        # forward + backward + optimize
        
        outs = model(history_sequences['sequence'])
        
        loss = criterion(outs, target_sequences['sequence'])
        loss.backward()
        
        optimizer.step()
        
        _,lengths = pad_packed_sequence(history_sequences['sequence'])
        
        n = lengths.sum().item()
        
        total_loss += loss.item() * n
        total_n += n
    return total_loss / total_n


def train_one_epochV2(model, train_loader, epoch, criterion, optimizer):
    """
    Trains one epoch and returns mean loss over training
    
    criterion has to have reduction='none'
    """
    model.train()
    
    total_loss = 0
    total_n = 0
    for i, batch in enumerate(iter(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        history_sequences, target_sequences = batch['train_sequences'],batch['target_sequences']

        # zero the parameter gradients
        model.zero_grad()

        # forward + backward + optimize
        inputs = history_sequences['sequence']
        outs = model(inputs)
        
        loss = criterion(outs, target_sequences['sequence'])
        
        # zero-out positions of the loss corresponding to padded inputs
        # if a sequence has all zeros it is considered to be a padding.
        # Comment: safer way to do this would be a solution using the lengths...
        sequences,lengths = pad_packed_sequence(inputs,batch_first=True)
        mask = ~sequences.any(dim=2).unsqueeze(2).repeat(1,1,sequences.shape[-1])
        loss.masked_fill_(mask, 0)
        
        loss = loss.sum() / (lengths.sum()*sequences.shape[-1])

        loss.backward()
        
        optimizer.step()
        
        _,lengths = pad_packed_sequence(history_sequences['sequence'])
        
        n = lengths.sum().item()
        
        total_loss += loss.item() * n
        total_n += n
    return total_loss / total_n


############################
###### METRICS #############
############################

def compute_metrics(preds, targets,level_interest=None,k_interest=None):
    """ 
    Computes recall for a batch of predictions. 
    Returns the average of each metric at the end in the format {metric:avg,metric2:avg,etc..}
    """
    
    levels = ['1 adm','2 adm','3 adm','>3 adm','last adm']
    recall_at = [10,20,30]
    
    if level_interest is not None:
        if level_interest in levels:
            levels = [level_interest]
        else:
            raise ValueError(f'Expecting one of {levels}. Got {level_interest}')
    
    if k_interest is not None:
        if k_interest in recall_at:
            recall_at = [k_interest]
        else:
            raise ValueError()
    
    
    res = dict()
    for key in levels:
        res[key] = {f'recall{k}':[] for k in recall_at}
        
        
        
    for idx_pat, pat in enumerate(targets):
        for idx_adm,adm in enumerate(pat):
            for k in recall_at:
                if idx_adm+1 == len(pat):
                    res['last adm'][f'recall{k}'].append(Metrics.recall(adm,preds[idx_pat][idx_adm],k=k))
    """
    for idx_pat, pat in enumerate(targets):
        for idx_adm,adm in enumerate(pat):
            for k in recall_at:
                if idx_adm +1 <=3:
                    res[f'{idx_adm+1} adm'][f'recall{k}'].append(Metrics.recall(adm,preds[idx_pat][idx_adm],k=k))
                else:
                    res[f'>3 adm'][f'recall{k}'].append(Metrics.recall(adm,preds[idx_pat][idx_adm],k=k))
                
                # we are at the last admission.
                if idx_adm+1 == len(pat):
                    res['last adm'][f'recall{k}'].append(Metrics.recall(adm,preds[idx_pat][idx_adm],k=k))
    """
    return res


def concat_metrics(old_metrics:dict,new_metrics:dict):
    # dicts are passed by reference
    # so i just update whatever was passed to <old_metrics>
    for level in old_metrics:
        for metric in old_metrics[level]:
            old_metrics[level][metric].extend(new_metrics[level][metric])
    return None