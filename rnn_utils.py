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

from Metrics import Metrics
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, accuracy_score

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
    
    def forward(self, input,**kwargs):
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

def eval_model(model, dataloader, dataset, decision_thresholds, metrics, only_loss=False,name=None):
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
    
    model_outputs,golden = outs2df(model, dataloader, dataset, return_golden=True)
    
    loss = compute_loss(model, dataloader)
    
    if only_loss:
        return loss
    
    predictions = make_predictions(model_outputs,golden,decision_thresholds)
    
    metrics = compute_metrics(model_outputs,predictions,golden,metrics)
    
    if name is not None:
        metrics.name = name
    return loss,metrics


def compute_loss(model,dataloader):
    """
    Computes loss of a model on a particular dataloader. 
    Ignores padded positions to obtain a more correct loss.
    """
    
    # get n_labels
    for batch in iter(dataloader):
        n_labels = batch['target_sequences']['sequence'].shape[-1]
        break
        
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    total_loss = total_sequences = 0
    with torch.no_grad():
        for i, batch in enumerate(iter(dataloader)):
            
            inputs, targets = batch['train_sequences']['sequence'],batch['target_sequences']['sequence']
            outs = model(inputs,take_mc_average=True)
            loss = criterion(outs, targets)
            
            # zero-out positions of the loss corresponding to padded inputs
            sequences,lengths = pad_packed_sequence(inputs,batch_first=True)
            mask = ~sequences.any(dim=2).unsqueeze(2).repeat(1,1,sequences.shape[-1])
            loss.masked_fill_(mask, 0)
            
            # compute loss
            total_loss += loss.sum()
            total_sequences += lengths.sum()
        
    reducted_loss = total_loss / (total_sequences*n_labels)
    return reducted_loss

def outs2df(model,dataloader,dataset,return_golden=False):
    """
    Generates model outputs on a dataloader and returns as a pd.DataFrame.
    
    If <return_golden> is True, also returns a dataframe with the golden labels.
    """
    model.eval()
    
     # eg:: ccs, icd9, etc..
    code_type = dataset.grouping
    
    int2code = dataset.grouping_data[code_type]['int2code']
    n_labels = len(int2code)
    
    col_names = ['diag_' + str(code) for code in int2code.keys()]
    
    sigmoid = nn.Sigmoid()
    
    flatten_list = lambda x: [item for sublist in x for item in sublist]
    
    full_df = full_golden = None
    with torch.no_grad():
        for i, batch in enumerate(iter(dataloader)):
            
            inputs, targets = batch['train_sequences']['sequence'],batch['target_sequences']['sequence']
            outs = model(inputs,take_mc_average=True)
            
            # Turn to pandas to store the <model_output> data
            
            # we want to ignore the padded sequences
            _,lengths = pad_packed_sequence(inputs,batch_first=True)
            relevant_positions = [[i+idx*max(lengths) for i in range(e)] for idx,e in enumerate(lengths)]
            relevant_positions = flatten_list(relevant_positions)
            
            outs_flattened = outs.view(1,-1,outs.size()[2])
            relevant_outs = outs_flattened[:,relevant_positions,:]
            relevant_outs = sigmoid(relevant_outs).detach().numpy()[0,:,:]

            df = (pd.DataFrame(relevant_outs,
                               columns=col_names)
                  .assign(pat_id=batch['target_pids'])
                 )
            full_df = df if full_df is None else pd.concat([full_df,df])
            
            if return_golden:
                targets_flattened = targets.view(1,-1,targets.size()[2])
                relevant_targets = targets_flattened[:,relevant_positions,:].detach().numpy()[0,:,:]
                golden_df = (pd.DataFrame(relevant_targets,
                                        columns=col_names)
                             .assign(pat_id=batch['target_pids'])
                            )
                full_golden = golden_df if full_golden is None else pd.concat([full_golden,golden_df])
                    
        full_df['adm_index'] = full_df.groupby(['pat_id']).cumcount()+1
        full_df = full_df.reset_index(drop=True)
        full_df[['pat_id','adm_index']] = full_df[['pat_id','adm_index']].astype(int)
        # reorder columns
        full_df = full_df.set_index(['pat_id','adm_index']).sort_index()
        
        if return_golden:
            full_golden['adm_index'] = full_golden.groupby(['pat_id']).cumcount()+1
            full_golden = full_golden.reset_index(drop=True)
            full_golden[['pat_id','adm_index']] = full_golden[['pat_id','adm_index']].astype(int)
            # reorder columns
            full_golden = full_golden.set_index(['pat_id','adm_index']).sort_index()
            
            return full_df,full_golden
    return full_df


def outs2df_mc(model, dataloader, dataset, return_golden=False):
    """
    Generates model outputs on a dataloader and returns as a pd.DataFrame.
    
    If <return_golden> is True, also returns a dataframe with the golden labels.
    """
    model.train() # needs to be active for mc_dropout
    
     # eg:: ccs, icd9, etc..
    code_type = dataset.grouping
    
    int2code = dataset.grouping_data[code_type]['int2code']
    n_labels = len(int2code)
    
    col_names = ['diag_' + str(code) for code in int2code.keys()]
    
    sigmoid = nn.Sigmoid()
    
    flatten_list = lambda x: [item for sublist in x for item in sublist]
    
    full_df = full_golden = None
    for i, batch in enumerate(iter(dataloader)):

        inputs, targets = batch['train_sequences']['sequence'],batch['target_sequences']['sequence']
        outs = model(inputs,take_mc_average=False)

        # Turn to pandas to store the <model_output> data

        # we want to ignore the padded sequences
        _,lengths = pad_packed_sequence(inputs,batch_first=True)
        relevant_positions = [[i+idx*max(lengths) for i in range(e)] for idx,e in enumerate(lengths)]
        relevant_positions = flatten_list(relevant_positions)

        outs_flattened = outs.view(outs.size()[0],1,-1,outs.size()[-1])
        relevant_outs = outs_flattened[:,:,relevant_positions,:]
        relevant_outs = sigmoid(relevant_outs).detach().numpy()[:,0,:]
        
        # merge the N passes
        full_df_npass = None
        for i in range(relevant_outs.shape[0]):
            df = pd.DataFrame(relevant_outs[i,:,:],columns=col_names)
            df = df.assign(n_pass=i+1,pat_id=batch['target_pids'])
            full_df_npass = df if full_df_npass is None else pd.concat([full_df_npass,df])
        
        full_df = full_df_npass if full_df is None else pd.concat([full_df,full_df_npass])

        if return_golden:
            targets_flattened = targets.view(1,-1,targets.size()[2])
            relevant_targets = targets_flattened[:,relevant_positions,:].detach().numpy()[0,:,:]
            golden_df = (pd.DataFrame(relevant_targets,
                                    columns=col_names)
                         .assign(pat_id=batch['target_pids'])
                        )
            full_golden = golden_df if full_golden is None else pd.concat([full_golden,golden_df])

    full_df['adm_index'] = full_df.groupby(['pat_id','n_pass']).cumcount()+1
    full_df = full_df.reset_index(drop=True)
    full_df[['pat_id','adm_index']] = full_df[['pat_id','adm_index']].astype(int)
    # reorder columns
    full_df = full_df.set_index(['pat_id','adm_index','n_pass']).sort_index()

    if return_golden:
        full_golden['adm_index'] = full_golden.groupby(['pat_id']).cumcount()+1
        full_golden = full_golden.reset_index(drop=True)
        full_golden[['pat_id','adm_index']] = full_golden[['pat_id','adm_index']].astype(int)
        # reorder columns
        full_golden = full_golden.set_index(['pat_id','adm_index']).sort_index()

        return full_df,full_golden
    return full_df


def get_prediction_thresholds(model_outputs : pd.DataFrame, golden_data : pd.DataFrame, method='roc gm'):
    
    supported_methods = ['roc gm','max f1']
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == 'max f1':
            curve = precision_recall_curve
            metric = lambda precision,recall: 2* (precision*recall) / (precision + recall)
        elif method == 'roc gm':
            curve = roc_curve
            metric = lambda fpr, tpr: np.sqrt(tpr * (1-fpr))
        else:
            raise ValueError(f'method must be one of the following: {supported_methods}. Got {method}')

        thresholds_data = None
        for diag in model_outputs.filter(like='diag_').columns:
            y_true = golden.loc[:,diag].to_numpy().reshape((-1,1))
            probas_pred = model_outputs.loc[:,diag].to_numpy().reshape((-1,1))

            x, y, thresholds = curve(y_true, probas_pred);

            if len(thresholds) == 1:
                threshold = pd.DataFrame(data=[[0.5,np.nan]],columns=['threshold',method],index=[diag])
            else:
                scores = metric(x,y)
                if pd.Series(scores).isna().all():
                    threshold = pd.DataFrame(data=[[0.5,np.nan]],columns=['threshold',method],index=[diag])
                else:
                    ix = pd.Series(scores).idxmax()

                    threshold = pd.DataFrame(data=[[thresholds[ix],scores[ix]]],columns=['threshold',method],index=[diag])

            if thresholds_data is None:
                thresholds_data = threshold
            else:
                thresholds_data = pd.concat([thresholds_data,threshold])
        return thresholds_data

def make_predictions(model_outputs, golden, decision_thresholds):
    

    def predict(predictions: pd.Series, threshold : float):
        return predictions.apply(lambda x: 1 if x > threshold else 0)

    preds = model_outputs.apply(lambda x: predict(x, decision_thresholds.loc[x.name,'threshold']),axis=0)
    
    return preds


def train_one_epoch(model, train_loader, epoch, criterion, optimizer, take_mc_average=True):
    """
    Trains one epoch and returns mean loss over training
    
    criterion has to have reduction='none'
    """
    model.train()
    
    # get n_labels
    n_labels = next(iter(train_loader))['target_sequences']['sequence'].shape[-1]

    
    total_loss = []
    for i, batch in enumerate(iter(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        history_sequences, target_sequences = batch['train_sequences'],batch['target_sequences']

        # zero the parameter gradients
        model.zero_grad()

        # forward + backward + optimize
        inputs = history_sequences['sequence']
        outs = model(inputs,take_mc_average)
        
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
        
        # compute loss
        total_loss.append(loss.item())
    return np.mean(total_loss) #no weighted average but whatever.. only the lat batch has different size


############################
###### METRICS #############
############################

def compute_metrics(model_outputs,model_predictions,golden,metrics):
    """
    all input dataframes must be of the form:
    double index of (<pat_id>,>adm_index>)
    and columns are the diagnostics. eg: diag_0,...,diag_272
    
    returns several metrics in a dataframe
    
    
    Parameters:
    -----------
    
    metrics : list
        ['roc,avgprec','acc','recall','precision','f1']
    """
    accepted = ['roc','avgprec','acc','recall','accuracy','precision','f1','recall@','precision@','f1@']
    
    diag_weights = golden.sum(axis=0)
    adm_weights = golden.sum(axis=1)
    
    if metrics == 'all':
        metrics = accepted
    
    assert len(metrics) > 0
    assert any([e in metrics for e in accepted]) or any([e for e in metrics if 'recall@' in e])
    
    # threshold independent
    diag_metrics = list()
    adm_metrics = list()
    
    if 'roc' in metrics:
        roc_diag = model_outputs.apply(lambda col: roc_auc_score(golden[col.name],col) if any(golden[col.name] == 1) else np.nan).rename('roc_diag')
        roc_adm = model_outputs.apply(lambda row: roc_auc_score(golden.loc[row.name],row) if any(golden.loc[row.name] == 1) else np.nan,axis=1).rename('roc_adm')
        diag_metrics.append(roc_diag)
        adm_metrics.append(roc_adm)
    
    if 'avgprec' in metrics:
        avgprec_diag = model_outputs.apply(lambda col: average_precision_score(golden[col.name],col) if any(golden[col.name] == 1) else np.nan).rename('avgprec_diag')
        avgprec_adm = model_outputs.apply(lambda row: average_precision_score(golden.loc[row.name],row) if any(golden.loc[row.name] == 1) else np.nan,axis=1).rename('avgprec_adm')
        diag_metrics.append(avgprec_diag)
        adm_metrics.append(avgprec_adm)

    # threshold dependent
    
    if 'accuracy' in metrics:
        accuracy_diag = model_predictions.apply(lambda col: accuracy_score(golden[col.name],col) if any(golden[col.name] == 1) else np.nan).rename('accuracy_diag')
        accuracy_adm = model_predictions.apply(lambda row: accuracy_score(golden.loc[row.name],row) if any(golden.loc[row.name] == 1) else np.nan,axis=1).rename('accuracy_adm')
        diag_metrics.append(accuracy_diag)
        adm_metrics.append(accuracy_adm)

    if 'recall' in metrics:
        recall_diag = model_predictions.apply(lambda col: recall_score(golden[col.name],col,zero_division=0)).rename('recall_diag')
        recall_adm = model_predictions.apply(lambda row: recall_score(golden.loc[row.name],row,zero_division=0),axis=1).rename('recall_adm')
        diag_metrics.append(recall_diag)
        adm_metrics.append(recall_adm)

    if 'precision' in metrics:
        precision_diag = model_predictions.apply(lambda col: precision_score(golden[col.name],col) if any(model_predictions[col.name] == 1) else np.nan).rename('precision_diag')
        precision_adm = model_predictions.apply(lambda row: precision_score(golden.loc[row.name],row) if any(model_predictions.loc[row.name] == 1) else np.nan,axis=1).rename('precision_adm')
        diag_metrics.append(precision_diag)
        adm_metrics.append(precision_adm)

    if 'f1' in metrics:
        f1_diag = model_predictions.apply(lambda col: f1_score(golden[col.name],col) if any(golden[col.name] == 1) else np.nan).rename('f1_diag')
        f1_adm = model_predictions.apply(lambda row: f1_score(golden.loc[row.name],row) if any(golden.loc[row.name] == 1) else np.nan,axis=1).rename('f1_adm')
        diag_metrics.append(f1_diag)
        adm_metrics.append(f1_adm)
    
    # i.e. if recall@k in metrics
    if any(filter(lambda x: re.match('\w+@\d+',x), metrics)):
        
        matches = [e[0] for e in [re.findall('\w+@\d+',e) for e in metrics] if e] # get all <metric>@k in metrics (there may be multiple)
        for match in matches:
            
            k = int(re.findall('\w+@(\d+)',match)[0])
            metric = re.findall('(\w+)@\d+',match)[0]
            
            topk_outputs = model_outputs.apply(lambda row: row.nlargest(k),axis=1)

            # fix missing columns from previous operation
            missing_cols = [col for col in model_outputs.columns if col not in topk_outputs.columns]
            topk_outputs_all_cols = pd.concat([topk_outputs,pd.DataFrame(columns=missing_cols)])
            topk_outputs_all_cols = topk_outputs_all_cols[model_outputs.columns]
            
            ## sometimes k > (#logits>0) so we will turn all 0 logits into nan so that the following lines don't convert them to predictions
            topk_outputs_all_cols = topk_outputs_all_cols.mask(topk_outputs_all_cols == 0,np.nan)
            # done, continuing...

            topk_predictions = np.where(topk_outputs_all_cols.isna(),0,1)
            topk_predictions = pd.DataFrame(data=topk_predictions,columns=model_predictions.columns,index=model_predictions.index)

            if metric == 'recall':

                metric_at_k_diag = topk_predictions.apply(lambda col: recall_score(golden[col.name],col,zero_division=0)).rename(f'recall@{k}_diag')
                metric_at_k_adm = topk_predictions.apply(lambda row: recall_score(golden.loc[row.name],row,zero_division=0),axis=1).rename(f'recall@{k}_adm')
            
            elif metric == 'precision':
                metric_at_k_diag = topk_predictions.apply(lambda col: precision_score(golden[col.name],col) if any(topk_predictions[col.name] == 1) else np.nan).rename(f'precision@{k}_diag')
                metric_at_k_adm = topk_predictions.apply(lambda row: precision_score(golden.loc[row.name],row) if any(topk_predictions.loc[row.name] == 1) else np.nan,axis=1).rename(f'precision@{k}_adm')
                
            elif metric == 'f1':
                metric_at_k_diag = topk_predictions.apply(lambda col: f1_score(golden[col.name],col) if any(golden[col.name] == 1) else np.nan).rename(f'f1@{k}_diag')
                metric_at_k_adm = topk_predictions.apply(lambda row: f1_score(golden.loc[row.name],row) if any(golden.loc[row.name] == 1) else np.nan,axis=1).rename(f'f1@{k}_adm')
            
            else:
                print('what is happening')
                print(metric)

            diag_metrics.append(metric_at_k_diag)
            adm_metrics.append(metric_at_k_adm)        
    
    # take weighted average
    """
    diag_metrics_wavg = (pd.concat(diag_metrics,axis=1)
                         .multiply(diag_weights,axis=0)
                         .sum(axis=0)
                         .divide(
                             diag_weights.sum()
                         )
                        )
    
    adm_metrics_wavg = (pd.concat(adm_metrics,axis=1)
                        .multiply(adm_weights,axis=0)
                        .sum(axis=0)
                        .divide(
                            adm_weights.sum()
                        )
                       )
    """
    diag_metrics_wavg = (pd.concat(diag_metrics,axis=1)
                         .mean(axis=0)
                        )
    
    adm_metrics_wavg = (pd.concat(adm_metrics,axis=1)
                         .mean(axis=0)
                        )

    res = pd.concat([diag_metrics_wavg,adm_metrics_wavg])
    
    res.index.name = 'metrics'
    
    return res


def compute_metricsV2(model_outputs,model_predictions,golden,metrics):
    """
    all input dataframes must be of the form:
    double index of (<pat_id>,>adm_index>)
    and columns are the diagnostics. eg: diag_0,...,diag_272
    
    returns several metrics in a dataframe
    
    
    Parameters:
    -----------
    
    metrics : list
        ['roc,avgprec','acc','recall','precision','f1']
    """
    accepted = ['roc','avgprec','acc','recall','accuracy','precision','f1']
    
    diag_weights = golden.sum(axis=0)
    adm_weights = golden.sum(axis=1)
    
    if metrics == 'all':
        metrics = accepted
    
    assert len(metrics) > 0
    assert any([e in metrics for e in accepted])
    
    # threshold independent
    diag_metrics = list()
    adm_metrics = list()
    
    diag_condition = lambda diag: any([any(golden[diag] == 1),any(model_predictions[diag] == 1)])
    adm_condition =  lambda adm: any( [any(golden.loc[adm] == 1), any(model_predictions[adm] == 1)])
    
    if 'roc' in metrics:
        roc_diag = model_outputs.apply(lambda col: roc_auc_score(golden[col.name],col) if diag_condition(col.name) else np.nan).rename('roc_diag')
        roc_adm = model_outputs.apply(lambda row: roc_auc_score(golden.loc[row.name],row) if adm_condition(row.name) else np.nan,axis=1).rename('roc_adm')
        diag_metrics.append(roc_diag)
        adm_metrics.append(roc_adm)
    
    if 'avgprec' in metrics:
        avgprec_diag = model_outputs.apply(lambda col: average_precision_score(golden[col.name],col) if diag_condition(col.name) else np.nan).rename('avgprec_diag')
        avgprec_adm = model_outputs.apply(lambda row: average_precision_score(golden.loc[row.name],row) if adm_condition(row.name) else np.nan,axis=1).rename('avgprec_adm')
        diag_metrics.append(avgprec_diag)
        adm_metrics.append(avgprec_adm)

    # threshold dependent
    
    if 'accuracy' in metrics:
        accuracy_diag = model_predictions.apply(lambda col: accuracy_score(golden[col.name],col) if diag_condition(col.name) else np.nan).rename('accuracy_diag')
        accuracy_adm = model_predictions.apply(lambda row: accuracy_score(golden.loc[row.name],row) if adm_condition(row.name) else np.nan,axis=1).rename('accuracy_adm')
        diag_metrics.append(accuracy_diag)
        adm_metrics.append(accuracy_adm)

    if 'recall' in metrics:
        recall_diag = model_predictions.apply(lambda col: recall_score(golden[col.name],col) if diag_condition(col.name) else np.nan).rename('recall_diag')
        recall_adm = model_predictions.apply(lambda row: recall_score(golden.loc[row.name],row) if adm_condition(row.name) else np.nan,axis=1).rename('recall_adm')
        diag_metrics.append(recall_diag)
        adm_metrics.append(recall_adm)

    if 'precision' in metrics:
        precision_diag = model_predictions.apply(lambda col: precision_score(golden[col.name],col,zero_division=0) if diag_condition(col.name) else np.nan).rename('precision_diag')
        precision_adm = model_predictions.apply(lambda row: precision_score(golden.loc[row.name],row) if adm_condition(row.name) else np.nan,axis=1).rename('precision_adm')
        diag_metrics.append(precision_diag)
        adm_metrics.append(precision_adm)

    if 'f1' in metrics:
        f1_diag = model_predictions.apply(lambda col: f1_score(golden[col.name],col) if diag_condition(col.name) else np.nan).rename('f1_diag')
        f1_adm = model_predictions.apply(lambda row: f1_score(golden.loc[row.name],row) if adm_condition(row.name) else np.nan,axis=1).rename('f1_adm')
        diag_metrics.append(f1_diag)
        adm_metrics.append(f1_adm)
    
    # take weighted average
    diag_metrics_wavg = (pd.concat(diag_metrics,axis=1)
                         .multiply(diag_weights,axis=0)
                         .sum(axis=0)
                         .divide(
                             diag_weights.sum()
                         )
                        )
    
    adm_metrics_wavg = (pd.concat(adm_metrics,axis=1)
                        .multiply(adm_weights,axis=0)
                        .sum(axis=0)
                        .divide(
                            adm_weights.sum()
                        )
                       )

    res = pd.concat([diag_metrics_wavg,adm_metrics_wavg])
    
    res.index.name = 'metrics'
    
    return res