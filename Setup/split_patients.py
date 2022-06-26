#! /home/debian/Simao/miniconda3/envs/thesis/bin/python3

import sys
import os
cwd = os.getcwd()
print(cwd)

# protection against running this cell multiple times
assert os.path.dirname(cwd).split('/')[-1] == 'master-thesis','Oops, directory already changed previously as indended. Ignoring...'

new_cwd = os.path.dirname(cwd) # parent directory
sys.path.append(new_cwd)

import copy
import json

from os.path import basename
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from rnn_utils import DiagnosesDataset, MYCOLLATE, split_dataset

from torch.utils.data import DataLoader

import argparse



def main():

    parser = argparse.ArgumentParser(description='Split patients into train-test-validation')
    parser.add_argument('-i','--input_file', type=str, help='file with patient ids')
    parser.add_argument('-o','--output_path', type=str, help='path to save the splits')
    parser.add_argument('-v','--val_size', type=float, help='size of the validation set in fraction')
    parser.add_argument('-t','--test_size', type=float, help='size of the test set in fraction')
    
    
    

    args = parser.parse_args()
    print(args.input_file)
    print(args.output_path)
    print(args.val_size)
    print(args.test_size)
    val_size = args.val_size
    test_size = args.test_size
    
    assert os.path.isdir(args.output_path), 'Output path doesnt exist'
    
    seed = 426
    print(f'{seed=}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_size = 1 - (test_size + val_size)
    assert test_size + val_size < 1, 'Oops'
    print('train_size=',train_size)
    

    with open(args.input_file,'r') as fp:
        dataset = json.load(fp)
    
    pat_ids = list(dataset.keys())

    whole_train,test = train_test_split(pat_ids,test_size=test_size)

    val_size_corrected = val_size/(1-test_size)
    train,val = train_test_split(whole_train,test_size=val_size_corrected)

    print(f"{len(whole_train)=}")
    print(f"{len(train)=}")
    print(f"{len(val)=}")
    print(f"{len(test)=}")
    
    def generate_subset_data(original,inclusion_list):
        df = pd.DataFrame(original)
        subset_original = df.loc[:,inclusion_list].to_dict()
        return subset_original

    whole_train_subset = generate_subset_data(dataset,whole_train)
    train_subset = generate_subset_data(dataset,train)
    val_subset = generate_subset_data(dataset,val)
    test_subset = generate_subset_data(dataset,test)

    # sanity checks

    print(f"{len(whole_train_subset)=}")
    print(f"{len(train_subset)=}")
    print(f"{len(val_subset)=}")
    print(f"{len(test_subset)=}")

    # file suffix with metadata
    params = {'train':train_size,
              'eval':val_size,
              'test':test_size,
              'rseed':seed,
             }
    
        # assign filename to each subset
    names = {'whole_train_subset':whole_train_subset,
             'train_subset':train_subset,
             'val_subset':val_subset,
             'test_subset':test_subset
            }

    # Save (finally!)
    for name in names:

        filename = os.path.join(args.output_path,name)
        with open(filename+'.json','w') as fp:
            json.dump(names[name],fp)

    with open(os.path.join(args.output_path,'splits_metadata.json'),'w') as fp:
        json.dump(params,fp)
        
    print('Done')

if __name__ == '__main__':
    main()
