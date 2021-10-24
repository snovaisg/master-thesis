# +
import os

import numpy as np
import pandas as pd
from datetime import datetime
from IPython.display import display
from config import Settings

class Mimic():
    """
    Class that reads and preprocesses the Mimic dataset
    """
    def __init__(self, settings : Settings, grouper=None):
        
        self.path_dataset_folder = settings.mimic_path

        assert os.path.isdir(self.path_dataset_folder), f'Error: Please input a valid path to the dataset. Got: {self.path_dataset_folder}'
            
        self.grouper = grouper
        
        self.filename_icd9chapters = 'icd9cm_chapters.csv'
        self.filename_admissions = 'ADMISSIONS.csv.gz'
        self.filename_diagnoses = 'DIAGNOSES_ICD.csv.gz'
        self.filename_procedures = 'PROCEDURES_ICD.csv.gz'
        self.filename_icd9_procedure_codes = 'D_ICD_PROCEDURES.csv.gz'
        self.filename_icd9_diagnoses_codes = 'D_ICD_DIAGNOSES.csv.gz'
        
        self.diagnoses = self.__read_diagnoses()
        self.admissions = self.__read_admissions()
        
    def read_icd9_procedures_codes(self):
        full_file_path = os.path.join(self.path_dataset_folder,self.filename_icd9_procedure_codes)
        df = pd.read_csv(full_file_path,
                         compression='gzip',
                         usecols=['ICD9_CODE','SHORT_TITLE','LONG_TITLE'],
                         index_col='ICD9_CODE',
                         dtype={'ICD9_CODE':str,
                                'SHORT_TITLE':str,
                                'LONG_TITLE':str
                               }
                        )
        return df
    
    def read_icd9_diagnoses_codes(self):
        full_file_path = os.path.join(self.path_dataset_folder, self.filename_icd9_diagnoses_codes)
        df = pd.read_csv(full_file_path,
                         compression='gzip',
                         usecols=['ICD9_CODE','SHORT_TITLE','LONG_TITLE'],
                         index_col='ICD9_CODE',
                         dtype={'ICD9_CODE':str,
                                'SHORT_TITLE':str,
                                'LONG_TITLE':str
                               }
                        )
        
        if self.grouper is not None:
            groups = self.grouper.get_available_groupers()
            
            for g in groups:
                df[g] = self.grouper.lookup(g,df['ICD9_CODE'])
        return df
        
    def read_procedures(self):
        full_file_path = os.path.join(self.path_dataset_folder,self.filename_procedures)
        df = pd.read_csv(full_file_path,
                         compression='gzip',
                         index_col='ROW_ID'
                        )
        return df
    def read_procedures_all(self):
        df = self.read_procedures()
        icd9_proc = self.read_icd9_procedures_codes()
        df = df.join(icd9_proc,on='ICD9_CODE')
        return df
    
    
    def read_admissions(self):
        return self.admissions
    
    def __read_admissions(self):
        full_file_path = os.path.join(self.path_dataset_folder,self.filename_admissions)
        
        date_parser=lambda x: pd.to_datetime(x,format='%Y-%m-%d %H:%M:%S')
        
        df = pd.read_csv(full_file_path,
                         compression='gzip',
                         parse_dates=[
                             "ADMITTIME",
                             "DISCHTIME",
                             "EDREGTIME",
                             "EDOUTTIME",
                             "DEATHTIME"
                         ],
                         date_parser=date_parser
                        ).drop(columns='ROW_ID')
        
        # Consider "Urgent" to be the same as "Emergency"
        # https://mimic.mit.edu/docs/iii/tables/admissions/#admission_type
        df['ADMISSION_TYPE'] = df['ADMISSION_TYPE'].replace('URGENT','EMERGENCY')
        
        # sorts the admissions. inside each patient they will be sorted when subselecting patients
        df = df.sort_values('ADMITTIME', ascending=True)
        return df
    
    def get_diagnoses_for_admission(self,hadm_id: int) -> pd.DataFrame:
        return self.diagnoses[self.diagnoses.HADM_ID == hadm_id]
    
    def get_admissions_for_patient(self, patient_id: int) -> list:
        admissions_list = (self.admissions
                           .loc[self.admissions['SUBJECT_ID'] == patient_id,
                                ['HADM_ID','ADMITTIME']
                               ]
                           .sort_values('ADMITTIME',
                                        ascending=True
                                       )
                           .loc[:,'HADM_ID'].tolist()
                          )
                  
                  
        # predicting future bugs
        assert type(admissions_list) == list and len(admissions_list) > 0, ' Oopsie'
        
        return admissions_list
        
    
    def read_diagnoses(self):
        return self.diagnoses
    
    def __read_diagnoses(self):
        full_file_path = os.path.join(self.path_dataset_folder,self.filename_diagnoses)
        df = pd.read_csv(full_file_path,
                         compression='gzip',
                         dtype={'ICD9_CODE':str}
                        )
        
        if self.grouper is not None:
            groups = self.grouper.get_available_groupers()
            
            for g in groups:
                df[g] = self.grouper.lookup(g,df['ICD9_CODE'])
                
        # sorts the diagnoses by seq. inside each patient they will be sorted when subselecting patients
        
        df = df.sort_values('SEQ_NUM',ascending=True)
        return df
    
    def read(self,filename):
        full_file_path = os.path.join(self.path_dataset_folder,filename)
        return pd.read_csv(full_file_path,
                           compression='gzip')
