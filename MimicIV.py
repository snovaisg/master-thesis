import numpy as np
import pandas as pd
from config import Settings; settings = Settings()
import os

class MimicIV:
    def __init__(self,settings,grouper=None):
        
        self.grouper = grouper
        self.path_dataset_folder = os.path.join(settings.mimic_path,settings.mimic_iv_path)

        assert os.path.isdir(self.path_dataset_folder), f'Error: Please input a valid path to the dataset. Got: {self.path_dataset_folder}'
        
        self.filepath_admissions = 'core/admissions.csv.gz'
        
        self.filepath_meta_diagnoses = 'hosp/d_icd_diagnoses.csv.gz'
        self.filepath_diagnoses = 'hosp/diagnoses_icd.csv.gz'
        
        
        self.admissions = self.__read_admissions() # important that this is ran first
        self.diagnoses = self.__read_diagnoses()
        
    def __read_admissions(self):
        filepath = os.path.join(self.path_dataset_folder,self.filepath_admissions)
        
        date_parser=lambda x: pd.to_datetime(x,format='%Y-%m-%d %H:%M:%S')
        
        df = pd.read_csv(filepath,
                         parse_dates=[
                             'admittime',
                             'dischtime'
                         ],
                         date_parser=date_parser,
                         compression='gzip')
        
        df = df.sort_values('admittime', ascending=True)
        df['hadm_index'] = df.groupby('subject_id').admittime.cumcount()
        return df
    
    def read_meta_diagnoses(self):
        filepath = os.path.join(self.path_dataset_folder,self.filepath_meta_diagnoses)
        df = pd.read_csv(filepath,
                         compression='gzip')
        return df
    
    def get_diagnoses_for_admission(self,hadm_id: int) -> pd.DataFrame:
        return self.diagnoses[self.diagnoses.hadm_id == hadm_id]
    
    def __read_diagnoses(self):
        filepath = os.path.join(self.path_dataset_folder,self.filepath_diagnoses)
        df = pd.read_csv(filepath,
                         compression='gzip')
        
        # temporary measure: remove all diagnoses with icd10 coding
        df = df[df.icd_version == 9]
        
        #df = pd.merge(df,self.admissions[['hadm_id','admittime']],left_on='hadm_id',right_on='hadm_id')
        
        if self.grouper is not None:
            groups = self.grouper.get_available_groupers()
            
            for g in groups:
                df[g] = self.grouper.lookup(g,df['icd_code'])
                
        # sorting which helps for other operations
        df = df.sort_values('seq_num',ascending=True)
        df = pd.merge(df,self.admissions[['hadm_id','hadm_index']],left_on='hadm_id',right_on='hadm_id')
        df = df.sort_values('hadm_index')
        return df
    
    def read_diagnoses(self):
        return self.diagnoses
    def read_admissions(self):
        return self.admissions
