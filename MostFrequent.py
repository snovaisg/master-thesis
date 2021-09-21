import pandas as pd

class MostFrequent():
    """
    This class is a baseline that predicts diagnoses for the last visit
    by retrieving all the diagnostics in the history of 
    the patient. Diagnostics are ordered by highest frequency.
    
    
    For now, unfortunately, depends on a Mimic object
    """
    @staticmethod
    def predict(patient_id: int, code:str, mimic_obj):
        """
        predicts diagnoses of the last visit of a patient using their history.
        
        
        Parameters
        ----------
        
        patient_id : int
            The patient id from mimic
            
        code : str
            which coding scheme to use (eg.: ccs, icd9chapters, etc..)
            
        mimic_obj : Mimic
            mimic object to access the dataset
        Returns
        --------
        predicted : list
            list of predicted diagnoses for the last visit
        golden : list, no repeats
            list of actual diagnoses the patient got on the last visit
        """
        
        # get all admissions of the patient except the last one 
        admissions = mimic_obj.get_admissions_for_patient(patient_id)

        all_diagnoses = [mimic_obj.get_diagnoses_for_admission(adm)[code].tolist() for adm in admissions]
        
        train_diagnoses,test_diagnoses = all_diagnoses[:-1],all_diagnoses[-1]
        
        # flatten list
        train_diagnoses = [e for p in train_diagnoses for e in p]

        # sort by frequency
        all_diagnoses_counts = pd.Series(train_diagnoses).value_counts().sort_values(ascending=False)

        # return predictions and actual
        return all_diagnoses_counts.index.tolist(), list(set(test_diagnoses))


