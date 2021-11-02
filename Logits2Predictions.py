import pandas as pd
import numpy as np

class Logits2Predictions():
    """
    This class transforms logit matrices (outputs of a model)
    into predictions based on a prediction method, eg: topk-based, threshold-based,etc...
    
    Receives dataframes as input and outputs a prediction daframe of same shape
    """
    def __init__(self,logits):
        """
        logits can be: dataframe (tensor to be supported)
        """
        self.logits = logits
    
    def topk(self,k:int=30):
        """
        k : int
            top@k
        """
        temp = self.logits.copy().filter(like='diag_')
        
        # get k-largest logit of each example
        temp['cutoff'] = temp.apply(lambda row: row.nlargest(k).iloc[-1],axis=1)
        
        prediction_data = temp.apply(lambda row: np.where(row >= row.cutoff,1,0),axis=1).tolist()
        
        prediction_df = pd.DataFrame(index=temp.index,
                                     data=prediction_data,
                                     columns=temp.columns
                                    ).drop(columns=['cutoff'])
        
        return prediction_df
