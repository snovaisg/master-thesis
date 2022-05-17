import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import warnings

def plot_reliability(data, accuracy=False, reliability=False,show_percent=False):
    """
    Creates a reliability plot.
    
    data : dict
        data must be the result of calling reliability.loc[idx[diag,:],:].reset_index().to_dict(orient='list')
        on the reliability dataframe.
    
    y_values : list
        list of mean accuracy OR relative frequency of positive examples of each bin
    
    """
    
    goal = {'accuracy':accuracy,'reliability':reliability}
    filtered = [e for e in goal if goal[e]]
    
    
    assert any(goal), 'Must choose at least 1'
    
    options = {'accuracy':dict(name='accuracies',y_label='accuracy',title='Acc. vs conf.',perc_col='perc_samples_predicted_class',perc_y_label='% predicted samples'),
               'reliability':dict(name='rel_freq_positive_examples',y_label='Rel. freq. positives',title='Reliability',perc_col='perc_samples_positive_class', perc_y_label='% positive samples')
              }
    
    
    conf = np.linspace(0,1,data['nbins'][0]+1)[:-1]+0.05
    
    ncols = len(filtered)
    nrows = 2 if show_percent else 1
    
    if not show_percent:
        fig, ax = plt.subplots(1, 2, figsize=(8, 8));
        for idx,e in enumerate(filtered):

            ax[idx].plot([0,1], [0,1], 'k--');
            ax[idx].plot(conf, data[options[e]['name']]);
            ax[idx].set_xticks((np.arange(0, 1.1, step=0.2)));
            ax[idx].set_ylabel(options[e]['y_label']);
            ax[idx].set_xlabel(r'confidence');
            ax[idx].set_xticks((np.arange(0, 1.1, step=0.2)));
            ax[idx].set_yticks((np.arange(0, 1.1, step=0.2)));
            ax[idx].set_title(options[e]['title'])
            ax[idx].set_aspect('equal')

            fig.tight_layout(pad=4.0);
            fig.suptitle(f'Reliability plots for diagnostic {data["diag"][0]}',y=0.75,size=15)
    else:
        fig, ax = plt.subplots(nrows, 2, figsize=(8, 8));
        fig.tight_layout(h_pad=3, w_pad=3)
        
        for idx,e in enumerate(filtered):
            
            #ax[1,idx].plot([0,1], [0,1], 'k--') ;
            ax[1,idx].bar(conf, data[options[e]['perc_col']],width=0.1);
            ax[1,idx].set_xticks((np.arange(0, 1.1, step=0.2)));
            ax[1,idx].set_ylabel('% samples');
            ax[1,idx].set_xlabel(r'confidence');
            ax[1,idx].set_xticks((np.arange(0, 1.1, step=0.2)));
            ax[1,idx].set_yticks((np.arange(0, 1.1, step=0.2)));
            ax[1,idx].set_aspect('equal')
            
            ax[0,idx].plot([0,1], [0,1], 'k--',label='perfect calibration');
            ax[0,idx].plot(conf, data[options[e]['name']]);
            ax[0,idx].set_xticks((np.arange(0, 1.1, step=0.2)));
            ax[0,idx].set_ylabel(options[e]['y_label']);
            ax[0,idx].set_xlabel(r'confidence');
            ax[0,idx].set_xticks((np.arange(0, 1.1, step=0.2)));
            ax[0,idx].set_yticks((np.arange(0, 1.1, step=0.2)));
            ax[0,idx].set_title(options[e]['title'],size=15)
            ax[0,idx].set_aspect('equal')
            ax[0,idx].legend()
            
    fig.suptitle(f'Reliability plots for {data["diag"][0]}',y=1.08,size=20)
    return fig, ax



def get_prediction_thresholds(prediction_data : pd.DataFrame, golden_data : pd.DataFrame, method='roc gm'):
    
    thresholds_data = None
    for diag in prediction_data.filter(like='diag_').columns:
        testy = golden_data.loc[:,diag].to_numpy().reshape((-1,1))
        yhat = prediction_data.loc[:,diag].to_numpy().reshape((-1,1))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No positive samples in y_true, true positive value should be meaningless")
            fpr, tpr, thresholds = roc_curve(testy, yhat);

        #geometric mean between sensitivity and specificity
        gmeans = np.sqrt(tpr * (1-fpr))

        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)

        threshold = pd.DataFrame(data=[[thresholds[ix],gmeans[ix]]],columns=['threshold','gmean (roc)'],index=[diag])

        if thresholds_data is None:
            thresholds_data = threshold
        else:
            thresholds_data = pd.concat([thresholds_data,threshold])
    return thresholds_data


def ece(logits: pd.Series, preds : pd.Series, goldens : pd.Series, nbins:int = 10):
    
    # confidences of predicted class, not positive class
    confidences = logits.where(preds==1,1-logits)
    
    accuracies = preds == goldens.to_numpy()
    
    ece = 0
    
    bins = np.linspace(0,1,nbins+1)
    for left,right in zip(bins[:-1],bins[1:]):
        
        in_bin = ((confidences > left) & (confidences < right)).values
        
        any_in_bin = in_bin_predicted_mask.sum() > 0
        
        acc_in_bin = accuracies[in_bin].mean() if any_in_bin_predicted else 0
        
        avg_confidence_in_bin = confidences[in_bin].mean()
        
        weight = in_bin.sum() / preds.shape[0]
        ece += weight * abs(acc_in_bin - avg_confidence_in_bin)
    
    return ece

def ECE(logits : pd.DataFrame, preds : pd.DataFrame, golden : pd.DataFrame, nbins=10):
    
    # confidence of predicted class
    confidences = logits.where(preds==1,1-logits)

    accuracies = preds == golden.to_numpy()
    ece = np.zeros(shape=(accuracies.shape[1],))

    bins = np.linspace(0,1,nbins+1)

    for left,right in zip(bins[:-1],bins[1:]):
        in_bin = ((confidences > left) & (confidences < right)).values

        avg_acc_in_bin = accuracies[in_bin].mean(axis=0).to_numpy()
        avg_confidence_in_bin = confidences[in_bin].mean(axis=0).to_numpy()

        weight = in_bin.sum(axis=0) / preds.shape[0]

        ece += weight * np.abs(avg_acc_in_bin - avg_confidence_in_bin)
    return ece