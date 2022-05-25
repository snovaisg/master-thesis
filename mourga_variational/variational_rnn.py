import sys
import os
this_dir = os.path.abspath(os.path.dirname(__file__))
if this_dir not in sys.path:
    sys.path.append(this_dir)

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class VariationalRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_labels,
                 num_layers=1,
                 rnn_type='rnn',
                 dropouti=0., # dropout probability to inputs
                 dropoutw=0., # dropout probability to network units
                 dropouto=0., # dropout probability to outputs between layers
                ):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        We use batch_first=True for our implementation.
        Tensors are are shape (batch_size, sequence_length, feature_size).
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(VariationalRNN, self).__init__()

        self.lockdrop = LockedDropout()

        assert rnn_type in ['lstm', 'gru', 'rnn'], 'type of RNN is not supported'

        if not isinstance(hidden_size, list):
            nhidden = [hidden_size]

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_labels = n_labels
        self.dropouti = dropouti               # dropout probability of inputs
        self.dropoutw = dropoutw               # dropout probability to network units
        self.dropouto = dropouto               # dropout probability to outputs of layers
        
        self.N = 15 # 15 forward passes in MC dropout
        #self.T = nn.Parameter(torch.tensor(1.0))
        
        # create model
        args = lambda l: dict(input_size=input_size if l == 0 else hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              batch_first=True
                             )
            
        if rnn_type == 'lstm':
            self.rnns = [nn.LSTM(**args(l)) for l in range(num_layers)]
        
        elif rnn_type == 'gru':
            self.rnns = [nn.GRU(**args(l)) for l in range(num_layers)]
            
        elif rnn_type == 'rnn':
            self.rnns = [nn.RNN(**args(l)) for l in range(num_layers)]
            

        # Dropout to netowork units (matrices weight_hh AND weight_ih of each layer of the RNN)
        if dropoutw:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0', 'weight_ih_l0'],
                                    dropout=dropoutw) for rnn in self.rnns]

        self.rnns = nn.ModuleList(self.rnns)
        
        self.lin = nn.Linear(in_features = hidden_size,
                             out_features= n_labels
                            )

    def forward(self, batch, take_mc_average):
        """

        Parameters
        ----------
        
        batch: torch.tensor, size = (batch_size,seq_length,seq_size)
        
        mc_dropout : Bool
            if True then perform self.N forward passes
        
        :return:
        """
        
        res = list()
        for npass in range(self.N):
            # Dropout to inputs of the RNN (dropouti)
            out = self.lockdrop(batch, self.dropouti)

            # for each layer of the RNN
            for l, rnn in enumerate(self.rnns):
                # calculate hidden states and output from the l RNN layer
                out, _ = rnn(out)

                # apply dropout to the output of the l-th RNN layer (dropouto)
                out = self.lockdrop(out, self.dropouto)

            out = self.lin(pad_packed_sequence(out,batch_first=True)[0])

            res.append(out)
            
        res = torch.stack(res)
        if take_mc_average:
            res = res.mean(0)
        return res

    
def outs2df_mc(model,dataloader,dataset,return_golden=False):
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
            outs = model(inputs,take_mc_average=False)
            
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