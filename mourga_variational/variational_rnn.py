import sys
import os
this_dir = os.path.abspath(os.path.dirname(__file__))
if this_dir not in sys.path:
    sys.path.append(this_dir)

from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class VariationalRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_labels,
                 num_layers=1,
                 rnn_type='RNN',
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

        assert rnn_type in ['LSTM', 'GRU', 'RNN'], 'type of RNN is not supported'

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
        
        # create model
        args = lambda l: dict(input_size=input_size if l == 0 else hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              batch_first=True
                             )
            
        if rnn_type == 'LSTM':
            self.rnns = [nn.LSTM(**args(l)) for l in range(num_layers)]
        
        elif rnn_type == 'GRU':
            self.rnns = [nn.GRU(**args(l)) for l in range(num_layers)]
            
        elif rnn_type == 'RNN':
            self.rnns = [nn.RNN(**args(l)) for l in range(num_layers)]
            

        # Dropout to netowork units (matrices weight_hh AND weight_ih of each layer of the RNN)
        if dropoutw:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0', 'weight_ih_l0'],
                                    dropout=dropoutw) for rnn in self.rnns]

        self.rnns = nn.ModuleList(self.rnns)
        
        self.lin = nn.Linear(in_features = hidden_size,
                             out_features= n_labels
                            )

    def forward(self, batch):
        """

        :param x: tensor of shape (batch_size, seq_len, embedding_size)
        :param hidden: tuple (h0, c0), each of shape (num_layers * num_directions, batch_size, hidden_size)
        :param lengths: tensor (size 1 with true lengths)
        :return:
        """

        # Dropout to inputs of the RNN (dropouti)
        out = self.lockdrop(batch, self.dropouti)

        # for each layer of the RNN
        for l, rnn in enumerate(self.rnns):
            # calculate hidden states and output from the l RNN layer
            out, _ = rnn(out)
            
            # apply dropout to the output of the l-th RNN layer (dropouto)
            out = self.lockdrop(out, self.dropouto)
        
        out = self.lin(pad_packed_sequence(out,batch_first=True)[0])

        return out
