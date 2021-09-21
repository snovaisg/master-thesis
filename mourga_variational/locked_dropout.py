import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence

"""
Code from https://github.com/salesforce/awd-lstm-lm
paper: https://arxiv.org/pdf/1708.02182.pdf (see Section 4.2)
"""


class LockedDropout(nn.Module):
    """
    This function applies dropout to the input tensor x.
    The shape of the tensor x in our implementation is (batch_size, seq_len, feature_size)
    (contrary to Merity's AWD that uses (seq_len, batch_size, feature_size)).
    So, we sample a mask from the 'feature_size' dimension,
    but a different mask for each 'batch_size' dimension,
    and expand it in the 'sequence_length' dimension so that
    we apply the SAME mask FOR EACH TIMESTEP of the RNN (= 'seq_len' dim.).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        if type(x) == torch.nn.utils.rnn.PackedSequence:
            x, lengths = pad_packed_sequence(x, batch_first=True)
            batch_size, seq_length, feat_size = x.size()
            
            m = x.data.new(batch_size, 1, feat_size).bernoulli_(1 - dropout)
            mask = Variable(m, requires_grad=False) / (1 - dropout)
            mask = mask.expand_as(x)
            out = x * mask
            x = pack_padded_sequence(out,
                                     lengths=lengths,
                                     batch_first=True,
                                     enforce_sorted=False
                                    )
            return x
        
        # else: sequences are not packed
        batch_size, seq_length, feat_size = x.size()
        m = x.data.new(batch_size, 1, feat_size).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x