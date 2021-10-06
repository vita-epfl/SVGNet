import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Union
import torch.nn.functional as F



class EncoderCell(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 256):
        """Initialize the encoder network.
        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
        """
        super(EncoderCell, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden=None) -> Any:
        """Run forward propagation.
        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden 
        """


        embedded = F.relu(self.linear1(x))
        if hidden:
            hidden = self.lstm1(embedded, hidden)
        else:
            hidden = self.lstm1(embedded,)


        return hidden

    
    
class EncoderLSTM(nn.Module):

    def __init__(self,):

        super(EncoderLSTM, self).__init__()
        self.encoder_cell = EncoderCell()
    def forward(self, x) -> Any:
        

        input_length = x.shape[1]
        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = x[:, ei, :]
            if ei==0:
                encoder_hidden = self.encoder_cell(encoder_input,)
            else:
                encoder_hidden = self.encoder_cell(encoder_input, encoder_hidden)


        return encoder_hidden[0]
    
    

