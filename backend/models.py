"""

"""

__all__ = ['AutoEncoderMLP']

import torch
from typing import List, Union, Callable


class MLP(torch.nn.Module):

    def __init__(
            self,
            input_size: int,                            # Length of input tensor
            hidden_size: Union[int, List[int]],         # Length of HL outputs.
            output_size: int,                           # z_dim
            activation_fn: Callable = torch.nn.ReLU,    # Activation function
            keep_prob: float = 1.0,                     # Dropout, called after activation.
    ):
        super().__init__()
        # Determine input/output sizes of each hidden layer
        hidden_size = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        in_dim = [input_size] + hidden_size
        out_dim = hidden_size + [output_size]
        self.n_layers = len(in_dim)

        # Construct Layers
        self.layer_names = [f'{self.__class__.__name__}_hl_{i:02.0f}' for i in range(self.n_layers)]
        for name, in_s, hl_s in zip(self.layer_names, in_dim, out_dim):
            self.__setattr__(name=name, value=torch.nn.Linear(in_features=in_s, out_features=hl_s, bias=True))

        self.activation = activation_fn()
        self.dropout = torch.nn.Dropout(p=keep_prob)

    def forward(self, x):
        """"""
        for i, ln in enumerate(self.layer_names):
            x = self.__getattr__(ln)(x)
            x = self.activation(x) if i < (self.n_layers - 1) else x
            # x = self.dropout(x)
        return x


class AutoEncoderMLP(torch.nn.Module):

    def __init__(
            self,
            input_size: int,                            # Size of input into network (i.e. encoder)
            enc_hidden_size: Union[int, List[int]],     # Length of encoder hidden layer outputs
            latent_size: int,                           # Length of latent representation vector.
            dec_hidden_size: Union[int, List[int]],     # Length of decoder hidden layer outputs
            output_size: int,                           # Length of network output.
            activation_fn: Callable = torch.nn.ReLU,    # Activation function
            keep_prob: float = 0.8,                     # Dropout, called after activation.
    ):
        super().__init__()

        # Network
        self.encoder = MLP(input_size, enc_hidden_size, latent_size, activation_fn, keep_prob)
        self.decoder = MLP(latent_size, dec_hidden_size, output_size, activation_fn, keep_prob)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.nn.Sigmoid()(x)
        return x






