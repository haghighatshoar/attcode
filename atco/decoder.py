# -----------------------------------------------------------------------------------------------------------------------
# This file provides a trainable decoder module based on the attention mechanism.
# 
#
#
# (C) Saeid Haghighatshoar
# email: haghighatshoar@gmail.com
#
#
# last update: 29.01.2024
# ------------------------------------------------------------------------------------------------------------------------
from atco.encoder import Encoder
from atco.channel import Channel, BSC

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Softmax
from torch.nn import MultiheadAttention, Sequential, Linear
from typing import List

import matplotlib.pyplot as plt


class AttLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int, **kwargs):
        """this module simulates a linear layer as it used in attention mechanism.

        Args:
            in_features (int): nmber of input features in the linear layer in each attention head.
            out_features (int): number of output features in the linear layer in each attention head.
            num_heads (int): number of attention heads.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.all_out_features = self.out_features * self.num_heads

        # build the inner linear module for simulation
        # NOTE: we have 3 x since we need to compute query, key, value tensors for each batch
        self.linear = Linear(in_features=self.in_features, out_features=3 * self.all_out_features, **kwargs)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """this function passes the input from the linear layer and decompose into `num_heads` outputs for `num_heads`
        attention heads in the layer afterwards.

        Args:
            input (torch.tensor): input tensor of dim `batch_size x num_token x token_dim`

        Returns:
            torch.tensor: output tensor of dim `batch_size x num_token x num_heads x token_embed_dim` where `token_embed_dim` is the
            dimension of the token after being embedded thtough the linear layer.
        """
        # check dimensions 
        try:
            batch, num_token, token_dim = input.shape
        except:
            raise ValueError("input should be of dimension `batch_size x num_token x token_dim`")

        # compute the output
        # NOTE: we need to repeat the last dim 3-times as it is needed for query, key, value generation
        output = self.linear(input)

        # decompose the output into query, key, value tuple
        output = output.reshape(batch, num_token, -1)

        return output


class Attention(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        this class is a wrapper around the MultiHeadAttention class.
        Args:
            all the inputs are as in the multi-head attention.
        """
        super().__init__()

        self.att = MultiheadAttention(*args, **kwargs)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        this function implement the forward function for MuliHeadAttention
        Args:
            input (torch.tensor): input tensor of dimension `B x T x (3 x embed_dim) where 3 is because it expect to
            receive the full input containing the query, key, and values.

        Returns:
            torch.tensor: resulting attention output.
        """
        try:
            B, T, input_dim = input.shape
            if input_dim % (3 * self.att.num_heads) != 0:
                raise ValueError("")
        except:
            raise ValueError("input tensor should be of dim `B x T x (3 x embed_dim)`")

        # extract the weights and values
        qkv_dim = input_dim // 3
        q = input[:, :, :qkv_dim]
        k = input[:, :, qkv_dim:2 * qkv_dim]
        v = input[:, :, 2 * qkv_dim:]

        output, weights = self.att(q, k, v)

        return output


class Decoder(torch.nn.Module):
    def __init__(self, k: int, n:int, num_tokens:int, num_att_heads: List[int]):
        """This class builds a trainable channel decoder based on attention mechanism used in LLMs.

        Args:
            k (int): input information block length.
            n (int): encoded codewords block length.
            num_tokens (int): number of tokens used for decomposing the encoded block.
            num_att_heads (List[int]): a list containing the number of attention heads in each layer.

        Raises:
            ValueError: if there is any mismatch between the parameters of various blocks.
        """
        super().__init__()

        # input and output bit length of the encoder
        self.k = k
        self.n = n

        if n % num_tokens != 0:
            raise ValueError(
                "encoded block size should divide number of tokens otherwise one cannot build attention layers of the same dim!")

        self.num_tokens = num_tokens
        self.num_token_bits = n // num_tokens

        # number of layers needed
        self.num_layers = len(num_att_heads)
        self.num_att_heads = num_att_heads

        # build the network
        nets = []
        inout_dim = self.num_token_bits

        for num_heads in self.num_att_heads:
            # linear layer
            linear = AttLinear(
                in_features=inout_dim,
                out_features=inout_dim,
                num_heads=num_heads,
                bias=False,
            )

            # build the multi-head attention layer
            embed_dim = inout_dim * num_heads
            att = Attention(embed_dim=embed_dim, num_heads=num_heads, bias=False, batch_first=True)

            # update the input dim for the next layer
            inout_dim = embed_dim

            # add both layers to the list of layers
            nets.extend([linear, att])

        # add the final linear layer and do softmax to convert the tokens into 0-1 bits
        one_hot_linear_dim = nets[-1].att.embed_dim
        num_one_hot_output = 2 ** self.num_token_bits

        one_hot_linear_layer = Linear(in_features=one_hot_linear_dim, out_features=num_one_hot_output, bias=False)
        nets.append(one_hot_linear_layer)

        softmax = Softmax(dim=-1)
        nets.append(softmax)

        self.net = Sequential(*nets)

    def forward(self, input:torch.tensor) -> torch.tensor:
        return self.net(input)

    def parameters(self):
        return self.net.parameters()
