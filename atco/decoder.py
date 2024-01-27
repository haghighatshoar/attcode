# -----------------------------------------------------------------------------------------------------------------------
# This file provides a trainable decoder module based on the attention mechanism.
# 
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 26.01.2024
# ------------------------------------------------------------------------------------------------------------------------
from atco.encoder import Encoder
from atco.channel import Channel 

import numpy as np

import torch
from torch.nn import MultiheadAttention, Sequential, Linear
from torch.optim import Adam
from typing import List


class AttLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features:int, num_heads:int, **kwargs):
        """this module simulates a linear layer as it used in attention mechanism.

        Args:
            in_features (int): nmber of input features in the linear layer in each attention head.
            out_features (int): number of output features in the linear layer in each attention head.
            num_heads (int): number of attention heads.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads 
        
        self.all_out_features = self.out_features * self.num_heads
        
        # build the inner linear module for simulation
        # NOTE: we have 3 x since we need to compute query, key, value tensors for each batch
        self.linear = Linear(in_features=3*self.in_features, out_features=self.all_out_features, **kwargs) 
    
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
        output = self.linear(input)
        
        # decomose the output into query, key, value tuple
        output = output.reshape(batch, num_token, self.num_heads, self.out_features)
        
        return output
        
        
        

class Decoder:
    def __init__(self, k:int, n:int, num_tokens: int, enc: Encoder, chan: Channel, num_att_heads:List[int]):
        """This class builds a trainable channeld decoder based on attention mechanism used in LLMs.

        Args:
            k (int): input information block length.
            n (int): encoded codewords block length.
            num_tokens (int): number of tokens used for decomposing the encoded block.
            enc (Encoder): encoder used for encoding the bits.
            chan (Channel): channel for which the training is done.
            num_att_heads (List[int]): a list containing the number of attention heads in each layer.

        Raises:
            ValueError: if there is any mismatch between the parameters of various blocks.
        """
        # input and output bit length of the encoder
        self.k = k
        self.n = n 

        if n % num_tokens != 0:
            raise ValueError("encoded block size should divide number of tokens otherwise one cannot build attention layers of the same dim!")

        self.num_tokens = num_tokens
        self.num_token_bits = n // num_tokens

        self.chan = chan 
        self.enc = enc 

        # some sanity checks
        if self.enc.k != self.k or self.enc.n != self.n:
            raise ValueError("input and output block length in encoder and decoder are not the same!")

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
            )

            # build the multi-head attention layer
            embed_dim = inout_dim * num_heads
            att = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

            # update the input dim for the next layer
            inout_dim = embed_dim

            # add both layers to the list of layers
            nets.extend([linear, att])

        self.net = Sequential(*nets)
    
    def train(self, num_epoch: int, num_batch: int):
        """this function trains the network with random input bits.

        Args:
            num_epoch (int): number of training epochs.
            num_batch (int): number of batch sizes used in training.
        """
        
        # build the optimizer
        lr = 0.001
        opt = Adam(self.net.parameters(), lr=lr)
        loss_func = None
        
        for epoch in range(num_epoch):
            # build a random input bit
            input_bits = (np.random.randn(self.k * num_batch) < 0.0).astype(np.int64).reshape(self.k, num_batch)
            
            # encode the bits
            encoded_bits = self.enc(input_bits)
            
            # compute noisy bits
            noisy_output = self.chan(encoded_bits)
            
            # compute the corrsponding LLR
            llr = self.chan.llr(noisy_output)
            
            llr = llr.reshape(num_batch, self.num_tokens, self.num_token_bits)
            
            # decompose the LLR into `num_tokens x token_size`
            # give llr to the input of the networks and compute the decoded block
            decoded_bits = self.net(llr)
            
            # compute the loss
            loss = loss_func(decoded_bits, encoded_bits)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
    
            


