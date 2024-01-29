# -----------------------------------------------------------------------------------------------------------------------
# In this file, we apply attention-based channel decoder training to design a good channel decoder over various channels.
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
from atco.decoder import Decoder

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mlflow import log_metric


class Trainer:
    def __init__(self, enc: Encoder, chan: Channel, dec: Decoder):
        # some sanity checks
        if enc.k != dec.k or enc.n != dec.n:
            raise ValueError("encoded and decoder have different block-size parameters!")

        self.enc = enc
        self.dec = dec
        self.chan = chan

        self.k = enc.k
        self.n = enc.n

    def train(self, num_epoch: int, num_batch: int, lr: float):
        """this function trains the decoder network with random bits as input.

        Args:
            num_epoch (int): number of training epochs.
            num_batch (int): number of batch sizes used in training.
            lr (float): the learning to be adopted for training.
        """

        # build the optimizer
        opt = Adam(self.dec.parameters(), lr=lr)
        loss_func = torch.nn.CrossEntropyLoss()

        loss_vec = []

        for epoch in range(num_epoch):
            # build a random input bit
            input_bits = (np.random.randn(self.k * num_batch) < 0.0).astype(np.int64).reshape(self.k, num_batch)

            # encode the bits
            encoded_bits = self.enc(input_bits)

            # compute noisy bits
            noisy_output = self.chan(encoded_bits)

            # compute the corresponding LLR
            llr = self.chan.llr(noisy_output)

            llr = llr.T.reshape(num_batch, self.dec.num_tokens, self.dec.num_token_bits)
            llr = torch.tensor(data=llr, dtype=torch.float32)

            # decompose the LLR into `num_tokens x token_size`
            # give llr to the input of the networks and compute the decoded block
            decoded_tokens_one_hot = self.dec(llr)

            # compute the one-hot encoding of the original bits/tokens
            encoded_tokens = encoded_bits.T.reshape(num_batch, self.dec.num_tokens, self.dec.num_token_bits)
            encoded_token_binary = torch.tensor(
                data=np.sum(encoded_tokens * 2 ** np.arange(self.dec.num_token_bits).reshape(1, 1, -1), axis=-1))

            encoded_tokens_one_hot = F.one_hot(encoded_token_binary.reshape(1, -1),
                                               num_classes=2 ** self.dec.num_token_bits).reshape(num_batch,
                                                                                                 self.dec.num_tokens,
                                                                                                 -1).float()

            # compute the loss
            # loss = loss_func(decoded_tokens_one_hot, encoded_tokens_one_hot)
            loss = torch.mean(torch.log(1.0 / decoded_tokens_one_hot[encoded_tokens_one_hot==1]))

            loss.backward()
            opt.step()
            opt.zero_grad()

            # compute the error probability as well
            # first find the one-hot-encodings with maximum likelihood and encode them as binary strings
            decoded_one_hot_index = torch.argmax(encoded_tokens_one_hot, axis=-1).numpy()
            decode_str_bits = [''.join([np.binary_repr(num, width=self.dec.num_token_bits) for num in batch_data]) for
                               batch_data in decoded_one_hot_index]

            # join the binary strings and convert them into bits
            decoded_bits = np.array([[int(ch) for ch in bit_string] for bit_string in decode_str_bits],
                                    dtype=np.int64).T

            # compute error rate across codeowords in each batch
            bit_err_rate = np.sum(np.abs(decoded_bits.ravel() - encoded_bits.ravel())) / encoded_bits.size

            # save the loss and track it as well
            loss_vec.append(loss.item())
            log_metric("loss", loss.item())
            log_metric("bit error rate", bit_err_rate)

            print(f"epoch: {epoch} / {num_epoch}, loss:{loss.item()}, bit-err-rate:{bit_err_rate}")


def train_decoder():
    # channel info
    eps = 0.1
    chan = BSC(eps=eps)
    chan_cap = 1 + (eps * np.log2(eps) + (1 - eps) * np.log2(1 - eps))

    # choose code rate
    backoff = 3
    rate = chan_cap / backoff

    # channel and code info
    k = 20
    n = k / rate
    n = int(10 * np.ceil(n / 10))

    # encoder
    enc = Encoder(k=k, n=n)

    # decoder
    num_att_heads = [5,]
    num_tokens = 20

    dec = Decoder(
        k=k,
        n=n,
        num_tokens=num_tokens,
        num_att_heads=num_att_heads,
    )

    # trainer module
    trainer = Trainer(enc=enc, chan=chan, dec=dec)

    # training
    num_epochs = 100_000
    num_batch = 10
    lr = 0.0001

    trainer.train(num_epoch=num_epochs, num_batch=num_batch, lr=lr)


def main():
    train_decoder()


if __name__ == '__main__':
    main()
