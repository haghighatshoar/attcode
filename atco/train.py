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
            # NOTE: here as a test/validation, we put a condition that from each 100 only a single one passes through
            # the channel noise and see how this affects things
            if (epoch + 1)%10 == 0:
                # it is time to throw some noise and see if training has happened
                noisy_output = self.chan(encoded_bits)
            else:
                noisy_output = encoded_bits.copy()

            # compute the corresponding LLR
            llr = self.chan.llr(noisy_output)

            llr = llr.T.reshape(num_batch, self.dec.num_tokens, self.dec.num_token_bits)
            llr = torch.tensor(data=llr, dtype=torch.float32)

            # decompose the LLR into `num_tokens x token_size`
            # give llr to the input of the networks and compute the decoded block
            decoded_tokens_one_hot = self.dec(llr)

            # compute the one-hot encoding of the original bits/tokens
            encoded_tokens_binary = encoded_bits.T.reshape(num_batch, self.dec.num_tokens, self.dec.num_token_bits)
            encoded_tokens_decimal = torch.tensor(
                data=np.sum(encoded_tokens_binary * 2 ** np.arange(self.dec.num_token_bits-1, -1, step=-1).reshape(1, 1, -1), axis=-1))

            encoded_tokens_one_hot = F.one_hot(encoded_tokens_decimal.reshape(1, -1),
                                               num_classes=2 ** self.dec.num_token_bits).reshape(num_batch,
                                                                                                 self.dec.num_tokens,
                                                                                                 -1).float()

            # compute the loss
            # loss = loss_func(decoded_tokens_one_hot, encoded_tokens_one_hot)
            loss = torch.mean(torch.log(1.0 / decoded_tokens_one_hot[encoded_tokens_one_hot==1]))

            loss.backward()
            opt.step()
            opt.zero_grad()

            #================================================
            #          compute bit error probability
            #================================================
            # compute the error probability as well
            # first find the one-hot-encodings with maximum likelihood and encode them as binary strings
            decoded_tokens_decimal = torch.argmax(decoded_tokens_one_hot, axis=-1).numpy()
            decode_str_bits = [''.join([np.binary_repr(num, width=self.dec.num_token_bits) for num in batch_data]) for
                               batch_data in decoded_tokens_decimal]

            # join the binary strings and convert them into bits
            decoded_bits = np.array([[int(ch) for ch in bit_string] for bit_string in decode_str_bits],
                                    dtype=np.int64).T

            # compute error rate across codeowords in each batch
            bit_err_rate = np.sum(np.abs(decoded_bits.ravel() - encoded_bits.ravel())) / encoded_bits.size
            
            #================================================
            #        compute block error probability
            #================================================
            decoded_tokens_binary = decoded_bits.T.reshape(num_batch, self.dec.num_tokens, self.dec.num_token_bits)
            decoded_tokens_decimal = np.sum(decoded_tokens_binary * (2**np.arange(self.dec.num_token_bits-1,-1,step=-1)).reshape(1,1,-1), axis=-1)
            
            token_err_rate = np.sum(decoded_tokens_decimal != encoded_tokens_decimal.numpy())/encoded_tokens_decimal.numel()
            

            # save the loss and track it as well
            loss_vec.append(loss.item())
            log_metric("loss", loss.item())
            log_metric("bit error rate", bit_err_rate)
            log_metric("token error rate", token_err_rate)

            print(f"epoch: {epoch:5}/{num_epoch:5}, loss:{loss.item():1.5f}, bit-err-rate:{bit_err_rate:0.6f}, token-err-rate:{token_err_rate:0.6f}")


def train_decoder():
    # channel info
    eps = 0.1
    chan = BSC(eps=eps)
    chan_cap = 1 - (-eps * np.log2(eps) - (1 - eps) * np.log2(1 - eps))

    # choose code rate
    backoff = 4
    rate = min([chan_cap / backoff, 0.2])

    # channel and code info
    k = 40
    n = k / rate
    n = int(10 * np.ceil(n / 10))

    # encoder
    enc = Encoder(k=k, n=n)

    # decoder
    num_att_heads = [10]
    num_tokens = n//10

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
    num_batch = 5
    lr = 0.0001

    trainer.train(num_epoch=num_epochs, num_batch=num_batch, lr=lr)


def main():
    train_decoder()


if __name__ == '__main__':
    main()
