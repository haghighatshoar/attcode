# -----------------------------------------------------------------------------------------------------------------------
# This file provides a collection of basic and well-known channels used in communication systems.
# 
#
#
# (C) Saeid Haghighatshoar
# email: haghighatshoar@gmail.com
#
#
# last update: 29.01.2024
# ------------------------------------------------------------------------------------------------------------------------
import numpy as np


class Channel:
    def __init__(self, *args, **kwargs):
        pass

    def process(self, input: np.ndarray) -> np.ndarray:
        """this module processes the input data to the channel and generates the correspinding random output.

        Args:
            input (np.ndarray): an array containing the input bit sequence.

        Raises:
            NotImplementedError: this should be implemented in each instance of the channel.

        Returns:
            np.ndarray: an array containing the noisy outputs received from the channel.
        """
        raise NotImplementedError("this function needs to be implemented in each instance of the channel!")

    def llr(self, output: np.ndarray) -> np.ndarray:
        """this method computes the LLR for the outputs received from the channel.

            Args:
                output (np.ndarray): _description_

            Raises:
                NotImplementedError: this method needs to be implemented in each channel.

            Returns:
                np.ndarray: an array containing the LLR values for the ouputs received from the channel.
            """
        raise NotImplementedError("this method needs to be implemented for all the sub-classes!")

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """ this is the same as the process function. """
        return self.process(*args, **kwargs)


class BSC(Channel):
    def __init__(self, eps: float):
        """this class represents a binary symmetric channel with an error rate eps.

        Args:
            eps (float): bit error rate of the channel.
        """
        if eps < 0.0 or eps > 1.0:
            raise ValueError("error rate of channel needed to be in the range [0.0, 1.0]!")

        self.eps = eps

    def process(self, input: np.ndarray) -> np.ndarray:
        # generate random bits
        err = (np.random.rand(*input.shape) < self.eps).astype(np.int64)

        output = (input + err) % 2

        return output

    def llr(self, bits: np.ndarray) -> np.ndarray:
        if not set(bits.ravel()).issubset({0, 1}):
            raise ValueError("the outputs received from this channel need to be 0/1 bits!")

        llr_vec = bits * np.log(1 - self.eps) + (1 - bits) * np.log(self.eps)
        return llr_vec


class Gaussian(Channel):
    def __init__(self, sigma: float):
        """this class represents a binary Gaussian channel whose input is anti-podal {+1, -1} and whose output is real-valued.

        Args:
            sigma (float): std of the Gaussian noise of the channel.
        """
        if sigma < 0.0:
            raise ValueError("variance of Gaussian channel needs to be positive!")

        self.sigma = sigma

    def process(self, input: np.ndarray) -> np.ndarray:
        # generate Gaussian noise
        # also apply anti-podal encoding for the input bits
        anti_podal = 2 * input - 1

        noise = self.sigma * np.random.randn(*input.size)

        output = anti_podal + noise

        return output

    def llr(self, output: np.ndarray) -> np.ndarray:
        """this function computes the log-likelihood ratio for binary Gaussian channel.
        NOTE: in computing LLR we assume the following bit to anti-podal mapping: 0 -> -1, 1 -> +1

        Args:
            output (np.ndarray): an array contating the output values received from the channel.

        Returns:
            np.ndarray: an array containing the LLR values.
        """
        llr_vec = 2 * output / self.sigma ** 2

        return llr_vec
