# -----------------------------------------------------------------------------------------------------------------------
# This file provides basic encoding module for designing channel codes based on attention mechanism used in LLM.
# 
#
#
# (C) Saeid Haghighatshoar
# email: haghighatshoar@gmail.com
#
#
# last update: 26.01.2024
# ------------------------------------------------------------------------------------------------------------------------
import numpy as np 


class Encoder:
  def __init__(self, k:int, n:int):
    """this module builds a simple encoder for encoding the information bits.

    Args:
        k (int): number of information bits.
        n (int): number of encoded bits.
    """
    self.k = k
    self.n = n
    
    # code rate
    self.rate = k/n

    # build a random encoding matrix
    # NOTE: we assume that the encoder is a linear one.
    # we also make sure that the encoder matrix has a nonzero determinants
    while True:
        G = (np.random.randn(n,k) < 0.0).astype(np.int64)
        
        GG = (G.T @ G) % 2
        
        det = np.linlag.det(GG) % 2
        if det != 0:
            break
    
    self.G = G
    

  def encode(self, bits: np.ndarray) -> np.ndarray:
    """this modules encodes the information bits.

    Args:
        bits (np.ndarray): input array of size `k x B` consisting of `B`batch each of size `k`.

    Returns:
        np.ndarray: `n x B` encoded bits.
    """
    if len(bits.shape) == 1:
      # there is only a single stream to encode
      bits = bits.reshape(-1,1)
    
    bits_k, num_chan = bits.shape

    if bits_k != self.k:
      raise ValueError("number of bits in the input does not match the block size in the encoder!")
    
    bits_out = (self.G @ bits) % 2

    # remove the number of batches if there is only one
    bits_out = bits_out.squeeze()

    return bits_out

  def __call__(self, *args, **kwargs)->np.ndarray:
    """this module is the same as encode module.

    Returns:
        np.ndarray: encoded bits.
    """
    return self.encode(*args, **kwargs)
  