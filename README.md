# attcode
This project develops channel codes based on attention mechanism as is used in LLMs.

The main idea is in fact very simple:
- based on the output of the channel, we construct log-likelihood ratios (LLRs).
- these LLRs are decomposed into seveal tokens each of smaller number of LLRs.
- the attention mechanism is applied to find out the connection between the LLRs as imposed by the grammer of the code.
- this underlying connection between the symbols is then used to decode the tokens and recover the original bits.

It is interesting to note that some sort of attention mechanism appears in iterative decoding in sparse codes where
the local info is each check node is combined and is passed as messages to nodes in the next step. 
This is known as message passing and works quite well in practice.

In this project, we are trying to understand if we can build a counter-part of such itertaive method when the code is not
locally sparse.
