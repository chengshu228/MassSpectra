import os
from d2l import torch as d2l
import matplotlib.pyplot as plt

from utils_function import split
from con_function import read_smiles, tokenize, load_corpus_smiles

lines = read_smiles()


corpus, vocab = load_corpus_smiles(lines)
print(len(corpus), len(vocab))

# import random
import torch
from d2l import torch as d2l 

print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
# plt.plot(freqs, xlabel='token:x', label='frequency: n(x)', xscale='log', yscale='log')


bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])

# d2l.

src_vocab = d2l.Vocab(lines, min_freq=0,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(len(src_vocab))

# print(type(lines), lines[:3])
# print(lines)
# print(corpus, vocab)

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    print(line)
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token]*(num_steps - len(line))

from con_function import new_lines, load_corpus_smiles
lines = new_lines(lines)
corpus, vocab = load_corpus_smiles(lines, max_tokens=-1)
print(lines[0], src_vocab[lines[0]])
print(truncate_pad(vocab[lines[0]], 10, src_vocab['<pad>']))

