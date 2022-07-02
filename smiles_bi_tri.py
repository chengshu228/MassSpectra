import os
from d2l import torch as d2l

from utils_function import read_smiles, load_corpus, new_lines, truncate_pad
import config

data_loc = config.data_loc
smiles_file = os.path.join(data_loc, 'smiles.txt')
lines = read_smiles(smiles_file)

corpus, vocab = load_corpus(lines, type_data='smiles', max_tokens=-1)
print(len(corpus), len(vocab), (vocab.token_freqs[:10]))

freqs = [freq for token, freq in vocab.token_freqs]

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

src_vocab = d2l.Vocab(lines, min_freq=0, reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(len(src_vocab))

lines = new_lines(lines)
corpus, vocab = load_corpus(lines, type_data='smiles', max_tokens=-1)
print(lines[0], src_vocab[lines[0]])
print(truncate_pad(vocab[lines[0]], 10, src_vocab['<pad>']))
