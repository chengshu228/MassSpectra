
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

from build_vocab import Vocab


def plot_token_frequency(seq_lengths, name):
    plt.figure()
    plt.hist(seq_lengths)
    plt.grid()
    plt.xlabel('# tokens per sequence')
    plt.ylabel('frequency: n(x)')
    # plt.yscale('log')
    plt.savefig(os.path.join('figure', f'token_frequency_{name}.png'))
    plt.show()

def plot_bigram(corpus, vocab, name):
    freqs = [freq for token, freq in vocab.token_freqs]

    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    print(f"bigram_vocab: {bigram_vocab.token_freqs[:10]}")

    trigram_tokens = [triple for triple in zip(
        corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print(f"trigram_vocab: {trigram_vocab.token_freqs[:10]}")
    print(trigram_vocab.token_freqs[:10])

    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

    plt.figure()
    h1, = plt.plot(np.arange(len(freqs)), freqs)
    h2, = plt.plot(np.arange(len(bigram_freqs)), bigram_freqs)
    h3, = plt.plot(np.arange(len(trigram_freqs)), trigram_freqs)
    plt.grid()
    plt.xlabel('# token: x')
    plt.ylabel('frequency: n(x)')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(handles=[h1, h2, h3], labels=['unigram', 'bigram', 'trigram'], loc='upper right')
    plt.savefig(os.path.join('figure', f'token_frequency_diffrent_{name}.png'))
    plt.show()