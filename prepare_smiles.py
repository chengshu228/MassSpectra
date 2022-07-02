import os
import torch
from d2l import torch as d2l
import numpy as np

from utils_function import new_lines, truncate_pad, numberFile, load_corpus
from utils_plot import plot_token_frequency, plot_bigram

import config

data_loc = config.data_loc
pad_smiles = config.pad_smiles

def output_smiles():    
    with open(os.path.join(data_loc, 'smiles.txt'), 'r', encoding='utf-8') as f_smiles:
        lines = f_smiles.readlines()
    smiles = [smi.replace('\n', '') for smi in lines]
    smiles_tokens = new_lines(smiles)
    len_smiles = list(map(len, smiles))
    sub = np.array(len_smiles)<=pad_smiles
    print(f"number of len_smiles<=pad_smiles: {np.sum(sub)}")
    print(f"percent of total: {np.sum(sub)/len(smiles)*100}")
    # print(f"set(len_smiles): {set(len_smiles)}")
    print(f"max(len_smiles): {max(len_smiles)}")

    plot_token_frequency(seq_lengths=len_smiles, name='smiles')
    
    tgt_corpus, tgt_vocab = load_corpus(smiles, type_data='smiles', max_tokens=-1)
    print(tgt_vocab['<unk>'], tgt_vocab['<pad>'], tgt_vocab['<bos>'], tgt_vocab['<eos>'])
    print(f"len(tgt_corpus): {len(tgt_corpus)}, \tlen(tgt_vocab): {len(tgt_vocab)}")
    print(f"tgt_vocab.token_freqs[:10]: \n\t{tgt_vocab.token_freqs[:10]}")

    plot_bigram(tgt_corpus, tgt_vocab, name='smiles')

    tensor_smiles = torch.tensor([truncate_pad(tgt_vocab[token], pad_smiles, tgt_vocab['<pad>']) for token in smiles_tokens])
    # print(tensor_smiles[:1])

    return tensor_smiles, smiles

# output_smiles()