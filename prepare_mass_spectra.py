
import json
import numpy as np
import torch

from utils_function import tokenize, new_lines, truncate_pad, load_corpus
from utils_plot import plot_token_frequency, plot_bigram
import config

data_loc = r'/data/cshu/mass_spectra/smiles_intensity'

with open(data_loc+'/mass_spectra.json') as f_ms:
    mass_spectras = json.load(f_ms)
print(f'the length of mass spectras: {len(mass_spectras)}')

new_mass_spectras = [list(map(float, mass_spectra.split())) 
    for mass_spectra in mass_spectras]
length_new_mass_spectras = [len(list(map(float, mass_spectra.split()))) 
    for mass_spectra in mass_spectras]
print(f"max length of mass spectras: {max(length_new_mass_spectras)}, " +\
    "\\min length of mass spectras: {min(length_new_mass_spectras)}")

count = 0
for i in range(len(new_mass_spectras)):
    if length_new_mass_spectras[i] == 0:
        count += 1
print(f'the length of new mass spectras: {len(count)}')

len_new_mass_spectras = list(map(len, new_mass_spectras))
sub = np.array(len_new_mass_spectras)<=100
print(f"number of len_new_mass_spectras<=100: {np.sum(sub)}")
print(f"percent of total: {np.sum(sub)/len(new_mass_spectras)*100}")
print(f"set(len_new_mass_spectras): {set(len_new_mass_spectras)}")

plot_token_frequency(seq_lengths=len_new_mass_spectras, name='new_mass_spectras')
 
new_mass_spectras_tokens = new_lines(new_mass_spectras, type='ms')
print(f"0 new mass spectras tokens: {new_mass_spectras_tokens[0]}")
src_corpus, src_vocab = load_corpus(new_mass_spectras, type='ms', max_tokens=-1)
print(f"len(src_corpus): {len(src_corpus)} \tlen(src_vocab): {len(src_vocab)}")
print(src_vocab['<unk>'], src_vocab['<pad>'], src_vocab['<bos>'], src_vocab['<eos>'])

for i in range(1):
    print(f"\t{truncate_pad(src_vocab[new_mass_spectras_tokens[i]], max(length_new_mass_spectras), src_vocab['<pad>'])}")

numpy_mass_spectra = [truncate_pad(src_vocab[token], max(length_new_mass_spectras), src_vocab['<pad>']) 
    for token in new_mass_spectras_tokens]
print(len(numpy_mass_spectra), len(new_mass_spectras_tokens))

new_mass_spectras_tokens = list(new_mass_spectras_tokens) 
for i in np.arange(len(new_mass_spectras_tokens)):
    for j in np.arange(len(new_mass_spectras_tokens[i])):
        numpy_mass_spectra[i][j] = new_mass_spectras_tokens[i][j]

numpy_mass_spectra = numpy_mass_spectra[:][:config.pad_ms]
tensor_mass_spectra = torch.tensor(numpy_mass_spectra)

print(tensor_mass_spectra[0])
