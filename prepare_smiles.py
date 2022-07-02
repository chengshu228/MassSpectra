import os
import torch
from d2l import torch as d2l
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles

from utils_function import split
from con_function import read_smiles, tokenize, load_corpus_smiles, new_lines, truncate_pad
from utils_plot import plot_token_frequency, plot_bigram

# 读取所有的smiles

def numberFile(file_location):
    """文件夹内的文件数量"""
    number = 0
    for _ in os.listdir(file_location):
        number += 1
    return number

print(os.getcwd())
print(numberFile(os.path.join(os.getcwd(), 'SMILES')))
print(numberFile(os.path.join(os.getcwd(), 'new_mass_spectra')))


current_path = os.getcwd()
with open(os.path.join(current_path, 'data/smiles.txt'), 'r', encoding='utf-8') as f_smiles:
    lines = f_smiles.readlines()
# print(lines)
print(len(lines))
smiles = [smi.replace('\n', '') for smi in lines]
# return np.array(smiles)

# smiles = read_smiles()
# print(f"smiles:\n\t{smiles}\ntype(smiles): {type(smiles)}, \nsmiles[:1]: \n\t{smiles[:1]}")
print(len(smiles))

len_smiles = list(map(len, smiles))
sub = np.array(len_smiles)<=200
print(f"number of len_smiles<=200: {np.sum(sub)}")
print(f"percent of total: {np.sum(sub)/len(smiles)*100}")
print(f"set(len_smiles): {set(len_smiles)}")
print(f"max(len_smiles): {max(len_smiles)}")

# 绘制smiles的长度分布情况
plot_token_frequency(seq_lengths=len_smiles, name='smiles')
 
# corpus: 索引号     vocab：所有元素
tgt_corpus, tgt_vocab = load_corpus_smiles(smiles, max_tokens=-1)
print(tgt_vocab['<unk>'], tgt_vocab['<pad>'], tgt_vocab['<bos>'], tgt_vocab['<eos>'])

print(f"len(tgt_corpus): {len(tgt_corpus)}, \tlen(tgt_vocab): {len(tgt_vocab)}")
print(f"tgt_vocab.token_freqs[:10]: \n\t{tgt_vocab.token_freqs[:10]}")

# 绘制n_bigram的长度分布情况
plot_bigram(tgt_corpus, tgt_vocab, name='smiles')

smiles_tokens = new_lines(smiles)

for i in range(2):
    print(f"\t{truncate_pad(tgt_vocab[smiles_tokens[i]], 200, tgt_vocab['<pad>'])}")

tensor_smiles = torch.tensor([truncate_pad(tgt_vocab[token], 200, tgt_vocab['<pad>']) for token in smiles_tokens])
print(tensor_smiles[:2])