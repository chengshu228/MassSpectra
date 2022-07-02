

import torch
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

import os
from d2l import torch as d2l
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles
import json

import con_function
from con_function import tokenize, new_lines_ms, truncate_pad, load_corpus_ms
from utils_plot import plot_token_frequency, plot_bigram

with open('data/mass_spectra.json') as f_ms:
    mass_spectras = json.load(f_ms)
print(len(mass_spectras))


new_mass_spectras = [list(map(float, mass_spectra.split())) for mass_spectra in mass_spectras]
length_new_mass_spectras = [len(list(map(float, mass_spectra.split()))) for mass_spectra in mass_spectras]
# 有几个未取到数值？？？
# print(f"max_length_mass_spectras: {max(length_new_mass_spectras)}, min_length_mass_spectras: {min(length_new_mass_spectras)}")
print(f"max_length_mass_spectras: {max(length_new_mass_spectras)}")


len_new_mass_spectras = list(map(len, new_mass_spectras))
sub = np.array(len_new_mass_spectras)<=100
print(f"number of len_new_mass_spectras<=100: {np.sum(sub)}")
print(f"percent of total: {np.sum(sub)/len(new_mass_spectras)*100}")
print(f"set(len_new_mass_spectras): {set(len_new_mass_spectras)}")

# 绘制new_mass_spectras的长度分布情况
plot_token_frequency(seq_lengths=len_new_mass_spectras, name='new_mass_spectras')
 
new_mass_spectras_tokens = new_lines_ms(new_mass_spectras)
print(f"new_mass_spectras_tokens: {new_mass_spectras_tokens[0]}")
# corpus: 索引号     vocab：所有元素
src_corpus, src_vocab = load_corpus_ms(new_mass_spectras, max_tokens=-1)
print(f"len(src_corpus): {len(src_corpus)} \tlen(src_vocab): {len(src_vocab)}")


print(src_vocab['<unk>'], src_vocab['<pad>'], src_vocab['<bos>'], src_vocab['<eos>'])

# for i in range(1):
#     print(f"\t{truncate_pad(src_vocab[new_mass_spectras_tokens[i]], max(length_new_mass_spectras), src_vocab['<pad>'])}")

numpy_mass_spectra = [truncate_pad(src_vocab[token], max(length_new_mass_spectras), src_vocab['<pad>']) for token in new_mass_spectras_tokens]
print(len(numpy_mass_spectra), len(new_mass_spectras_tokens))
new_mass_spectras_tokens = list(new_mass_spectras_tokens) 
for i in np.arange(len(new_mass_spectras_tokens)):
    for j in np.arange(len(new_mass_spectras_tokens[i])):
        numpy_mass_spectra[i][j] = new_mass_spectras_tokens[i][j]
# print(numpy_mass_spectra.shape)

numpy_mass_spectra = numpy_mass_spectra[:][:100]
tensor_mass_spectra = torch.tensor(numpy_mass_spectra)

print(tensor_mass_spectra[0])


src_num_steps, tgt_num_steps = max(length_new_mass_spectras), 200


#############################################
################### smiles ##################
#############################################
# 读取所有的smiles
smiles = con_function.read_smiles()
# corpus: 索引号     vocab：所有元素
tgt_corpus, tgt_vocab = con_function.load_corpus_smiles(smiles, max_tokens=-1)
# smiles添加特殊符号
smiles_tokens = con_function.new_lines(smiles)
# 填充smiles
tensor_smiles = torch.tensor([con_function.truncate_pad(tgt_vocab[token], 200, tgt_vocab['<pad>']) for token in smiles_tokens])
# print(f'tensor_smiles[0]:  \n\t{tensor_smiles[0]}')

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
# src_num_steps, tgt_num_steps = 100, 200
# batch_size, lr, num_epochs, device = 8, 0.005, 2, d2l.try_gpu()
batch_size, lr, num_epochs  = 8, 0.005, 2

split_ratio = int(len(new_mass_spectras)*0.8)

train_iter, src_vocab, tgt_vocab = con_function.load_data_nmt(
    batch_size=batch_size, source=new_mass_spectras, target=smiles, 
    src_num_steps=src_num_steps, tgt_num_steps=tgt_num_steps)
# train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=16, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    # print(X)
    print('X:', X.type(torch.int32))
    print('Y:', Y.type(torch.int32))
    # print('X的有效长度:', X_valid_len)
    # print('Y:', Y.type(torch.int32))
    # print('Y的有效长度:', Y_valid_len)
    break



import seq_to_seq
from seq_to_seq import Seq2SeqEncoder, Seq2SeqDecoder, train_seq2seq
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)




train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab,  device)


src_test = new_mass_spectras[int(len(tensor_mass_spectra)*0.8):]
tgt_test = smiles[int(len(tensor_smiles)*0.8):]
# source = new_mass_spectras[split_ratio:]
# target=smiles[split_ratio:]
# source = new_lines_ms(source)
# target = con_function.new_lines(target)
# src_test, src_valid_len = con_function.build_array_nmt(source, src_vocab, src_num_steps)
# tgt_test, tgt_valid_len = con_function.build_array_nmt(target, tgt_vocab, tgt_num_steps)

test_iter, src_vocab, tgt_vocab = con_function.load_data_nmt(
    batch_size=batch_size, source=new_mass_spectras[split_ratio:], target=smiles[split_ratio:], 
    src_num_steps=src_num_steps, tgt_num_steps=tgt_num_steps)

for src, tgt in zip(src_test, tgt_test):
# for src, tgt in zip(src_test, tgt_test):
    # translation, attention_weight_seq = seq_to_seq.predict_seq2seq(net, src, src_vocab, tgt_vocab, src_num_steps, tgt_num_steps, device)
    translation, attention_weight_seq = con_function.predict_seq2seq(net, src, src_vocab, tgt_vocab, src_num_steps, tgt_num_steps, device)
    print(f'{src} => {translation}, bleu {seq_to_seq.bleu(translation, tgt, k=2):.3f}')

# for eng, fra in zip(engs, fras):
#     translation, attention_weight_seq = predict_seq2seq(
#         net, eng, src_vocab, tgt_vocab, num_steps, device)