from d2l import torch as d2l

data_loc = r'/data/cshu/mass_spectra/smiles_intensity'
device = d2l.try_all_gpus()

pad_ms = 100
pad_smiles = 200

src_num_steps = 100
tgt_num_steps = 100

embed_size = 32
num_hiddens = 32
num_layers = 2
dropout =  0.1
batch_size = 8
lr = 0.005
num_epochs = 1

split_ratio = 0.8

