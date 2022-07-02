import os
import math
import numpy as np
from d2l import torch as d2l
import torch
from rdkit import Chem
import collections
from urllib import error, request

from build_vocab import Vocab

def numberFile(file_location):
    '''number of files'''
    number = 0
    for _ in os.listdir(file_location):
        number += 1
    return number

def readFile(parent_path):
    '''the list of file's paths'''
    file_path_list = []
    file_list=os.listdir(parent_path)
    for file in file_list:
        file_path = os.path.join(parent_path, file)
        file_path_list.append(file_path)
        print(file_path)
    return file_path_list

def get_redirect_url(query, html_loc, filename):
    '''return the adress of URL'''
    url = f"https://hmdb.ca/unearth/q?query={query[0]}&searcher=metabolites&button="
    print('\turl: ', url)
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29"}
    try :
        req = request.Request(url, headers=headers) 
        res = request.urlopen(req) 
        # try:                                       
        #     buffer = res.read()
        # except http_client.IncompleteRead as e:
        #     buffer = e.partial
        # result_html=buffer.decode("utf-8")
        result_html = res.read().decode('utf-8')
        with open(f"{html_loc}/{filename}.html", 'w', encoding='utf-8') as f:
            f.write(result_html)
            f.close()
    except error.HTTPError as e:
        print('\tHTTPError:{}'.format(e.reason))
        print('\tHTTPError:{}'.format(e))    
    except error.URLError as e:
        print('\tURLError:{}'.format(e.reason))
        print('\tURLError:{}'.format(e))
    except Exception as e:
        print(e)
    return None

def train_test_split(data, test_size=0.1):
    '''train_test_split'''
    train_length = int(len(data)*(1-test_size))
    return data[:train_length,:], data[train_length:,:]

def read_smiles(smiles_file):
    '''read smiles'''
    with open(smiles_file, 'r', encoding='utf-8') as f_smiles:
        lines = f_smiles.readlines()
    smiles = [smi.replace('\n', '') for smi in lines]
    return np.array(smiles)

# def write_smiles(smiles, smiles_file):
#     """
#     Write a list of SMILES to a line-delimited file.
#     """
#     # write sampled SMILES
#     with open(smiles_file, 'w') as f:
#         for sm in smiles:
#             _ = f.write(sm + '\n')

def tokenize(lines, type_data='smiles', token='word'):
    '''split text as word or char'''
    if token == 'word':
        return [line for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print(f"Error: unkown token type - {token}. ")

def new_lines(lines, type_data='smiles'):
    if type_data == 'smiles':
        tokens = tokenize(lines, token='word')
        new_tokens = [split_smiles(split_smiles(tokens[line])).split() + ['<eos>'] for line in range(len(lines))]
    else:
        new_tokens = [lines[line] for line in range(len(lines))]
    return new_tokens

def load_corpus(lines, type_data='smiles', max_tokens=-1):
    '''get corpus and vocab'''
    tokens = new_lines(lines, type_data=type_data)
    if type_data == 'smiles':        
        vocab = Vocab(tokens, min_freq=0, reserved_tokens=['<pad>', '<eos>'])
        corpus = [vocab[token] for line in tokens for token in line]
    else:
        vocab = Vocab(tokens, min_freq=0)
        corpus = [token for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def truncate_pad(line, num_steps, padding_token):
    '''truncate or pad sequence'''
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token]*(num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):
    '''mini-batch'''
    lines = [vocab[line] for line in lines]
    lines = [[vocab['<bos>']] + line + [vocab['<eos>']] for line in lines]
    array = torch.tensor([truncate_pad(
        line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, source, target, src_num_steps, tgt_num_steps):
    '''return data iter and vocab'''
    source = new_lines(source, type_data='ms')
    target = new_lines(target, type_data='smiles')
    src_vocab = Vocab(source, min_freq=0    )
    tgt_vocab = Vocab(target, min_freq=0,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, src_num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, tgt_num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    '''predition of seq2seq'''
    net.eval() # set net as eval mode
    src_tokens = src_vocab[src_sentence]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])    
    enc_X = torch.unsqueeze(  # add axis
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor(  # add axis
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']: break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k): 
    '''calculate BLEU'''
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def split_smiles(sm):
    '''
    Split SMILES into words. 
    Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    '''
    arr = []
    i = 0
    while i < len(sm)-1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', \
                        'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F', '@']:
            arr.append(sm[i])
            i += 1
        elif sm[i]=='%':
            arr.append(sm[i:i+3])
            i += 3
        elif sm[i]=='@' and sm[i+1]=='@':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='b':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='X' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='L' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='s':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='T' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='Z' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='t' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='H' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='K' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='F' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm)-1:
        arr.append(sm[i])
    return ' '.join(arr) 
