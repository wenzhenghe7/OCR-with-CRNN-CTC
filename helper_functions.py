import json
import os
import shutil
import torch.nn as nn
import numpy as np

def load_json(json_path):
    '''
    load json file into python dictionary
    :param json_path(str): path to json file.
    :return: python dictionary converted from json file
    '''
    
    with open(json_path, 'r') as f:
        ret_dict = json.load(f)
    return ret_dict

def get_lexicon(alphabet='abcdefghilmnoprstuwy'):
    '''
    given a alphabet return a token_to_index mapping dictionary and a index_to_token mapping
    :return: token2index(dict) index2token(list) mapping dictionary
    '''
    
    index2token = [token for token in alphabet]
    token2index = {}
    for i, token in enumerate(index2token):
        token2index[token] = i
    return token2index, index2token


def pred_to_token_indicies(pred):
    '''
    convert logits for one sample to label index string
    :param pred(numpy array): the output logits for one sample(max_seq_len, lexicon_size)
    :return: the predicted string in list of indicies form (seq_len) ex if predicition is 'by' this function returns (1, 19)
    '''
    
    seq = []
    for i in range(pred.shape[0]):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    return out


def pred_to_token_indicies_batch(pred):
    '''
    decode logits output of ctc model to a list of "indicies setences"

    :param pred(numpy array): logits output from model(batch_size, w', n_vocab)
    :return: a list that contains sequence of token indicies for all samples in the batch.
    '''
    
    seqs = []

    for i in range(pred.shape[0]):
        out = pred_to_token_indicies(pred[i])
        seqs.append(out)

    return seqs


def calc_accy(logits, labels, label_lengths):
    '''
    calculate accy given logits and labels

    :param logits(torch tensor): output logits from transcription model (w', batch_size, n_vocab)
    :param labels(torch tensor): a tensor of shape (batch_size, max_seq_len) each entry on first dimension
                 corresponds to label for one sample
    :label_lengths(torch tensor): length of each label (batch_size)
    :return:accuracy(int) on the batch
    '''

    logits = logits.permute(1, 0, 2).cpu().data.numpy()
    seqs = pred_to_token_indicies_batch(logits)
    n_sample = labels.size(0)
    correct = 0
    for i, seq in enumerate(seqs):
        label = (labels[i].detach().cpu().numpy()[:label_lengths[i].item()] - 1).tolist()
        if seq == label:
            correct += 1
    return correct / n_sample