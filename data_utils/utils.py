import torch
import numpy as np
import json
from typing import List

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device, non_blocking=True)

def make_sentence_mask(seqs, seqs_len):
    mask = torch.zeros(*seqs.size())
    if len(mask.size()) == 2:
        # seqs (batchsize, max_len)
        # seqs_len (batchsize,)
        B, L = mask.size()
        for i in range(B):
            mask[i,:seqs_len[i]] = 1.0
    elif len(mask.size()) == 3:
        # seqs (batchsize, 5, max_len)
        # seqs_len (bachsize, 5)
        B, L1, L2 = seqs.size()
        for i in range(B):
            for j in range(L1):
                mask[i,j,:seqs_len[i,j]] = 1.0
    return mask

def pad_seq(seqs, pad_token, max_q_length = 40):
    assert isinstance(seqs, list)
    assert isinstance(seqs[0], torch.Tensor)
    max_length = max([x.size(0) for x in seqs])
    if max_length <= max_q_length:
        max_length = max_q_length
    else:
        raise ValueError("input length {}, max_length {}".format(max_length, max_q_length))
    output = []
    for seq in seqs:
        padded_seq = torch.ones(max_length, dtype=seq.dtype) * pad_token
        padded_seq[: seq.size(0)] = seq
        output.append(padded_seq)

    return list2tensor(output)

def ndarraylist2tensor(arraylist: List):
    return torch.from_numpy(np.array(arraylist))

def list2tensor(tensorlist):
    assert isinstance(tensorlist, list), print(type(tensorlist))
    assert isinstance(tensorlist[0], torch.Tensor)
    out_tensor = [t.unsqueeze(0) for t in tensorlist]
    return torch.cat(out_tensor, dim = 0)

def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab


def decoded_sentence(sentence_token, vocab):
    decoded_sen = []
    vocab = invert_dict(vocab)
    for token in sentence_token:
        decoded_sen.append(vocab[token])
    return decoded_sen