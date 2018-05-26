import torch
import torch.autograd as autograd

import numpy as np
#####################################################################
# Helper functions to make the code more readable.


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs).cuda()
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def logsumexp2d(a, axis=0):
    """Compute the log of the sum of exponentials of input elements.
    like: scipy.misc.logsumexp

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int
        Axis over which the sum is taken.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way.
    """
    a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max


def create_variables(args, X_word_ids_batch, y_sentences_batch, X_gaz_ids_batch, X_pos_ids_batch, X_caps_ids_batch, X_suf_ids_batch, X_pre_ids_batch, X_char_ids_batch, X_charngram_ids_batch):
    X_word_ids_batch = torch.LongTensor(X_word_ids_batch).squeeze(0)
    y_sentences_batch = torch.LongTensor(y_sentences_batch).squeeze(0)
    if args.gazetter:
        X_gaz_ids_batch = torch.LongTensor(X_gaz_ids_batch).squeeze(0)
    if args.pos:
        X_pos_ids_batch = torch.LongTensor(X_pos_ids_batch).squeeze(0)
    if args.chunk:
        X_chunk_ids_batch = torch.LongTensor(X_chunk_ids_batch).squeeze(0)
    if args.caps:
        X_caps_ids_batch = torch.LongTensor(X_caps_ids_batch).squeeze(0)
    if args.pre:
        X_pre_ids_batch = torch.LongTensor(X_pre_ids_batch).squeeze(0)
    if args.suf:
        X_suf_ids_batch = torch.LongTensor(X_suf_ids_batch).squeeze(0)
    if args.cuda:
        X_word_ids_batch = X_word_ids_batch.cuda() 
        y_sentences_batch = y_sentences_batch.cuda()
        if args.gazetter:
            X_gaz_ids_batch = X_gaz_ids_batch.cuda()
        if args.pos:
            X_pos_ids_batch = X_pos_ids_batch.cuda()
        if args.chunk:
            X_chunk_ids_batch = X_chunk_ids_batch.cuda()
        if args.caps:
            X_caps_ids_batch = X_caps_ids_batch.cuda()
        if args.pre:
            X_pre_ids_batch = X_pre_ids_batch.cuda()
        if args.suf:
            X_suf_ids_batch = X_suf_ids_batch.cuda()
    X_word_ids_batch = autograd.Variable(X_word_ids_batch)
    # y_sentences_batch = autograd.Variable(y_sentences_batch, requires_grad = False)
    if args.gazetter:
        X_gaz_ids_batch = autograd.Variable(X_gaz_ids_batch)
    if args.pos:
        X_pos_ids_batch = autograd.Variable(X_pos_ids_batch)
    if args.chunk:
        X_chunk_ids_batch = autograd.Variable(X_chunk_ids_batch)
    if args.caps:
        X_caps_ids_batch = autograd.Variable(X_caps_ids_batch)
    if args.pre:
        X_pre_ids_batch = autograd.Variable(X_pre_ids_batch)
    if args.suf:
        X_suf_ids_batch = autograd.Variable(X_suf_ids_batch)
    return X_word_ids_batch, y_sentences_batch, X_gaz_ids_batch, X_pos_ids_batch, X_caps_ids_batch, X_suf_ids_batch, X_pre_ids_batch, X_char_ids_batch, X_charngram_ids_batch
