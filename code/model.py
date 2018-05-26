import torch
import torch.autograd as autograd
from torch.autograd import Variable
from utils import *

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, args):
        super(BiLSTM_CRF, self).__init__()
        self.args = args
        self.use_gpu = self.args.use_gpu
        self.embedding_size = self.args.pretrained_embedding_size
        self.pretrained_embedding_size = self.args.pretrained_embedding_size
        self.hidden_size = self.args.hidden_size
        self.vocab_size = self.args.vocab_size
        self.tag_to_id = self.args.tag_to_id
        self.others_embedding_size = self.args.others_embedding_size
        self.crf = self.args.crf
        self.tagset_size = len(args.tag_to_id)
        self.out_channels = self.args.char_lstm_size
        self.char_mode = self.args.char_mode
        self.char_embedding_size=25

        #EMBEDDINGS
        if self.args.gazetter:
            self.gazetter_embeddings = nn.Embedding(len(args.gaz_embedding_matrix), self.args.others_embedding_size)
            init_embedding(self.gazetter_embeddings.weight)
            self.embedding_size += self.args.others_embedding_size
            print ('self.gazetter_embeddings:' , len(args.gaz_embedding_matrix), self.args.others_embedding_size)
            # self.gazetter_embeddings.weight.data.copy_(torch.from_numpy(args.gaz_embedding_matrix))
        if self.args.pos:
            self.pos_embeddings = nn.Embedding(len(args.pos_embedding_matrix), self.args.others_embedding_size)
            init_embedding(self.pos_embeddings.weight)
            self.embedding_size += self.args.others_embedding_size
            print('self.pos_embeddings:',len(args.pos_embedding_matrix), self.args.others_embedding_size)
            # self.pos_embeddings.weight.data.copy_(torch.from_numpy(args.pos_embedding_matrix))
        if self.args.chunk:
            self.chunk_embeddings = nn.Embedding(len(args.chunk_embedding_matrix), self.args.others_embedding_size)
            init_embedding(self.chunk_embeddings.weight)
            self.embedding_size += self.args.others_embedding_size
            print('self.chunk_embeddings:', len(args.chunk_embedding_matrix), self.args.others_embedding_size)
            # self.chunk_embeddings.weight.data.copy_(torch.from_numpy(args.chunk_embedding_matrix))
        if self.args.caps:
            self.caps_embeddings = nn.Embedding(len(args.caps_embedding_matrix), self.args.others_embedding_size)
            init_embedding(self.caps_embeddings.weight)
            self.embedding_size += self.args.others_embedding_size
            print('self.caps_embeddings:', len(args.caps_embedding_matrix), self.args.others_embedding_size)
            # self.caps_embeddings.weight.data.copy_(torch.from_numpy(args.caps_embedding_matrix))
        if self.args.pre:
            self.pre_embeddings = nn.Embedding(len(args.pre_embedding_matrix), self.args.others_embedding_size)
            init_embedding(self.pre_embeddings.weight)
            self.embedding_size += self.args.others_embedding_size
            print('self.pre_embeddings:', len(args.pre_embedding_matrix), self.args.others_embedding_size)
            # self.pre_embeddings.weight.data.copy_(torch.from_numpy(args.pre_embedding_matrix))
        if self.args.suf:
            self.suf_embeddings = nn.Embedding(len(args.suf_embedding_matrix), self.args.others_embedding_size)
            init_embedding(self.suf_embeddings.weight)
            self.embedding_size += self.args.others_embedding_size
            print('self.suf_embeddings:', len(args.suf_embedding_matrix), self.args.others_embedding_size)
            # self.suf_embeddings.weight.data.copy_(torch.from_numpy(args.suf_embedding_matrix))
        #END EMBEDDINGS

        if self.args.include_chars:
            self.char_lstm_size = self.args.char_lstm_size
            self.char_embeds = nn.Embedding(len(args.char_to_id), self.args.char_embedding_size)
            init_embedding(self.char_embeds.weight)
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(args.char_embedding_size, self.args.char_lstm_size, num_layers=1, bidirectional=True)
                init_lstm(self.char_lstm)
                self.embedding_size += self.args.char_lstm_size*2
                print('self.char_lstm: nn.LSTM(args.char_embedding_size, self.args.char_lstm_size): ', args.char_embedding_size, self.args.char_lstm_size)
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, self.args.char_embedding_size), padding=(2,0))
                self.embedding_size += self.out_channels
                print('self.char_cnn3 = ', 'in_channels=',1, 'out_channels=',self.out_channels, 'kernel_size=', 3, self.args.char_embedding_size)

        self.word_embeds = nn.Embedding(self.vocab_size, self.pretrained_embedding_size)
        print('self.word_embeds =',  self.vocab_size, self.pretrained_embedding_size)
        if self.args.pre_word_embeds is not None:
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(args.pre_word_embeds))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True)
        print('self.lstm = nn.LSTM(self.embedding_size, self.hidden_size): ', self.embedding_size, self.hidden_size)
        init_lstm(self.lstm)
        self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        self.h2_h1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(self.hidden_size*2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden2tag)
        init_linear(self.hw_gate)
        init_linear(self.hw_trans)

        if self.crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[self.tag_to_id[START_TAG], :] = -10000
            self.transitions.data[:, self.tag_to_id[STOP_TAG]] = -10000

    def _score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_id[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_id[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_id[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_id[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def _get_embeddings_for_sentence(self, sentence, chars2, caps, chars2_length, d,  gaz_ids, pos_ids, pre_ids, suf_ids):

        # t = self.hw_gate(chars_embeds)
        # g = nn.functional.sigmoid(t)
        # h = nn.functional.relu(self.hw_trans(chars_embeds))
        # chars_embeds = g * h + (1 - g) * chars_embeds

        embeds = self.word_embeds(sentence)
        if self.args.include_chars:
            if self.char_mode == 'LSTM':
                # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_size, bidirection=True, batchsize=chars2.size(0))
                chars_embeds = self.char_embeds(chars2).transpose(0, 1)
                packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
                lstm_out, _ = self.char_lstm(packed)
                outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
                outputs = outputs.transpose(0, 1)
                chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
                if self.use_gpu:
                    chars_embeds_temp = chars_embeds_temp.cuda()
                for i, index in enumerate(output_lengths):
                    chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_size], outputs[i, 0, self.char_lstm_size:]))
                chars_embeds = chars_embeds_temp.clone()
                for i in range(chars_embeds.size(0)):
                    chars_embeds[d[i]] = chars_embeds_temp[i]

            if self.char_mode == 'CNN':
                chars_embeds = self.char_embeds(chars2).unsqueeze(1)
                chars_cnn_out3 = self.char_cnn3(chars_embeds)
                chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                     kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)
        if self.args.gazetter:
            gazetter_embedding = self.gazetter_embeddings(gaz_ids)
        if self.args.pos:
            pos_embedding = self.pos_embeddings(pos_ids)
        if self.args.caps:
            caps_embedding = self.caps_embeddings(caps)
        if self.args.pre:
            pre_embedding = self.pre_embeddings(pre_ids)
        if self.args.suf:
            suf_embedding = self.suf_embeddings(suf_ids)

        if self.args.include_chars:
            if self.args.gazetter and self.args.pos and self.args.caps and self.args.suf and self.args.pre:
                embeds = torch.cat((embeds, chars_embeds, gazetter_embedding, pos_embedding, caps_embedding, suf_embedding, pre_embedding), 1)
            elif self.args.gazetter and self.args.pos and self.args.caps and self.args.suf:
                embeds = torch.cat((embeds, chars_embeds, gazetter_embedding, pos_embedding, caps_embedding, suf_embedding), 1)
            elif self.args.gazetter and self.args.pos and self.args.caps:
                embeds = torch.cat((embeds, chars_embeds, gazetter_embedding, pos_embedding, caps_embedding), 1)
            elif self.args.gazetter and self.args.pos:
                embeds = torch.cat((embeds, chars_embeds, gazetter_embedding, pos_embedding), 1)
            elif self.args.gazetter:
                embeds = torch.cat((embeds, chars_embeds, gazetter_embedding), 1)
        else:
            if self.args.gazetter and self.args.pos and self.args.caps and self.args.suf and self.args.pre:
                embeds = torch.cat((embeds, gazetter_embedding, pos_embedding, caps_embedding, suf_embedding, pre_embedding), 1)
            elif self.args.gazetter and self.args.pos and self.args.caps and self.args.suf:
                embeds = torch.cat((embeds, gazetter_embedding, pos_embedding, caps_embedding, suf_embedding), 1)
            elif self.args.gazetter and self.args.pos and self.args.caps:
                embeds = torch.cat((embeds, gazetter_embedding, pos_embedding, caps_embedding), 1)
            elif self.args.gazetter and self.args.pos:
                embeds = torch.cat((embeds, gazetter_embedding, pos_embedding), 1)
            elif self.args.gazetter:
                embeds = torch.cat((embeds, gazetter_embedding), 1)
        # if self.args.include_chars :

        return embeds

    def _get_lstm_features(self, sentence, chars2, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):

        embeds = self._get_embeddings_for_sentence(sentence, chars2, caps, chars2_length, d,  gaz_ids, pos_ids, pre_ids, suf_ids)
        #print ('concatenated embeds size: ', embeds.size())
        embeds = embeds.unsqueeze(1)
        #print ('unsequeezed embeds size: ', embeds.size())
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_size*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_id[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_id[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_id[START_TAG]] = 0
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_id[STOP_TAG]]
        terminal_var.data[self.tag_to_id[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_id[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_id[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, chars2, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self._get_lstm_features(sentence, chars2, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids)
        # print ('feats or scores: ', feats, feats.size())
        if self.crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores


    def forward(self, sentence, chars, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):
        feats = self._get_lstm_features(sentence, chars, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids)
        # viterbi to get tag_seq
        if self.crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq
