import torch
import torch.autograd as autograd
from torch.autograd import Variable
from utils import *
import torch.nn.functional as F

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


class BiLSTM_ATTN_ADVR_CRF(nn.Module):

    def __init__(self, args):
        super(BiLSTM_ATTN_ADVR_CRF, self).__init__()
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

        self.dropout_embeds = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True)
        print('self.lstm = nn.LSTM(self.embedding_size, self.hidden_size): ', self.embedding_size, self.hidden_size)
        init_lstm(self.lstm)
        self.dropout_lstm = nn.Dropout(0.5)
        self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        self.h2_h1 = nn.Linear(self.hidden_size*4, self.hidden_size*2)
        self.tanh = nn.Tanh()
        self.hidden4H2tag = nn.Linear(self.hidden_size*4, self.tagset_size)
        self.hidden2tag = nn.Linear(self.hidden_size*2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden4H2tag)
        init_linear(self.hidden2tag)
        init_linear(self.hw_gate)
        init_linear(self.hw_trans)

        if self.crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[self.tag_to_id[START_TAG], :] = -10000
            self.transitions.data[:, self.tag_to_id[STOP_TAG]] = -10000

        # Choose attention model
        if args.attn_model != 'none':
            self.attn = Attn(args.attn_model, args.hidden_size*2)
            self.dropout_attn = nn.Dropout(0.5)

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

    def _get_lstm_features(self, embeds):

        lstm_out, lstm_hidden = self.lstm(embeds) # S X B X 2H , 2 X 1 X H = SxBxE
        # print ('lstm_out.size():', lstm_out.size()) # 18L, 1L, 400L SxBx2H
        # print ('lstm_out[-1]: ', lstm_out[-1]) 
        #print ('lstm_hidden.size():', lstm_hidden.size())
        # print ('lstm_hidden.shape:', lstm_hidden) #  S times 2x1x200  

        #lstm_out = lstm_out.view(len(embeds), 1, self.hidden_size*2)
        lstm_out = self.dropout_lstm(lstm_out)
        seq_len = len(lstm_out)
        if self.args.attn_model != 'none':
            # print ('lstm_out:',lstm_out.size()) # (18L, 1L, 400L)  SxBx2H
            # lstm_out = lstm_out.squeeze(1) # S=1 x B x N -> B x N
            context = Variable(torch.zeros(seq_len, self.hidden_size*2)) # S x 2H
            if self.use_gpu: context = context.cuda()
            for i in range(seq_len): # L
                attn_weights = self.attn(lstm_out[i], lstm_out) # 1 X 1 X S = B x 2H, S x B x 2H
                # print ('attn_weights:', attn_weights.size())
                context[i] = attn_weights.bmm(lstm_out.transpose(0, 1)) # 2H  = B X 1 X S , B x S x 2H
                # print ('context[i].size():', context[i].size()) #(400L,) 2H
            
            # Final output layer (next word prediction) using the RNN hidden state and context vector
            
            context = context.unsqueeze(1) # S x B x 2H
            # print ('lstm_out, context:',lstm_out.size(), context.size()) #(18L, 1L, 400L), (18L, 1L, 400L)
            lstm_out = torch.cat((lstm_out, context), 2) # SxBx4H = SxBx2H, SxBx2H
            # print ('combine (lstm_out, context):',lstm_out.size()) #SxBx4H
            # lstm_out = self.h2_h1(lstm_out) # SxBx2H = SxBx4H
            # lstm_out = self.dropout_attn(lstm_out)
        lstm_out = lstm_out.squeeze(1) # S x B x 2H -> S x 2H
        return lstm_out

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

    def get_embeds(self, sentence, tags, chars2, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):
        # sentence, tags is a list of ints
        embeds = self._get_embeddings_for_sentence(sentence, chars2, caps, chars2_length, d,  gaz_ids, pos_ids, pre_ids, suf_ids)
        #print ('concatenated embeds size: ', embeds.size())
        embeds = embeds.unsqueeze(1)
        #print ('unsequeezed embeds size: ', embeds.size())
        self.embeds = self.dropout_embeds(embeds)
        # self.embeds.retain_grad()
        return self.embeds

    def neg_log_likelihood(self, sentence, tags, chars2, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):
        # sentence, tags is a list of ints
        embeds = self._get_embeddings_for_sentence(sentence, chars2, caps, chars2_length, d,  gaz_ids, pos_ids, pre_ids, suf_ids)
        #print ('concatenated embeds size: ', embeds.size())
        embeds = embeds.unsqueeze(1)
        #print ('unsequeezed embeds size: ', embeds.size())
        self.embeds = self.dropout_embeds(embeds)
        self.embeds.retain_grad()
        return self.neg_log_likelihood_embeds(self.embeds, tags)

    def neg_log_likelihood_embeds(self, embeds, tags):
        # features is a 2D tensor, len(sentence) * self.tagset_size
        lstm_feats = self._get_lstm_features(embeds) # (18L, 1L, 200L) SxBx2H = SxBxE

        if self.args.attn_model != 'none':
            feats = self.hidden4H2tag(lstm_feats) #(18L, 1L, 13L) SxBxT = SxBx4H  
        else:
            feats = self.hidden2tag(lstm_feats) #(18L, 1L, 13L) SxBxT = SxBx2H  
        # print ('pred tags or scores: ', feats.size()) 
        if self.crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores


    def forward(self, sentence, chars, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):
        embeds = self._get_embeddings_for_sentence(sentence, chars, caps, chars2_length, d,  gaz_ids, pos_ids, pre_ids, suf_ids)
        #print ('concatenated embeds size: ', embeds.size())
        embeds = embeds.unsqueeze(1)
        #print ('unsequeezed embeds size: ', embeds.size())
        embeds = self.dropout_embeds(embeds) #SxBxE
        lstm_feats = self._get_lstm_features(embeds) #(18L, 1L, 200L) SxBx2H = SxBxE
        if self.args.attn_model != 'none':
            feats = self.hidden4H2tag(lstm_feats) #(18L, 1L, 13L) SxBxT = SxBx2H  
        else:
            feats = self.hidden2tag(lstm_feats) #(18L, 1L, 13L) SxBxT = SxBx2H  
        # viterbi to get tag_seq
        if self.crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq


class CustomAttn(nn.Module):
    def __init__(self, method, hidden_size):#, max_length=MAX_LENGTH):
        super(CustomAttn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        # self.embedding_size = embedding_size
        self.use_gpu = True
        
        if self.method == 'tanh':
            self.attn = nn.Tanh()
            self.attn2 = nn.Linear(hidden_size*4, 1)

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size*2, hidden_size)
            self.attn2 = nn.Linear(hidden_size, 1)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, encoder_outputs):
        seq_len = len(encoder_outputs)

        print('seq_len: ', seq_len)
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len, seq_len)) # B x 1 x S
        print ('attn_energies.size():', attn_energies.size())
        hidden = Variable(torch.zeros(seq_len, 1, self.hidden_size))
        if self.use_gpu: 
            attn_energies = attn_energies.cuda()
            hidden = hidden.cuda()

        #hidden = self.attn(encoder_outputs) # B x 1 x S
        #attn_energies = self.attn2(hidden) # B x 1 x 1
        # Calculate energies for each encoder output
        for i in range(seq_len):
            for j in range(seq_len):
                print ('encoder_outputs[i].size(), encoder_outputs.size():' ,encoder_outputs[i].size(), encoder_outputs[j].size())
                score = self.score(encoder_outputs[i], encoder_outputs[j]) # [[batch, seq_len] = [batch, hidden_size*2], [batch, hidden_size*2]
                score = score.unsqueeze(1)
                attn_energies[i][j] = self.attn2(score)

        print ('attn energies before softmax: ', attn_energies.size()) #[seq_len, seq_len]
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=0)#.unsqueeze(0).unsqueeze(0)

    def forward_old(self, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        hidden = Variable(torch.zeros(seq_len, 1, self.hidden_size))
        if self.use_gpu: 
            attn_energies = attn_energies.cuda()
            hidden = hidden.cuda()

        hidden = self.attn(encoder_outputs) # B x 1 x S
        attn_energies = self.attn2(hidden) # B x 1 x 1
        # Calculate energies for each encoder output
        # for i in range(seq_len):
            # attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # print ('attn energies before softmax: ', attn_energies.size())
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=0)#.unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'tanh':
            print ('hidden.size(), encoder_output.size():', hidden.size(), encoder_output.size())
            print ('torch.cat((hidden, encoder_output))).size():', torch.cat((hidden, encoder_output)).size())
            energy = self.attn(torch.cat((hidden, encoder_output)))
            print ('energy.size(): ', energy.size())
            return energy

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            # energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.use_gpu = True
        
        if self.method == 'tanh':
            self.attn = nn.Tanh()
            # self.attn2 = nn.Linear(hidden_size*4, 1)

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if self.use_gpu: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i]) # 1 
        # S
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'tanh':
            # print ('hidden.size(), encoder_output.size():', hidden.size(), encoder_output.size()) #Bx2H, Bx2H
            # print ('torch.cat((hidden, encoder_output))).size():', torch.cat((hidden, encoder_output),1).size())
            # energy = self.attn(torch.cat((hidden, encoder_output), 1)) #Bx4H = Bx4H = Bx2H, Bx2H
            # print ('energy.size() b4 linear: ', energy.size())
            encoder_output = self.attn(encoder_output) # Bx2H
            energy = hidden.dot(encoder_output) #1 = Bx2H , Bx2H
            return energy

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output) # Bx2H
            energy = hidden.dot(energy) #1 = Bx2H , Bx2H
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            # print ("self.other, energy", self.other, energy)
            energy = self.other.dot(energy)
            return energy


class BiLSTM_SENTENCE(nn.Module):
    def __init__(self, args):
        super(BiLSTM_SENTENCE, self).__init__()
        self.args = args
        self.use_gpu = self.args.use_gpu
        self.embedding_size = self.args.pretrained_embedding_size
        self.pretrained_embedding_size = self.args.pretrained_embedding_size
        self.hidden_size = self.args.hidden_size
        self.vocab_size = self.args.vocab_size
        self.tag_to_id = self.args.tag_to_id
        self.others_embedding_size = self.args.others_embedding_size
        self.crf = self.args.crf
        self.tagset_size = 1
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

        self.dropout_embeds = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=True)
        print('self.lstm = nn.LSTM(self.embedding_size, self.hidden_size): ', self.embedding_size, self.hidden_size)
        init_lstm(self.lstm)
        self.h2_h1 = nn.Linear(self.hidden_size*4, self.hidden_size*2)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(self.hidden_size*2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden2tag)

        # Choose attention model
        if args.attn_model != 'none':
            self.attn = Attn(args.attn_model, args.hidden_size*2)


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

    def _get_lstm_features(self, embeds):

        lstm_out, lstm_hidden = self.lstm(embeds) # S X B X 2H , 2 X 1 X H = SxBxE
        # print ('lstm_out.size():', lstm_out.size()) # 18L, 1L, 400L SxBx2H
        # print ('lstm_out[-1]: ', lstm_out[-1]) 
        #print ('lstm_hidden.size():', lstm_hidden.size())
        # print ('lstm_hidden.shape:', lstm_hidden) #  S times 2x1x200  

        #lstm_out = lstm_out.view(len(embeds), 1, self.hidden_size*2)
        #lstm_out = self.dropout_embeds(lstm_out)
        seq_len = len(lstm_out)
        if self.args.attn_model != 'none':
            # print ('lstm_out:',lstm_out.size()) # (18L, 1L, 400L)  SxBx2H
            # lstm_out = lstm_out.squeeze(1) # S=1 x B x N -> B x N
            context = Variable(torch.zeros(seq_len, self.hidden_size*2)) # S x 2H
            if self.use_gpu: context = context.cuda()
            for i in range(seq_len): # L
                attn_weights = self.attn(lstm_out[i], lstm_out) # 1 X 1 X S = B x 2H, S x B x 2H
                # print ('attn_weights:', attn_weights.size())
                context[i] = attn_weights.bmm(lstm_out.transpose(0, 1)) # 2H  = B X 1 X S , B x S x 2H
                # print ('context[i].size():', context[i].size()) (400L,) 2H
            
            # Final output layer (next word prediction) using the RNN hidden state and context vector
            
            context = context.unsqueeze(1) # S x B x 2H
            # print ('lstm_out, context:',lstm_out.size(), context.size()) #(18L, 1L, 400L), (18L, 1L, 400L)
            lstm_out = torch.cat((lstm_out, context), 2) # SxBx4H = SxBx2H, SxBx2H
            # print ('combine (lstm_out, context):',lstm_out.size()) #SxBx4H
            lstm_out = self.h2_h1(lstm_out) # SxBx2H = SxBx4H

        lstm_out = lstm_out.squeeze(1) # S x B x 2H -> S x 2H
        return lstm_out

    def get_embeds(self, sentence, tags, chars2, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):
        # sentence, tags is a list of ints
        embeds = self._get_embeddings_for_sentence(sentence, chars2, caps, chars2_length, d,  gaz_ids, pos_ids, pre_ids, suf_ids)
        #print ('concatenated embeds size: ', embeds.size())
        embeds = embeds.unsqueeze(1)
        #print ('unsequeezed embeds size: ', embeds.size())
        self.embeds = self.dropout_embeds(embeds)
        return self.embeds

    def get_pred_output(self, embeds):
        # features is a 2D tensor, len(sentence) * self.tagset_size
        lstm_feats = self._get_lstm_features(embeds) # (18L, 1L, 200L) SxBx2H = SxBxE
        pred_tag_prob = self.hidden2tag(lstm_feats[-1]) #(18L, 1L, 13L) SxBxT = 2H
        # print ('train_pred_tag_prob, train_pred_tag_prob.size(): ', pred_tag_prob, pred_tag_prob.size()) 
        # pred_tag_prob = F.log_softmax(pred_tag_prob)
        return F.sigmoid(pred_tag_prob)


    def forward(self, sentence, chars, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids):
        embeds = self._get_embeddings_for_sentence(sentence, chars, caps, chars2_length, d,  gaz_ids, pos_ids, pre_ids, suf_ids)
        #print ('concatenated embeds size: ', embeds.size())
        embeds = embeds.unsqueeze(1)
        #print ('unsequeezed embeds size: ', embeds.size())
        embeds = self.dropout_embeds(embeds) #SxBxE
        lstm_feats = self._get_lstm_features(embeds) #(18L, 1L, 200L) SxBx2H = SxBxE
        pred_tag_prob = self.hidden2tag(lstm_feats[-1]) #BxT = Bx2H #gives some value including -ve values
        pred_tag = F.sigmoid(pred_tag_prob).cpu().data[0] # value in between 0 and 1
        pred_tag = 1 if pred_tag > 0.5 else 0 #value either 0 or 1
        # print ('test_pred_tag_prob, test_pred_tag:', pred_tag_prob, pred_tag)
        return pred_tag_prob, pred_tag

