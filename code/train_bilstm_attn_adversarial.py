# coding=utf-8
from __future__ import print_function
import optparse
import itertools
from collections import OrderedDict
import loader
import torch
import time
import cPickle
from torch.autograd import Variable
# important for ubuntu server as it does not have display
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sys
import visdom
from utils import *
from loader import *
from model import BiLSTM_CRF
from model_attn_adversarial import BiLSTM_ATTN_ADVR_CRF
from seq_crf_model import SEQ_CRF
import shutil

import torch.nn.functional as F
import datetime

t = time.time()
models_path = "models/"

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="dataset/eng.train",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="dataset/eng.testa",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="dataset/eng.testb",
    help="Test set location"
)
optparser.add_option(
    '--test_train', default='dataset/eng.train54019',
    help='test train'
)
optparser.add_option(
    '--score', default='evaluation/temp/score.txt',
    help='score file location'
)
optparser.add_option(
    "-s", "--tag_scheme", default="iob",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_embedding_size", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_size", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--pretrained_embedding_size", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--lstm_output_size", default="200",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="/home/raghavendra/BackUP/tools/GloVe-master/glove.6B.100d.txt",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="1",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_size", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--loss', default='loss.txt',
    help='loss file location'
)
optparser.add_option(
    '--name', default='test',
    help='model name'
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
optparser.add_option(
    '--test_crfFeatures_dir', default='CNN',
    help='test_crfFeatures_dir'
)
optparser.add_option(
    '--output_crfFeatures_dir', default='CNN',
    help='output_crfFeatures_dir'
)
optparser.add_option(
    '--clef_eval_script', default='',
    help='clef eval script'
)
optparser.add_option(
    '--clef_eval_script_arg', default='',
    help='clef eval script_arg'
)

optparser.add_option('--posfile', type=str, default='/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/word2VecFiles/posTags.txt', help='pos tags')
optparser.add_option('--chunkfile', type=str, default='/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/word2VecFiles/chunkTags.txt', help='chunk tags')
# optparser.add_option('--capfil2', type=str, default='False', help='capitalized tags')
optparser.add_option('--gazetterfile', type=str, default='/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/train/train.gazetter', help='is in gazetter')
optparser.add_option('--suffile', type=str, default='/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/word2VecFiles/suffixes_medical.txt', help='suffix')
optparser.add_option('--prefile', type=str, default='/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/finaldata/word2VecFiles/prefixes_medical.txt', help='prefix')
optparser.add_option('--caps', action='store_true', help='caps')
optparser.add_option('--pos', action='store_true', help='pos tags')
optparser.add_option('--chunk',action='store_true', help='chunk tags')
optparser.add_option('--gazetter', action='store_true', help='is in gazetter')
optparser.add_option('--suf', action='store_true', help='suffix')
optparser.add_option('--pre', action='store_true', help='prefix')
optparser.add_option('--others_embedding_size', type=int, default=5,
                            help='Size of the other embeddings 5')
optparser.add_option('--include_chars', action='store_true', help='include chars')
optparser.add_option('--lr', type=float, default=0.01, help='learning rate')
optparser.add_option('--lr_method', type=str, default='sgd', help='optimizer')
optparser.add_option('--nb_epoch', type=int, default=1000, help='number of epochs')
optparser.add_option('--min_occur', type=int, default=3, help='min_occur of word to include in embeddings')

optparser.add_option('--attn_model', type=str, default='none', help='attn model type')
optparser.add_option("--perturb", default="0", type='int', help="perturb embeddings for adverserial training (1 to enable)")

args = optparser.parse_args()[0]
args.padding = False
parameters = OrderedDict()
parameters['tag_scheme'] = args.tag_scheme
parameters['lower'] = args.lower == 1
parameters['zeros'] = args.zeros == 1
parameters['char_embedding_size'] = args.char_embedding_size
parameters['char_lstm_size'] = args.char_lstm_size
parameters['char_bidirect'] = args.char_bidirect == 1
parameters['pretrained_embedding_size'] = args.pretrained_embedding_size
parameters['lstm_output_size'] = args.lstm_output_size
parameters['word_bidirect'] = args.word_bidirect == 1
parameters['pre_emb'] = args.pre_emb
parameters['all_emb'] = args.all_emb == 1
parameters['cap_size'] = args.cap_size
parameters['crf'] = args.crf == 1
parameters['dropout'] = args.dropout
parameters['reload'] = args.reload == 1
parameters['name'] = args.name
parameters['char_mode'] = args.char_mode

parameters['use_gpu'] = args.use_gpu == 1 and torch.cuda.is_available()
use_gpu = parameters['use_gpu']

mapping_file = 'models/mapping.pkl'

name = parameters['name']
model_name = models_path + name #get_name(parameters)
tmp_model = model_name + '.tmp'


assert os.path.isfile(args.train)
assert os.path.isfile(args.dev)
assert os.path.isfile(args.test)
assert parameters['char_embedding_size'] > 0 or parameters['pretrained_embedding_size'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['pretrained_embedding_size'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

args.train_crfFeaturesFile = args.train
from thyme_code_oct2016 import preprocessingscript_thyme
preprocessingscript_thyme.loadFeatures2Tokens(args)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

train_sentences = loader.load_sentences(args.train, lower, zeros)
dev_sentences = loader.load_sentences(args.dev, lower, zeros)
test_sentences = loader.load_sentences(args.test, lower, zeros)
test_train_sentences = loader.load_sentences(args.test_train, lower, zeros)

update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)
update_tag_scheme(test_train_sentences, tag_scheme)

dico_words_train = word_mapping(train_sentences+test_sentences, lower, args.min_occur)[0]

dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences + test_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences + test_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower, args.padding, args
)
# dev_data = prepare_dataset(
#     dev_sentences, word_to_id, char_to_id, tag_to_id, lower, args.padding, args
# )
# test_data = prepare_dataset(
#     test_sentences, word_to_id, char_to_id, tag_to_id, lower, args.padding, args
# )
# test_train_data = prepare_dataset(
#     test_train_sentences, word_to_id, char_to_id, tag_to_id, lower, args.padding, args
# )

print("%i / sentences in train / dev / test." % (
    len(train_data)))#, len(dev_data), len(test_data)))

pre_embeds = {}
for i, line in enumerate(codecs.open(args.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['pretrained_embedding_size'] + 1:
        pre_embeds[s[0]] = np.array([float(i) for i in s[1:]])

word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), args.pretrained_embedding_size))
for w in word_to_id:
    if w in pre_embeds:
        word_embeds[word_to_id[w]] = pre_embeds[w]
    elif w.lower() in pre_embeds:
        word_embeds[word_to_id[w]] = pre_embeds[w.lower()]



print('Loaded %i pretrained embeddings.' % len(pre_embeds))
print('Total word ids: %i' % len(word_to_id))
print('tag_to_id:', tag_to_id)
#tag_to_id = {'<START>': 10, '<STOP>': 9, 'TB-M': 8, 'I-M': 2, 'THB-M': 7, 'O': 0, 'DI-M': 5, 'DHI-M': 6, 'B-M': 1, 'DB-M': 4, 'DHB-M': 3, 'THI-M': 11, 'TI-M': 12}


with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    cPickle.dump(mappings, f)

args.embedding = True
args.hidden_size=parameters['lstm_output_size']
args.word_to_id = word_to_id
args.tag_to_id = tag_to_id
args.use_gpu = use_gpu
args.char_to_id = char_to_id
args.word_embeds = word_embeds
args.pre_word_embeds = word_embeds
args.char_mode = parameters['char_mode']
args.vocab_size=len(word_to_id)
model = BiLSTM_ATTN_ADVR_CRF(args)
# vocab_size=len(word_to_id),
#                    tag_to_ix=tag_to_id,
#                    embedding_size=parameters['pretrained_embedding_size'],
#                    hidden_size=parameters['lstm_output_size'],
#                    use_gpu=use_gpu,
#                    char_to_ix=char_to_id,
#                    pre_word_embeds=word_embeds,
#                    crf=parameters['crf'],
#                    char_mode=parameters['char_mode'])
                   # n_cap=4,
                   # cap_embedding_size=10)
if parameters['reload']:
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()
learning_rate = args.lr
#args.lr_method = "adadelta"#"momentum"#"adadelta"#"adagrad"#"sgd"
if args.lr_method == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
elif args.lr_method == "adadelta":
    optimizer = torch.optim.Adadelta(model.parameters(), learning_rate, rho=0.95, eps=1e-06)
losses = []
loss = 0.0
best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0
all_F = [[0, 0, 0]]
plot_every = 500
eval_every = 5000
count = 0
vis = visdom.Visdom()
sys.stdout.flush()


def evaluating(model, datas, best_F, predf=None):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) , len(tag_to_id) ))#torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        caps = Variable(torch.LongTensor(data['caps']))
        gaz_ids = Variable(torch.LongTensor(data['gaz_ids']))
        pos_ids = Variable(torch.LongTensor(data['pos_ids']))
        pre_ids = Variable(torch.LongTensor(data['pre_ids']))
        suf_ids = Variable(torch.LongTensor(data['suf_ids']))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d, gaz_ids.cuda(), pos_ids.cuda(), pre_ids.cuda(), suf_ids.cuda())
        else:
            val, out = model(dwords, chars2_mask, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = '\t'.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            #print (line)
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    if predf is None:
        predf = eval_temp + '/pred.' + name
    scoref = eval_temp + '/score.' + name

    with open(predf, 'wb') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    for i, line in enumerate(eval_lines):
        #print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                #print('the best F is ', new_F)

    # print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
    #     "ID", "NE", "Total",
    #     *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    # ))
    # for i in range(confusion_matrix.size(0)):
    #     print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
    #         str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
    #         *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
    #           ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
    #     ))
    return best_F, new_F, save

def evaluating_test_clef(args, model):
    listFiles = os.listdir(args.test_crfFeatures_dir)
    if os.path.exists(args.output_crfFeatures_dir):
        shutil.rmtree(args.output_crfFeatures_dir)
    os.makedirs(args.output_crfFeatures_dir)
    for file in listFiles:
        temp_sentences = loader.load_sentences(os.path.join(args.test_crfFeatures_dir, file), lower, zeros)
        temp_data = prepare_dataset(
            temp_sentences, word_to_id, char_to_id, tag_to_id, lower, args.padding, args
        )
        evaluating(model, temp_data, -1, 
            os.path.join(args.output_crfFeatures_dir, file))
    #clef_eval_script = '/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/bmner/bmner_clef2013_code/NER-pytorch-master/commandline_scripts/eval_java_clef_seq.sh'
    os.system('sh %s %s' % (args.clef_eval_script, args.clef_eval_script_arg))

    
start_time = datetime.datetime.now()
epoch_start_time = datetime.datetime.now()
model.train(True)
for epoch in range(1, args.nb_epoch):
    print('epoch: ', epoch, "out of ", args.nb_epoch)
    print ("starting on epoch =", datetime.datetime.now() - epoch_start_time)
    epoch_start_time = datetime.datetime.now()
    #for i, index in enumerate(np.random.permutation(len(train_data))):
    perturb_embeds_list = []
    perturb_data = []
    for i in range(len(train_data)):
        # print ("starting on training data =", datetime.datetime.now() - start_time)
        # start_time = datetime.datetime.now()
        index = i

        tr = time.time()
        count += 1
        data = train_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']
        chars2 = data['chars']

        # if args.include_chars:
        ######### char lstm
        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        # ######## char cnn
        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))
        # else: 
        #     chars2_mask = Variable(torch.LongTensor(0))
        #     chars2_length= []

        targets = torch.LongTensor(tags)
        caps = Variable(torch.LongTensor(data['caps']))
        gaz_ids = Variable(torch.LongTensor(data['gaz_ids']))
        # print (data['pos_ids'])
        pos_ids = Variable(torch.LongTensor(data['pos_ids']))
        pre_ids = Variable(torch.LongTensor(data['pre_ids']))
        suf_ids = Variable(torch.LongTensor(data['suf_ids']))
        # time = 0.0
        # print ('targets1:', targets, targets.size())
        if use_gpu:
            sentence_in, targets, chars2_mask, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids = sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d, gaz_ids.cuda(), pos_ids.cuda(), pre_ids.cuda(), suf_ids.cuda()
        #    neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d, gaz_ids.cuda(), pos_ids.cuda(), pre_ids.cuda(), suf_ids.cuda())
        #else:
        # print ('sentence_in.requires_grad', sentence_in.requires_grad)
        # print ('data[str_words]:', data['str_words'])
        # print ('sentence_in:', sentence_in)
        # print ('targets2:', targets, targets.size())
        # print ("Time for loading tensors =", datetime.datetime.now() - start_time)
        # start_time = datetime.datetime.now()

        embeds = model.get_embeds(sentence_in, targets, chars2_mask, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids)
        # print ("Time to model.get_embeds() =", datetime.datetime.now() - start_time)
        # start_time = datetime.datetime.now()
        # print ('embeds.size(): ', embeds.size()) # 18L, 1L, 100L , SxBxE senlen x batch x embsize
        # print ('targets3:', targets, targets.size()) # 18L S senlen
        neg_log_likelihood = model.neg_log_likelihood_embeds(embeds, targets) 
        # print ("Time to model.neg_log_likelihood_embeds() =", datetime.datetime.now() - start_time)
        # start_time = datetime.datetime.now()
        # print ('neg_log_likelihood: ', neg_log_likelihood)
        # print ('neg_log_likelihood[0]: ', neg_log_likelihood.data[0])
        # neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d, gaz_ids, pos_ids, pre_ids, suf_ids)
        loss += neg_log_likelihood.data[0] / len(data['words'])
        neg_log_likelihood.backward()
        # print ("Time to neg_log_likelihood.backward() =", datetime.datetime.now() - start_time)
        # start_time = datetime.datetime.now()

        # print ('neg_log_likelihood:', neg_log_likelihood)
        # print ('type(model.embeds): ', type(model.embeds))
        # print ('model.embeds.size(): ', model.embeds.size())
        # print ('embeds.requires_grad: ', model.embeds.requires_grad)
        # print ('embeds.grad', model.embeds.grad)
        # print ('model.embeds.is_leaf:', model.embeds.is_leaf)
        # print ('embeds.grad.size()', model.embeds.grad.size())
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        # print ('2embeds.requires_grad: ', model.embeds.requires_grad)
        # print ('2embeds.grad', model.embeds.grad)
        # print ('2embeds.grad.size()', model.embeds.grad.size())
        optimizer.step()
        # print ("Time to optimizer.step() =", datetime.datetime.now() - start_time)
        # start_time = datetime.datetime.now()
        # print ('neg_log_likelihood.grad_variables: ', neg_log_likelihood.grad_variables)
        # print ('model.embeds.grad:', model.embeds.grad)
        # print ('model.word_embeds.weight.grad.index_select(1, sentence_in).size(): ', model.word_embeds.weight.grad.index_select(0, sentence_in).size())
        # print ('model.word_embeds.weight.grad.index_select(1, sentence_in): ', model.word_embeds.weight.grad.index_select(0, sentence_in))
        # print ("grads:", sentence_in.grad, model.word_embeds.weight.grad)
        # print ('embeds.requires_grad: ', model.embeds.requires_grad)
        # print ('embeds.grad', model.embeds.grad)
        # print ('embeds.grad.size()', model.embeds.grad.size())

        # print ('after model.zero_grad(): embeds.grad', model.embeds.grad)
        # print ('after model.zero_grad(): embeds.grad.size()', embeds.grad.size())
        if args.perturb:
            if args.tag_to_id[u'DB-M'] in tags or (args.tag_to_id[u'THB-M']) in tags:
                raw_perturb = model.embeds.grad
                # model.zero_grad()
                # model.embeds.grad.data.zero_()
                # print ('words:', data['str_words'])
                # print('perturb tags:', [id_to_tag[w] for w in tags])
                # print('perturb tags:', tags)
                adv_eps = Variable(torch.FloatTensor([1]))
                if use_gpu:
                    adv_eps = adv_eps.cuda()
                perturb = adv_eps * F.normalize(raw_perturb, p=2, dim=1)
                # print('sentence_in.size(), perturb.size():', sentence_in.size(), perturb.size())
                perturb_embeds = torch.add(model.embeds, 1, perturb).data
                perturb_embeds_list.append(perturb_embeds) 
                perturb_data.append(train_data[index])
        #make grad zero after backprop
        #model.zero_grad()
        #model.embeds.grad.data.zero_()

        # if count %plot_every == 0:
        #     loss /= plot_every
        #     print(count, ': ', loss)
        #     if losses == []:
        #         losses.append(loss)
        #     losses.append(loss)
        #     text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
        #     losswin = 'loss_' + name
        #     textwin = 'loss_text_' + name
        #     vis.line(np.array(losses), X=np.array([plot_every*i for i in range(len(losses))]),
        #          win=losswin, opts={'title': losswin, 'legend': ['loss']})
        #     vis.text(text, win=textwin, opts={'title': textwin})
        #     loss = 0.0

         #if count % (eval_every) == 0 and count > (eval_every * 20) or \
         #        count % (eval_every*4) == 0 and count < (eval_every * 20) or \
        if count % len(train_data) == 0 and epoch > 50:
            model.train(False)
            best_train_F, new_train_F, _ = 0,0,True#evaluating(model, test_train_data, best_train_F)
            best_dev_F, new_dev_F, save = 0,0,True#evaluating(model, dev_data, best_dev_F)
            # if save:
            #     torch.save(model, model_name)
            #best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)
            sys.stdout.flush()

            all_F.append([new_train_F])
            #all_F.append([new_train_F, new_dev_F, new_test_F])
            # all_F.append([new_dev_F, new_test_F])
            Fwin = 'F-score of {train, dev, test}_' + name
            # vis.line(np.array(all_F), win=Fwin,
            #      X=np.array([eval_every*i for i in range(len(all_F))]),
            #      opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})

            #EVALUATING TEST CLEF
            evaluating_test_clef(args, model)
            model.train(True)

        if count % len(train_data) == 0:
            adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))

    if args.perturb:
        print ('Running on perturb embeds... ')
        for j in range(len(perturb_embeds_list)):
            data = perturb_data[j]
            model.zero_grad()
            model.embeds.grad.data.zero_()
            tags = data['tags']
            targets = torch.LongTensor(tags)
            targets = targets.cuda()
            perturb_embeds = Variable(torch.cuda.FloatTensor(perturb_embeds_list[j]))
            # if use_gpu:
            #     perturb_embeds = perturb_embeds.cuda()
            neg_log_likelihood1 = model.neg_log_likelihood_embeds(perturb_embeds, targets)
            loss += neg_log_likelihood1.data[0] / len(data['words'])
            neg_log_likelihood1.backward()
            # print ("grads:", sentence_in.grad, model.word_embeds.weight.grad)
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()   
        if epoch > 50:
            model.train(False)
            evaluating_test_clef(args, model)
            model.train(True)

# print(time.time() - t)
# plotname = 'plot_seqcrf'
# plt.plot(losses)
# plt.savefig("%s" % plotname, bbox_inches='tight')
#plt.show()
