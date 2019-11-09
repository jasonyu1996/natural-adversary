import argparse
import json
from models import MLPClassifier, Baseline_Embeddings
from models import Seq2Seq, MLP_D, MLP_G, MLP_I, MLP_I_AE, JSDistance, Seq2SeqCAE, Baseline_Embeddings, Baseline_LSTM
from utils import to_gpu, Corpus, batchify, SNLIDataset, collate_snli
import random
import pickle as pkl
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

#-------------------------------------------------------
import random

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from settings import EXPERIMENTS_DIR
from experiment import Experiment
from utils2 import to_device, load_weights, load_embeddings, create_embeddings_matrix
from vocab import Vocab
from train2 import create_model
from preprocess import load_dataset, create_dataset_reader
#------------------------------------------------------

parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--model_type', type=str, default="emb",
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=20,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--packed_rep', type=bool, default=True,
                    help='pad all sentences to fixed maxlen')
parser.add_argument('--train_mode', action='store_true',
                    help='set training mode')
parser.add_argument('--maxlen', type=int, default=10,
                    help='maximum sentence length')
parser.add_argument('--lr', type=float, default=1e-03,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1111,
                    help='seed')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--save_path', type=str, required=True,
                    help='used for saving the models')
parser.add_argument('--vocab_size', type=int, default=11004,
                    help='vocabulary size')
parser.add_argument('--classifier_path', type=str, required=True,
                    help='path to classifier files ./models')
parser.add_argument('--z_size', type=int, default=100,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--load_pretrained', type=str, required=True,
                    help='load a pre-trained encoder and decoder to train the inverter')
parser.add_argument('--surrogate-layers', type=str, default='100-50',
                    help='hidden layer sizes of the surrogate model')
parser.add_argument('--perturb-budget', type=float, default=3e-2,
                    help='the budget for making perturbations by the adversary')
parser.add_argument('--new_encoder', action='store_true',
		    help='try the new encoder!')
args = parser.parse_args()

#cur_dir = './output/%s/' % args.load_pretrained
cur_dir = args.load_pretrained

with open(cur_dir + '/vocab.json', 'r') as fin:
    corpus_vocab = json.load(fin)

corpus_train = SNLIDataset(train=True, vocab_size=args.vocab_size-4, path=args.data_path)
corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size-4, path=args.data_path)
trainloader= torch.utils.data.DataLoader(corpus_train, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=True)
train_iter = iter(trainloader)
testloader= torch.utils.data.DataLoader(corpus_test, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=False)
#GAME
test_iter = iter(testloader)
#
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

EPS = args.perturb_budget

autoencoder = torch.load(open(cur_dir + '/models/autoencoder_model.pt', 'rb'))
#gan_gen = torch.load(open(cur_dir + '/models/gan_gen_model.pt', 'rb'))
#gan_disc = torch.load(open(cur_dir + '/models/gan_disc_model.pt', 'rb'))
inverter = torch.load(open(cur_dir + '/models/inverter_model.pt', 'rb'))

classifier1 = Baseline_Embeddings(100, vocab_size=args.vocab_size)
#classifier1 = Baseline_LSTM(100,300,maxlen=args.maxlen, gpu=args.cuda)
classifier1.load_state_dict(torch.load(args.classifier_path + "/baseline/emb.pt"))
vocab_classifier1 = pkl.load(open(args.classifier_path + "/vocab.pkl", 'rb'))

mlp_classifier = MLPClassifier(args.z_size * 2, 3, layers=args.surrogate_layers)
if not args.train_mode:
    mlp_classifier.load_state_dict(torch.load(args.save_path+'/surrogate{0}.pt'.format(args.surrogate_layers)))

#----------------------------------------------
exp_id = 'train.3ir9y_e3'
exp = Experiment.load(EXPERIMENTS_DIR, exp_id)
preprocess_exp = Experiment.load(EXPERIMENTS_DIR, exp.config.preprocess_exp_id)
dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb = load_dataset(preprocess_exp)
dataset_reader = create_dataset_reader(preprocess_exp.config)
model = create_model(exp.config, vocab, style_vocab, dataset_train.max_len, W_emb)
load_weights(model, exp.experiment_dir.joinpath('best.th'))
#---------------------------------------------

print(classifier1)
print(autoencoder)
print(inverter)
print(mlp_classifier)

optimizer = optim.Adam(mlp_classifier.parameters(),
                           lr=args.lr,
                           betas=(args.beta1, 0.999))

from torch.autograd import Variable

def evaluate_model():
    classifier1.eval()

    test_iter = iter(trainloader)
    correct=0
    total=0
    for batch in test_iter:
        premise, hypothesis, target, _, _, _, _ = batch
        
        if args.cuda:
            premise=premise.cuda()
            hypothesis = hypothesis.cuda()
            target = target.cuda()
            
        prob_distrib = classifier1.forward((premise, hypothesis))
        predictions = np.argmax(prob_distrib.data.cpu().numpy(), 1)
        correct+=len(np.where(target.data.cpu().numpy()==predictions)[0])
        total+=premise.size(0)
    acc=correct/float(total)
    print("Accuracy:{0}".format(acc))
    return acc
        

if args.cuda:
    autoencoder.gpu = True
    autoencoder = autoencoder.cuda()
    autoencoder.start_symbols = autoencoder.start_symbols.cuda()
    #gan_gen = gan_gen.cuda()
    #gan_disc = gan_disc.cuda()
    classifier1 = classifier1.cuda()
    inverter = inverter.cuda()
    mlp_classifier = mlp_classifier.cuda()
else:
    autoencoder.gpu = False
    autoencoder = autoencoder.cpu()
    autoencoder.start_symbols = autoencoder.start_symbols.cpu()
    #gan_gen = gan_gen.cpu()
    #gan_disc = gan_disc.cpu()
    classifier1 = classifier1.cpu()
    inverter = inverter.cpu()
    mlp_classifier = mlp_classifier.cpu()

def train_process(premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length):
    #mx = target.max().item()
    #assert(mx >= 0 and mx < 3)
    #for s, s_w in zip(premise, premise_words):
    #    for i, w in zip(s, s_w):
    #        assert(corpus_vocab.get(w, 3) == i)
    #print(hypothesis_words, flush=True)
    autoencoder.eval()
    inverter.eval()
    classifier1.eval()
    mlp_classifier.train()

    #print(premise.max().item(), flush=True)
    #print(hypothesis.max().item(), flush=True)

    premise_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in premise_words]).cuda()
    hypothesis_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in hypothesis_words]).cuda()

    c_prem = autoencoder.encode(premise_idx, premise_length, noise=False)
    z_prem = inverter(c_prem).detach()

    c_hypo = autoencoder.encode(hypothesis_idx, hypothesis_length, noise=False)
    z_hypo = inverter(c_hypo).detach()

    # z_comb = nn.cat((z_prem, z_hypo), 0).detach()

    output = mlp_classifier(z_prem, z_hypo)
    gold = classifier1((premise, hypothesis)).detach()

    #print(output.shape, flush=True)
    #print(gold.shape, flush=True)

    acc = (torch.argmax(gold, 1) == target).to(torch.float32).mean().item()
    acc_surrogate = (torch.argmax(output, 1) == target).to(torch.float32).mean().item()


    loss = -torch.mean(torch.sum(output * F.softmax(gold, dim=1), 1), 0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc, acc_surrogate


def perturb(criterion, premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length):
    autoencoder.eval()
    inverter.eval()
    classifier1.eval()
    mlp_classifier.eval()

    premise_words = [premise_words]
    hypothesis_words = [hypothesis_words]
    premise_length = [premise_length]
    hypothesis_length = [hypothesis_length]


    premise_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in premise_words]).cuda()
    hypothesis_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in hypothesis_words]).cuda()

    c_prem = autoencoder.encode(premise_idx, premise_length, noise=False)
    z_prem = inverter(c_prem).detach()
    print('c_prem = ' + str(c_prem))
    print('z_prem = ' + str(z_prem))
    c_hypo = autoencoder.encode(hypothesis_idx, hypothesis_length, noise=False).detach()
    c_hypo.requires_grad = True
    z_hypo = inverter(c_hypo)


    premise = premise.unsqueeze(0)
    hypothesis = hypothesis.unsqueeze(0)
    target = target.unsqueeze(0)
    
    output = mlp_classifier(z_prem, z_hypo)

    loss = criterion(output, target)
    mlp_classifier.zero_grad()
    inverter.zero_grad()
    loss.backward()

    direction = torch.sign(c_hypo.grad)
    nc_hypo = c_hypo + EPS * direction
    nhypo_idx = autoencoder.generate(nc_hypo, 10, False)

    return nhypo_idx.squeeze(0).cpu().numpy()


def classifier_pred(pw, hw):
    classifier1.eval()

    premise_idx = torch.tensor([vocab_classifier1.get(w, 3) for w in pw]).cuda().unsqueeze(0)
    hypothesis_idx = torch.tensor([vocab_classifier1.get(w, 3) for w in hw]).cuda().unsqueeze(0)

    return F.softmax(classifier1((premise_idx, hypothesis_idx)), 1).squeeze(0).cpu().detach().numpy()
    
def maximum(arr):
    a = -1
    t = 0
    for i in range(len(arr)):
        if arr[i] > a:
            a = arr[i]
            t = i
    return t

def create_inputs(instances):
    if not isinstance(instances, list):
        instances = [instances,]
        
    if not isinstance(instances[0], dict):
        sentences = [
            dataset_reader.preprocess_sentence(dataset_reader.spacy( dataset_reader.clean_sentence(sent)))
            for sent in instances
        ]
        
        style = list(style_vocab.token2id.keys())[0]
        instances = [
            {
                'sentence': sent,
                'style': style,
            }
            for sent in sentences
        ]
        
        for inst in instances:
            inst_encoded = dataset_train.encode_instance(inst)
            inst.update(inst_encoded)            
    
    
    instances = [
        {
            'sentence': inst['sentence_enc'],
            'style': inst['style_enc'],
        } 
        for inst in instances
    ]
    
    instances = default_collate(instances)
    instances = to_device(instances)      
    
    return instances

def get_sentences(outputs):
    predicted_indices = outputs["predictions"]
    end_idx = vocab[Vocab.END_TOKEN]
    
    if not isinstance(predicted_indices, np.ndarray):
        predicted_indices = predicted_indices.detach().cpu().numpy()

    all_predicted_tokens = []
    for indices in predicted_indices:
        indices = list(indices)

        # Collect indices till the first end_symbol
        if end_idx in indices:
            indices = indices[:indices.index(end_idx)]

        predicted_tokens = [vocab.id2token[x] for x in indices]
        all_predicted_tokens.append(predicted_tokens)
        
    return all_predicted_tokens

def perturb_new(criterion, premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length):
#     autoencoder.eval()
    inverter.eval()
    classifier1.eval()
    mlp_classifier.eval()
    model.eval()
    
    premise_words = [premise_words]
    hypothesis_words = [hypothesis_words]
    premise_length = [premise_length]
    hypothesis_length = [hypothesis_length]

    premise_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in premise_words]).cuda()
    hypothesis_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in hypothesis_words]).cuda()
#     print('premise_idx = ' + str(premise_idx))
#     print('hypothesis-idx = ' + str(hypothesis_idx))
#     c_prem = autoencoder.encode(premise_idx, premise_length, noise=False)
#     print("OLD C_PREM")
#     print(c_prem.shape)
#     print(premise_words[0])
    c_prem_full = model(create_inputs(' '.join(premise_words[0])))
    c_prem = c_prem_full['style_hidden']
#     print("C_PREM")
#     print(c_prem)
#     print("----")
#     c_prem = create_inputs(' '.join(premise_words[0]))
    z_prem = inverter(c_prem).detach()
#     print('c_prem = ' + str(c_prem))
#     print('z_prem = ' + str(z_prem))
#     c_hypo = autoencoder.encode(hypothesis_idx, hypothesis_length, noise=False).detach()
    c_hypo_full = model(create_inputs(' '.join(hypothesis_words[0])))
    c_hypo = c_hypo_full['style_hidden'].detach()
#     print("C_HYPO")
#     print(c_hypo)
#     print("------")
    c_hypo.requires_grad = True
    z_hypo = inverter(c_hypo)

    premise = premise.unsqueeze(0)
    hypothesis = hypothesis.unsqueeze(0)
    target = target.unsqueeze(0)

    output = mlp_classifier(z_prem, z_hypo)
    print("OUTPUT")
    print(output)

    loss = criterion(output, target)
    mlp_classifier.zero_grad()
    inverter.zero_grad()
    loss.backward()

    direction = torch.sign(c_hypo.grad)
    nc_hypo = c_hypo + EPS * direction
#     nhypo_idx = autoencoder.generate(nc_hypo, 10, False)
    c_hypo_full['style_hidden'] = nc_hypo

#     nhypo_idx = model.decode(nc_hypo)
    nhypo_idx = model.decode(c_hypo_full)
#     return nhypo_idx.squeeze(0).cpu().numpy()
    return nhypo_idx

if args.new_encoder:
    criterion = nn.CrossEntropyLoss().cuda()

    niter = 0
    evaluate = 0
    nevaluate = 0
    idx2words = dict(map(lambda x: (x[1], x[0]), corpus_vocab.items()))
    while niter < len(testloader):
        niter += 1
        batch = train_iter.next()
        for p, h, t, pw, hw, pl, hl in zip(*batch):
            nh = perturb(criterion, p.cuda(), h.cuda(), t.cuda(), pw, hw, pl, hl)
            print('--------------------------------')
            print('Target ', t)
            print(' '.join(pw))
            print(' '.join(hw))
#         nhw = (['<sos>'] + [idx2words[i] for i in nh])[:10]
            nhw = '<sos> ' + ' '.join(get_sentences(model.decode(nh))[0])
            print(nhw)
            print('Old ', classifier_pred(pw, hw))
            print('New ', classifier_pred(pw, nhw))
            print('Old Pred: ', maximum(classifier_pred(pw, hw)))
            print('New Pred: ', maximum(classifier_pred(pw, nhw)))
            print('Good: ', maximum(classifier_pred(pw, hw)) != maximum(classifier_pred(pw, nhw)))
            evaluate = evaluate + int(maximum(classifier_pred(pw, hw)) != maximum(classifier_pred(pw, nhw)))
            nevaluate = nevaluate + 1
    print(evaluate / nevaluate)
    print(nevaluate)



elif args.train_mode:
    # evaluate_model()

    for epoch in range(0, args.epochs):
        niter = 0
        loss_total = 0
        accuracy_sum = 0.0
        accuracy_surrogate_sum = 0.0
        while niter < len(trainloader):
            niter+=1
            premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length = train_iter.next()
            #print(premise)
            #print(hypothesis)
            if args.cuda:
                premise=premise.cuda()
                hypothesis = hypothesis.cuda()
                target = target.cuda()

            loss, acc, acc_surrogate = train_process(premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length)
            loss_total += loss
            accuracy_sum += acc
            accuracy_surrogate_sum += acc_surrogate
            if niter % 10 == 0:
                print(loss_total/10, accuracy_sum/10, accuracy_surrogate_sum/10)
                loss_total = 0.0
                accuracy_sum = 0.0
                accuracy_surrogate_sum = 0.0
        train_iter = iter(trainloader)
        # curr_acc = evaluate_model()
        # if curr_acc > best_accuracy:
        print("saving model...")
        with open(args.save_path+'/surrogate{}.pt'.format(args.surrogate_layers), 'wb') as f:
            torch.save(mlp_classifier.state_dict(), f)
        # best_accuracy = curr_acc

    # print("Best accuracy :{0}".format(best_accuracy))
else:
    # gen perturbations

    criterion = nn.CrossEntropyLoss().cuda()
    
    niter = 0
    evaluate = 0
    nevaluate = 0
    idx2words = dict(map(lambda x: (x[1], x[0]), corpus_vocab.items()))
    while niter < len(testloader):
        niter += 1
        batch = train_iter.next()
        print(batch)
        for p, h, t, pw, hw, pl, hl in zip(*batch):
            nh = perturb(criterion, p.cuda(), h.cuda(), t.cuda(), pw, hw, pl, hl)
            print('--------------------------------')
            print('Target ', t)
            print(' '.join(pw))
            print(' '.join(hw))
            nhw = (['<sos>'] + [idx2words[i] for i in nh])[:10]
            print(' '.join(nhw))
            print('Old ', classifier_pred(pw, hw))
            print('New ', classifier_pred(pw, nhw))
            print('Old Pred: ', maximum(classifier_pred(pw, hw)))
            print('New Pred: ', maximum(classifier_pred(pw, nhw)))
            print('Good: ', maximum(classifier_pred(pw, hw)) != maximum(classifier_pred(pw, nhw)))
            evaluate = evaluate + int(maximum(classifier_pred(pw, hw)) != maximum(classifier_pred(pw, nhw)))
            nevaluate = nevaluate + 1
    print(evaluate / nevaluate)
    print(nevaluate)
