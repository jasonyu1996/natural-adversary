import argparse
import json
from models import MLPClassifier, Baseline_Embeddings
from models import Seq2Seq, MLP_D, MLP_G, MLP_I, MLP_I_AE, JSDistance, Seq2SeqCAE, Baseline_Embeddings, Baseline_LSTM
from utils import to_gpu, Corpus, batchify, SNLIDataset, collate_snli
import random
import pickle as pkl
import torch
import scipy
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

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
parser.add_argument('--c_size', type=int, default=300,
                    help='dimension of code c to feed into inverter and decoder')
parser.add_argument('--load_pretrained', type=str, required=True,
                    help='load a pre-trained encoder and decoder to train the inverter')
parser.add_argument('--surrogate-layers', type=str, default='100-50',
                    help='hidden layer sizes of the surrogate model')
parser.add_argument('--perturb-budget', type=float, default=3e-2,
                    help='the budget for making perturbations by the adversary')
parser.add_argument('--alpha', type=float, default=0.4,
                    help='regulariser for retaining the original meaning of the input instance')
#parser.add_argument('--perturb-z', action='store_true', 
#                    help='add perturbation in the z space rather than in the c space')
parser.add_argument('--input-c', action='store_true',
                    help='use the c space as the direct input to the surrogate model')
parser.add_argument('--hsearch', action='store_true',
                    help='directly constructs an adversarial example')
parser.add_argument('--perturb-itern', type=int, default=50,
                    help='perturbation iteration number')
parser.add_argument('--perturb-rinitn', type=int, default=1,
                    help='perturbation random initialisation number')
parser.add_argument('--perturb-random', action='store_true',
                    help='use purely random perturbations')
parser.add_argument('--lstm', action='store_true',
                    help='use LSTM as the victim model (rather than the embedding)')
args = parser.parse_args()

cur_dir = './output/%s/' % args.load_pretrained

ALPHA = args.alpha

with open(cur_dir + '/vocab.json', 'r') as fin:
    corpus_vocab = json.load(fin)

corpus_train = SNLIDataset(train=True, vocab_size=args.vocab_size-4, path=args.data_path)
corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size-4, path=args.data_path)
trainloader= torch.utils.data.DataLoader(corpus_train, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=True)
train_iter = iter(trainloader)
testloader= torch.utils.data.DataLoader(corpus_test, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=False)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

EPS = args.perturb_budget



autoencoder = torch.load(open(cur_dir + '/models/autoencoder_model.pt', 'rb'))
gan_gen = torch.load(open(cur_dir + '/models/gan_gen_model.pt', 'rb'))
#gan_disc = torch.load(open(cur_dir + '/models/gan_disc_model.pt', 'rb'))
inverter = torch.load(open(cur_dir + '/models/inverter_model.pt', 'rb'))

if not args.lstm:
    classifier1 = Baseline_Embeddings(100, vocab_size=args.vocab_size)
    classifier1.load_state_dict(torch.load(args.classifier_path + "/baseline/emb.pt"))
else:
    classifier1 = Baseline_LSTM(100,300,maxlen=args.maxlen, gpu=args.cuda)
    classifier1.load_state_dict(torch.load(args.classifier_path + '/baseline/model_lstm.pt'))
vocab_classifier1 = pkl.load(open(args.classifier_path + "/vocab.pkl", 'rb'))


if args.input_c:
    mlp_classifier = MLPClassifier(args.c_size * 2, 3, layers=args.surrogate_layers)
else:
    mlp_classifier = MLPClassifier(args.z_size * 2, 3, layers=args.surrogate_layers)
if not args.train_mode:
    mlp_classifier.load_state_dict(torch.load(args.save_path+'/surrogate{0}.pt'.format(args.surrogate_layers)))

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
    gan_gen = gan_gen.cuda()
    #gan_disc = gan_disc.cuda()
    classifier1 = classifier1.cuda()
    inverter = inverter.cuda()
    mlp_classifier = mlp_classifier.cuda()
else:
    autoencoder.gpu = False
    autoencoder = autoencoder.cpu()
    autoencoder.start_symbols = autoencoder.start_symbols.cpu()
    gan_gen = gan_gen.cpu()
    #gan_disc = gan_disc.cpu()
    classifier1 = classifier1.cpu()
    inverter = inverter.cpu()
    mlp_classifier = mlp_classifier.cpu()

def train_process(premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length, input_c):
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

    c_hypo = autoencoder.encode(hypothesis_idx, hypothesis_length, noise=False)

    if input_c:
        c_prem = c_prem.detach()
        c_hypo = c_hypo.detach()
        output = mlp_classifier(c_prem, c_hypo)
    else:
        z_prem = inverter(c_prem).detach()
        z_hypo = inverter(c_hypo).detach()
        output = mlp_classifier(z_prem, z_hypo)

    # z_comb = nn.cat((z_prem, z_hypo), 0).detach()

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


def classifier_pred(pw, hw, idx=False):
    classifier1.eval()

    if idx:
        premise_idx = pw
        hypothesis_idx = hw
    else:
        premise_idx = torch.tensor([vocab_classifier1.get(w, 3) for w in pw]).cuda().unsqueeze(0)
        hypothesis_idx = torch.tensor([vocab_classifier1.get(w, 3) for w in hw]).cuda().unsqueeze(0)

    return F.softmax(classifier1((premise_idx, hypothesis_idx)), 1).squeeze(0).cpu().detach().numpy()
    

def normalised_decode(nh):
    nhw = (['<sos>'] + [idx2words[i] for i in nh])[:10]
    try:
        idx = nhw.index('<eos>')
    except:
        idx = len(nhw)
    nhw = nhw[:idx]
    for i in range(idx, 10):
        nhw.append('<pad>')
    return nhw

def compute_kl_div(premise_words, z_hypo, gold):
    #print(z_hypo.shape)
    nhw = normalised_decode(autoencoder.generate(gan_gen(z_hypo), 10, False).squeeze(0).cpu().numpy())
    #print(nhw)
    out = classifier_pred(premise_words[0], nhw)
    return F.kl_div(torch.log(torch.tensor(out, device=gold.device)), gold).cpu().item()

def perturb(criterion, premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length, input_c, hsearch):
    autoencoder.eval()
    inverter.eval()
    classifier1.eval()
    mlp_classifier.eval()
    gan_gen.eval()

    premise_words = [premise_words]
    hypothesis_words = [hypothesis_words]
    premise_length = [premise_length]
    hypothesis_length = [hypothesis_length]


    premise_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in premise_words]).cuda()
    hypothesis_idx = torch.tensor([[corpus_vocab.get(w, 3) for w in s] for s in hypothesis_words]).cuda()

    c_prem = autoencoder.encode(premise_idx, premise_length, noise=False)
    c_hypo = autoencoder.encode(hypothesis_idx, hypothesis_length, noise=False)

    if hsearch:
        ref_z_hypo = inverter(c_hypo)
        best = (1e10, torch.randn_like(ref_z_hypo))
        for k in range(0, args.perturb_rinitn):
            z_hypo = torch.randn_like(ref_z_hypo)
            z_hypo.requires_grad = True
            if input_c:
                z_prem = c_prem.detach()
                gold = mlp_classifier(c_prem, c_hypo).detach()
            else:
                z_prem = inverter(c_prem).detach()
                gold = mlp_classifier(z_prem, inverter(c_hypo)).detach()
            gold = torch.exp(gold)
            if not args.perturb_random:
# when doing random perturbations, do not use any information from the surrogate model
                perturb_optim = optim.SGD([z_hypo], lr=1e-2, momentum=0.9)
                for i in range(0, args.perturb_itern):
                    perturb_optim.zero_grad()
                    if input_c:
                        rz_hypo = gan_gen(z_hypo) # actually c
                    else:
                        rz_hypo = z_hypo
                    n_output = mlp_classifier(z_prem, rz_hypo)
                    loss = -torch.sum(n_output * gold)
                    
                    loss.backward()
                    perturb_optim.step()
            kl_div = compute_kl_div(premise_words, z_hypo, gold)
            if kl_div < best[0]:
                best = (kl_div, z_hypo)
        
        _, z_hypo = best
        nc_hypo = gan_gen(z_hypo)
    else:
        if input_c:
            z_prem = c_prem.detach()
            z_hypo = c_hypo.detach()
        else:
            z_prem = inverter(c_prem).detach()
            z_hypo = inverter(c_hypo).detach()
        z_hypo.requires_grad = True

        # forgot why I did this
        premise = premise.unsqueeze(0)
        hypothesis = hypothesis.unsqueeze(0)
        target = target.unsqueeze(0)
        reg_target = torch.ones(1, dtype=torch.int64).cuda()
        
        output = mlp_classifier(z_prem, z_hypo)

        z_hypo_detached = z_hypo.detach()
        reg_output1 = mlp_classifier(z_hypo_detached, z_hypo)
        reg_output2 = mlp_classifier(z_hypo, z_hypo_detached)
        loss = criterion(output, target) - ALPHA * (criterion(reg_output1, reg_target) + criterion(reg_output2, reg_target))
        mlp_classifier.zero_grad()
        inverter.zero_grad()
        loss.backward()

        direction = torch.sign(z_hypo.grad)
        
        nc_hypo = z_hypo + EPS * direction
        if not input_c:
            nc_hypo = gan_gen(nc_hypo)
    nhypo_idx = autoencoder.generate(nc_hypo, 10, False)

    return nhypo_idx.squeeze(0).cpu().numpy()

if args.train_mode:
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

            loss, acc, acc_surrogate = train_process(premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length, args.input_c)
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

    idx2words = dict(map(lambda x: (x[1], x[0]), corpus_vocab.items()))

    succ_dec = 0
    old_correct = 0
    new_correct = 0
    tot = 0
    classes_cnt = [0, 0, 0]
    classes_correct_cnt = [[[0 for i in range(0, 3)] for j in range(0, 3)] for k in range(0, 3)]
    good_list = []
    kl_div = 0.0

    while niter < len(testloader):
        niter += 1
        batch = train_iter.next()
        for p, h, t, pw, hw, pl, hl in zip(*batch):
            tot += 1
            classes_cnt[t.item()] += 1
            nh = perturb(criterion, p.cuda(), h.cuda(), t.cuda(), pw, hw, pl, hl, args.input_c, args.hsearch)
            print('--------------------------------')
            print('Target ', t)
            print(' '.join(pw))
            print(' '.join(hw))
            nhw = normalised_decode(nh)
            print(' '.join(nhw))

            old_pred = classifier_pred(pw, hw)
            new_pred = classifier_pred(pw, nhw)
            print('Old ', old_pred)
            print('New ', new_pred)
            
            if old_pred[t.item()] > new_pred[t.item()]:
                succ_dec += 1

            old_label = np.argmax(old_pred)
            new_label = np.argmax(new_pred)
            if old_label == t.item():
                old_correct += 1

            if new_label == t.item():
                new_correct += 1

            kl_div += F.kl_div(torch.log(torch.tensor(new_pred)), torch.tensor(old_pred)).item()
 
            classes_correct_cnt[t.item()][old_label][new_label] += 1
            if t.item() == 0 and old_label == 0 and new_label == 0:
                good_list.append((pw, hw, nhw))

    print('Success rate: %.3f' % (100.0 * succ_dec / tot))
    print('Old accuracy: %.3f' % (100.0 * old_correct / tot))
    print('New accuracy: %.3f' % (100.0 * new_correct / tot))
    print('Class cnt: ', [x / tot for x in classes_cnt])

    for i in range(0, 3):
        old_correct_cnt = sum(classes_correct_cnt[i][i])
        print('Class %d adversarial accuracy = %.5f%% (%d/%d)' % (i, classes_correct_cnt[i][i][i] / old_correct_cnt * 100, \
                classes_correct_cnt[i][i][i],\
                old_correct_cnt))
    print('Average KL divergence: %.5f' % (kl_div / tot))

    with open('hit.pkl', 'wb') as fout:
        pkl.dump(good_list, fout)
    

