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
parser.add_argument('--train_mode', type=bool, default=True,
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
args = parser.parse_args()

cur_dir = './output/%s/' % args.load_pretrained

with open(cur_dir + '/vocab.json', 'r') as fin:
    corpus_vocab = json.load(fin)

corpus_train = SNLIDataset(train=True, vocab_size=args.vocab_size, path=args.data_path, reset_vocab=corpus_vocab)
corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size, path=args.data_path, reset_vocab=corpus_vocab)
trainloader= torch.utils.data.DataLoader(corpus_train, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=True)
train_iter = iter(trainloader)
testloader= torch.utils.data.DataLoader(corpus_test, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=False)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)



autoencoder = torch.load(open(cur_dir + '/models/autoencoder_model.pt', 'rb'))
#gan_gen = torch.load(open(cur_dir + '/models/gan_gen_model.pt', 'rb'))
#gan_disc = torch.load(open(cur_dir + '/models/gan_disc_model.pt', 'rb'))
inverter = torch.load(open(cur_dir + '/models/inverter_model.pt', 'rb'))

classifier1 = Baseline_Embeddings(100, vocab_size=args.vocab_size)
classifier1.load_state_dict(torch.load(args.classifier_path + "/baseline/model_emb.pt"))
vocab_classifier1 = pkl.load(open(args.classifier_path + "/vocab.pkl", 'rb'))

mlp_classifier = MLPClassifier(args.z_size * 2, 3, layers='100-50')

print(classifier1)
print(autoencoder)
print(inverter)
print(mlp_classifier)

optimizer = optim.Adam(mlp_classifier.parameters(),
                           lr=args.lr,
                           betas=(args.beta1, 0.999))

if args.cuda:
    autoencoder = autoencoder.cuda()
    #gan_gen = gan_gen.cuda()
    #gan_disc = gan_disc.cuda()
    classifier1 = classifier1.cuda()
    inverter = inverter.cuda()
    mlp_classifier = mlp_classifier.cuda()
else:
    autoencoder = autoencoder.cpu()
    #gan_gen = gan_gen.cpu()
    #gan_disc = gan_disc.cpu()
    classifier1 = classifier1.cpu()
    inverter = inverter.cpu()
    mlp_classifier = mlp_classifier.cpu()




def train_process(premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length):
    #print(hypothesis_words, flush=True)
    autoencoder.eval()
    inverter.eval()
    classifier1.eval()
    mlp_classifier.train()

    #print(premise.max().item(), flush=True)
    #print(hypothesis.max().item(), flush=True)

    c_prem = autoencoder.encode(premise, premise_length, noise=False)
    z_prem = inverter(c_prem).detach()

    c_hypo = autoencoder.encode(hypothesis, hypothesis_length, noise=False)
    z_hypo = inverter(c_hypo).detach()

    # z_comb = nn.cat((z_prem, z_hypo), 0).detach()

    premise_idx = torch.tensor([[vocab_classifier1.get(w, 3) for w in s] for s in premise_words]).cuda()
    hypothesis_idx = torch.tensor([[vocab_classifier1.get(w, 3) for w in s] for s in hypothesis_words]).cuda()

    output = mlp_classifier(z_prem, z_hypo)
    gold = classifier1((premise_idx, hypothesis_idx)).detach()

    #print(output.shape, flush=True)
    #print(gold.shape, flush=True)


    loss = -torch.mean(torch.sum(output * F.softmax(gold, dim=1), 1), 0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


best_accuracy = 0
if args.train_mode:
    for epoch in range(0, args.epochs):
        niter = 0
        loss_total = 0
        while niter < len(trainloader):
            niter+=1
            premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length = train_iter.next()
            #print(premise)
            #print(hypothesis)
            if args.cuda:
                premise=premise.cuda()
                hypothesis = hypothesis.cuda()
                target = target.cuda()

            loss_total += train_process(premise, hypothesis, target, premise_words, hypothesis_words, premise_length, hypothesis_length)
        print(loss_total/float(niter))
        train_iter = iter(trainloader)
        # curr_acc = evaluate_model()
        # if curr_acc > best_accuracy:
        print("saving model...")
        with open(args.save_path+'/surrogate.pt', 'wb') as f:
            torch.save(mlp_classifier.state_dict(), f)
        # best_accuracy = curr_acc

    # print("Best accuracy :{0}".format(best_accuracy))
