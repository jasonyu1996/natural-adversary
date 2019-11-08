import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import to_gpu, load_embeddings
import json
import os
import numpy as np


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.gpu = gpu

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        if x.__class__.__name__ =="ndarray":
            x = Variable(torch.FloatTensor(x)).cuda()
            #x = x.cpu()
        if x.__class__.__name__ =="FloatTensor":
            x = Variable(x).cuda()
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_I(nn.Module):
    # separate Inverter to map continuous code back to z
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_I, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_I_AE(nn.Module):
    # separate Inverter to map continuous code back to z (mean & std)
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_I_AE, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.gpu = gpu
        noutput_mu = noutput
        noutput_var = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.linear_mu = nn.Linear(noutput, noutput_mu)
        self.linear_var = nn.Linear(noutput, noutput_var)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        mu = self.linear_mu(x)
        logvar = self.linear_var(x)
        std = 0.5*logvar
        std = std.exp_()                                        # std
        epsilon = Variable(std.data.new(std.size()).normal_())  # normal noise with the same type and size as std.data
        if self.gpu:
            epsilon = epsilon.cuda()

        sample = mu + (epsilon * std)

        return sample

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

        self.linear_mu.weight.data.normal_(0, init_std)
        self.linear_mu.bias.data.fill_(0)
        self.linear_var.weight.data.normal_(0, init_std)
        self.linear_var.bias.data.fill_(0)


class Seq2SeqCAE(nn.Module):
    # CNN encoder, LSTM decoder
    def __init__(self, emsize, nhidden, ntokens, nlayers, conv_windows="5-5-3", conv_strides="2-2-2",
                 conv_layer="500-700-1000", activation=nn.LeakyReLU(0.2, inplace=True),
                 noise_radius=0.2, hidden_init=False, dropout=0, gpu=True):
        super(Seq2SeqCAE, self).__init__()
        self.nhidden = nhidden      # size of hidden vector in LSTM
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.arch_conv_filters = conv_layer
        self.arch_conv_strides = conv_strides
        self.arch_conv_windows = conv_windows
        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        conv_layer_sizes = [emsize] + [int(x) for x in conv_layer.split('-')]
        conv_strides_sizes = [int(x) for x in conv_strides.split('-')]
        conv_windows_sizes = [int(x) for x in conv_windows.split('-')]
        self.encoder = nn.Sequential()

        for i in range(len(conv_layer_sizes) - 1):
            layer = nn.Conv1d(conv_layer_sizes[i], conv_layer_sizes[i + 1], \
                              conv_windows_sizes[i], stride=conv_strides_sizes[i])
            self.encoder.add_module("layer-" + str(i + 1), layer)

            bn = nn.BatchNorm1d(conv_layer_sizes[i + 1])
            self.encoder.add_module("bn-" + str(i + 1), bn)

            self.encoder.add_module("activation-" + str(i + 1), activation)

        self.linear = nn.Linear(1000, emsize)

        decoder_input_size = emsize + nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.linear_dec = nn.Linear(nhidden, ntokens)

        # 9-> 7-> 3 -> 1
    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)        # training stage
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)

        output, state = self.decoder(augmented_embeddings, state)

        decoded = self.linear_dec(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        if hidden.ndimension() == 1:
            hidden = hidden.unsqueeze(0)
        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        if not self.gpu:
            self.start_symbols = self.start_symbols.cpu()
        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear_dec(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            if indices.ndimension()==1:
                indices = indices.unsqueeze(1)
            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[0].zero()
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        embeddings = embeddings.transpose(1,2)
        c_pre_lin = self.encoder(embeddings)
        c_pre_lin = c_pre_lin.squeeze(2)
        hidden = self.linear(c_pre_lin)
        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        if norms.ndimension()==1:
            norms=norms.unsqueeze(1)
        hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            if self.gpu:
                gauss_noise = gauss_noise.cuda()

            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2) # (hidden, cell)

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False, generator=None, inverter=None):
        if not generator:   # only enc -> dec
            batch_size, maxlen = indices.size()
            self.embedding.weight.data[0].fill_(0)
            self.embedding_decoder.weight.data[0].fill_(0)
            hidden = self.encode(indices, lengths, noise)
            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)
        else:               # enc -> inv -> gen -> dec
            batch_size, maxlen = indices.size()
            self.embedding.weight.data[0].fill_(0)
            self.embedding_decoder.weight.data[0].fill_(0)
            hidden = self.encode(indices, lengths, noise)
            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            z_hat = inverter(hidden)
            c_hat = generator(z_hat)

            decoded = self.decode(c_hat, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=True):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2)) # (hidden, cell)

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False, generator=None, inverter=None):
        if not generator:
            batch_size, maxlen = indices.size()

            hidden = self.encode(indices, lengths, noise)

            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            decoded = self.decode(hidden, batch_size, maxlen,
                                  indices=indices, lengths=lengths)
        else:
            batch_size, maxlen = indices.size()
            self.embedding.weight.data[0].fill_(0)
            self.embedding_decoder.weight.data[0].fill_(0)
            hidden = self.encode(indices, lengths, noise)
            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            z_hat = inverter(hidden)
            c_hat = generator(z_hat)

            decoded = self.decode(c_hat, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        if norms.ndimension()==1:
            norms=norms.unsqueeze(1)
        hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            if indices.ndimension()==1:
                indices = indices.unsqueeze(1)
            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


def load_models(load_path):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    print('Loading models from' + load_path+"/models")
    ae_path = os.path.join(load_path+"/models/", "autoencoder_model.pt")
    inv_path = os.path.join(load_path+"/models/", "inverter_model.pt")
    gen_path = os.path.join(load_path+"/models/", "gan_gen_model.pt")
    disc_path = os.path.join(load_path+"/models/", "gan_disc_model.pt")

    autoencoder = torch.load(ae_path)
    inverter = torch.load(inv_path)
    gan_gen = torch.load(gen_path)
    gan_disc = torch.load(disc_path)
    return model_args, idx2word, autoencoder, inverter, gan_gen, gan_disc


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        noise = Variable(z, volatile=True)
    elif type(z) == np.ndarray:
        noise = Variable(torch.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences


class JSDistance(nn.Module):
    def __init__(self, mean=0, std=1, epsilon=1e-5):
        super(JSDistance, self).__init__()
        self.epsilon = epsilon
        self.distrib_type_normal = True

    def get_kl_div(self, input, target):
        src_mu = torch.mean(input)
        src_std = torch.std(input)
        tgt_mu = torch.mean(target)
        tgt_std = torch.std(target)
        kl = torch.log(tgt_std/src_std) - 0.5 +\
                    (src_std ** 2 + (src_mu - tgt_mu) ** 2)/(2 * (tgt_std ** 2))
        return kl

    def forward(self, input, target):
        ##KL(p, q) = log(sig2/sig1) + ((sig1^2 + (mu1 - mu2)^2)/2*sig2^2) - 1/2
        if self.distrib_type_normal:
            d1=self.get_kl_div(input, target)
            d2=self.get_kl_div(target, input)
            return 0.5 * (d1+d2)
        else:
            input_num_zero = input.data[torch.eq(input.data, 0)]
            if input_num_zero.dim() > 0:
                input_num_zero = input_num_zero.size(0)
                input.data = input.data - (self.epsilon/input_num_zero)
                input.data[torch.lt(input.data, 0)] = self.epsilon/input_num_zero
            target_num_zero = target.data[torch.eq(target.data, 0)]
            if target_num_zero.dim() > 0:
                target_num_zero = target_num_zero.size(0)
                target.data = target.data - (self.epsilon/target_num_zero)
                target.data[torch.lt(target.data, 0)] = self.epsilon/target_num_zero
            d1 = torch.sum(input * torch.log(input/target))/input.size(0)
            d2 = torch.sum(target * torch.log(target/input))/input.size(0)
            return (d1+d2)/2


class Baseline_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, maxlen=10, dropout= 0, vocab_size=11004, gpu=False):
        super(Baseline_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = 1
        self.gpu = gpu
        self.maxlen = maxlen
        self.embedding_prem = nn.Embedding(vocab_size+4, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size+4, emb_size)
        self.premise_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        print(self.premise_encoder)
        self.hypothesis_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.layers = nn.Sequential()
        layer_sizes = [2*hidden_size, 400, 100]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.add_module("bn" + str(i + 1), bn)

            self.layers.add_module("activation" + str(i + 1), nn.ReLU())

        layer = nn.Linear(layer_sizes[-1], 3)
        self.layers.add_module("layer" + str(len(layer_sizes)), layer)

        self.layers.add_module("softmax", nn.Softmax())

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding_prem.weight.data.uniform_(-initrange, initrange)
        self.embedding_hypo.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.premise_encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.hypothesis_encoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2)) # (hidden, cell)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad


    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        batch_size = premise_indices.size(0)
        state_prem= self.init_hidden(batch_size)
        state_hypo= self.init_hidden(batch_size)
        premise = self.embedding_prem(premise_indices)
        output_prem, (hidden_prem, _) = self.premise_encoder(premise, state_prem)
        hidden_prem= hidden_prem[-1]
        if hidden_prem.requires_grad:
            hidden_prem.register_hook(self.store_grad_norm)

        hypothesis = self.embedding_hypo(hypothesis_indices)
        output_hypo, (hidden_hypo, _) = self.hypothesis_encoder(hypothesis, state_hypo)
        hidden_hypo= hidden_hypo[-1]
        if hidden_hypo.requires_grad:
            hidden_hypo.register_hook(self.store_grad_norm)

        concatenated = torch.cat([hidden_prem, hidden_hypo], 1)
        probs = self.layers(concatenated)
        return probs


class Baseline_Embeddings(nn.Module):
    def __init__(self, emb_size, vocab_size=11004):
        super(Baseline_Embeddings, self).__init__()
        self.embedding_prem = nn.Embedding(vocab_size, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size*2, 3)
        embeddings_mat = load_embeddings()
        self.embedding_prem.weight.data.copy_(embeddings_mat)
        self.embedding_hypo.weight.data.copy_(embeddings_mat)

    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        enc_premise = self.embedding_prem(premise_indices)
        enc_hypothesis = self.embedding_hypo(hypothesis_indices)
        enc_premise = torch.mean(enc_premise,1).squeeze(1)
        enc_hypothesis = torch.mean(enc_hypothesis,1).squeeze(1)

        concatenated = torch.cat([enc_premise, enc_hypothesis], 1)
        probs = self.linear(concatenated)
        return probs

class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, layers=None):
        super(MLPClassifier, self).__init__()
        last_size = input_size
        layer_sizes = list(map(int, layers.split('-')))

        self.layers = nn.Sequential()
        for i, lsize in enumerate(layer_sizes):
            layer = nn.Linear(last_size, lsize)
            self.layers.add_module('layer' + str(i), layer)
            bn = nn.BatchNorm1d(lsize, eps=1e-05, momentum=0.1)
            self.layers.add_module('bn' + str(i), bn)
            self.layers.add_module("activation" + str(i), nn.ReLU())
            last_size = lsize

        self.linear = nn.Linear(last_size, output_size)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, z_prem, z_hypo):
        x = torch.cat((z_prem, z_hypo), 1)
        return self.log_softmax(self.linear(self.layers(x)))


import numpy as np
import torch
import torch.nn.functional as F

from losses import SequenceReconstructionLoss, StyleEntropyLoss, MeaningZeroLoss
from utils2 import get_sequences_lengths, to_device


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers=1, bidirectional=False, return_sequence=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def zero_state(self, batch_size):
        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = self.num_layers if not self.bidirectional else self.nb_layers * 2
        state_shape = (nb_layers, batch_size, self.hidden_size)

        # shape: (num_layers, batch_size, hidden_dim)
        h = to_device(torch.zeros(*state_shape))

        # shape: (num_layers, batch_size, hidden_dim)
        c = torch.zeros_like(h)

        return h, c

    def forward(self, inputs, lengths):
        batch_size = inputs.shape[0]

        # shape: (num_layers, batch_size, hidden_dim)
        h, c = self.zero_state(batch_size)

        lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
        inputs_sorted = inputs[inputs_sorted_idx]

        # pack sequences
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths_sorted.detach(), batch_first=True)

        # shape: (batch_size, sequence_len, hidden_dim)
        outputs, (h, c) = self.lstm(packed, (h, c))

        # concatenate if bidirectional
        # shape: (batch_size, hidden_dim)
        h = torch.cat([x for x in h], dim=-1)

        # unpack sequences
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        _, inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)
        outputs = outputs[inputs_unsorted_idx]
        h = h[inputs_unsorted_idx]

        if self.return_sequence:
            return outputs
        else:
            return h


class Squeeze(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()

        self.dim = dim

    def forward(self, inputs):
        inputs = inputs.squeeze(self.dim)
        return inputs


class SpaceTransformer(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.Dropout(dropout),
            # torch.nn.ELU(),
            torch.nn.Hardtanh(-10, 10),
        )

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs):
        outputs = self.classifier(inputs)
        return outputs


class Seq2Seq(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, max_len, scheduled_sampling_ratio,
                 start_index, end_index, pad_index, trainable_embeddings, W_emb=None, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.dropout = dropout
        self.scheduled_sampling_ratio = scheduled_sampling_ratio
        self.trainable_embeddings = trainable_embeddings

        self.start_index = start_index
        self.end_index = end_index
        self.pad_index = pad_index

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_index)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(embedding_size, hidden_size, dropout)
        self.decoder_cell = torch.nn.LSTMCell(embedding_size, hidden_size)
        self.output_projection = torch.nn.Linear(hidden_size, vocab_size)

        self._xent_loss = SequenceReconstructionLoss(ignore_index=pad_index)

    def encode(self, inputs):
        # shape: (batch_size, sequence_len)
        sentence = inputs['sentence']

        # shape: (batch_size, )
        lengths = get_sequences_lengths(sentence)

        # shape: (batch_size, sequence_len, embedding_size)
        sentence_emb = self.embedding(sentence)

        # shape: (batch_size, hidden_size)
        decoder_hidden = self.encoder(sentence_emb, lengths)

        output_dict = {
            'decoder_hidden': decoder_hidden
        }

        return output_dict

    def decode(self, state, targets=None):
        # shape: (batch_size, hidden_size)
        decoder_hidden = state['decoder_hidden']
        decoder_cell = torch.zeros_like(decoder_hidden)

        batch_size = decoder_hidden.size(0)

        if targets is not None:
            num_decoding_steps = targets.size(1)
        else:
            num_decoding_steps = self.max_len

        # shape: (batch_size, )
        last_predictions = decoder_hidden.new_full((batch_size,), fill_value=self.start_index).long()
        # shape: (batch_size, sequence_len, vocab_size)
        step_logits = []
        # shape: (batch_size, sequence_len, )
        step_predictions = []

        for timestep in range(num_decoding_steps):
            # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio during training.
            # shape: (batch_size,)
            decoder_input = last_predictions
            if timestep > 0 and self.training and torch.rand(1).item() > self.scheduled_sampling_ratio:
                decoder_input = targets[:, timestep - 1]

            # shape: (batch_size, embedding_size)
            decoder_input = self.embedding(decoder_input)

            # shape: (batch_size, hidden_size)
            decoder_hidden, decoder_cell = self.decoder_cell(decoder_input, (decoder_hidden, decoder_cell))

            # shape: (batch_size, vocab_size)
            output_projection = self.output_projection(decoder_hidden)

            # list of tensors, shape: (batch_size, 1, vocab_size)
            step_logits.append(output_projection.unsqueeze(1))

            # shape (predicted_classes): (batch_size,)
            last_predictions = torch.argmax(output_projection, 1)

            # list of tensors, shape: (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, max_len, vocab_size)
        logits = torch.cat(step_logits, 1)
        # shape: (batch_size, max_len)
        predictions = torch.cat(step_predictions, 1)

        state.update({
            "logits": logits,
            "predictions": predictions,
        })

        return state

    def calc_loss(self, output_dict, inputs):
        # shape: (batch_size, sequence_len)
        targets = inputs['sentence']
        # shape: (batch_size, sequence_len, vocab_size)
        logits = output_dict['logits']

        loss = self._xent_loss(logits, targets)

        output_dict['loss'] = loss

        return output_dict

    def forward(self, inputs):
        state = self.encode(inputs)
        output_dict = self.decode(state, inputs['sentence'])

        output_dict = self.calc_loss(output_dict, inputs)

        return output_dict


class Seq2SeqMeaningStyle(Seq2Seq):
    def __init__(self, meaning_size, style_size, nb_styles, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.meaning_size = meaning_size
        self.style_size = style_size
        self.nb_styles = nb_styles

        self.hidden_meaning = SpaceTransformer(self.hidden_size, self.meaning_size, self.dropout)
        self.hidden_style = SpaceTransformer(self.hidden_size, self.meaning_size, self.dropout)
        self.meaning_style_hidden = SpaceTransformer(meaning_size + style_size, self.hidden_size, self.dropout)

        # D - discriminator: discriminates the style of a sentence
        self.D_meaning = Discriminator(meaning_size, self.hidden_size, nb_styles, self.dropout)
        self.D_style = Discriminator(style_size, self.hidden_size, nb_styles, self.dropout)

        # P - predictor: predicts the meaning of a sentence (word embeddings)
        self.P_meaning = Discriminator(meaning_size, self.hidden_size, self.embedding_size, self.dropout)
        self.P_style = Discriminator(style_size, self.hidden_size, self.embedding_size, self.dropout)

        # P_bow - predictor_bow: predicts the meaning of a sentence (BoW)
        self.P_bow_meaning = Discriminator(meaning_size, self.hidden_size, self.vocab_size, self.dropout)
        self.P_bow_style = Discriminator(style_size, self.hidden_size, self.vocab_size, self.dropout)

        # Discriminator for gaussian z
        self.D_hidden = Discriminator(self.hidden_size, self.hidden_size, 2, self.dropout)

        self._D_loss = torch.nn.CrossEntropyLoss()
        self._D_adv_loss = StyleEntropyLoss()

        self._P_loss = torch.nn.MSELoss()
        self._P_adv_loss = MeaningZeroLoss()

        self._P_bow_loss = torch.nn.BCEWithLogitsLoss()
        self._P_bow_adv_loss = StyleEntropyLoss()

    def encode(self, inputs):
        state = super().encode(inputs)

        # shape: (batch_size, hidden_size)
        decoder_hidden = state['decoder_hidden']

        # shape: (batch_size, hidden_size)
        meaning_hidden = self.hidden_meaning(decoder_hidden)

        # shape: (batch_size, hidden_size)
        style_hidden = self.hidden_style(decoder_hidden)

        state['meaning_hidden'] = meaning_hidden
        state['style_hidden'] = style_hidden

        return state

    def combine_meaning_style(self, state):
        # shape: (batch_size, hidden_size * 2)
        decoder_hidden = torch.cat([state['meaning_hidden'], state['style_hidden']], dim=-1)

        # shape: (batch_size, hidden_size)
        decoder_hidden = self.meaning_style_hidden(decoder_hidden)

        state['decoder_hidden'] = decoder_hidden

        return state

    def decode(self, state, targets=None):
        state = self.combine_meaning_style(state)

        output_dict = super().decode(state, targets)
        return output_dict

    def calc_discriminator_loss(self, output_dict, inputs):
        output_dict['loss_D_meaning'] = self._D_loss(output_dict['D_meaning_logits'], inputs['style'])
        output_dict['loss_D_style'] = self._D_loss(output_dict['D_style_logits'], inputs['style'])

        if 'meaning_embedding' in inputs:
            output_dict['loss_P_meaning'] = self._P_loss(output_dict['P_meaning'], inputs['meaning_embedding'])
            output_dict['loss_P_style'] = self._P_loss(output_dict['P_style'], inputs['meaning_embedding'])

        if 'meaning_bow' in inputs:
            output_dict['loss_P_bow_meaning'] = self._P_bow_loss(output_dict['P_bow_meaning'], inputs['meaning_bow'])
            output_dict['loss_P_bow_style'] = self._P_bow_loss(output_dict['P_bow_style'], inputs['meaning_bow'])

        return output_dict

    def calc_discriminator_adv_loss(self, output_dict, inputs):
        output_dict['loss_D_adv_meaning'] = self._D_adv_loss(output_dict['D_meaning_logits'])
        output_dict['loss_D_adv_style'] = self._D_loss(output_dict['D_style_logits'], inputs['style'])

        if 'meaning_embedding' in inputs:
            output_dict['loss_P_adv_meaning'] = self._P_loss(output_dict['P_meaning'], inputs['meaning_embedding'])
            output_dict['loss_P_adv_style'] = self._P_adv_loss(output_dict['P_style'])

        if 'meaning_bow' in inputs:
            output_dict['loss_P_bow_adv_meaning'] = self._P_bow_loss(
                output_dict['P_bow_meaning'], inputs['meaning_bow'])
            output_dict['loss_P_bow_adv_style'] = self._P_bow_adv_loss(output_dict['P_bow_style'])

        return output_dict

    def discriminate(self, output_dict, inputs, adversarial=False):
        output_dict['D_meaning_logits'] = self.D_meaning(output_dict['meaning_hidden'])
        output_dict['D_style_logits'] = self.D_style(output_dict['style_hidden'])

        if 'meaning_embedding' in inputs:
            output_dict['P_meaning'] = self.P_meaning(output_dict['meaning_hidden'])
            output_dict['P_style'] = self.P_style(output_dict['style_hidden'])

        if 'meaning_bow' in inputs:
            output_dict['P_bow_meaning'] = self.P_bow_meaning(output_dict['meaning_hidden'])
            output_dict['P_bow_style'] = self.P_bow_style(output_dict['style_hidden'])

        # calc loss
        if not adversarial:
            output_dict = self.calc_discriminator_loss(output_dict, inputs)
        else:
            output_dict = self.calc_discriminator_adv_loss(output_dict, inputs)

        return output_dict


class StyleClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, trainable_embeddings, pad_index, nb_styles,
                 W_emb=None, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.trainable_embeddings = trainable_embeddings
        self.nb_styles = nb_styles

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_index)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(embedding_size, hidden_size, dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, nb_styles),
        )

        self._xent_loss = torch.nn.CrossEntropyLoss()

    def encode(self, inputs):
        # shape: (batch_size, sequence_len)
        sentence = inputs['sentence']

        # shape: (batch_size, )
        lengths = get_sequences_lengths(sentence)

        # shape: (batch_size, sequence_len, embedding_size)
        sentence_emb = self.embedding(sentence)

        # shape: (batch_size, hidden_size)
        decoder_hidden = self.encoder(sentence_emb, lengths)

        output_dict = {
            'decoder_hidden': decoder_hidden
        }

        return output_dict

    def classify(self, state):
        # shape: (batch_size, hidden_size)
        hidden = state['decoder_hidden']

        # shape: (batch_size, nb_classes)
        logits = self.classifier(hidden)
        predictions = torch.argmax(logits, 1)

        state.update({
            "logits": logits,
            "predictions": predictions,
        })

        return state

    def calc_loss(self, output_dict, inputs):
        # shape: (batch_size, sequence_len)
        targets = inputs['style']
        # shape: (batch_size, sequence_len, vocab_size)
        logits = output_dict['logits']

        loss = self._xent_loss(logits, targets)

        output_dict['loss'] = loss

        return output_dict

    def forward(self, inputs):
        state = self.encode(inputs)
        output_dict = self.classify(state)

        output_dict = self.calc_loss(output_dict, inputs)

        return output_dict
