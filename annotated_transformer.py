import copy
import math
import time
from typing import Optional

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data, datasets

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


#####################################################################################################################


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


#####################################################################################################################


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


#####################################################################################################################


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask_local = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_local) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


#####################################################################################################################


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


#####################################################################################################################


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#####################################################################################################################
# Training
#####################################################################################################################

# Batches and Masking


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# Training Loop


def run_epoch(data_iter, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# Optimizer

class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]


#####################################################################################################################
# Regularization
#####################################################################################################################

# We implement label smoothing using the KL div loss. Instead of using a one-hot target distribution, we create a
# distribution that has confidence of the correct word and the rest of the smoothing mass distributed throughout
# the vocabulary


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        # self.criterion = nn.NLLLoss()    # iohavoc
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


#####################################################################################################################
# Synthetic Data

# batch == batch size, ie. num of sequences
# 10 == sequence length
# nbatches equals how many batches are there in our epoch, since this generator is called from a run_epoch()


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


#####################################################################################################################
# Real world example
#####################################################################################################################


def load_and_preprocess_spacy_de_en():
    spacy_de = spacy.load('de_core_news_sm',
                          disable=['parser' 'tagger', 'entity', 'ner',
                                   'entity_linker', 'entity_ruler', 'textcat',
                                   'sentencizer', 'merge_noun_chunks',
                                   'merge_entities', 'merge_subtokens'])
    spacy_en = spacy.load('en_core_web_sm',
                          disable=['parser' 'tagger', 'entity', 'ner',
                                   'entity_linker', 'entity_ruler', 'textcat',
                                   'sentencizer', 'merge_noun_chunks',
                                   'merge_entities', 'merge_subtokens'])

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
                                                 vars(x)['trg']) <= MAX_LEN)

    print(vars(train[0]))
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.tgt, min_freq=MIN_FREQ)
    return SRC, TGT, train, val, test


# data.Iterator is a torchtext dataset iterator


class BatchingIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, tgt = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, tgt, pad_idx)


#####################################################################################################################
# IOHAVOC EN-FR data
#####################################################################################################################

def load_and_preprocess_spacy_fr_en():
    # tokenizer models
    spacy_fr = spacy.load('fr_core_news_sm',
                          disable=['parser' 'tagger', 'entity', 'ner',
                                   'entity_linker', 'entity_ruler', 'textcat',
                                   'sentencizer', 'merge_noun_chunks',
                                   'merge_entities', 'merge_subtokens'])
    spacy_en = spacy.load('en_core_web_sm',
                          disable=['parser' 'tagger', 'entity', 'ner',
                                   'entity_linker', 'entity_ruler', 'textcat',
                                   'sentencizer', 'merge_noun_chunks',
                                   'merge_entities', 'merge_subtokens'])

    def tokenize_fr(text):
        return [tok.text for tok in spacy_fr.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_fr, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)

    train, val, test = data.TabularDataset.splits(path='./',
                                                  train='en-fr-audio-algorithms-train.txt',
                                                  validation='en-fr-audio-algorithms-val.txt',
                                                  test='en-fr-audio-algorithms-train.txt',
                                                  format='tsv', fields=[('src', SRC), ('trg', TGT)])
    # check an example
    print(vars(train[0]))

    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.tgt, min_freq=MIN_FREQ)
    return SRC, TGT, train, val, test


#####################################################################################################################
# Loss Computation


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


# Greedy Decoding - This code predicts a translation using greedy decoding for simplicity.
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# Train the simple copy task.
def train_copy_task():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src_global = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask_global = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src_global, src_mask_global, max_len=10, start_symbol=1))


#####################################################################################################################
#####################################################################################################################
# PyTorch Lightning implementation of the FR-EN realwork translation task
#####################################################################################################################

class AnnotatedTransformer(pl.LightningModule):

    def __init__(self, batch_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(AnnotatedTransformer, self).__init__()
        # make model
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.batch_size = batch_size

        # Setup data: FR-EN NMT task using spacy
        self.SRC, self.TGT, self.train_dataset, self.val_dataset, self.test = load_and_preprocess_spacy_fr_en()
        self.train_iter = BatchingIterator(self.train_dataset, batch_size=self.batch_size, device=0, repeat=False,
                                           sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn,
                                           train=True)
        self.valid_iter = BatchingIterator(self.val_dataset, batch_size=self.batch_size, device=0, repeat=False,
                                           sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn,
                                           train=False)
        self.pad_idx = self.TGT.vocab.stoi["<blank>"]

        # initialize model data structures
        self.encoder_decoder = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, len(self.SRC.vocab)), c(position)),
            nn.Sequential(Embeddings(d_model, len(self.TGT.vocab)), c(position)),
            Generator(d_model, len(self.TGT.vocab)))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.encoder_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # self.optimizer = NoamOpt(self.encoder_decoder.src_embed[0].d_model, 1, 2000,
        #                          torch.optim.Adam(self.encoder_decoder.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        self.criterion = criterion = LabelSmoothing(size=len(self.TGT.vocab), padding_idx=self.pad_idx, smoothing=0.1)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.encoder_decoder(x.src, x.tgt, x.src_mask, x.tgt_mask)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        fixed_batch = rebatch(self.pad_idx, batch)
        out = self.forward(fixed_batch)
        norm = fixed_batch.ntokens

        x = self.encoder_decoder.generator(out)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), fixed_batch.tgt_y.contiguous().view(-1)) / norm
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.optimizer.zero_grad()
        # self.log("train_loss", loss.item() * norm, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss
            )
        )

    def configure_optimizers(self):
        # pass  # manual optimization requires this to exist but do nothing ¯\_(ツ)_/¯
        return torch.optim.Adam(self.encoder_decoder.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.valid_iter

# Unused because we are using TorchText Iterators
# class AnnotatedTransformerDataModule(pl.LightningDataModule):
#
#     def __init__(self, batch_size):
#         super().__init__()
#         self.batch_size = batch_size
#
#     def setup(self, stage: Optional[str] = None):
#         # transforms for images
#         self.pad_idx = self.TGT.vocab.stoi["<blank>"]
#         SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
#         TGT = data.Field(tokenize=tokenize_fr, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)
#         train, val, test = data.TabularDataset.splits(path='./')
#
#     def train_dataloader(self):
#         return DataLoader(self.train, batch_size=self.batch_size)
#
#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=self.batch_size)
#
#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size)


def train_real_world_task():
    # SRC, TGT, train, val, test = load_and_preprocess_spacy_de_en()    #  IWSLT DE-EN NMT task using spacy
    SRC, TGT, train, val, test = load_and_preprocess_spacy_fr_en()  # FR-EN NMT task using spacy

    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

    BATCH_SIZE = 12000
    train_iter = BatchingIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False,
                                  sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
    valid_iter = BatchingIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False,
                                  sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        # train mode
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model,
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))

        # validate mode
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                         SimpleLossCompute(model.generator, criterion, opt=None))
        print(loss)


if __name__ == "__main__":
    # Old PyTorch code
    # train_copy_task()         # original toy data copy task
    # train_real_world_task()     # real world FR-EN or IWSLT DE-EN NMT task using spacy

    # New PyTorch Lightning
    # fr_en_task = AnnotatedTransformerDataModule(batch_size=12000)

    transfomer = AnnotatedTransformer(batch_size=1200)
    # trainer = pl.Trainer(automatic_optimization=False)
    trainer = pl.Trainer()
    trainer.fit(transfomer)
