"""
Assignment #2-Natural Language Processing-CS-4765

Implemented models: Uniform, Unigram, Bigram, Interpolated
                    0        1        2       interp

1.) Create the Model `m`
2.) Train `m` from the training data
3.) Test `m` in log-space for the dev-data and test-data

Developer: Vaasudevan Srinivasan, MEng. GGE
Python   : 2.7
Email    : vaasu.devan@unb.ca
Date     : Oct 12, 2018

"""
from __future__ import print_function, division
from collections import Counter
import math
import sys

class Model(object):  # Common class with methods for all the models

    def check(self, EPS=0.0001):
        Total_p = 0
        for w in self.vocab:
            p = math.exp(self.log_prob(w))
            assert 0-EPS < p < 1+EPS
            Total_p += p
        assert 1-EPS < Total_p < 1+EPS
    
    def read_corpora(self, infile):
        sent = []
        for t in open(infile):
            token = t.strip().lower()
            if token:
                sent.append(token)
            if token == '</s>':
                yield sent
                sent = []

    @staticmethod
    def Perplexity(infile):
        logprobs = [float(x.split()[1]) for x in open(infile)]
        logP = sum(logprobs)
        N = len(logprobs)
        HW = (-1/N) * logP
        perplexity = math.exp(HW)
        print(str(perplexity))
        return perplexity
    

class UniformModel(Model):
    
    def __init__(self, infile):
        self.train(infile)

    def train(self, infile):
        Sntncs = self.read_corpora(infile)  # Inherited method
        self.vocab = {tok for sen in Sntncs for tok in sen if tok!=r'<s>'}
        self.vocab.add('UNK')
        self.V = len(self.vocab) # Types

    def log_prob(self, w):       # w is not used because prob is uniform
        return math.log(1/self.V)

    def test(self, infile):
        for sent in self.read_corpora(infile):
            for token in sent[1:]:    # sent[0] is <s>
                log_p = self.log_prob(token)
                print("%s %s"%(token, log_p))


class UnigramModel(Model):

    def __init__(self, infile):
        self.vocab = set()
        self.tokens = []
        self.train(infile)

    def train(self, infile):
        for sent in self.read_corpora(infile):
            for token in sent:
                if token != '<s>':
                    self.tokens.append(token) # No prob estimate for <s>
                    self.vocab.add(token)
        self.vocab.add('UNK')
        self.counter = Counter(self.tokens)
        self.denom = len(self.tokens) + len(self.vocab) # N + V

    def log_prob(self, w):
        word = w if w in self.vocab else 'UNK'
        count = self.counter[word]
        prob = (count+1) / self.denom # C(w)+1/N+V : Laplace Smoothing
        return math.log(prob)

    def test(self, infile):
        for sent in self.read_corpora(infile):
            for token in sent[1:]:    # sent[0] is <s>
                log_p = self.log_prob(token)
                print("%s %s"%(token, log_p))


class BigramModel(Model):

    def __init__(self, infile, k=0.006045):
        self.k = k     # This is the dial for this model
        self.vocab = set()
        self.tokens, self.biTokens = [], []
        self.train(infile)

    def train(self, infile):
        for sent in self.read_corpora(infile):
            for pair in zip(sent, sent[1:]):
                self.biTokens.append(pair)
            for token in sent:
                self.tokens.append(token)       # include <s>
                if token != '<s>':
                    self.vocab.add(token)
        self.vocab.add('UNK')
        self.counter = Counter(self.tokens)     # counts every tokens
        self.biCounter = Counter(self.biTokens) # counts every token pairs
        self.V = len(self.vocab)                # Total no of types

    def log_prob(self, w, given):               # P(Wi|Wi-1)
        word = w if w in self.vocab else 'UNK'
        given = given if (given in self.vocab or given=='<s>') else 'UNK'
        bicount = self.biCounter[(given, word)] # C(Wi-1 Wi) 
        count = self.counter[given]             # C(Wi-1)
        prob = (bicount+self.k) / (count+self.k*self.V) # Add-k Smoothing
        return math.log(prob)

    def test(self, infile):
        for sent in self.read_corpora(infile):
            for word,given in zip(sent[1:], sent): # sent[0] is <s>
                log_p = self.log_prob(word, given)
                print("%s %s"%(word, log_p))


class InterpolatedModel(Model):

    def __init__(self, infile, k=0.006045, l=0.001):
        self.k = k
        self.lmbda = l   # This is the dial for this model
        self.vocab = set()
        self.tokens, self.biTokens = [], []
        self.train(infile)

    def train(self, infile):
        for sent in self.read_corpora(infile):
            for pair in zip(sent, sent[1:]):
                self.biTokens.append(pair)
            for token in sent:
                self.tokens.append(token)       # <s> could be given
                if token != '<s>':
                    self.vocab.add(token)
        self.vocab.add('UNK')
        self.counter = Counter(self.tokens)     # counts every tokens
        self.biCounter = Counter(self.biTokens) # counts every token pairs
        self.V = len(self.vocab)                # Total no of types
        self.N = len(self.tokens)- self.counter['<s>'] #  Total tokens
        self.denom = self.N + self.V

    def log_prob(self, w, given):
        # Unigram-Model
        word = w if w in self.vocab else 'UNK'
        count = self.counter[word]
        uniGram = (count+1) / self.denom # C(w)+1/N+V : Laplace Smoothing
        uniGram = math.log(uniGram)

        # Bigram-Model
        word = w if w in self.vocab else 'UNK'
        given = given if (given in self.vocab or given=='<s>') else 'UNK'
        bicount = self.biCounter[(given, word)] # C(Wi-1 Wi) 
        count = self.counter[given]             # C(Wi-1)
        biGram = (bicount+self.k) / (count+self.k*self.V) # Add-k Smoothing
        biGram = math.log(biGram)

        # Interpolated-Model implementing both Unigram and Bigram
        prob = math.log(((1-self.lmbda)*math.exp(biGram)) + (self.lmbda*math.exp(uniGram)))
        return prob

    def test(self, infile):
        for sent in self.read_corpora(infile):
            for word,given in zip(sent[1:], sent): # sent[0] is <s>
                log_p = self.log_prob(word, given)
                print("%s %s"%(word, log_p))

if __name__ == '__main__':

    Models = {
             '0': UniformModel,
             '1': UnigramModel,
             '2': BigramModel,
        'interp': InterpolatedModel,
     }
   
    M = Models.get(sys.argv[1])(sys.argv[2]) # <model-no> <training-data>
    # M.check()                              # Check Prob cond for the model
    M.test(sys.argv[3])                      # <test-data>

# EOF
