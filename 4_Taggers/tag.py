"""
Assignment #4- Natural Language Processing-CS-6765
--------------------------------------------------
Implemented POS taggers
========================
most frequest tag-for-word baseline : baseline
Hidden Markov Model                 : hmm
"""

from __future__ import print_function, division
from collections import Counter, defaultdict as ddict
from sys import argv as arg, version_info
from math import log, exp
import itertools

__author__ = "Vaasudevan Srinivasan"
__about__  = "MEngg Geodesy and Geomatics-GGE (Fall term)"
__email__  = "vaasu.devan@unb.ca"
__about__  = "Part of Speech (POS) taggers"
__date__   = "Nov 20, 2018"


class Taggers(object):

    @staticmethod
    def sent_from_file(fname, tags=False, sep='\t'):
        sents, curr_tokens = [], []
        with open(fname) as f:
            for line in f:
                if line.strip():
                    line = [x.strip() for x in line.split(sep)]
                    token = (line[0], line[1]) if tags else line[0]
                    curr_tokens.append(token)
                else:
                    sents.append(curr_tokens)
                    curr_tokens = []
        return sents

    @staticmethod
    def accuracy(gold_file, tag_file):
        """
        Calculates the accuracy of the tagger
        """
        with open(gold_file) as g, open(tag_file) as t:

             gold_lines = [x.strip() for x in g if x.strip()]
             sys_lines = [x.strip().split('\t') for x in t if x.strip()]

        sys_words = [x[0] for x in sys_lines]
        sys_tags = [x[1] for x in sys_lines]
        gold_tags = gold_lines

        # Accuracy
        num = len([g for g,s in zip(gold_tags, sys_tags) if g == s])
        denom = len(gold_tags)
        acc = num / denom
        print("Accuracy: ", round(acc, 3))
        
        # Ambiguous words accuracy (words with more than one tag)
        tag_dict = ddict(set)
        for w,t in zip(sys_words,gold_tags):
            tag_dict[w].add(t)
        ambig_words = set([w for w in tag_dict if len(tag_dict[w]) > 1])
        zipped = zip(gold_tags, sys_tags, sys_words)
        num = len([g for g,s,w in zipped if g == s and w in ambig_words])
        denom = len([w for w in sys_words if w in ambig_words])
        acc = num / denom
        print("Accuracy (ambiguous tokens N=%s): %s" % (denom, round(acc, 3)))


class Baseline(Taggers):

    def __init__(self, train_file):

        self.mft_for_word = {}
        self.train(train_file)

    def train(self, train_file):
        """
        Most frequent tag for the word or the mft of the doc
        """
        # Compute (word: tags counts) and tag counts
        sents = self.sent_from_file(train_file, tags=True)
        word_tag_freq = ddict(lambda: ddict(int)) # d[w][t] = 0
        tag_freq = ddict(int) # d[t] = 0
        for sent in sents:
            for word_tag in sent:
                word, tag = word_tag
                word_tag_freq[word][tag] += 1
                tag_freq[tag] += 1

        # Most frequent tag for each word in training document
        for word,tags in word_tag_freq.iteritems():
            self.mft_for_word[word] = max(tags, key=lambda t:tags[t])

        # Most Frequent tag in the entire training document
        self.mft = max(tag_freq, key=lambda t:tag_freq[t])

    def tag(self, test_file, sep='\t'):

        sents = self.sent_from_file(test_file, tags=False)
        for sent in sents:
            for word in sent:
                print(word, self.mft_for_word.get(word, self.mft), sep=sep)
            print()


class HiddenMarkovModel(Taggers):

    def __init__(self, train_file, kti=0.95, ke=0.05, sep='\t'):

        self.kti = kti # k for Transition and Initial probs
        self.ke = ke   # k for Emission prob
        self.tag_tag_matrix = ddict(dict)  # A
        self.word_tag_matrix = ddict(dict) # B
        self.tag_set, self.word_set = set(), {'UNK'}
        self.sep = sep
        self.train(train_file)

    def check_prob(self, matrix):

        for t in self.tag_set:
           # print(t, sum([math.exp(matrix[k][t]) for k in matrix]))
           assert round(sum([exp(matrix[k][t]) for k in matrix]),2) == 1.0

    def train(self, train_file):
        """
        A: Transition Probability (Bigram Model)
        B: Emission Probability
        
        use this to view the matrix artistically
        print(__import__('pandas').DataFrame(matrix))
        """
        tags, word_tag = ["<s>"], []

        # Read the training_file; Dump words and tags
        with open(train_file) as f:
            for line in f:
                if line.strip():
                    w,t = (x.strip() for x in line.split(self.sep))
                    word_tag.append((t,w))
                    tags.append(t)
                    self.tag_set.add(t)
                    self.word_set.add(w)
                else:
                    tags.extend(("</s>", "<s>"))
        tags = tags[:-1] if tags[-1]=="<s>" else tags+["<s>"]

        # Counter class for tag-tag and word-tag
        izip = zip if version_info.major == 3 else itertools.izip
        ttCounter = Counter(izip(tags, tags[1:])) # tag-tag
        wtCounter = Counter(word_tag)             # word-tag
        tCounter = Counter(tags)                  # tag

        # Transition & Initial matrix with add-k smoothing
        tag_space = ['<s>'] + list(self.tag_set)
        kV = self.kti * len(self.tag_set) 
        for t1 in sorted(self.tag_set): # tag
            for t2 in tag_space:        # tag-1
                numer = ttCounter[(t2,t1)] + self.kti
                denom = tCounter[t2] - ttCounter[(t2,"</s>")] + kV
                self.tag_tag_matrix[t1][t2] = log(numer/denom)
        self.check_prob(self.tag_tag_matrix)

        # Emission matrix with add-k smoothing
        kW = self.ke * len(self.word_set)
        for w in self.word_set:             # word
            for t in self.tag_set:          # tag
                numer = wtCounter[(t,w)] + self.ke
                denom = tCounter[t] + kW
                self.word_tag_matrix[w][t] = log(numer/denom)
        self.check_prob(self.word_tag_matrix)

    def tag(self, test_file, sep='\t'):
        """
        Tagger based on Viterbi Algorithm
        """
        ttMat = self.tag_tag_matrix
        wtMat = self.word_tag_matrix
        tSet = sorted(self.tag_set)
        tag_prod = tuple(itertools.product(tSet, tSet))

        for sent in self.sent_from_file(test_file, tags=False):

            predicted_wt = []
            for ix,word in enumerate(sent, start=1):
                w = word if word in self.word_set else "UNK"

                # Initial state
                if ix == 1:
                    prob = {t: ttMat[t]["<s>"]+wtMat[w][t] for t in tSet}

                # Other states
                else:
                    s = ddict(dict)
                    for t1,t2 in tag_prod:
                        s[t1][t2] = prob[t1] + ttMat[t2][t1] + wtMat[w][t2]
                    prob = {t: max([s[k][t] for k in tSet]) for t in tSet}

                # Predict tag which has max prob for the word w
                T = max(prob, key=lambda x: prob[x])
                predicted_wt.append((word, T))

            # Write the predicted tags for the sent to stdout
            for w, t in predicted_wt:
                print(w, t, sep=sep)
            print()


if __name__ == '__main__':

    taggers = {
        'baseline': Baseline,
             'hmm': HiddenMarkovModel
    }
    # args: tag.py TRAIN TEST METHOD
    POStag = taggers.get(arg[3])(arg[1])
    POStag.tag(arg[2])

# EOF
