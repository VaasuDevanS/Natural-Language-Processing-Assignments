"""
Assignment #5- Natural Language Processing-CS-6765
--------------------------------------------------
Implemented chat bots
=====================
Overlap-based method  : overlap
Word embeddings-w2v   : w2v
"""

from __future__ import print_function, division
from collections import defaultdict as ddict
from math import log, exp, sqrt
from sys import argv as arg
from itertools import izip
import re

__author__ = "Vaasudevan Srinivasan"
__about__  = "MEngg Geodesy and Geomatics-GGE (Fall term)"
__email__  = "vaasu.devan@unb.ca"
__about__  = "Chatbot"
__date__   = "Dec 4, 2018"


class Chatbot(object):

    @staticmethod
    def tokenize(s):
        tokens = s.lower().split()
        trimmed_tokens = []
        for t in tokens:
            if re.search('\w', t):
                t = re.sub('^\W*', '', t) # Leading non-alphanumeric chars
                t = re.sub('\W*$', '', t) # Trailing non-alphanumeric chars
            trimmed_tokens.append(t)
        return tuple(trimmed_tokens)


class Overlap(Chatbot):
    """
    This method of chatbot replies from the actual
    tweets. It replies based on the Most similar
    Response.
    """
    def __init__(self, responses, vectors=None):
        # This reads the entire responses and tokenizes it
        self.responses_types = []
        with open(responses) as f:
            for res in f:
                self.responses_types.append((res, set(self.tokenize(res))))

    def reply(self, query):
        q_tokenized = self.tokenize(query)
        max_sim = 0
        max_resp = "Sorry, I don't understand"
        for res,r_tokenized_set in self.responses_types:
            sim = len(r_tokenized_set.intersection(q_tokenized))
            if sim > max_sim:
                max_sim, max_resp = sim, res
        return max_resp


class W2V(Chatbot):
    """
    This method of chatbot chooses reply from the
    tweets based on cosine value between all the
    normalized res vectors and normalized query vector
    """
    def __init__(self, responses, vectors):
        self.type_vectors = ddict(int) # FastText vector file
        self.res_nvec = {}             # Responses-Normalised vectors
        self.load_vectors(vectors)
        self.normalize_responses(responses)
    
    def load_vectors(self, fname):
        # Modified version of load_vectors from
        # https://fasttext.cc/docs/en/crawl-vectors.html
        for line in open(fname):
            tkns = line.rstrip().strip().split(' ')
            self.type_vectors[tkns[0]] = tuple((float(i) for i in tkns[1:]))

    def mag(self, vec):
        return sqrt(sum((x*x for x in vec)))

    def sum_vectors(self, vecs):
        return tuple((sum(i) for i in izip(*vecs)))

    def mul_vectors(self, vec1, vec2):
        return sum((i*j for i,j in izip(vec1,vec2)))

    def div_vectors(self, vec, denom):
        return tuple((i/denom for i in vec))

    def normalize_doc(self, doc):
        # Normalizes each doc tokens from FastText-type vectors
        tok_vec = []
        n = 0
        for tok in self.tokenize(doc):
            vec = self.type_vectors[tok]
            n += 1
            if vec != 0:
                mag = self.mag(vec) # magnitude is the length
                nvec = self.div_vectors(vec, mag)
                tok_vec.append(nvec)
        if len(tok_vec) != 0:
            return self.div_vectors(self.sum_vectors(tok_vec), n)
        else:
            return False

    def normalize_responses(self, responses):
        # Normalize all the responses (tweets) and store in res_vec
        with open(responses) as f:
            for res in f:
                n_doc = self.normalize_doc(res)
                if n_doc:
                    self.res_nvec[res] = n_doc

    def cosine(self, res_vec, query_vec):
        numer = self.mul_vectors(res_vec, query_vec)
        denom = self.mag(res_vec) * self.mag(query_vec)
        cos = numer / denom
        assert -1.0 <= round(cos,2) <= 1.0, "Cosine value error..!!"
        return cos

    def reply(self, query):
        qVec = self.normalize_doc(query)
        if not qVec:
            return "Sorry, I don't understand", 0.0
        reply = {}
        for res,nvec in self.res_nvec.iteritems():
            reply[res] = self.cosine(nvec, qVec)
        return max(reply, key=lambda x:reply[x]), max(reply.values())


if __name__ == "__main__":

    # python chatbot.py METHOD
    responses, vectors = ("Data/tweets-en-filtered-sample.txt",
                          "Data/cc.en.300.vec.10k")

    if arg[1] == "both":
        # Initialize both the bots
        overlap = Overlap(responses, vectors)
        w2v = W2V(responses, vectors)
        # Start interacting with the user
        print("\nBot: Hi! Let's chat\n")
        while True:
            query = raw_input("YOU: ")
            # Get cosine between query and reply for the overlap bot
            oreply = overlap.reply(query)
            vq, vr = w2v.normalize_doc(query), w2v.normalize_doc(oreply)
            ocos = w2v.cosine(vr, vq) if vq else 0.0
            print("Overlap: %s -(%s)" % (oreply.strip(), round(ocos,3)))

            w2vreply, w2vcos = w2v.reply(query)
            print("W2V: %s -(%s)" % (w2vreply.strip(), round(w2vcos,3)))
            print()

    elif arg[1] == "overlap":
        # Overlap method
        bot = Overlap(responses, vectors)
        print("\nOverlap: Hi! Let's chat")
        while True:
            query = raw_input("    YOU: ")
            reply = bot.reply(query)
            print("%s: %s" % (arg[1].capitalize(), reply.strip()))

    elif arg[1] == "w2v":
        # Word2Vec method
        bot = W2V(responses, vectors)
        print("\nw2v: Hi! Let's chat")
        while True:
            query = raw_input("YOU: ")
            reply, cos = bot.reply(query)
            print("%s: %s" % (arg[1].capitalize(), reply.strip()))
    print()

# EOF
