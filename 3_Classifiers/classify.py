"""
Assignment #3- Natural Language Processing-CS-6765
--------------------------------------------------
Implemented classifiers
========================
most frequest class baseline : baseline
         logistic regression : lr
  sentiment lexicon baseline : lexicon
                 naive bayes : nb
       binarized naive bayes : nbbin
"""

from __future__ import print_function, division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter, defaultdict
from itertools import izip
from sys import argv as arg
import math
import re

__author__ = "Vaasudevan Srinivasan"
__about__  = "MEngg Geodesy and Geomatics-GGE (Fall term)"
__email__  = "vaasu.devan@unb.ca"
__about__  = "Sentiment Analysis Classifiers"
__date__   = "Oct 30, 2018"

class Classifier(object):

    @staticmethod
    def tokenize(sent):
        tokens = sent.lower().split()
        trimmed_tokens = []
        for t in tokens:
            if re.search('\w', t):
                t = re.sub('^\W*', '', t)
                t = re.sub('\W*$', '', t)
            trimmed_tokens.append(t)
        return tuple(trimmed_tokens)

    def test(self, test_doc):
        with open(test_doc) as file_:
            for sent in file_:
                print(self.predict_class(sent))

    @staticmethod
    def score(system, gold):
        # Computes Macro-average score for the classifier
        predictions = [i.strip() for i in open(system)]
        gold = [i.strip() for i in open(gold)]
        assert len(predictions) == len(gold)
        
        confusion_matrix = Counter(zip(predictions,golds))
        assert sum(confusion_matrix.values()) == len(predictions)

        all_classes = set(predictions).union(golds)
        acc, macro_p, macro_r, macro_f = 0, 0, 0, 0 
        for k in all_classes:
            num = confusion_matrix[(k,k)]
            acc += num
            
            p_denom = sum([confusion_matrix[(k,c)] for c in all_classes])
            p = 0 if p_denom==0 else num/p_denom

            r_denom = sum([confusion_matrix[(c,k)] for c in all_classes])
            r = 0 if r_denom==0 else num/r_denom

            f_denom = p_denom + r_denom
            f = 0 if f_denom==0 else 2*p*r/f_denom # Harmonic-mean

            macro_p += p
            macro_r += r
            macro_f += f
            print(k)
            print("P: %.3f" % p)
            print("R: %.3f" % r)
            print("F: %.3f" % f)

        acc /=  sum(confusion_matrix.values())
        macro_p /= len(all_classes)
        macro_r /= len(all_classes)
        macro_f /= len(all_classes)
        print("Accuracy: %.3f" % acc)
        print("Macro averaged P: %.3f" % macro_p)
        print("Macro averaged R: %.3f" % macro_r)
        print("Macro averaged F: %.3f" % macro_f)


class Baseline(Classifier):
    """
    Baseline predicts the class only with the
    most frequent class.
    """
    def __init__(self, train_doc, train_class):
        self.train(train_doc, train_class)

    def train(self, train_doc, train_class):
        with open(train_class) as f:
            counts = Counter(f)
        self.mfc = counts.most_common(1)[0][0].strip()

    def predict_class(self, sent):
        # returns the most frequent class
        return self.mfc


class LogisticRegression_(Classifier):

    def __init__(self, train_doc, train_class):
        self.train(train_doc, train_class)

    def train(self, train_doc, train_class):
        # read the files
        train_texts = [x.strip() for x in open(train_doc)]
        train_classes = [x.strip() for x in open(train_class)]

        self.count_vectorizer = CountVectorizer(analyzer=self.tokenize)
        train_counts = self.count_vectorizer.fit_transform(train_texts)
        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self.clf = lr.fit(train_counts, train_classes)

    def test(self, test_doc):
        test_texts = [x.strip() for x in open(test_doc)]
        test_counts = self.count_vectorizer.transform(test_texts)
        results = self.clf.predict(test_counts)
        for r in results:
            print(r)


class SentimentLexicon(Classifier):
    """
    https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
    This classifier predicts from the positive and negative 
    words compiled in the Opinion Lexicon by Bing Liu.
    """
    def __init__(self, train_doc, train_class):
        self.train(train_doc, train_class)

    def train(self, train_doc, train_class):
        self.pos = [i.strip() for i in open("Data/pos-words.txt")]
        self.neg = [i.strip() for i in open("Data/neg-words.txt")]

    def predict_class(self, sent):
        pos, neg = 0, 0
        for token in self.tokenize(sent):
            pos += token in self.pos # True:1 False:0
            neg += token in self.neg
        return [("positive", "negative")[neg>pos], "neutral"][pos==neg]


class NaiveBayes(Classifier):
    
    def __init__(self, train_doc, train_class):
        self.cls_tokens = defaultdict(list)
        self.cls_counter = {}
        self.N_cls = {}
        self.train(train_doc, train_class)

    def train(self, train_doc, train_class):
        classes = []
        self.types = set()
        with open(train_class) as f1, open(train_doc) as f2:
            for cls,doc in izip(f1, f2):
                cls = cls.strip()
                classes.append(cls)
                for t in self.tokenize(doc):
                    self.types.add(t)
                    self.cls_tokens[cls].append(t)
        self.counts = Counter(classes)
        self.total = sum(self.counts.values()) # Total classes
        self.V = len(self.types)               # Total types
        for cls,tokens in self.cls_tokens.iteritems():
            self.cls_counter[cls] = Counter(tokens)
            self.N_cls[cls] = len(tokens)

    def predict_class(self, sent):
        Prob = {}
        for cls,counter in self.cls_counter.iteritems():
            Prob[cls] = 0
            for token in self.tokenize(sent):
                if token not in self.types:
                    continue
                t_count = counter[token]
                Prob[cls] += math.log((t_count+1)/(self.N_cls[cls]+self.V))
            Prob[cls] += math.log(self.counts[cls]/self.total)
        return max(Prob, key=lambda k: Prob[k])
    

class BinarizedNaiveBayes(NaiveBayes):

    def __init__(self, train_doc, train_class):
        NaiveBayes.__init__(self, train_doc, train_class)

    def train(self, train_doc, train_class):
        classes = []
        self.types = set()
        with open(train_class) as f1, open(train_doc) as f2:
            for cls,doc in izip(f1, f2):
                cls = cls.strip()
                classes.append(cls)
                for t in set(self.tokenize(doc)):
                    self.types.add(t)
                    self.cls_tokens[cls].append(t)
        self.counts = Counter(classes)
        self.total = sum(self.counts.values()) # Total classes
        self.V = len(self.types)               # Total types
        for cls,bin_tokens in self.cls_tokens.iteritems():
            self.cls_counter[cls] = Counter(bin_tokens)
            self.N_cls[cls] = len(bin_tokens)
        
        def predict_class(self, sent):
        Prob = {}
        for cls,counter in self.cls_counter.iteritems():
            Prob[cls] = 0
            for token in set(self.tokenize(sent)):
                if token not in self.types:
                    continue
                t_count = counter[token]
                Prob[cls] += math.log((t_count+1)/(self.N_cls[cls]+self.V))
            Prob[cls] += math.log(self.counts[cls]/self.total)
        return max(Prob, key=lambda k: Prob[k])


if __name__ == '__main__':

    classifiers = {
        'baseline': Baseline,
              'lr': LogisticRegression_,
         'lexicon': SentimentLexicon,
              'nb': NaiveBayes,
           'nbbin': BinarizedNaiveBayes,
    }
    # *arg -> method train-doc train-class test-doc
    c = classifiers.get(arg[1])(arg[2], arg[3])
    c.test(arg[4])

# EOF
