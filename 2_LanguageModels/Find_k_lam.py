from __future__ import print_function
import numpy as np
import lm
import os

TrainingData = 'Data/reuters-train.txt' 
TestData = 'Data/reuters-dev.txt'

if False: # Bigram-Model

    File = open('mic-k.csv', 'w')
    model = lm.BigramModel(TrainingData)
    for k in np.arange(0.006, 0.0061, 0.000001):
        model.test(TestData, k)
        per = lm.Model.Perplexity('out.txt')
        File.write('%s,%s\n'%(k,per))
        os.remove('out.txt')
    File.close()

if True: # Interpolated-Model

    File = open('reu-lam.csv', 'w')
    model = lm.InterpolatedModel(TrainingData)
    for lam in np.arange(0.01, 0.1, 0.001):
        model.test(TestData, lam)
        per = lm.Model.Perplexity('out.txt')
        File.write('%s,%s\n'%(lam,per))
        os.remove('out.txt')
    File.close()

os.remove('lm.pyc')

