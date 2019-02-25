# Natural-Language-Processing-Assignments
University of New Brunswick Fall-2018 CS6765: Natural Language Processing

This Repository contains the python code for the Fall Term Assignments.  
No usage of numpy/nltk in any of the code and developed using Python2.7 (built-in modules)  
sklearn is used only in Assignment3 for Logistic Regression

## Getting started

| No  | Python-file  | Usage
|:-:|:-:|:-:|
| 1  | tokenize.py<br> count.py  | python tokenize.py FILE > FILE.tokens<br> python count.py FILE.tokens > FILE.freqs      
| 2 |  lm.py<br>perplexity.py |  python lm.py MODEL TRAIN_FILE TEST_FILE > OUTPUT<br>python perplexity.py OUTPUT
| 3 | classify.py<br>score.py  | python classify.py METHOD TRAIN_DOCS TRAIN_CLASSES TEST_DOCS > PREDICTED_CLASSES<br> python score.py PREDICTED_CLASSES TRUE_CLASSES
| 4 | tag.py<br>accuracy.py  | python tag.py TRAIN_FILE TEST_FILE METHOD > SYSTEM_OUTPUT<br>python accuracy.py TRUE_TAGS SYSTEM_OUTPUT
| 5 | chatbot.py |  python chatbot.py METHOD  

## Arguments

| No  | Arguments  | File-Location (in Individual Assignment folder)
|:-:|:-:|:-:|
| 1  | FILE | Data/tweets-en.txt.gz      
| 2 |  MODEL<br>TRAIN_FILE<br>TEST_FILE |  <b>1</b> or <b>2</b> or <b>interp</b><br>Data/reuters-train.txt<br>Data/reuters-dev.txt
| 3 | METHOD<br>TRAIN_DOCS<br>TRAIN_CLASSES<br>TEST_FILE<br>TRUE_CLASSES  | <b>baseline</b> or <b>lr</b> or <b>lexicon</b> or <b>nb</b> or <b>nbbin</b><br>Data/train.docs.txt<br>Data/train.classes.txt<br>Data/dev.docs.txt<br>Data/dev.classes.txt
| 4 | TRAIN_FILE<br>TEST_FILE<br>METHOD<br>TRUE_TAGS  |Data/train.en.txt<br>Data/dev.en.words.txt<br><b>baseline</b> or <b>hmm</b><br>Data/dev.en.tags.txt
| 5 | METHOD |  overlap<br>w2v<br>both

Assignment 2: - 
MODEL  
* <b>1</b> represents <b>Unigram (with Add-1 smoothing)</b>
* <b>2</b> represents <b>Bigram (with Add-k smoothing)</b>
* <b>3</b> represents <b>Interpolated (both Unigram and Bigram)</b>

Assignment 3: - 
METHOD  
* <b>baseline</b> represents <b>Most-Frequent-Class-Baseline</b>
* <b>lr</b> represents <b>Logistic Regression (used from skimage)</b>
* <b>lexicon</b> represents <b>Sentiment Lexicon containing + and - words</b>
* <b>nb</b> represents <b>Naive Bayes Model (with add-k smoothing)</b>
* <b>nbbin</b> represents <b>Binarized Naive Bayes</b>

Assignment 4: - 
METHOD 
* <b>baseline</b> represents <b>Most-Frequent-Tag-Baseline</b>
* <b>2</b> represents <b>Hidden Markov Model (Bigram with add-k smoothing) and Viterbi Algorithm</b>

Assignment 5: - 
METHOD  
* <b>overlap</b> represents <b>Chatbot responses based on the word overlap</b>
* <b>w2v</b> represents <b>Response with highest Cosine value (from pre-trained vectors from fastText)</b>
* <b>both</b> represents <b>both responses from overlap and w2v with their Cosine values</b>
