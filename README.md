# Natural-Language-Processing-Assignments

This Repository contains the python code for the Fall Term Assignments.  
No usage of Numpy in any of the code and developed using Python2.7 (built-in modules)

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
| 5 | METHOD |  python chatbot.py METHOD
