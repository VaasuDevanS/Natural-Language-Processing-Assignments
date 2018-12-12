"""
Assignment 1 - Natural Language Processing
------------------------------------------
Developer: Vaasudevan Srinivasan, GGE

Program to find the count the tokens based on
the output produced by tokenize.py
"""

from __future__ import print_function   # Python 2.7.x and Python 3.x
from collections import defaultdict     # dictionaries with default values
import sys                              # Reading command-line arguments
import re                               # Regular Expression

Count = defaultdict(int)                # Initialise new key's value as int
with open(sys.argv[1]) as f:
    for word in f:
        if word != '\n':                # To eliminate any new lines
            word = word.strip().lower() # Every token to lowercase
            Count[word] += 1            # Increment every word's count by 1

# Sorted  : to sort the dictionary
# key     : based on what - In this case by the key's value
# reverse : non-ascending order
for word in sorted(Count, key=lambda i:Count[i], reverse=True):
    print(word, Count[word])

# EOF