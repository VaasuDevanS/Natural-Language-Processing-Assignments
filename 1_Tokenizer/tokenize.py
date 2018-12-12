"""
Assignment 1 - Natural Language Processing
------------------------------------------
Developer: Vaasudevan Srinivasan, GGE

Program to find the tokens based on below rules
* Any sequence of alphanumeric characters, underscores, hyphens, or
  apostrophes, that optionally begins with # or @, is a token
* Any sequence of other non-whitespace characters is a token
* Any sequence of whitespace characters is a token boundary
"""
from __future__ import print_function # Python 2.7.x and Python 3.x
import gzip                           # Reading compressed text files
import sys                            # Reading command-line arguments
import re                             # Regular Expression

# gzip.open for compressed files like .gz or normal open
Open = gzip.open if sys.argv[1].endswith('.gz') else open

# `with` statement to close the file automatically after the end
with Open(sys.argv[1]) as f:
    for line in f:
        # re.findall    : To find everyting that matches the regex
        # [#@]?         : Optionally if starts with # or @
        # [\w\-']+      : Match sequence of [a-zA-Z0-9_] Hyphens Apostrophes
        # [^#@\w\-'\s]+ : Not match # or @ or [a-zA-Z0-9_] or - or ' or space
        tokens = re.findall(r"(?:[#@]?[\w\-']+|[^#@\w\-'\s]+)", str(line))
        for t in tokens:
            print(t)    # Prints the matched tokens line by line
        print()         # Prints a new line for maintaining 'vertical' format

# EOF