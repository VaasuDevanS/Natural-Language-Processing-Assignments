import sys

gold_fname = sys.argv[1]
sys_fname = sys.argv[2]

# Read in files, split lines, and ignore blank lines
gold_lines = [x.strip() for x in open(gold_fname) if x.strip()]
sys_lines = [x.strip().split('\t') for x in open(sys_fname) if x.strip()]

# Do some sanity checks
if len(gold_lines) != len(sys_lines):
    sys.exit("Number of lines in system output does not match gold standard")

if not all([len(x) == 2 for x in sys_lines]):
    sys.exit("System output line does not contain 2 columns")

sys_words = [x[0] for x in sys_lines]

gold_tags = gold_lines
sys_tags = [x[1] for x in sys_lines]

num = len([g for g,s in zip(gold_tags, sys_tags) if g == s])
denom = len(gold_tags)
acc = num / float(denom)

print "Accuracy: ", round(acc, 3)

tag_dict = {}
for w,t in zip(sys_words,gold_tags):
    if w in tag_dict:
        tag_dict[w].add(t)
    else:
        tag_dict[w] = set([t])
ambiguous_words = set([w for w in tag_dict if len(tag_dict[w]) > 1])

num = len([g for g,s,w in zip(gold_tags, sys_tags, sys_words) \
           if g == s and w in ambiguous_words])
denom = len([w for w in sys_words if w in ambiguous_words])
acc = num / float(denom)

print "Accuracy (ambiguous tokens N=" + str(denom) + "): ", round(acc, 3)
