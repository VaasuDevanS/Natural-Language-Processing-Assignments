def score(predictions, golds):
    def my_round(x):
        return '%.3f' % x

    assert len(predictions) == len(golds)

    confusion_matrix = {}
    for p,g in zip(predictions, golds):
        confusion_matrix[(p,g)] = confusion_matrix.get((p,g), 0) + 1

    assert sum(confusion_matrix.values()) == len(predictions)

    all_classes = set(predictions).union(golds)
    acc = 0
    macro_p = 0
    macro_r = 0
    macro_f = 0
    for k in all_classes:
        num = confusion_matrix.get((k,k), 0)
        acc += num

        p_denom = sum([confusion_matrix.get((k,x), 0) for x in all_classes])
        if p_denom == 0:
            # print "WARNING: P undefined: Setting P to 0"
            p = 0
        else:
            p = num / float(p_denom)

        r_denom = sum([confusion_matrix.get((x,k), 0) for x in all_classes])
        if r_denom == 0:
            # print "WARNING: R undefined: Setting R to 0"
            r = 0
        else:
            r = num / float(r_denom)
            
        f_denom = p + r
        if f_denom == 0:
            # print "WARNING: F undefined: Setting F to 0"
            f = 0
        else:
            f = 2 * p * r / float(p + r)

        macro_p += p
        macro_r += r
        macro_f += f
        # """  
        print k
        print "P:", my_round(p)
        print "R:", my_round(r)
        print "F:", my_round(f)
        print
        # """
    acc = acc / float(sum(confusion_matrix.values()))
    macro_p = macro_p / float(len(all_classes))
    macro_r = macro_r / float(len(all_classes))
    macro_f = macro_f / float(len(all_classes))
    print "Accuracy:", my_round(acc)
    print "Macro averaged P:", my_round(macro_p)
    print "Macro averaged R:", my_round(macro_r)
    print "Macro averaged F:", my_round(macro_f)


if __name__ == '__main__':
    import sys

    predictions_fname = sys.argv[1]
    gold_standard_fname = sys.argv[2]

    predictions = [x.strip() for x in open(predictions_fname)]
    golds = [x.strip() for x in open(gold_standard_fname)]

    score(predictions, golds)
