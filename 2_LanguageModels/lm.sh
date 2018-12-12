#!/usr/bin/env bash

touch sam.out
rm *.out

mic_train='Data/mic-train.txt'
mic_test='Data/mic-dev.txt'

reu_train='Data/reuters-train.txt'
reu_test='Data/reuters-dev.txt'

echo -e "\nPerplexities\n------------"
for arg in 0 1 2 interp;
do
    # Output names
    mic_out=mic.${arg}.out;
    reu_out=reu.${arg}.out;

    # Create Models
    python lm.py ${arg} ${mic_train} ${mic_test} > ${mic_out};
    python lm.py ${arg} ${reu_train} ${reu_test} > ${reu_out};
    
    # if [ ${arg} -eq 0 ]; then echo Perplexity; fi;

    # Print Perplexity
    echo ${mic_out}: `python -c "import lm;lm.Model.Perplexity('${mic_out}')"`
    echo ${reu_out}: `python -c "import lm;lm.Model.Perplexity('${reu_out}')"`
    echo
done

# remove tmp file
rm lm.pyc
rm *.out

