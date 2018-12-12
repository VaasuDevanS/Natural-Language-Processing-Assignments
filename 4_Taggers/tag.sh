
if [ $1 -eq 1 ]; then reset; fi;
echo -e "\nAccuracy\n========"
declare -A arr
arr=( ["baseline"]="-----------" ["hmm"]="------" )

for l in en fr; # languages {'en': 'English', 'fr': 'French'}
do
    for m in baseline hmm; # Taggers {'hmm': 'Hidden Markov Model'}
    do
        # Variables
        echo -e "\n${m}-${l}\n${arr[${m}]}"
        train="Data/train.${l}.txt"
        dev="Data/dev.${l}.words.txt"
        dev_c="Data/dev.${l}.tags.txt"
        out=${l}_${m}.out
        # echo ${train} ${dev} ${dev_c} ${out}

        t=`(time python tag.py ${train} ${dev} ${m} > ${out})2>&1 | grep real`
        ttaken=`echo ${t} | cut -d' ' -f2`
        echo "Time taken: ${ttaken}"
        python Data/acc.py ${dev_c} ${out} 
        # start, stop_=`date +%s`
        # echo $((stop_-start)) | awk '{print int($1/60)"m:"int($1%60)"s"}'
        # \time -f "%e"
        # break
    done
done
echo
rm *.out
