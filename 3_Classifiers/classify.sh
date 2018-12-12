
if [ $1 -eq 1 ]; then reset; fi;

train_data='Data/train.docs.txt'
train_class='Data/train.classes.txt'
test_data='Data/dev.docs.txt'
test_class='Data/dev.classes.txt'

# Declaring an dict-array
declare -A arr
arr=( ["baseline"]="--------" ["lr"]="---" )
arr+=( ["lexicon"]="-------" ["nb"]="---" ["nbbin"]="------" )

echo -e '\nScore\n=====\n'

for i in baseline lr lexicon nb nbbin;
do
    start=$(date +%s)
    echo -e "${i}\n${arr[${i}]}"
    out=${i}.out
    python -W ignore classify.py ${i} ${train_data} ${train_class} ${test_data} > ${out}
    end=$(date +%s)
    python Data/score.py ${out} ${test_class}
    echo
    # echo -n "Time taken-> " 
    # echo $((end-start)) | awk '{print int($1/60)"m:"int($1%60)"s\n"}'
done

rm *.out
