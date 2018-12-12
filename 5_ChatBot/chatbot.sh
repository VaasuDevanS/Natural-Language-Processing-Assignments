
if [ $1 -eq 0 ]; then reset; fi

declare -A arr
arr=( ["overlap"]="-------" ["w2v"]="---" )

for m in overlap w2v;
do
    echo -e "\n${m}\n${arr[${m}]}"
    start=`date +%s`
    python chatbot.py ${m} "$2"
    end=`date +%s`
    echo -n "Time taken: "
    echo $((end-start)) | awk '{print int($1%60)"sec"}'
    echo
done

# EOF
