
eval_folder_root="./experiments/mnist/"

for dir in ${eval_folder_root}/*/*/
do
    echo "${dir}"
    python evaluation.py --folder ${dir}
done

