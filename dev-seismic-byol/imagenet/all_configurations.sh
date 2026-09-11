#!/bin/bash

#full:
version=(modern default)
repetition=(0 1 2)
red_mode=full
task=PretrainEval
for v in "${version[@]}"; do
for r in "${repetition[@]}"; do
    FLAGS=" --reduction_mode ${red_mode} --version ${v} --repetition ${r}"
    ROOT=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/${task}/${v}/repetition_${r}/${red_mode}/
    mkdir -p ${ROOT}

#default
nrepetition=(0 1 2)
version=(modern traditional)
num_classes=(1000 500 477 250 200 125 80 9)
red_mode=default
task=PretrainEval

for v in "${version[@]}"; do
for r in "${repetition[@]}"; do
for n in "${num_classes[@]}"; do

    if [ ${n} -eq 1000 ]; then
        per_class=(640 320 160 80 40 20)

    elif [ ${n} -eq 500 ]; then
        per_class=(1300 640 320 160 80 40)

    elif [ ${n} -eq 250 ]; then
        per_class=(1300 640 320 160 80)
    
    elif [ ${n} -eq 125 ]; then
        per_class=(1300 640 320 160)
    else
        per_class=(1300)

    fi

    for p in ${per_class[@]}; do

    FLAGS="--reduction_mode ${red_mode} --version ${v} --per_class ${p} --num_classes ${n} --repetition ${r} --eval_acc"
    ROOT=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/${task}/${v}/repetition_${r}/${red_mode}/${n}
    mkdir -p ${ROOT}
    end
end
end
end

#taxonomic
repetition=(0 1 2)
level=(9 7 6 3)
red_mode=taxonomic
version=(modern traditional)
task=PretrainEval
for v in "${version[@]}"; do
for r in "${repetition[@]}"; do
for l in "${level[@]}"; do

    FLAGS="--reduction_mode ${red_mode} --version ${v} --top_down --level ${l} --repetition ${r} --eval_acc"
    ROOT=/petrobr/parceirosbr/home/joao.frare/workspace/spfm/Seismic-Byol/dev-seismic-byol/imagenet/jobs_out/${task}/${v}/repetition_${r}/${red_mode}/
    mkdir -p ${ROOT}

end
end
end