#!/bin/bash
#Uskerim

mkdir -p -- "outputs"

dataset="cifar10"
skip="20"
arch="resnet110"
job_id=$(cat utils/exp_stat.db | tr -d ' \n')


cnt=0
for noise in '0.12' '0.25' '0.50' '1.00' ; do
    job_name="${job_id}_${noise}_`date`.txt"
    CUDA_VISIBLE_DEVICES="${cnt}" python  "certify.py" --dataset "${dataset}" \
        --checkpoint "models/${dataset}/${arch}/noise_${noise}/checkpoint.pth.tar" \
        --sigma "${noise}" --skip "${skip}"  > "outputs/${job_name}.txt" &
	cnt=$((cnt+1))
	sleep 1
	echo "${cnt}"
done
