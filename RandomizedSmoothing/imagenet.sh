#!/bin/bash
#Uskerim

mkdir -p -- "outputs"

dataset="imagenet"
export IMAGENET_DIR="PUT Directory Address for ImageNet dataset here!"
skip="1000"
arch="resnet50"

job_id=$(cat utils/exp_stat.db | tr -d '\n ')
gpus="0,1,2,3"


for noise in '0.25', '0.5', '1.0'; do
    job_name="${job_id}_${noise}_`date`.txt"
    CUDA_VISIBLE_DEVICES="${gpus}" python  "certify.py" --dataset "${dataset}" \
        --checkpoint "models/${dataset}/${arch}/noise_${noise}/checkpoint.pth.tar" \
        --sigma "${noise}"  --skip "${skip}"  #> "outputs/${job_name}.txt" &
    real_id="$!"
    wait "${real_id}"

done
