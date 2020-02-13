#!/bin/bash
#Uskerim

mkdir -p -- "output"

#dataset="imagenet"
dataset="imagenet"
freq="1000"
steps="300"
noise="0.25"
duplicate="True"
tv_lam="0.3"
c_lam="20.0"
s_lam="5.0"
batch="1"
export IMAGENET_DIR="/scratch0/aminjun/ImageNet/"
#export IMAGENET_DIR="/home/alishafahi/datasets/ImageNet/"
if [ "${dataset}" == "cifar10" ] ; then
    arch="resnet110"
else
    arch="resnet50"
fi
 
job_name="`date`:${data_set}_${noise}_${tv_lam}_${c_lam}_${steps}_${batch}_${s_lam}"
CUDA_VISIBLE_DEVICES=0,1 python  "code/certify.py" "${dataset}" "models/${dataset}/${arch}/noise_${noise}/checkpoint.pth.tar" "${noise}" \
        "output/${job_name}" --skip "${freq}" -s "${steps}" -d "${duplicate}" --tv_lam "${tv_lam}" \
	 --c_lam "${c_lam}" --batch "${batch}" --s_lam "${s_lam}"  #> "${job_name}.txt" &
