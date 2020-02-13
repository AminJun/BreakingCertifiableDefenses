#!/bin/bash
#Uskerim

mkdir -p -- "output"

#dataset="imagenet"
dataset="cifar10"
freq="20"
steps="300"
noise="0.25"
duplicate="True"
tv_lam="0.3"
c_lam="20.0"
s_lam="5.0"
batch="10000"
export IMAGENET_DIR="/scratch0/aminjun/ImageNet/"
#export IMAGENET_DIR="/home/alishafahi/datasets/ImageNet/"
if [ "${dataset}" == "cifar10" ] ; then
    arch="resnet110"
else
    arch="resnet50"
fi
 
cnt=0
for noise in '0.12' '0.25' '0.50' '1.00' ; do 
job_name="`date`_${dataset}_${noise}"
CUDA_VISIBLE_DEVICES="${cnt}" python  "code/certify.py" "${dataset}" "models/${dataset}/${arch}/noise_${noise}/checkpoint.pth.tar" "${noise}" \
        "output/${job_name}" --skip "${freq}" -s "${steps}" -d "${duplicate}" --tv_lam "${tv_lam}" \
	 --c_lam "${c_lam}" --batch "${batch}" --s_lam "${s_lam}"  > "${job_name}.txt" &
	cnt=$((cnt+1))
	echo $cnt
done
