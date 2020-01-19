#!/bin/bash
if [ ! -e "crown-ibp_models" ] ; then 
	wget "https://download.huan-zhang.com/models/crown-ibp/models_crown-ibp.tar.gz"; 
	tar -zxf "models_crown-ibp.tar.gz"
	rm "models_crown-ibp.tar.gz"
fi
# CUDA_VISIBLE_DEVICES=0 python eval.py --config config/cifar_crown_eps.json       --path_prefix crown-ibp_models/cifar_crown_0.00784  
CUDA_VISIBLE_DEVICES=1 python eval.py --config config/cifar_crown.json           --path_prefix crown-ibp_models/cifar_crown_0.03137       > "c8s`date`.txt" & 
CUDA_VISIBLE_DEVICES=2 python eval.py --config config/cifar_crown_large.json     --path_prefix crown-ibp_models/cifar_crown_large_0.03137 > "c8l`date`.txt" &
CUDA_VISIBLE_DEVICES=3 python eval.py --config config/cifar_crown_eps.json       --path_prefix crown-ibp_models/cifar_crown_0.00784       > "c2s`date`.txt" & 
CUDA_VISIBLE_DEVICES=1 python eval.py --config config/cifar_crown_large_eps.json --path_prefix crown-ibp_models/cifar_crown_large_0.00784 > "c2l`date`.txt'" &
