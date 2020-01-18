#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python eval.py --config config/cifar_crown.json           --path_prefix crown-ibp_models/cifar_crown_0.03137  
CUDA_VISIBLE_DEVICES=0 python eval.py --config config/cifar_crown.json           --path_prefix crown-ibp_models/cifar_crown_0.03137       > "c8s`date`.txt" & 
CUDA_VISIBLE_DEVICES=1 python eval.py --config config/cifar_crown_large.json     --path_prefix crown-ibp_models/cifar_crown_large_0.03137 > "c8l`date`.txt" 
CUDA_VISIBLE_DEVICES=0 python eval.py --config config/cifar_crown_eps.json       --path_prefix crown-ibp_models/cifar_crown_0.00784       > "c2s`date`.txt" & 
CUDA_VISIBLE_DEVICES=1 python eval.py --config config/cifar_crown_large_eps.json --path_prefix crown-ibp_models/cifar_crown_large_0.00784 > "s2l`date`.txt'" 
#sleep 1 
#nvidia-smi
