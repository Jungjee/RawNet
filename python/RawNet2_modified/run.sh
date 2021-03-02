#!/bin/bash

######
## 2020.12.03.
## w/ apex
#name="aam_33"
#python main.py -name ${name} \
#  -epoch 160 -bs 400 \
#  -module_name model_RawNet2_rm_cls_head -model_name get_RawNet2 \
#  -m_layers 1 1 1 2 1 2 \
#  -m_nb_filters 128 128 256 256 512 512 \
#  -nb_iter_per_log 50 -lr_decay cosine -early_stop 0 \
#  -load_model 0 -load_model_dir /exp/DNNs \
#  -m_margin 0.1 -m_scale 30 \
#  -augment 1
##eer: 1.91%, current best!!

