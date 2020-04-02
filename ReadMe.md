# Notice
This repository is under update to reflect codes for reproducing our recent paper. 
For reproduction of the original RawNet paper, please refer to `RawNet1' folder.

# Overview
This github project includes PyTorch implementation for reproducing experiments and DNN models used in the paper
[Improved RawNet with Filter-wise Re-scaling for Text-independent Speaker Verification using Raw Waveforms]( https://arxiv.org/pdf/2004.00526.pdf ) 
which is submitted to Interspeech2020 as a conference paper. 
Pre-trained model is available at 'Pre-trained_model/rawnet2_best_weights.pt' and extracted speaker embeddings are available at 'spk_embd/'. 

# Usage

## Environment Setting
We used Nvidia GPU Cloud for conducting our experiments. We used the 'nvcr.io/nvidia/pytorch:19.10-py3' image. Refer to 'launch_ngc.sh'. We used two Titan V GPUs for training. 

## Training RawNet2

1. (selectively) Enter virtual environment using NGC. 
2. Run 'train_RawNet2.py -name NAME'

##  Evaluate Pre-trained Model

1. Go into Pre-trained_model folder. 
2. Download extracted RawNet2 speaker embeddings for the VoxCeleb1 devset [Here]( https://www.dropbox.com/s/2y4k5rap8cztcrf/TTA_vox1_dev.pk?dl=0 )
(Too big to upload in Github)

3. Run 'evaluate_pretrained_RawNet2.py'





##### Email jeewon.leo.jung@gmail.com for other details :-).

# Citation

TBA

```
@article{jung2019RawNet,
  title={RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification},
  author={Jung, Jee-weon and Heo, Hee-soo and Kim, ju-ho and Shim, Hye-jin and Yu, Ha-jin},
  journal={Proc. Interspeech 2019},
  pages={1268--1272},
  year={2019}
}
```


# Log
- 2020.04.01. : initial commit
