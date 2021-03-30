# Overview
This github project includes PyTorch implementation for reproducing experiments and DNN models used in the paper
[Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms]( https://arxiv.org/pdf/2004.00526.pdf ) 
which is for presentation at Interspeech2020 as a conference paper. 
Trained model is available at 'Pre-trained_model/rawnet2_best_weights.pt' and extracted speaker embeddings are available at *spk_embd/*. 

**For reproduction of the original RawNet paper, please refer to 'RawNet1' folder.**

# Usage

## Environment Setting
We used Nvidia GPU Cloud for conducting our experiments. We used the 'nvcr.io/nvidia/pytorch:19.10-py3' image. Refer to *launch_ngc.sh*. We used two Titan V GPUs for training. 

## Training RawNet2

1. Download VoxCeleb1&2 datasets and move to *DB/*.       
(or just give directories to your DB as arguments using *--DB DIR_TO_VOX1* and *--DB_vox2 DIR_TO_VOX2*)    
Filetree will be added as reference in meantime. 

2. (selectively) Enter virtual environment using NGC. 
3. Run *train_RawNet2.py -name NAME*

##  Evaluating the Trained Model to achieve EER reported in the paper.

1. Go into Pre-trained_model folder. 
2. Download extracted RawNet2 speaker embeddings for the VoxCeleb1 devset [Here]( https://www.dropbox.com/sh/0p9nwrw7hzq0pwm/AACA0W2iE--9uSS85qJ7fahUa?dl=0 )
(Too big to upload in Github)
3. Move downloaded speaker embedding to *spk_embd/*
4. Run *evaluate_pretrained_RawNet2.py*    

## Utilizing Extracted Speaker Embeddings. 
We encourage to use the extracted speaker embeddings for further speaker embedding enhancement studies or back-end studies since RawNet2 paper adopts simple cosine similarity for back-end classification.     

Speaker embeddings are located under *spk_embd/* and are saved using pickle, where it contains a dictionay.    
Key   : Utterance ID (Spk/videoID/segID)
Value : Speaker embedding


##### Email jeewon.leo.jung@gmail.com for other details :-).

# BibTex

This reposity provides the code for reproducing below papers. 
```
@article{jung2020improved,
  title={Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms},
  author={Jung, Jee-weon and Kim, Seung-bin and Shim, Hye-jin and Kim, Ju-ho and Yu, Ha-Jin},
  journal={Proc. Interspeech 2020},
  pages={3583--3587},
  year={2020}
}
```
```
@article{jung2019RawNet,
  title={RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification},
  author={Jung, Jee-weon and Heo, Hee-soo and Kim, ju-ho and Shim, Hye-jin and Yu, Ha-jin},
  journal={Proc. Interspeech 2019},
  pages={1268--1272},
  year={2019}
}
```

# TO-DO
1. Add comments to codes. 

# Log
- 2020.04.01. : Initial commit
- 2020.04.02. : Evaluate Pre-trained Model validated
- 2020.04.02. : Evaluated training
- 2020.07.10. : Add filetree of Datasets
- 2020.07.27. : Revise citation and current status
