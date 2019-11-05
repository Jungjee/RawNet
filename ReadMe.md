# Overview

This github project includes codes for reproducing experiments and DNN models used in the paper
[RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification]( https://isca-speech.org/archive/Interspeech_2019/pdfs/1982.pdf ) 
which was presented at Interspeech2019 as a conference paper.
For following the implementation of the paper, refer to "Keras" folder. 
"PyTorch" folder contains scripts using VoxCeleb2 dataset with a few modifications (baseline is uploaded currently).
	

#  Reproduction of the system in the paper
	1. Script 'Keras/lunch_ngc.sh' is used to create a virtual environment for DNN training using NGC(nvidia gpu cloud).
	2. Script 'Keras/00-pre_process_waveforms.py' was conducted in another workstation when we reproduced experiemnts regarding RawNet.
	3. For back-end research or front-end verification, we provide speaker embeddings extracted with RawNet at 'Keras/data/speaker_embeddings_RawNet'. 
		Cosine similarity metric with this embeddings demonstrate EER of 4.8 % on the VoxCeleb1 evaluation set. 
		This file can also obtained by running script 'Keras/01-trn_RawNet.py' (minor differences can occur due to random seed).

# To use pre-trained RawNet embeddings.

'Keras/data/speaker_embeddings_RawNet_4.8eer' contains speaker embeddings extracted using RawNet. 
Load it using python pickle library, a dictionary will be obtained. 
It has two keys: ['dev_dic_embeddings', 'eval_dic_embeddings'] where each value corresponding to the key is a dictionary that has speaker embeddings.
Decoding with cosine similarity with VoxCeleb1 dataset will yield an EER of 4.8 %. 
In our paper, training a b-vector classifier using these embeddings yielded an EER of 4.0 %. 

For other back-end researches on speaker verification, using these speaker embeddings might be a good start :)

# PyTorch implementation of RawNet

Additional baseline using VoxCeleb2 for training and VoxCeleb1 for validation and evaluation is updated in 'PyTorch' folder. 
It shows an EER of 3.6% on VoxCeleb1 evaluation.
To run the PyTorch baseline,  


	1. Script 'PyTorch/lunch_ngc.sh' is used to create a virtual environment for DNN training using NGC(nvidia gpu cloud).
	2. Run train_RawNet.py (look yaml file for parameter configurations)


###### Other guidelines are currently being updated.
Email jeewon.leo.jung@gmail.com for other details :-).

# Citation

If you used the codes of this repository, please cite  [RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification]( https://isca-speech.org/archive/Interspeech_2019/pdfs/1982.pdf ) 

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
- 2019.04.17. : 01 script executing
- 2019.04.24. : 01 script verified
- 2019.04.29. : 02 script executing 
- 2019.04.29. : 02 script verified
- 2019.10.14. : short utterance preparation script added regarding ASRU 2019 paper
- 2019.10.22. : Previous scripts and data moved under "Keras"
- 2019.10.22. : Add citation guidelines
- 2019.10.22. : Initial commit of PyTorch scripts
- 2019.11.05. : PyTorch baseline on VoxCeleb2 
