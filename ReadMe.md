# (2019.10.22.)PyTorch based Scripts are going to be uploaded including VoxCeleb2 results!
## Scripts and data using Keras moved to folder "Keras"

# This github project includes codes for reproducing experiments and DNN models used in the paper

RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification
- Currently accepted to Interspeech2019 as a conference paper.
- arXiv pre-print is at https://arxiv.org/abs/1904.08104
	

Below are few notes for reproduction
------------------------------------
	1. Script 'lunch_ngc.sh' is used to create a virtual environment for DNN training using NGC(nvidia gpu cloud).
	2. Script '00-pre_process_waveforms.py' was conducted in another workstation when we reproduced experiemnts regarding RawNet.
	3. For back-end research or front-end verification, we provide speaker embeddings extracted with RawNet at 'data/speaker_embeddings_RawNet'. 
		Cosine similarity metric with this embeddings demonstrate EER of 4.8 % on the VoxCeleb1 evaluation set. 
		This file can also obtained by running script '01-trn_RawNet.py' (minor differences can occur due to random seed).

For those who want to use RawNet embeddings.
--------------------------------------------

'data/speaker_embeddings_RawNet_4.8eer' contains speaker embeddings extracted using RawNet. 
Load it using python pickle library, a dictionary will be obtained. 
It has two keys: ['dev_dic_embeddings', 'eval_dic_embeddings'] where each value corresponding to the key is a dictionary that has speaker embeddings.
Decoding with cosine similarity with VoxCeleb1 dataset will yield an EER of 4.8 %. 
In our paper, training a b-vector classifier using these embeddings yielded an EER of 4.0 %. 

For other back-end researches on speaker verification, using these speaker embeddings might be a good start :)


# Other guidelines are currently being updated.
Email jeewon.leo.jung@gmail.com for other details :-).

# Citation

If you used the codes of this repository, please cite  [RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification]( https://isca-speech.org/archive/Interspeech_2019/pdfs/1982.pdf ) 

```
@article{jung2018avoiding,
  title={RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification},
  author={Jung, Jee-weon and Heo, Hee-soo and Kim, ju-ho and Shim, Hye-jin and Yu, Ha-jin},
  journal={Proc. Interspeech 2019},
  pages={1268--1272},
  year={2019}
}
```


Log
- 2019.04.17. : 01 script executing
- 2019.04.24. : 01 script verified.
- 2019.04.29. : 02 script executing 
- 2019.04.29. : 02 script verified.
- 2019.10.14. : short utterance preparation script added regarding ASRU 2019 paper.
- 2019.10.22. : Previous scripts and data moved under "Keras".
