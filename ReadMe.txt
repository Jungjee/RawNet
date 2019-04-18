This github project includes codes for reproducing experiments and DNN models used in the paper 
''RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification''
	* Currently submitted to Interspeech2019 as a conference paper.
	* arXiv pre-print is at https://arxiv.org/abs/1904.08104
	

Below are few notes for reproduction
	1. Script 'lunch_ngc.sh' is used to create a virtual environment for DNN training using NGC(nvidia gpu cloud).
	2. Script '00-pre_process_waveforms.py' was conducted in another workstation when we reproduced experiemnts regarding RawNet.
	3. For back-end research or front-end verification, we provide speaker embeddings extracted with RawNet at 'data/speaker_embeddings_RawNet'. 
		Cosine similarity metric with this embeddings demonstrate EER of 4.8 % on the VoxCeleb1 evaluation set. 
		This file can also obtained by running script '01-trn_RawNet.py' (minor differences can occur due to random seed).

Other guidelines are currently being updated.
Email jeewon.leo.jung@gmail.com for other details :-).

Log
	2019.04.18	: executing 01 script
