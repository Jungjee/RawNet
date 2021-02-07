# Overview
This repository includes implementations of speaker verification systems that input raw waveforms.

Currently, it has three systems in [python](python/). 
Detailed instructions on each system is described in individual `ReadME` files.

## RawNet2_modified
- Code refactoring
  - PyTorch ResNet alike model implementation
  - Deeper architecture
  - Improved feature map scaling method
    - [α-feature map scaling for raw waveform speaker verification]( https://doi.org/10.7776/ASK.2020.39.5.441 )
      - Only abstract is in English
  - Angular loss function adopted
- Performance
  - EER 1.91%
    - Trained using VoxCeleb2
    - VoxCeleb1 original trial
  - Will be used as a baseline system for authors' future works
## RawNet2

- Improved performance than RawNet
  - DNN speaker embedding extraction with raw waveform inputs
  - cosine similarity back-end
  - EER 4.8% -->> 2.56%
    - VoxCeleb1 original trial
- Uses a technique named feature map scaling
  - scales feature map alike squeeze-excitation
- Implemented in PyTorch.
- Published as a conference paper in Interspeech 2020. 
  - [Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms]( https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf ) 

```
@article{jung2020improved,
  title={Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms},
  author={Jung, Jee-weon and Kim, Seung-bin and Shim, Hye-jin and Kim, Ju-ho and Yu, Ha-Jin},
  journal={Proc. Interspeech 2020},
  pages={3583--3587},
  year={2020}
}
```
## RawNet
- DNN-based speaker embedding extractor used with another DNN-based classifier
  - Built on top of authors' previous works on raw waveform speaker verification
    - [ICASSP2018](https://ieeexplore.ieee.org/abstract/document/8462575) and [Interspeech2018](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1608.pdf)
  - EER 4.8% with cosine simaility back-end, 4.0% with proposed concat&mul back-end
    - VoxCeleb1 original trial
- Implemented in Keras and PyTorch
- Published as a conference paper in Interspeech 2019. 
  - [RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification]( https://isca-speech.org/archive/Interspeech_2019/pdfs/1982.pdf ) 

```
@article{jung2019RawNet,
  title={RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification},
  author={Jung, Jee-weon and Heo, Hee-soo and Kim, ju-ho and Shim, Hye-jin and Yu, Ha-jin},
  journal={Proc. Interspeech 2019},
  pages={1268--1272},
  year={2019}
}
```