# Overview

This system covers modified version of RawNet2. 
More explanation will be added soon.

## Training and evaluation
1. configure dataset path
  - give directories via arguments or 
  - move VoxCeleb1 to `DB/VoxCeleb1`, VoxCeleb2 to `DB/VoxCeleb2`, musan to `DB/augment`, RIR to `DB/augment/RIR_NOISES`
    - musan should be spliited for real-time loading in Dataloaders using script in [Voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer)
2. run `python main.py -name exp1 -module_name model_RawNet2 -model_name get_RawNet2`
  - Train using VoxCeleb2 dataset
  - EER: 1.91% on VoxCeleb1 original trial
