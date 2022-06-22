## Usage

RawNet3 is hosted via two repositories.
Inference of any utterance with 16k 16bit mono format and Vox1-O benchmark is 
supported in this repository.

Training recipe, on the other hand, will be supported in 
https://github.com/clovaai/voxceleb_trainer.

Model weight parameters are served by huggingface at 
https://huggingface.co/jungjee/RawNet3, which is used as a submodule here

To download the model, run:
`git submodule update --init --recursive`

### Single utterance inference
Run: `python inference.py --inference_utterance --input {YOUR_INPUT_FILE}`

Optionally, `--out_dir` can be set to direct where to save the extracted speaker embedding. (default: `./out.npy`)

### Benchmark on the Vox1-O evaluation protocol
Run: `python inference.py --vox1_o_benchmark --DB_dir`

Note that `DB_dir` should direct the directory of VoxCeleb1 dataset. 
For example, if `DB_dir`="/home/abc/db/VoxCeleb1",
VoxCeleb1 folder is expected to have 1,251 folders inside which corresponds to 1,251 speakers of the VoxCeleb1 dataset. 

If you successfully run the benchmark, you will get:
`Vox1-O benchmark Finished. EER: 0.8932, minDCF:0.06690`. 
