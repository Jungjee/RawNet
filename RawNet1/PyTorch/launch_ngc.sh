sudo nvidia-docker run -it --rm --ipc=host --shm-size 20G -v ~/Dropbox/exp_codes/RawNet:/code -v ~/DB:/DB -v ~/Exp/RawNet:/exp nvcr.io/nvidia/pytorch:19.05-py3
