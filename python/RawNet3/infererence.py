import argparse
import sys
from typing import Dict

import numpy as np
import soundfile as sf
import torch

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck


def main(args: Dict) -> None:
    model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )

    model.load_state_dict(
        torch.load(
            "./models/weights/model.pt",
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    print("RawNet3 initialised & weights loaded!")

    if args.inference_utterance:
        audio, sample_rate = sf.read(args.input)
        if len(audio.shape) > 1:
            raise ValueError(
                f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
            )

        if sample_rate != 16000:
            raise ValueError(
                f"RawNet3 supports 16k sampling rate only. Input data's sampling rate is {sample_rate}."
            )

        if (
            len(audio) < 48000
        ):  # RawNet3 was trained using utterances of 3 seconds
            print(
                f"RawNet3 requires utterance duration of at least 3 seconds. Input data has a duartion of {len(audio)/16000} seconds. Padding is conducted and the quality of extracted speaker embedding may drop."
            )
            shortage = 48000 - len(audio) + 1
            audio = np.pad(audio, (0, shortage), "wrap")

        audios = np.linspace(0, len(audio) - 48000, num=args.n_segments)
        outputs = model(torch.from_numpy(audios))
        print(outputs.size())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RawNet3 inference")

    parser.add_argument(
        "--inference_utterance", type=bool, default=False, action="store_true"
    )
    parser.add_argument("--input", type=str, default="")
    parser.add_argument(
        "--vox1_o_benchmark", type=bool, default=False, action="store_true"
    )
    parser.add_argument(
        "--n_segments", type=int, default=10, help="number of segments to make using each utterance"
    )
    args = parser.parse_args()

    assert args.inference_utterance or args.vox1_o_benchmark
    if args.inference_utterance:
        assert args.input != ""

    sys.exit(main(args))
