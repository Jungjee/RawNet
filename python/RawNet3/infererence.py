import argparse
import itertools
import os
import sys
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck
from utils import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf


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
    gpu = False

    model.load_state_dict(
        torch.load(
            "./models/weights/model.pt",
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    model.eval()
    print("RawNet3 initialised & weights loaded!")

    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    if args.inference_utterance:
        output = extract_speaker_embd(
            model,
            fn=args.input,
            n_samples=48000,
            n_segments=args.n_segments,
            gpu=gpu,
        ).mean(0)

        np.save(args.out_dir, output.detach().cpu().numpy())
        return

    if args.vox1_o_benchmark:
        with open("../../trials/cleaned_test_list.txt", "r") as f:
            trials = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in trials]))

        setfiles = list(set(files))
        setfiles.sort()

        embd_dic = {}
        for f in tqdm(setfiles):
            embd_dic[f] = extract_speaker_embd(
                model, os.path.join(args.DB_dir, f), n_samples=64000, gpu=gpu
            )

        labels, scores = [], []
        for line in trials:
            data = line.split()
            ref_feat = F.normalize(embd_dic[data[1]], p=2, dim=1)
            com_feat = F.normalize(embd_dic[data[2]], p=2, dim=1)

            if gpu:
                ref_feat = ref_feat.cuda()
                com_feat = com_feat.cuda()

            dist = (
                torch.cdist(
                    ref_feat.reshape((args.n_segments, -1)),
                    com_feat.reshape((args.n_segments, -1)),
                )
                .detach()
                .cpu()
                .numpy()
            )
            score = -1.0 * np.mean(dist)
            labels.append(int(data[0]))
            scores.append(score)

        result = tuneThresholdfromScore(scores, labels, [1, 0.1])

        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        p_target, c_miss, c_fa = 0.05, 1, 1
        mindcf, _ = ComputeMinDcf(
            fnrs, fprs, thresholds, p_target, c_miss, c_fa
        )
        print(
            "Vox1-O benchmark Finished. EER: %2.4f, minDCF:%.5f"
            % (result[1], mindcf)
        )


def extract_speaker_embd(
    model, fn: str, n_samples: int, n_segments: int = 10, gpu: bool = False
) -> np.ndarray:
    audio, sample_rate = sf.read(fn)
    if len(audio.shape) > 1:
        raise ValueError(
            f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"RawNet3 supports 16k sampling rate only. Input data's sampling rate is {sample_rate}."
        )

    if (
        len(audio) < n_samples
    ):  # RawNet3 was trained using utterances of 3 seconds
        shortage = n_samples - len(audio) + 1
        audio = np.pad(audio, (0, shortage), "wrap")

    audios = []
    startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
    for asf in startframe:
        audios.append(audio[int(asf) : int(asf) + n_samples])

    audios = torch.from_numpy(np.stack(audios, axis=0).astype(np.float32))
    if gpu:
        audios = audios.to("cuda")
    with torch.no_grad():
        output = model(audios)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RawNet3 inference")

    parser.add_argument(
        "--inference_utterance", default=False, action="store_true"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input file to extract embedding. Required when 'inference_utterance' is True",
    )
    parser.add_argument(
        "--vox1_o_benchmark", default=False, action="store_true"
    )
    parser.add_argument(
        "--DB_dir",
        type=str,
        default="",
        help="Directory for VoxCeleb1. Required when 'vox1_o_benchmark' is True",
    )
    parser.add_argument("--out_dir", type=str, default="./out.npy")
    parser.add_argument(
        "--n_segments",
        type=int,
        default=10,
        help="number of segments to make using each utterance",
    )
    args = parser.parse_args()

    assert args.inference_utterance or args.vox1_o_benchmark
    if args.inference_utterance:
        assert args.input != ""

    if args.vox1_o_benchmark:
        assert args.DB_dir != ""

    sys.exit(main(args))
