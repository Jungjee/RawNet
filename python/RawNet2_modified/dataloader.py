import glob
import os
import random
import warnings

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from utils import get_label_dic_Voxceleb, get_utt_list

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def get_loader(args):
    l_trn = sorted(get_utt_list(args.DB_vox2 + args.dev_wav))
    print("# train sample: {}".format(len(l_trn)))
    l_eval = sorted(get_utt_list(args.DB + args.eval_wav))
    d_label = get_label_dic_Voxceleb(l_trn)

    # define dataset generators
    trnset = Trainset(
        l_utt=l_trn,
        labels=d_label,
        nb_samp=args.nb_samp,
        base_dir=args.DB_vox2 + args.dev_wav,
        augment=args.augment,
        musan_dir=args.musan_dir,
        rir_dir=args.rir_dir,
    )
    trnset_sampler = torch.utils.data.DistributedSampler(
        trnset
    )
    evlset = EvaluationSet(
        l_utt=l_eval,
        nb_seg=args.nb_tta_seg,
        nb_samp=args.nb_samp,
        base_dir=args.DB + args.eval_wav,
    )
    evlset_sampler = torch.utils.data.DistributedSampler(evlset, shuffle=False)

    trnset_gen = torch.utils.data.DataLoader(
        trnset,
        batch_size=args.bs,
        shuffle=(trnset_sampler is None),
        sampler=trnset_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=args.nb_worker,
    )
    evlset_gen = torch.utils.data.DataLoader(
        evlset,
        batch_size=args.bs // 10,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.nb_worker,
        sampler=evlset_sampler,
    )
    return trnset_gen, trnset_sampler, evlset_gen, d_label


class Trainset(torch.utils.data.Dataset):
    def __init__(
        self,
        l_utt,
        labels,
        nb_samp=59049,
        base_dir="",
        augment=False,
        musan_dir="",
        rir_dir="",
    ):
        """
        arguments:
        l_utt    :list of strings (each string: utt key)
        labels   :dictionary where key: utt key and value: label integer
        nb_samp  :integer, the number of samples in each utterance for each mini-batch
        base_dir :directory of dataset
        """
        self.l_utt = l_utt
        self.labels = labels
        self.nb_samp = nb_samp + 1
        self.base_dir = base_dir
        self.augment = augment
        self.musan_dir = musan_dir
        self.rir_dir = rir_dir
        if augment:
            self.augment_wav = AugmentWAV(
                musan_dir=musan_dir, rir_dir=rir_dir, nb_samp=self.nb_samp
            )

    def __len__(self):
        return len(self.l_utt)

    def __getitem__(self, index):
        # get utterance id
        key = self.l_utt[index]

        # load utt
        try:
            x, _ = sf.read(self.base_dir + key)
            x = x.reshape(1, -1)
        except:
            raise ValueError("%s" % key)

        # adjust duration to "nb.samp" for mini-batch construction
        nb_actual_samp = x.shape[1]
        if nb_actual_samp > self.nb_samp:
            start_idx = np.random.randint(
                low=0, high=nb_actual_samp - self.nb_samp)
            x = x[:, start_idx: start_idx + self.nb_samp]
        elif nb_actual_samp < self.nb_samp:
            nb_dup = int(self.nb_samp / nb_actual_samp) + 1
            x = np.tile(x, (1, nb_dup))[:, : self.nb_samp]
        else:
            x = x

        # apply data augmentation
        if self.augment:
            augtype = random.randint(0, 4)
            if augtype == 1:
                x = self.augment_wav.reverberate(x)
            elif augtype == 2:
                x = self.augment_wav.additive_noise("music", x)
            elif augtype == 3:
                x = self.augment_wav.additive_noise("speech", x)
            elif augtype == 4:
                x = self.augment_wav.additive_noise("noise", x)

        # apply pre-emphasis
        x = pre_emphasis(x)  # 59050 to 59049

        # get label
        y = self.labels[key.split("/")[0]]

        return x.astype(np.float32), y


class EvaluationSet(torch.utils.data.Dataset):
    def __init__(self, l_utt, nb_seg=10, nb_samp=59049, base_dir=""):
        """
        l_utt       :list of strings (each string: utt key)
        nb_seg      :integer, the number of segments to extract from an utterance
        nb_samp     :integer, the number of samples in each utterance for each mini-batch
        base_dir    :directory of dataset
        """
        self.l_utt = l_utt
        self.nb_seg = nb_seg
        self.nb_samp = nb_samp
        self.base_dir = base_dir

    def __len__(self):
        return len(self.l_utt)

    def __getitem__(self, index):
        key = self.l_utt[index]
        try:
            x, _ = sf.read(self.base_dir + key)
            x = x.reshape(1, -1)
        except:
            raise ValueError("%s" % key)

        # apply pre-emphasis
        x = pre_emphasis(x)

        # match minimum required duration if too short
        nb_actual_samp = x.shape[1]
        if nb_actual_samp < self.nb_samp:
            nb_dup = int(self.nb_samp / nb_actual_samp) + 1
            x = np.tile(x, (1, nb_dup))[:, : self.nb_samp]
            nb_actual_samp = x.shape[1]

        # start indices of each segment
        stt_idx = np.linspace(0, nb_actual_samp - self.nb_samp, self.nb_seg)

        # list of segments
        l_x = []
        for idx in stt_idx:
            l_x.append(x[:, int(idx): int(idx) + self.nb_samp])
        x = np.stack(l_x, axis=0).astype(np.float32)  # (10, self.nb_samp)

        return x, key


#####
# Pre-emphasize an utterance (single & multi-channel)
# x : (numpy array or torch tensor) shape (#channel, #sample)
def pre_emphasis(x):
    return x[:, 1:] - 0.97 * x[:, :-1]


class AugmentWAV(object):
    """
    Acknowledgement: Github project 'clovaai/voxceleb_trainer'.
    link: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py

    Adjusted for RawNets
    """

    def __init__(self, musan_dir, rir_dir, nb_samp):
        self.nb_samp = nb_samp
        self.noisetypes = ["noise", "speech", "music"]
        self.noisesnr = {"noise": [0, 15],
                         "speech": [13, 20], "music": [5, 15]}
        self.numnoise = {"noise": [1, 1], "speech": [3, 7], "music": [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_dir, "*/*/*/*.wav"))

        for file in augment_files:
            if not file.split("/")[-4] in self.noiselist:
                self.noiselist[file.split("/")[-4]] = []
            self.noiselist[file.split("/")[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_dir, "*/*/*.wav"))

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(
            self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1])
        )

        noises = []
        for noise in noiselist:
            noiseaudio, _ = sf.read(noise)
            noiseaudio = noiseaudio.reshape(1, -1)
            nb_actual_samp = noiseaudio.shape[1]
            if nb_actual_samp > self.nb_samp:
                start_idx = np.random.randint(
                    low=0, high=nb_actual_samp - self.nb_samp)
                noiseaudio = noiseaudio[:, start_idx: start_idx + self.nb_samp]
            elif nb_actual_samp < self.nb_samp:
                nb_dup = int(self.nb_samp / nb_actual_samp) + 1
                noiseaudio = np.tile(noiseaudio, (1, nb_dup))[
                    :, : self.nb_samp]
            else:
                noiseaudio = noiseaudio

            noise_snr = random.uniform(
                self.noisesnr[noisecat][0], self.noisesnr[noisecat][1]
            )
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
            )

        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, _ = sf.read(rir_file)
        rir = rir.reshape(1, -1)
        nb_actual_samp = rir.shape[1]
        if nb_actual_samp > self.nb_samp:
            start_idx = np.random.randint(
                low=0, high=nb_actual_samp - self.nb_samp)
            rir = rir[:, start_idx: start_idx + self.nb_samp]
        elif nb_actual_samp < self.nb_samp:
            nb_dup = int(self.nb_samp / nb_actual_samp) + 1
            rir = np.tile(rir, (1, nb_dup))[:, : self.nb_samp]
        else:
            rir = rir

        rir = rir / np.sqrt(np.sum(rir ** 2))

        res = signal.convolve(audio, rir, mode="full")[:, : self.nb_samp]
        return res


if __name__ == "__main__":
    import soundfile as sf

    # wav = ta.load_wav('/DB/VoxCeleb1/dev_wav/id10210/7P4E-933KyY/00001.wav')[0]
    wav = sf.read("/DB/VoxCeleb1/dev_wav/id10210/7P4E-933KyY/00001.wav")
    print("soundfile:", wav)
    exit()
    wav = pre_emphasis(wav)
    print(wav)
    print(wav[0][0].dtype)

    print(wav.size())
    exit()
