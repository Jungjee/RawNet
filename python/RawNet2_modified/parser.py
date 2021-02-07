import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()
    # dir
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-module_name", type=str, required=True)
    parser.add_argument("-model_name", type=str, required=True)
    parser.add_argument("-save_dir", type=str, default="/exp/DNNs/")
    parser.add_argument("-DB", type=str, default="DB/VoxCeleb1/")
    parser.add_argument("-DB_vox2", type=str, default="DB/VoxCeleb2/")
    parser.add_argument("-dev_wav", type=str, default="wav/")
    parser.add_argument("-val_wav", type=str, default="dev_wav/")
    parser.add_argument("-eval_wav", type=str, default="eval_wav/")
    parser.add_argument("-ddp_port", type=str, default="8888")
    parser.add_argument("-musan_dir", type=str, default="DB/augment/musan_split")
    parser.add_argument(
        "-rir_dir", type=str, default="DB/augment/RIRS_NOISES/simulated_rirs"
    )

    # hyper-params
    parser.add_argument("-bs", type=int, default=400)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-lr_min", type=float, default=0.000005)
    parser.add_argument("-nb_samp", type=int, default=59049)
    parser.add_argument("-wd", type=float, default=0.0001)
    parser.add_argument("-epoch", type=int, default=160)
    parser.add_argument("-optimizer", type=str, default="Adam")
    parser.add_argument("-nb_worker", type=int, default=8)
    parser.add_argument("-seed", type=int, default=1234)
    parser.add_argument("-nb_tta_seg", type=int, default=10)
    parser.add_argument("-verbose", type=int, default=1)
    parser.add_argument("-patience", type=int, default=40)
    parser.add_argument("-nb_iter_per_log", type=int, default=50)
    parser.add_argument("-lr_decay", type=str, default="keras")
    parser.add_argument("-load_model_dir", type=str, default="")

    # DNN args
    parser.add_argument(
        "-m_nb_filters", type=int, nargs="+", default=[128,128,256,256,512,512]
    )
    parser.add_argument("-m_layers", type=int, nargs="+", default=[1,1,1,2,1,2])
    parser.add_argument("-m_code_dim", type=int, default=512)
    parser.add_argument("-m_nb_samp", type=int, default=59049)

    #####
    # loss
    #####
    parser.add_argument("-m_margin", type=float, default=0.1)
    parser.add_argument("-m_scale", type=float, default=30)

    #####
    # flag
    #####
    parser.add_argument("-amsgrad", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "-augment",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="apply data augmentation using reverberation from rir dataset and noise from MUSAN dataset",
    )
    parser.add_argument("-debug", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "-early_stop",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="when True, performs early stopping based on EER with patience",
    )
    parser.add_argument(
        "-do_lr_decay", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "-load_model",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="for resuming experiment. not complete.",
    )
    parser.add_argument(
        "-reproducible",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="when true, extact experiment will be reproducible, with the cost of much slower training",
    )
    parser.add_argument(
        "-use_cce",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="configures whether to add categorical cross-entropy to the final loss function",
    )

    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == "m_":
            args.model[k[2:]] = v
    return args