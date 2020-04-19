import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    #dir
    parser.add_argument('-name', type = str, required = True)
    parser.add_argument('-save_dir', type = str, default = 'DNNs/')
    parser.add_argument('-DB', type = str, default = 'DB/VoxCeleb1/')
    parser.add_argument('-DB_vox2', type = str, default = 'DB/VoxCeleb2/')
    parser.add_argument('-dev_wav', type = str, default = 'wav/')
    parser.add_argument('-val_wav', type = str, default = 'dev_wav/')
    parser.add_argument('-eval_wav', type = str, default = 'eval_wav/')
    
    #hyper-params
    parser.add_argument('-bs', type = int, default = 100)
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-nb_samp', type = int, default = 59049)
    parser.add_argument('-window_size', type = int, default = 11810)
    
    parser.add_argument('-wd', type = float, default = 0.0001)
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-optimizer', type = str, default = 'Adam')
    parser.add_argument('-nb_worker', type = int, default = 8)
    parser.add_argument('-temp', type = float, default = .5)
    parser.add_argument('-seed', type = int, default = 1234) 
    parser.add_argument('-nb_val_trial', type = int, default = 40000) 
    parser.add_argument('-lr_decay', type = str, default = 'keras')
    parser.add_argument('-load_model_dir', type = str, default = '')
    parser.add_argument('-load_model_opt_dir', type = str, default = '')

    #DNN args
    parser.add_argument('-m_first_conv', type = int, default = 251)
    parser.add_argument('-m_in_channels', type = int, default = 1)
    parser.add_argument('-m_filts', type = list, default = [128, [128,128], [128,256], [256,256]])
    parser.add_argument('-m_blocks', type = list, default = [2, 4])
    parser.add_argument('-m_nb_fc_att_node', type = list, default = [1])
    parser.add_argument('-m_nb_fc_node', type = int, default = 1024)
    parser.add_argument('-m_gru_node', type = int, default = 1024)
    parser.add_argument('-m_nb_gru_layer', type = int, default = 1)
    parser.add_argument('-m_nb_samp', type = int, default = 59049)
    
    #flag
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-make_val_trial', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-debug', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-comet_disable', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-mg', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-load_model', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-reproducible', type = str2bool, nargs='?', const=True, default = True)

    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            print(k, v)
            args.model[k[2:]] = v
    return args