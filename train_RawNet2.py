from tqdm import tqdm
from collections import OrderedDict

import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

from dataloader import *
from model_RawNet2 import RawNet2
from parser import get_args
from trainer import *
from utils import *

def main():
    #parse arguments
    args = get_args()

    #set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Device: {}'.format(device))

    #get utt_lists & define labels
    l_dev = sorted(get_utt_list(args.DB_vox2 + args.dev_wav))
    l_val = sorted(get_utt_list(args.DB + args.val_wav))
    l_eval = sorted(get_utt_list(args.DB + args.eval_wav))
    d_label_vox2 = get_label_dic_Voxceleb(l_dev)
    args.model['nb_classes'] = len(list(d_label_vox2.keys()))

    #def make_validation_trial(l_utt, nb_trial, dir_val_trial):
    if bool(args.make_val_trial):
        make_validation_trial(l_utt = l_val, nb_trial = args.nb_val_trial, dir_val_trial = args.DB + 'val_trial.txt')
    with open(args.DB + 'val_trial.txt', 'r') as f:
        l_val_trial = f.readlines()
    with open(args.DB + 'veri_test.txt', 'r') as f:
        l_eval_trial = f.readlines()

    #define dataset generators
    devset = Dataset_VoxCeleb2(list_IDs = l_dev,
        labels = d_label_vox2,
        nb_samp = args.nb_samp,
        base_dir = args.DB_vox2 + args.dev_wav)
    devset_gen = data.DataLoader(devset,
        batch_size = args.bs,
        shuffle = True,
        drop_last = True,
        num_workers = args.nb_worker)
    valset = Dataset_VoxCeleb2(list_IDs = l_val,
        return_label = False,
        nb_samp = args.nb_samp,
        base_dir = args.DB + args.val_wav)
    valset_gen = data.DataLoader(valset,
        batch_size = args.bs,
        shuffle = False,
        drop_last = False,
        num_workers = args.nb_worker)
    TA_evalset = TA_Dataset_VoxCeleb2(list_IDs = l_eval,
        return_label = False,
        window_size = args.window_size, # 20% of nb_samp
        nb_samp = args.nb_samp, 
        base_dir = args.DB+args.eval_wav)
    TA_evalset_gen = data.DataLoader(TA_evalset,
        batch_size = 1, 
        shuffle = False,
        drop_last = False,
        num_workers = args.nb_worker)

    #set save directory
    save_dir = args.save_dir + args.name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'results/'):
        os.makedirs(save_dir+'results/')
    if not os.path.exists(save_dir+'models/'):
        os.makedirs(save_dir+'models/')
        
    #log experiment parameters to local and comet_ml server
    f_params = open(save_dir + 'f_params.txt', 'w')
    for k, v in sorted(vars(args).items()):
        print(k, v)
        f_params.write('{}:\t{}\n'.format(k, v))
    for k, v in sorted(args.model.items()):
        print(k, v)
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.close()
    
    #define model
    if bool(args.mg):
        model_1gpu = RawNet2(args.model)
        if args.load_model: model_1gpu.load_state_dict(torch.load(args.load_model_dir))
        nb_params = sum([param.view(-1).size()[0] for param in model_1gpu.parameters()])
        model = nn.DataParallel(model_1gpu).to(device)
    else:
        model = RawNet2(args.model).to(device)
        if args.load_model: model.load_state_dict(torch.load(args.load_model_dir))
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    if not args.load_model: model.apply(init_weights)
    print('nb_params: {}'.format(nb_params))

    #set ojbective funtions
    criterion = {}
    criterion['cce'] = nn.CrossEntropyLoss()
    
    #set optimizer
    params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if 'bn' not in name
            ]
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if 'bn' in name
            ],
            'weight_decay':
            0
        },
    ]
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params,
            lr = args.lr,
            momentum = args.opt_mom,
            weight_decay = args.wd,
            nesterov = args.nesterov)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params,
            lr = args.lr,
            weight_decay = args.wd,
            amsgrad = args.amsgrad)
    else:
        raise NotImplementedError('Add other optimizers if needed')
    if args.load_model: optimizer.load_state_dict(torch.load(args.load_model_opt_dir))
    
    #set learning rate decay
    if bool(args.do_lr_decay):
        if args.lr_decay == 'keras':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
        elif args.lr_decay == 'cosine':
            raise NotImplementedError('Not implemented yet')
        else:
            raise NotImplementedError('Not implemented yet')
     
    ##########################################
    #Train####################################
    ##########################################
    best_val_eer = 99.
    best_TA_eval_eer = 99.
    f_eer = open(save_dir + 'eers.txt', 'a', buffering = 1)
    for epoch in tqdm(range(args.epoch)):
        #train phase
        train_model(model = model,
            db_gen = devset_gen,
            args = args,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            criterion = criterion,
            device = device,
            epoch = epoch)


        #validation phase
        val_eer = evaluate_model(mode = 'val',
            model = model,
            db_gen = valset_gen, 
            l_utt = l_val,
            save_dir = save_dir,
            epoch = epoch,
            device = device,
            l_trial = l_val_trial,
            args = args)
        f_eer.write('epoch:%d, val_eer:%.4f\n'%(epoch, val_eer))
        
        TA_eval_eer = time_augmented_evaluate_model(mode = 'eval',
            model = model,
            db_gen = TA_evalset_gen, 
            l_utt = l_eval,
            save_dir = save_dir,
            epoch = epoch,
            device = device,
            l_trial = l_eval_trial,
            args = args)
        f_eer.write('epoch:%d, TA_eval_eer:%.4f\n'%(epoch, TA_eval_eer))
        
        save_model_dict = model_1gpu.state_dict() if args.mg else model.state_dict()

        #record best validation model
        if float(val_eer) < best_val_eer:
            print('New best validation EER: %f'%float(val_eer))
            best_val_eer = float(val_eer)

            torch.save(save_model_dict, save_dir +  'models/best_val.pt')
            torch.save(optimizer.state_dict(), save_dir + 'models/best_opt_val.pt')
            
        if float(TA_eval_eer) < best_TA_eval_eer:
            print('New best TA_EER: %f'%float(TA_eval_eer))
            best_TA_eval_eer = float(TA_eval_eer)

            torch.save(save_model_dict, save_dir +  'models/TA_%d_%.4f.pt'%(epoch, TA_eval_eer))
            torch.save(optimizer.state_dict(), save_dir + 'models/best_opt_eval.pt')
        
    f_eer.close()

if __name__ == '__main__':
    main()
