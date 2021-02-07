import torch
import torch.cuda.amp as amp
import torch.optim as optim

from utils import cosine_annealing, keras_lr_decay


def get_optimizer(args, model):
    params = [
        {
            "params": [
                param for name, param in model.named_parameters() if "bn" not in name
            ]
        },
        {
            "params": [
                param for name, param in model.named_parameters() if "bn" in name
            ],
            "weight_decay": 0,
        },
    ]
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.opt_mom,
            weight_decay=args.wd,
            nesterov=args.nesterov,
        )
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad
        )
    else:
        raise NotImplementedError("Add other optimizers if needed")
    if args.load_model:
        pass  # not now
        # optimizer.load_state_dict(torch.load(args.load_model_opt_dir))

    # AMP init
    scaler = amp.GradScaler()
    if args.load_model:
        pass  # not now

    # set learning rate decay
    if bool(args.do_lr_decay):
        if args.lr_decay == "keras":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: keras_lr_decay(step)
            )
        elif args.lr_decay == "cosine":
            total_steps = args.epoch * args.nb_iter_per_epoch
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    total_steps,
                    1,  # since lr_lambda computes multiplicative factor
                    args.lr_min / args.lr,
                ),
            )
        elif args.lr_decay == "triangular2":
            lr_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=args.lr_min,
                max_lr=args.lr,
                step_size_up=args.nb_iter_per_epoch * (args.epoch_per_cycle // 2),
                mode="triangular2",
                cycle_momentum=False  # should be cycled?
                # https://github.com/pytorch/pytorch/issues/19003
            )
        else:
            raise NotImplementedError("Not implemented yet")

    return optimizer, scaler, lr_scheduler
