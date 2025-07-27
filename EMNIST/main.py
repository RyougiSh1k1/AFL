import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models
from torch.utils.data import Subset
from dataset import prepare_data
import numpy as np
from afl import LinearAnalytic, init_local, local_update, aggregation, clean_regularization, validate

# Basic Setup
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='DIR', nargs='?', default='cifar100',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
#Data Setup
parser.add_argument('--data', default='./data', type=str, metavar='PATH',
                    help='path of dataset')
parser.add_argument('--datadir', default='./dataset', type=str, metavar='PATH',
                    help='path to locate dataset split')
parser.add_argument( '--seed', default=1, type=int, metavar='N',
                    help='random seed for spliting data')
parser.add_argument( '--modelseed', default=1, type=int, metavar='N',
                    help='random seed for spliting data')
parser.add_argument( '--num_clients', default=50, type=int, metavar='N',
                    help='number of clients')
parser.add_argument( '--num_classes', default=100, type=int, metavar='N',
                    help='total number of classes')
parser.add_argument( '--niid',  dest='niid', action='store_true',
                    help='set data to non-iid')
parser.add_argument( '--balance',  dest='balance', action='store_true',
                    help='balance distribution')
parser.add_argument('--partition', default='dir', type=str,
                    help='dirstribution type of non-iid setting')
parser.add_argument( '--alpha', default=0.1, type=float,
                    help='skewness of dir distribution')
parser.add_argument( '--shred', default=10, type=int,
                    help='skewness of pat distribution')
parser.add_argument( '--rg', default=0, type=float,
                    help='regularization factor of analytic learning')
parser.add_argument( '--clean_reg',  dest='clean_reg', action='store_true',
                    help='clean regularization factor after aggregation')
# Continual Learning Setup
parser.add_argument('--continual', dest='continual', action='store_true',
                    help='enable continual learning setup')
parser.add_argument('--tasks_per_client', default=4, type=int,
                    help='number of tasks per client in continual learning')
parser.add_argument('--classes_per_task', default=20, type=int,
                    help='number of classes per task in continual learning')

best_acc1 = 0

def main():
    args = parser.parse_args()

    # Dataset-specific settings for continual learning
    if args.continual:
        if args.dataset == 'cifar100':
            args.num_clients = 10
            args.num_classes = 100
            args.tasks_per_client = 4
            args.classes_per_task = 20
            args.niid = False
            print("="*50)
            print("CIFAR-100 Continual Learning Configuration:")
            print(f"  Number of clients: {args.num_clients}")
            print(f"  Tasks per client: {args.tasks_per_client}")
            print(f"  Classes per task: {args.classes_per_task}")
            print("  Non-IID setting: Disabled")
            print("="*50)
        elif args.dataset == 'emnist':
            args.num_clients = 8
            args.num_classes = 26  # EMNIST-letters has 26 classes (A-Z)
            args.tasks_per_client = 6
            args.classes_per_task = 4
            args.niid = False
            print("="*50)
            print("EMNIST-letters Continual Learning Configuration:")
            print(f"  Number of clients: {args.num_clients}")
            print(f"  Number of classes: {args.num_classes}")
            print(f"  Tasks per client: {args.tasks_per_client}")
            print(f"  Classes per task: {args.classes_per_task}")
            print("  Non-IID setting: Disabled")
            print("="*50)

    if args.seed is not None:
        random.seed(args.modelseed)
        torch.manual_seed(args.modelseed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    import resnet
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model_pretrain = torchvision.models.__dict__[args.arch](pretrained=True)
        model = resnet.__dict__[args.arch](args.num_classes)
        model.load_state_dict(model_pretrain.state_dict(), strict=False)
        args.feat_size = model_pretrain.fc.weight.size(1)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch]()
    model = model.cuda(args.gpu)

    # dataset splitting
    result = prepare_data(args)
    if args.continual and args.dataset in ['cifar100', 'emnist']:
        train_total, train_data_idx, testset, task_info = result
    else:
        train_total, train_data_idx, testset = result
        task_info = None
    
    random.seed(args.modelseed)
    torch.manual_seed(args.modelseed)
    train_dataset = []
    for idx in range(args.num_clients):
        train_dataset_idx = torch.utils.data.Subset(train_total, train_data_idx[idx])
        train_dataset.append(train_dataset_idx)
    model.eval()

    global_model = LinearAnalytic(args.feat_size, args.num_classes).cuda(args.gpu)
    local_weights, local_R, local_C = [], [], []
    local_models = []
    local_train_acc = []
    
    print("\n" + "="*50)
    print("Training locally!")
    print("="*50)
    
    start = time.time()
    
    for idx in range(args.num_clients):
        print(f"\nClient #{idx}:")
        
        if args.continual and task_info is not None:
            # For continual learning, show task information
            print(f"  Tasks for this client:")
            for task_id, task_classes in enumerate(task_info[idx]):
                if args.dataset == 'emnist':
                    # Convert class indices to letters for EMNIST
                    letters = [chr(65 + c) for c in task_classes]  # 65 is ASCII for 'A'
                    print(f"    Task {task_id}: Classes {task_classes} (Letters: {letters})")
                else:
                    print(f"    Task {task_id}: Classes {task_classes}")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset[idx], 
            args.batch_size, 
            drop_last=False, 
            shuffle=True, 
            num_workers=8
        )
        
        # Train locally
        W, R, C = local_update(train_loader, model, global_model, args)
        local_model = init_local(args)
        local_model.fc.weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        local_models.append(local_model)
        
        # Validate
        correct, num_sample = validate(train_loader, model, local_model.cuda(), args)
        acc = correct / num_sample
        W = W.cpu()
        
        print(f"  Training Accuracy: {acc * 100:.2f}%")
        print(f"  Number of samples: {num_sample}")
        
        local_weights.append(W)
        local_R.append(R)
        local_C.append(C)
        local_train_acc.append(acc.cpu().item())
    
    endtime = time.time() - start
    print(f"\nElapsing time for local training: {endtime:.2f} seconds")
    
    # Aggregation
    print("\n" + "="*50)
    print("Aggregating!")
    print("="*50)
    
    global_weight, global_R, global_C = aggregation(local_weights, local_R, local_C, args)
    print('Aggregation done!')
    global_model.fc.weight = torch.nn.parameter.Parameter(torch.t(global_weight.float()))

    # Evaluate the global model
    print("\n" + "="*50)
    print("Evaluating global model!")
    print("="*50)

    val_loader = torch.utils.data.DataLoader(testset, args.batch_size, drop_last=False, shuffle=True, num_workers=8)
    correct, num_sample = validate(val_loader, model, global_model, args)
    acc = correct / num_sample * 100
    endtime_1 = time.time() - start
    
    print(f"\nElapsing time for training and aggregation: {endtime_1:.2f} seconds")
    print(f"Average accuracy on test set: {acc:.2f}%")
    print(f"Average local training accuracy: {np.mean(local_train_acc) * 100:.2f}%")
    
    acc_c = -100
    if args.clean_reg:
        print("\n" + "="*50)
        print("Cleaning regularization...")
        print("="*50)
        
        global_weight_clean = clean_regularization(global_weight, global_C, args)
        global_model.fc.weight = torch.nn.parameter.Parameter(torch.t(global_weight_clean.float()))
        val_loader = torch.utils.data.DataLoader(testset, args.batch_size, drop_last=False, shuffle=True)
        correct_c, num_sample = validate(val_loader, model, global_model, args)
        acc_c = correct_c / num_sample * 100
        print(f"Average accuracy after regularization cleaning: {acc_c:.2f}%")

    endtime_2 = time.time() - start
    print(f"\nElapsing time plus cleansing regularization: {endtime_2:.2f} seconds")
    
    # Save results
    import csv
    filename = f"{args.dataset}_{args.arch}_{args.num_clients}clients"
    if args.continual:
        filename += f"_continual_{args.tasks_per_client}tasks_{args.classes_per_task}classes"
    else:
        filename += f"_{args.alpha}_{args.shred}_{args.partition}"
    filename += ".csv"
    
    with open(filename, mode='a+', encoding="ISO-8859-1", newline='') as file:
        data = (
            str(local_train_acc), '-', 
            str(acc.cpu().item()), 
            str(acc_c.cpu().item()), '-', 
            str(endtime), 
            str(endtime_1), 
            str(endtime_2), '-', 
            str(args)
        )
        wr = csv.writer(file)
        wr.writerow(data)
    
    print(f'\nResults written to: {filename}')

if __name__ == '__main__':
    main()