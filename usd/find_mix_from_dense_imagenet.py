from __future__ import division
import cProfile, pstats
import argparse
import os
import time
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import sys
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import models
import torch

import os.path as osp
import numpy as np
from devkit.sparse_ops import SparseConv,SparseLinear
sys.path.append(osp.abspath(osp.join(__file__, '../')))

from devkit.core import (save_checkpoint, LRScheduler,get_sparsities_erdos_renyi,summary,conv_flops_counter_hook,normalize_erk_sparsity,update_erk_sparsity)

from devkit.core import get_layer_wise_dense_flops_params


from devkit.dataset.imagenet_dataset import ColorAugmentation

import socket
import math

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #socket.close()
        return s.getsockname()[1]

port = find_free_port()
# Fine mixed N:M from dense net
#local_rank = int(os.environ['LOCAL_RANK'])

parser = argparse.ArgumentParser(
    description='Pytorch CIFAR Training')
parser.add_argument('--config', default='configs/config_resnet50_2:4.yaml')
parser.add_argument("--local_rank", type=int)
parser.add_argument(
    '--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--M', default=16, type=int)
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epoch_rfap', default=10, type=int, help='value of how many epoch we continue to finetune model after target sparsity achieved')
parser.add_argument('--target_sparsity', default=0.0, type=float)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--data', type=str)
parser.add_argument('--iter_update_erk', default=100, type=int, help='iteration period for verifying and updating N and M values')
parser.add_argument('--erk_value', default=-1, type=float, help='erk value applied to all layers')
parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--alpha', type=float, default=-1, help='Divide every group by alpha to calculate NIntermediate.')                                                                                                                           

#My modification
parser.add_argument('--with_weight_penalty', default=0, type=int)
parser.add_argument('--initialisation', default=1, type=int)
parser.add_argument("--ste", action="store_true")

parser.add_argument('--Target_RMSI',  type=float)
parser.add_argument('--alpha_target_sparsity', default=0.5, type=float)

# New argument to specify optimizer
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer to use: sgd or adam')
parser.add_argument('--hf_model', type=str, default=None, help='HuggingFace model if not null loading weights to apply to target model')

args = parser.parse_args()


# start_epoch_complexity_loss = 20
#state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
# global layer_sparse_vis_dict
    
    
# layer_sparse_vis_dict = {}

num_iters = 0


target_sparsity = 0.0
target_sparsity_erk = 0.0


target_sparse_flops_ratio = 0.875

decision_dict = {}

vote_ratio = 0.75

num_params = 0
DenseParameters = 0
SparseParameters = 0
initialisationsetting = True
model_at_t_minus_1 = None	
layer_at_t_minus_1 = None	

rmsi_errorr = 0

stop = 0
# hyper parameters study can be found in our paper
w1=0.5 # erk 
w2=0.5 # flops 

erk_sparsity_dict = {}


def broadcast_params(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

file_path_train = "train_file.csv"
file_path_sp_layers = "sp_layers.csv"
file_path_schema = "schema_file.txt"
file_path_validate = "validate_file.csv"
    
def append_to_global_file(data, file_path):
    with open(file_path, "a") as file:
        file.write(data + "\n")
        file.close()
#with_weight_penalty = args.with_weight_penalty

def set_ste(net,ste):
    for mod in net.modules():
        if isinstance(mod, SparseConv): 
            mod.set_ste(ste)  
        elif isinstance(mod, SparseLinear): 
            mod.set_ste(False)  
			
def main():
    global args, best_prec1, with_weight_penalty, DenseParameters, ste, initialisationsetting, model_at_t_minus_1, layer_at_t_minus_1, rmsi_errorr, stop
    args = parser.parse_args()
    with_weight_penalty = args.with_weight_penalty
    with open(args.config) as f:
         config = yaml.safe_load(f)
        #config = yaml.load(f)
        

    for key in config:
        for k, v in config[key].items():
            
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
            #else:
            #    print("existing attr ", k)
    
    setattr(args, "N", args.M)

    # print(args.Ns)
    # exit(0)
    print('Enabled distributed training.')

    print("free port : " + str(port))
    
    ste = args.ste

    rank = 0#int(os.environ['RANK'])
    world_size = 1#int(os.environ['WORLD_SIZE'])

    #rank, world_size = init_dist(
    #    backend='nccl', port=port)
    print("free port : " + str(port))
    args.rank = rank
    args.world_size = world_size
    print("world size =  ..................... " + str( world_size))
    print("rank =  ..................... " + str( rank))
    
    print('Enabled distributed training.' + str(rank))

    num_classes = args.num_classes  # Set the desired number of output classes
    print("num classes = ...................... " + str(num_classes))

    data = args.data  # Set the desired number of output classes
    print("dataset data = ...................... " + data)
    
    #Initialisation du modèle
    if args.initialisation == 0:
        initialisationsetting = False
	

    # create model
    if rank == 0:
        print("=> creating model '{}'".format(args.model))
        print("=> M = ", args.M)
        print("=> N = ", args.N)
    
    model = models.__dict__[args.model](pretrained=initialisationsetting,N = args.N, M = args.M,search = True, num_new_classes = num_classes)
    
    #if args.hf_model:
        #print("loading from h model ......................................")
        # Load the model from Hugging Face
        #model_hf = ViTForImageClassification.from_pretrained(args.hf_model)
        #model.load_state_dict(model_hf.state_dict())

    # Count the number of parameters
    #num_params = sum(p.numel() for p in model.parameters())
	
    DenseParameters = model.get_dense_parametrers()
    print("****************************************")
    print ("DenseParameters = ",DenseParameters)
    print ("***************************************")
    
    set_ste(model, ste)

    # Print the number of parameters
    print("****Number of parameters before sparse: ", num_params)
    
    #print(model)
    model.reset_classifier(num_classes=num_classes)
										   

    if rank == 0:
        #print('Pruning with data layout ', args.layout)
        
        print("SGD weight decay = ", args.weight_decay )
        print("SGD momentum =", args.momentum)
        print("SGD learning rate = ", args.finetue_lr)
        print("Weight decay in Sparse Layer = ", args.sparse_decay)
        #print(model.check_N_M())
        print("Vote threshold = ",vote_ratio)
        #print(print_log_sparse_layer(model))

    #my modification    
    if with_weight_penalty == 1:  
      set_decay_sparse_layer(model,args.sparse_decay)
    set_alpha(model,args.alpha)                                

    model.set_datalayout(args.layout)
    if rank == 0:
        print("=> data layout '{}'".format(args.layout))



    global target_sparsity,target_sparsity_erk
    target_sparsity = args.target_sparsity
    global file_path_train,file_path_schema,file_path_validate,file_path_sp_layers
    name = ""
    if "cifar" in args.data.lower():
        name = "cifar" + str(args.num_classes)
    elif "imagenet" in args.data.lower():
        name = "imagenet" + str(args.num_classes)
    if ste:
        file_path_train = "./usd/train_file_spars_ste_" + name + "_" + args.model + "_" + str(target_sparsity) +"_"+  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation) + ".csv"
        file_path_sp_layers = "./usd/sp_layers_file_spars_ste_" + name + "_" + args.model + "_" + str(target_sparsity) +"_"+  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation) + ".csv"
        file_path_schema = "./usd/schemes/schema_file_spars_ste_" + name + "_" + args.model + "_" + str(target_sparsity) + "_" +  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation)  + ".txt"
        file_path_validate = "./usd/validate_file_spars_ste_" + name + "_" + args.model + "_" + str(target_sparsity) +"_"+  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation)  + ".csv"
    else:
        file_path_train = "./usd/train_file_spars_srste_" + name + "_" + args.model + "_" + str(target_sparsity) + "_"+  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation) + ".csv"
        file_path_sp_layers = "./usd/sp_layers_file_spars_srste_" + name + "_" + args.model + "_" + str(target_sparsity) +"_"+  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation) + ".csv"
        file_path_schema = "./usd/schemes/schema_file_spars_srste_" + name + "_" + args.model + "_" + str(target_sparsity) +"_"+  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation)  + ".txt"
        file_path_validate = "./usd/validate_file_spars_srste_" + name + "_" + args.model + "_" + str(target_sparsity) +"_"+  str(args.alpha_target_sparsity) + "_M_" + str(args.M) + "_alpha_" + str(args.alpha) + "_prtrained_" + str(args.initialisation)  + ".csv"

    # Create the file if it doesn't exist

    dir_path = os.path.dirname(file_path_train)
    os.makedirs(dir_path, exist_ok=True)

    dir_path = os.path.dirname(file_path_schema)
    os.makedirs(dir_path, exist_ok=True)

    dir_path = os.path.dirname(file_path_sp_layers)
    os.makedirs(dir_path, exist_ok=True)

    dir_path = os.path.dirname(file_path_validate)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path_schema, "w") as file:
        pass
    with open(file_path_train, "w") as file:
        pass
    with open(file_path_sp_layers, "w") as file:
        pass
    with open(file_path_validate, "w") as file:
        pass
    
    header = 'Search,Epoch,iteration,train_loader_data,batch_time,batch_time(AVG),data_time,data_time(AVG),Loss,Loss(AVG),Prec@1,Prec@1(AVG),Prec@5,Prec@5(AVG),Learning Rate,ErrorRMSI,sparse(FLOPS),dense(FLOPS), DenseParameters, SparseParameters'
    append_to_global_file(header, file_path_train)

    header = 'Search,Epoch,iteration,test_loader_data,batch_time,batch_time(AVG),Loss,Loss(AVG),Prec@1,Prec@1(AVG),Prec@5,Prec@5(AVG)'
    append_to_global_file(header, file_path_validate)

    header = 'Epoch,iteration' + model._get_sparse_layer_names()
    append_to_global_file(header, file_path_sp_layers)
        
    target_sparsity_erk = args.target_sparsity
    
    global erk_sparsity_dict

    #my modification    
    if with_weight_penalty == 1: 
      erk_sparsity_dict = get_sparsities_erdos_renyi(model.named_layers,target_sparsity_erk,[],list(model.dense_layers),True)
      if args.erk_value > 0:
          for key in erk_sparsity_dict:
              erk_sparsity_dict[key] = args.erk_value
    if rank == 0:
        print("Overall sparsity target of ERK Heuristic ", target_sparsity_erk)
        print("Heuristic ERK sparsity ")
        print(erk_sparsity_dict)
    #################################################################
   
    #exit(0)
    model.cuda()
    
    #broadcast_params(model)
    
    # define loss function (criterion) and optimizer
    if num_classes > 500:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
        
    print("optimizer :::::::::::::::::::::::: ", args.optimizer)
    # Modified optimizer initialization based on args.optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.finetue_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.finetue_lr,
                                     weight_decay=args.weight_decay)
        
    #optimizer = torch.optim.SGD(model.parameters(), args.finetue_lr, # change this to fine - tune learning rate 0.0010
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    #print(model.parameters())
    model_dir = args.model_dir
    
    start_epoch = 0
    if args.rank == 0 and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    


    if args.rank == 0:
        writer = SummaryWriter(model_dir)
    else:
        writer = None

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    traindir = os.path.join(args.data, 'train_img')
    valdir = os.path.join(args.data, 'test_img')
    print("*********************************")
    print (traindir)
    print("*********************************")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]))


    val_dataset =  datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    #################################################################
    print ("############################################")
    #train_sampler = DistributedSampler(train_dataset)
    #val_sampler = DistributedSampler(val_dataset)
    #train_sampler = SequentialSampler(train_dataset)
    #val_sampler = SequentialSampler(val_dataset)
    
    print ("num worker ..........................."+ str(args.workers))
    #train_loader = DataLoader(
    #    train_dataset, batch_size=args.batch_size//args.world_size, shuffle=True,
    #    num_workers=args.workers, pin_memory=False, sampler=train_sampler)
    #val_loader = DataLoader(
    #    val_dataset, batch_sisze=args.batch_size//args.world_size, shuffle=True,
    #    num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size//args.world_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=2,  # Make sure args.workers is set to a suitable value (4-8)
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch batches to further reduce waiting time
        )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size//args.world_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=2,  # Same number of workers as train loader
        persistent_workers=True
        )

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer)
        return

    niters = len(train_loader)
    
    # does not need adjust lr at this moment
    #lr_scheduler = LRScheduler(optimizer, niters, args)

    best_prec1 = 0.0
    lr_scheduler = LRScheduler(optimizer, niters, args)

    set_us(model,1) 
    display_weights(model)
    #print("initialize")
    initialize_mod_threshold(model)
    if "resnet" in args.model.lower():
        print("summary model....")
        sum_srt,_ = summary(model, input_size=(3, 224, 224))

    set_flops(model)


    flops_layers,params_layers = get_layer_wise_dense_flops_params(model)

    print('Dense Flops of each layer')
    print(flops_layers)

    print('Dense Parameters of each layer')
    print(params_layers)



    # exit(0)
    # print(sum_srt)

    # if rank == 0:
    #     print(sum_srt)

    #     total_sparse_flops,total_dense_flops = compute_flops_reduction(model)
    #     print('Current FLOPs: sparse - {:.4f} M , dense - {:.4f} M, sparse/dense - {:.4f}'.format(
    #         total_sparse_flops*1e-6,total_dense_flops*1e-6, total_sparse_flops/total_dense_flops
    #         )
    #         )





    decay_normalized = normalize_with_flops(model,include_first=True)
    if args.rank ==0:
        print("flops normalization")
        print(decay_normalized)
    

	#my modification    
    if with_weight_penalty == 1: 
      erk_normal = normalize_erk_sparsity(erk_sparsity_dict)
    
      avg_normalization = average_two_normalization(erk_normal,decay_normalized,w1,w2)
      if args.rank ==0:
        print("use (ERK * {} + flops * {}) as normalization".format(w1,w2))
        print(avg_normalization)

    #my modification    
    if with_weight_penalty == 1: 
       apply_normalized_factor(model,avg_normalization)
    



    # initialize_layer_sparse_vis_dict(model)

    
    # exit()
    sparseEpoch = -1
    is_best = False
    save_checkpoint('./usd/schemes/checkpoint', {
                'epoch': 0,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                #'arch_optimizer': arch_optimizer.state_dict(),
    }, is_best)

    append_to_global_file(str(model.check_N_M()), file_path_schema)
    
    for epoch in range(start_epoch, args.epochs):
        #train_sampler.set_epoch(epoch)
        #print("################################################")
        #print ("train....................................")
        #print("################################################")
        if epoch == 2:
            append_to_global_file(str(model.check_N_M()), file_path_schema)
        train(train_loader, model, criterion, optimizer, epoch, writer, iter_update_erk=args.iter_update_erk)

        #print("################################################")
        #print ("validate....................................")
        #print("################################################")
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

            #NewModif
            #save the model at iteration t-1
            #**********
            #torch.save(model.state_dict(), 'model_at_t_minus_1.pth')
												   
            #**********
        if stop == 0 and epoch % 10 == 0:
            save_checkpoint('./usd/schemes/checkpoint', {
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                #'arch_optimizer': arch_optimizer.state_dict(),
            }, False)
        if stop == 1 and is_best:
            save_checkpoint('./usd/schemes/checkpoint', {
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                #'arch_optimizer': arch_optimizer.state_dict(),
            }, is_best)
        
        if stop == 1 and sparseEpoch == -1:
            sparseEpoch = epoch 
        if sparseEpoch > 0 and epoch - sparseEpoch > args.epoch_rfap:
            break
    if rank == 0:
        print("Best accuracy is ",best_prec1 )

    print("file_path_train: ", file_path_train)
    print("file_path_schema: ", file_path_schema)
    print("file_path_sp_layers: ", file_path_sp_layers)
    print("file_path_validate: ", file_path_validate)
    # Count the number of parameters
    #num_params = sum(p.numel() for p in model.parameters())

    # Print the number of parameters
    #print("****Number of parameters : ", num_params)
    SparseParameters = model.get_sparse_parametrers()
    print("****************************************")
    print ("SparseParameters = ",SparseParameters)
    print ("***************************************")

    #print the value of FLOPS
    #num_flops = compute_flops_reduction(model)
    #print ("***Number of FLOPS : ",num_flops)

def train(train_loader, model, criterion, optimizer,epoch, writer, iter_update_erk):
    global DenseParameters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # complexity_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    SAD = AverageMeter()

    # switch to train mode
    model.train()
    world_size = args.world_size
    rank = args.rank

    apply_penalty_flag = False

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global num_iters

            #my modification   

        if with_weight_penalty == 1 and num_iters % 5 == 0  :
            # set apply penalty flag as true
            set_apply_penalty_flag(model,True)
            apply_penalty_flag = True
        num_iters  += 1

        # normally the search requires less than 30000 iters
        if with_weight_penalty == 1 and num_iters in [30000,40000,50000,60000,70000,80000]:
            if rank == 0:
                print('change args.sparse_decay *=2 at num iterations = {}'.format(num_iters))
            args.sparse_decay *= 2
            set_decay_sparse_layer(model,args.sparse_decay)

        # measure data loading time
        data_time.update(time.time() - end)
        #lr_scheduler.update(i, epoch)
        input = input.to('cuda', non_blocking=True)  # non_blocking=True enhances parallelism
        target = target.to('cuda', non_blocking=True)

        #target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / world_size
        current_lr = get_lr(optimizer)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        #dist.all_reduce_multigpu([reduced_loss])
        #dist.all_reduce_multigpu([reduced_prec1])
        #dist.all_reduce_multigpu([reduced_prec5])

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # if epoch > start_epoch_complexity_loss: 
        #     arch_optimizer.zero_grad()
        loss.backward()
        #loss_complexity.backward()
        
        #if epoch > 0:
                #model_at_t_minus_1_state_dict = load_checkpoint(args.model_dir)
        rmsi_errorr = 0
        if i % iter_update_erk == 0 and stop == 0:
            #set_debug_flag_sparseConv(model,tmp_flag)
            #adjust_N_M_of_each_layer(model,target_sparsity,epoch)
            adjust_N_M_of_each_layer_based_on_each_group(model,target_sparsity,epoch,i, erk_value=args.erk_value)  
            rmsi_errorr = model.Total_RMSI_ERROR()
            
        #updating the model's parameters
        optimizer.step()
        
        if apply_penalty_flag == True:
            set_apply_penalty_flag(model,False)
            apply_penalty_flag = False
            # dsiable penalty 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            try:
                total_sparse_flops,total_dense_flops = compute_flops_reduction(model)
            except:
                total_sparse_flops = 0
                total_dense_flops = 0
            
            SparseParameters = model.get_sparse_parametrers()
            #print("****************************************")
            #print ("SparseParameters = ",SparseParameters)
            #print ("***************************************")

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Complexity_Loss {complexity_losses.val:.4f} ({complexity_losses.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  #'Complexity Learning Rate {current_lr:.3f}\t'
                  'Learning Rate {current_lr:.4f}\t' 
                  'ErrorRMSI {errorRmsi:.7f}\t'
                  'SparseParameters {sp}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,current_lr=current_lr, errorRmsi=rmsi_errorr, sp=SparseParameters))
            
            data = '{0},{1},{2},{3},{batch_time.val:.3f},{batch_time.avg:.3f},{data_time.val:.3f},{data_time.avg:.3f},{loss.val:.4f},{loss.avg:.4f},{top1.val:.3f},{top1.avg:.3f},{top5.val:.3f},{top5.avg:.3f},{current_lr:.4f},{errorRmsi:.7f},{sparse:.4f} M,{dense:.4f} M, {denses:.4f} M, {sparses:.4f} M'.format(stop,epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, loss=losses, top1=top1, top5=top5,current_lr=current_lr,errorRmsi=rmsi_errorr, sparse=total_sparse_flops*1e-6,dense=total_dense_flops*1e-6,denses=DenseParameters,sparses=SparseParameters)
            append_to_global_file(data, file_path_train)
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Train/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Train/Avg_Top5', top5.avg / 100.0, niter)

def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    world_size = args.world_size
    rank = args.rank

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            
            #lr_scheduler.update(i, epoch)
            input = input.to('cuda', non_blocking=True)  # non_blocking=True enhances parallelism
            target = target.to('cuda', non_blocking=True)

            #target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)


            # compute output
            output = model(input_var)
            loss = criterion(output, target_var) / world_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1.clone() / world_size
            reduced_prec5 = prec5.clone() / world_size

            #dist.all_reduce_multigpu([reduced_loss])
            #dist.all_reduce_multigpu([reduced_prec1])
            #dist.all_reduce_multigpu([reduced_prec5])

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                
                
        if rank == 0:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            data = '{0},{1},{2},{3},{batch_time.val:.3f},{batch_time.avg:.3f},{loss.val:.4f},{loss.avg:.4f},{top1.val:.3f},{top1.avg:.3f},{top5.val:.3f},{top5.avg:.3f}'.format(stop,epoch, i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1, top5=top5)
            
            append_to_global_file(data, file_path_validate)    
            niter = (epoch + 1)
            writer.add_scalar('Eval/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Eval/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Eval/Avg_Top5', top5.avg / 100.0, niter)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#NewModif
#***************************************************************************************
#Compute the RMSI Error
def compute_rmsi_error_old(net, model_at_t_minus_1):
    rmsi_error = 0
    
    #model_at_t_minus_1 = torch.load('model_at_t_minus_1.pth')
    #model_at_t_minus_1 = models.__dict__[args.model](pretrained=initialisationsetting,N = args.N, M = args.M,search = True, num_new_classes = 100)
    #model_at_t_minus_1.load_state_dict(model_at_t_minus_1_state_dict)



    for (param_t, param_t_minus_1) in zip(net.parameters(), model_at_t_minus_1.parameters()):
        # Assuming param_t and param_t_minus_1 have the same shape
        rmsi_error += torch.sum((param_t - param_t_minus_1) ** 2).item()
    
    # Normalize by the total number of parameters
    total_params = sum(p.numel() for p in net.parameters())
    rmsi_error /= total_params
    
    # Take the square root
    rmsi_error = math.sqrt(rmsi_error)
    return rmsi_error
import hashlib

# Function to calculate the hash of the model's state_dict
def hash_state_dict(state_dict):
    # Concatenate all the parameter tensors into one continuous byte string
    params_bytes = b''.join(tensor.cpu().numpy().tobytes() for tensor in state_dict.values())
    # Compute the hash using hashlib
    return hashlib.sha256(params_bytes).hexdigest()
 
def clone_state_dict(state_dict):
  #return deepcopy(state_dict)
  return {key: tensor.clone() for key, tensor in state_dict.items()}

#***************************************    
#RMSI error
# **************************************    
def compute_rmsi_error(net, model_at_t_minus_1):
    #global model_at_t_minus_1
    rmsi_error = 0
   
    #print("hash of state dict :", hash_state_dict(net.state_dict()))
    if model_at_t_minus_1 == None:
        print ('Init model_at_t_minus_1 RMSI____________')
        #model_at_t_minus_1 = {k: v.to(device) for k, v in clone_state_dict(net.state_dict()).items()}
        model_at_t_minus_1 = {k: v.clone().to(next(net.parameters()).device) for k, v in net.state_dict().items()}
    
     # Ensure that all parameters from the state dictionary are on the same device as the model
    device = next(net.parameters()).device

    # Access state dictionaries of both models
    state_dict_current = net.state_dict()

    # Print keys for diagnostic purposes
    current_keys = set(state_dict_current.keys())
    print (f"Current Keys  in current state dict: {current_keys}")
    previous_keys = set(model_at_t_minus_1.keys())
    print(f"previous_keys in current state dict: {previous_keys}")
    
    missing_keys = previous_keys - current_keys
    additional_keys = current_keys - previous_keys

    if missing_keys:
        print(f"Keys missing in current state dict: {missing_keys}")
    if additional_keys:
        print(f"Additional keys in current state dict: {additional_keys}")

    
    # Compare each parameter between the current network and the state dictionary
    for name, param_t in state_dict_current.items():
        if name not in model_at_t_minus_1:
            raise ValueError(f"Parameter {name} not found in the model_at_t_minus_1 state dictionary")
        
        param_t_minus_1 = model_at_t_minus_1[name]
        rmsi_error += torch.sum((param_t - param_t_minus_1) ** 2).item()
    
    # Normalize by the total number of parameters
    total_params = sum(p.numel() for p in state_dict_current.values())
    rmsi_error /= total_params
    
    # Take the square root
    rmsi_error = math.sqrt(rmsi_error)
    model_at_t_minus_1 = {k: v.to(device) for k, v in clone_state_dict(net.state_dict()).items()}
    return rmsi_error

def count_zero_parameters(model):
    zero_params = 0
    total_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()

    return zero_params, total_params

def test_threshold(net):
    CountDensParam = 0
    CountZeroParam = 0
    CountSparseParam = 0
    global model_at_t_minus_1, layer_at_t_minus_1
    for mod in net.modules():
        #if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            # Compute the threshold for the layer
            #THRESHOLD = mod.compute_threshold_of_layer()
            #element_count = THRESHOLD.numel()
        mod.zero_parameters_below_threshold(layer_at_t_minus_1)
        print (mod)

            # print("Number of elements in the vector:", element_count)
            # print('THRESHOLD:', THRESHOLD[0, 0].item())

            # Apply the zeroing operation
            # layer_at_t_minus_1 = mod.zero_parameters_below_threshold(THRESHOLD[0, 0].item(),layer_at_t_minus_1)

            # print (layer_at_t_minus_1)
            # Count the zero parameters
            # zero_params, total_params = count_zero_parameters(mod)
            # CountDensParam = CountDensParam + total_params
            # CountZeroParam = CountZeroParam + zero_params
            # print(f"Nombre de paramètres zéro: {zero_params}")
            # print(f"Nombre total de paramètres: {total_params}")
            # print(f"Pourcentage de paramètres zéro: {100.0 * zero_params / total_params:.2f}%")
		
    # net_state_dict = net.state_dict()
    # rmsi_error, model_at_t_minus_1 = compute_rmsi_error(net, model_at_t_minus_1)
    # RMSI_Error = rmsi_error

    # print(f"Nombre de paramètres Dense: {CountDensParam}")	
    # print(f"Nombre de paramètres Sparse: {CountDensParam - CountZeroParam}")	
    # print(f"Taux de sparsity globale: {100.0 * CountZeroParam / CountDensParam:.2f}%")								 


    #print ("RMSI_ERROR NEW =  ", RMSI_Error)
    													
#================================================================Below are functions for searching===================================================#

# core function
#rajouter le parametre net_et_t_minus_1 pour cette fonction
def adjust_N_M_of_each_layer_based_on_each_group(net,target_sparsity,epoch,iterations, erk_value=-1):
    global stop
    continuous_sparsity_dict = {}

    change_ = False

    if iterations % 1000 == 0 and args.rank == 0:
        print('Total iterations {}'.format(num_iters))

    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            M = mod.M  
            c_N = mod.N # current c_N
            layer_name = mod.name
            if c_N == 1 : 
                continue
            # elif 'SparseConv0' in layer_name: # you can specify constraints for certain layers if you want
            #     continue
            else:
                #if mod.learned_threshold == None:# initialize threshold T for once
                #commenté by me pour que le threshold soit dynamique et pour initialiser le weight précédent
                if isinstance(mod, SparseConv):
                    mod.update_learned_sparsity()
                elif mod.learned_threshold == None:
                    mod.update_learned_sparsity()
                
                global vote_ratio
                #mod_at_t_minus_1 = None
                #modification de la fonction check_sparsity_each_group en considérant le mod_at_t_minus_1 dans net_at_t_minus_1

                #current_overall_sparsity = net.get_overall_sparsity() 
             
                #if target_sparsity < args.alpha_target_sparsity * target_sparsity:
                #    N_intermediate,flag_ , continuous_sparsity,N_inter_change_flag = mod.check_sparsity_each_group_RMSI(vote_ratio)
                #else:
                N_intermediate,flag_ , continuous_sparsity,N_inter_change_flag = mod.check_sparsity_each_group(vote_ratio)

                continuous_sparsity_dict[layer_name] = continuous_sparsity

                if flag_ == True :
                    #print('change {} to N:M {}:{}'.format(layer_name,mod.N,mod.M))
                    current_overall_sparsity = net.get_overall_sparsity() 
                    current_scheme = net.check_N_M()
                    change_ = True
                    current_flops_ratio = check_current_flops_ratio(net)
                    #if args.rank == 0:
                    #    print('Decision made at iterations {}: change scheme of layer {} to {} - {}'.format(
                    #    num_iters,mod.name,mod.N,mod.M))
                    #    print('Current sparse flops ratio is {}'.format(current_flops_ratio))
                    #    print('Current overall sparsity is {}'.format(current_overall_sparsity))
                    #    print('Schemes at iterations {} is '.format(num_iters))
                    #    print(net.check_N_M())

                    flops_normalization = normalize_with_flops(net,include_first=True)

                    global target_sparsity_erk,erk_sparsity_dict
                    if erk_value < 0:
                        erk_sparsity = update_erk_sparsity(erk_sparsity_dict,net)
                    else:
                        erk_sparsity = erk_sparsity_dict
                    #print("erk_sparsity : ", erk_sparsity)
                    erk_normal = normalize_erk_sparsity(erk_sparsity)

                    global w1,w2
                    
                    new_normalized = average_two_normalization(erk_normal,flops_normalization,w1,w2)

                    # new_normalized = flops_normalization

                    # if args.rank == 0:
                    #     print('new flops normalization')
                    #     print(flops_normalization)
                    #     print('new erk normalization')
                    #     print(erk_normal)
                    #     print("updateing normalizaion with (erk * {} + flops * {})".format(w1,w2))
                    #     print(new_normalized)
                    #     print('\n')

                   #my modification
                    if with_weight_penalty == 1: 
                      apply_normalized_factor(net,new_normalized)

                    decision_dict[current_overall_sparsity] = (iterations,current_scheme)
                    
                    
                    # use codes below when using FLOPs as optimization goal.
                    ###### FLOPS#####, remember to change target_sparse_flops_ratio before #####
                    # if current_flops_ratio >= target_sparse_flops_ratio-0.001:
                    #     if args.rank == 0:
                    #         print('Target target_sparse_flops_ratio {:.3f} has been achieved, current current_flops_ratio is {:.3f}'.format(target_sparse_flops_ratio, current_flops_ratio))
                    #         print('The schemes of each layer')
                    #         print(net.check_N_M())
                    #         total_sparse_flops,total_dense_flops = compute_flops_reduction(net)
                    #         print('Current FLOPs: sparse - {:.4f} M , dense - {:.4f} M, sparse/dense - {:.4f}'.format(
                    #             total_sparse_flops*1e-6,total_dense_flops*1e-6, total_sparse_flops/total_dense_flops
                    #             )
                    #             )
                    #        # print('Decision logs')
                    #        # print(decision_dict)
                    #     exit(0)

                    # use this when using model size as optimization goal.
                    #### Model size#####
                    if current_overall_sparsity >= target_sparsity-0.001:
                        if args.rank == 0:
                            print('Target Sparsity {:.5f} has been achieved, current sparsity is {:.5f}'.format(target_sparsity, current_overall_sparsity))
                            print('The schemes of each layer')
                            print(net.check_N_M())
                            append_to_global_file(str(net.check_N_M()), file_path_schema)
                            total_sparse_flops,total_dense_flops = compute_flops_reduction(net)
                            print('Current FLOPs: sparse - {:.4f} M , dense - {:.4f} M, sparse/dense - {:.4f}'.format(
                                total_sparse_flops*1e-6,total_dense_flops*1e-6, total_sparse_flops/total_dense_flops
                                )
                                )
                            SparseParameters = net.get_sparse_parametrers()
                            print("****************************************")
                            print ("SparseParameters = ",SparseParameters)
                            print ("***************************************")
                            # print('Decision logs')
                            # print(decision_dict)
                            
                            # Count the number of parameters
                            #num_params = sum(p.numel() for p in net.parameters())

                            # Print the number of parameters
                            #print("****Number of parameters after sparsification: ", num_params)
                        stop = 1
                        break
                        #exit(0)
    if iterations % 200 == 0 and args.rank == 0:
        strr = f'{epoch},{iterations},' + ",".join(map(lambda x: str(x.item()), continuous_sparsity_dict.values()))
        append_to_global_file(strr, file_path_sp_layers)
    
    values = np.array([x.item() for x in continuous_sparsity_dict.values()])
    avg_sparsity = np.mean(values)
    #print("avg_spars", avg_sparsity, args.alpha_target_sparsity * target_sparsity)
    if avg_sparsity >= args.alpha_target_sparsity * target_sparsity:
        set_us(net, 0)

    if stop == 1:
        for mod in net.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                mod.mix_from_dense = False

    

# def check_contributions_gradient_and

def average_two_normalization(norm_dict1,norm_dict2,w1=0.5,w2=0.5):
    if norm_dict1 == None:
        return norm_dict2
    elif norm_dict2 == None :
        return norm_dict1

    new_dict = {}
    for layer_name, value in norm_dict1.items():
        new_dict[layer_name] = (value*w1 + norm_dict2[layer_name]*w2)
    
    return new_dict

def set_apply_penalty_flag(net,flag):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            mod.apply_penalty = flag

def set_debug_flag_sparseConv(net,flag):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            mod.print_flag = flag

def initialize_mod_threshold(net):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
            if mod.learned_threshold == None:
                mod.update_learned_sparsity()

def set_flops(net):
    for mod in net.modules():
        if isinstance(mod, SparseConv): 
            # dense parameters
            in_shape = np.array(mod.input_shape)
            
            out_shape = np.array(mod.output_shape)
            mod.flops =  conv_flops_counter_hook(mod,in_shape,out_shape)

            #print('{} input-{} output-{} FLOPs-{}M'.format(mod.name,in_shape,out_shape,mod.flops*1e-6))
        elif isinstance(mod, SparseLinear): 
            input_ = mod.in_features
            # pytorch checks dimensions, so here we don't care much
            output_last_dim = mod.out_features
            bias_flops = output_last_dim if mod.bias is not None else 0

            mod.flops = int(input_ * output_last_dim + bias_flops)
            #print('{} input-{} output-{} FLOPs-{}M'.format(mod.name,input_,output_last_dim,mod.flops*1e-6))

# dense/sparse
def check_current_flops_ratio(net):
    sparse_flops = 0
    dense_flops = 0

    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 

            dense_flops += mod.flops
            sparse_flops += mod.flops * mod.N/mod.M   

    
    return 1.0  - (sparse_flops/dense_flops)

def apply_normalized_factor(net,decay_normalized):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            layer_name = mod.name
            mod.normalized_factor = decay_normalized[layer_name]

def print_log_sparse_layer(net):
    log_dict = {}
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            layer_name = mod.name
            log_dict[layer_name] = mod.log
    
    return log_dict

def get_N_inter_M(net):
    sparse_scheme = {}
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            layer_name = mod.name
            sparse_scheme[mod.get_name()] = list([mod.N_intermediate,mod.M])
    return sparse_scheme

def get_overall_sparsity_N_inter_M(net):
    sparse_paras = 0
    dense_paras = 0
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            sparse_paras += mod.dense_parameters * mod.N_intermediate / mod.M 
            dense_paras += mod.dense_parameters

    return 1.0 - (sparse_paras/dense_paras)  

def normalize_with_flops(net,include_first=True):
    decay_normalized = {}

# and include_first==True 
    normalized_factor = []
    layer_list = []
    valid_flops_list = []
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) : 
            layer_name = mod.name
            layer_list.append(layer_name)
            dense_flops = mod.flops
            sparse_flops = mod.flops * mod.N / mod.M  
            normalized_factor.append(sparse_flops)
            if 'SparseConv0' in layer_name:
                if include_first == True:
                    valid_flops_list.append(sparse_flops)
                else:
                    continue
            else:        
                valid_flops_list.append(sparse_flops)

        

    valid_flops_list = np.asarray(valid_flops_list)

    normalized_factor = np.asarray(normalized_factor)
    normalized_factor = normalized_factor/np.amax(valid_flops_list)

    normalized_factor = normalized_factor.tolist()
    zip_iterator = zip(layer_list, normalized_factor)
    decay_normalized = dict(zip_iterator)

    return decay_normalized

# return total sparse and dense flops
def compute_flops_reduction(net):
    total_dense_flops = 0
    total_sparse_flops = 0
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) : 
            layer_name = mod.name
            
            dense_flops = mod.flops
            sparse_flops = mod.flops * mod.N / mod.M  
            total_dense_flops += dense_flops
            total_sparse_flops += sparse_flops
    
    return total_sparse_flops,total_dense_flops
          
def set_decay_sparse_layer(net,decay):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            mod.decay = decay

def set_alpha(net,alpha):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            mod.alpha = alpha                              

def set_us(net,us):
    for mod in net.modules():
        if isinstance(mod, (SparseConv, SparseLinear)):
            mod.us = us

def display_weights(net):
    for mod in net.modules():
        if isinstance(mod, (SparseConv, SparseLinear)):
            print(mod.name, "   ", mod.weight.numel())
# def check_sparse_pattern(network,N,M):
def adjust_learning_rate(optimizer, epoch):
    #global state
    if epoch in args.schedule:
        args.finetue_lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.finetue_lr

def adjust_learning_rate_complexity(optimizer, epoch):
    #global state
    if epoch in args.schedule:
        args.lra *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lra

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
# Function to load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_state_dict = checkpoint['model_state_dict']
    return model_state_dict

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)  # top 20 functions by cumulative time