from __future__ import division
import os
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
import argparse
from time import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
from search import models
import numpy as np
from devkit.sparse_ops import SparseConv,SparseLinear
from devkit.core import (load_state_ckpt,set_sparse_scheme)
from devkit.dataset.imagenet_dataset import ImagenetDataset

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

#import wandb
#wandb.init(project="dominoSearchTraining")
# Sparse
import ast 
parser = argparse.ArgumentParser(
    description='Pytorch USD Testing Inference')
parser.add_argument('--config', default='configs/config_resnet50_2:4.yaml')
parser.add_argument('--checkpoint_path', default='resnet18/resnet18_sparse/model-best.pth')
parser.add_argument('--original_model', default='inference_benchmark/full_models/resnet18_cifar10_full.pth')
parser.add_argument('--schemes_file', default='schemes/schema_file.txt')
parser.add_argument('--print_freq', default=5, type=int)
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('-e', '--evaluate', default=True, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()
file_path_validate = "./inference_benchmark/test_file.csv"

def append_to_global_file(data, file_path):
    with open(file_path, "a") as file:
        file.write(data + "\n")
        file.close()

isCuda = torch.cuda.is_available()    

def main():
    global args, best_prec1, DenseParameters, file_path_validate
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print("checkpoint_path", args.checkpoint_path)
    rank = 0
    num_classes = args.num_classes  # Set the desired number of output classes
    print("num classes = ...................... " + str(num_classes))
    # create model
    decay = args.decay
    if rank == 0:
        print("=> creating model '{}'".format(args.model))

    model = models.__dict__[args.model](pretrained=False,N = args.N, M = args.M, num_classes = num_classes, num_new_classes = num_classes, search=False)
    #if not os.path.exists(args.original_model):
    #    print("original model parameter ( --original_model <original_model>) must be specified")
    #    exit(0)
    
    if not os.path.exists(args.checkpoint_path):
        print("checkpoint model parameter ( --checkpoint_path <checkpoint_path>) must be specified (finetuned model)")
        exit(0)
    if not os.path.exists(args.schemes_file):
        print("sparse schemes model parameter ( --schemes_file <schemes_file>) must be specified (N:M)")
        exit(0)
    
    cudnn.benchmark = True
    
    model_dict = torch.load(args.checkpoint_path, weights_only=True)
    model.load_state_dict(model_dict["state_dict"])
    DenseParameters = model.get_dense_parametrers()
    print("****************************************")
    print ("DenseParameters = ",DenseParameters)
    print ("***************************************")
    # switch to evaluate mode
    set_eval(model, evaluate=True)

    model.eval()

    N_file = args.N
    M_file = args.M

    batchs = (10, 20, 30, 35, 40, 45, 50, 60, 120, 240, 512, 600)

    file_path_validate = "./inference_benchmark/validate_file_" + args.model + "_" + str(N_file)+"_" + str(M_file) + ".csv"
    
    with open(file_path_validate, "w") as file:
        pass

    header = 'batch_size,batch_time,batch_time_sp,Loss,Loss_sp,Prec@1,Prec@1_sp,Prec@5,Prec@5_sp'
    append_to_global_file(header, file_path_validate)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = ImagenetDataset(
        args.val_root,
        args.val_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    if isCuda:
        criterion = nn.CrossEntropyLoss().cuda()
        model.cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    #if args.evaluate:
    #    print("evaluate the dense model for inference acceleration performance")
    #    for batch_size in batchs:
    #        val_loader = DataLoader(
    #            val_dataset, batch_size=batch_size, shuffle=True)
    #        validate(val_loader, model, batch_size, criterion, dense=1)

    with open(args.schemes_file) as f:
        first_line = f.readline()
    sparse_schemes = ast.literal_eval(first_line)   

    set_sparse_scheme(model,sparse_schemes)
    set_weight_penalty(model,decay)
    #print('Use schemes file {}'.format(args.schemes_file))
    #print("Start to train mixed Sparse NN")
    #print(model)
    print(model.check_N_M())
    

    sparseParameters = model.get_sparse_parametrers()
    print("****************************************")
    print ("SparseParameters = ",sparseParameters)
    print ("***************************************")

    model.eval()

    if args.evaluate:
        print("evaluate the dense and sparse model for inference acceleration performance")
        for batch_size in batchs:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True)
            validate(val_loader, model, batch_size, criterion, dense=0)
        exit(0)

def validate(val_loader, model, batchsize, criterion, dense=1):
    batch_time = AverageMeter()
    batch_time_sparse = AverageMeter()
    losses_sparse = AverageMeter()
    top1_sparse = AverageMeter()
    top5_sparse = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    world_size = 1
    rank = 0

    batch_times = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if isCuda:
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input.cuda())
            else:
                input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            set_dense(model, dense=True)
            output = model(input_var)
            start_time = time()
            output = model(input_var)
            end_time = time()
            batch_time.update(end_time - start_time, input.size(0))
            loss = criterion(output, target_var) / world_size
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data.clone()
            losses.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            #------------------------------------------------
            set_dense(model, dense=False)

            start_time = time()
            output = model(input_var)
            end_time = time()
            batch_time_sparse.update(end_time - start_time, input.size(0))
            loss = criterion(output, target_var) / world_size
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data.clone()
            losses_sparse.update(reduced_loss.item(), input.size(0))
            top1_sparse.update(prec1.item(), input.size(0))
            top5_sparse.update(prec5.item(), input.size(0))

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Time(SP) {batch_time_sparse.val:.3f} ({batch_time_sparse.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss(SP) {loss_sparse.val:.4f} ({loss_sparse.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@1(SP) {top1_sparse.val:.3f} ({top1_sparse.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      'Prec@5(SP) {top5_sparse.val:.3f} ({top5_sparse.avg:.3f})'.format(
                    i, len(val_loader), 
                    batch_time=batch_time, batch_time_sparse=batch_time_sparse, 
                    loss=losses,loss_sparse=losses_sparse,
                    top1=top1, top1_sparse=top1_sparse, 
                    top5=top5, top5_sparse=top5_sparse
                    ))
        if rank == 0:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            data = '{0},{batch_time.avg:.3f},{batch_time_sparse.avg:.3f},{loss.val:.4f},{loss_sparse.val:.4f},{top1.val:.3f},{top1_sparse.val:.3f},{top5.avg:.3f},{top5_sparse.avg:.3f}'.format(
                batchsize, batch_time= batch_time, batch_time_sparse=batch_time_sparse
                ,loss=losses,loss_sparse=losses_sparse
                ,top1=top1,top1_sparse=top1_sparse
                , top5=top5, top5_sparse=top5_sparse
                )
            append_to_global_file(data, file_path_validate)

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

def set_weight_penalty(model, decay):
    for mod in model.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) : 
            mod.decay = decay

def set_eval(net,evaluate=True):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            mod.evaluate = evaluate

def set_dense(net,dense=True):
    for mod in net.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear): 
            mod.dense = dense

if __name__ == '__main__':
    main()
