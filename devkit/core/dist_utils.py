import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

__all__ = [
    'init_dist', 'broadcast_params','average_gradients']

def init_dist(backend='nccl',
              master_ip='127.0.0.1',
              port=29500):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(port)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print("1------- Port : " + str(port))
    num_gpus = torch.cuda.device_count()
    print("2-------" + str(num_gpus))
    torch.cuda.set_device(rank % num_gpus)
    print("3-------" + str(num_gpus))

    if not dist.is_available():
      print("Distributed training is not available.")
    

    if dist.is_initialized():
      print("Distributed training is already initialized.")
      
    
    dist.init_process_group(backend='nccl', init_method='env://')
    print("4-------" + str(num_gpus))
    #torch.cuda.set_device(dist.get_rank())
    dist.init_process_group(backend=backend)
    print("5-------" + str(num_gpus))
    return rank, world_size



def average_gradients(model):
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)

def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

