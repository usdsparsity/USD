import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../../')))
from devkit.sparse_ops import SparseConv, SparseLinear
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNetV1', 'resnet18_sparse', 'resnet34_sparse', 'resnet50_sparse', 'resnet101_sparse',
           'resnet152_sparse', 'resnet56_sparse', 'resnet110_sparse']


model_urls = {
    'resnet18_sparse': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34_sparse': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50_sparse': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101_sparse': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152_sparse': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet56_sparse': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56-4bfd9763.th',
    'resnet110_sparse': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th',
}

def conv3x3(in_planes, out_planes, stride=1, N=2, M=4,search=False):
    """3x3 convolution with padding"""
    return SparseConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, N=N, M=M,search=search)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlockRes(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, option='A', N=2, M=4,search=False):
        super(BasicBlockRes, self).__init__()
        self.conv1 = SparseConv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, N=N, M=M,search=search)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SparseConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, N=N, M=M,search=search)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     SparseConv(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, N=N, M=M,search=search),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

from collections import OrderedDict
def load_state_dict_from_url_sort(url_model, progress):
    checkpoint = load_state_dict_from_url(url_model,
                                              progress=progress)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, N=2, M=4,search=False):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, N=N, M=M,search=search)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, N=N, M=M,search=search)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, N=2, M=4,search=False):
        super(Bottleneck, self).__init__()

        self.conv1 = SparseConv(inplanes, planes, kernel_size=1, bias=False, N=N, M=M,search=search)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SparseConv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, N=N, M=M,search=search)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SparseConv(planes, planes * 4, kernel_size=1, bias=False, N=N, M=M,search=search)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetV1(nn.Module):

    def __init__(self, block, layers, num_classes=1000,num_new_classes=1000,  N=2, M=4,search=False):
        super(ResNetV1, self).__init__()


        self.N = N
        self.M = M

        self.num_new_classes = num_new_classes

        self.named_layers = {} # layers have parameters like conv2d and linear
        self.dense_layers = {} # layers which will be kept as dense

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        #self.conv1 = SparseConv(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False, N=self.N, M=self.M,search=search)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], N = self.N, M = self.M,search=search)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, N = self.N, M = self.M,search=search)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, N = self.N, M = self.M,search=search)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, N = self.N, M = self.M,search=search)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = SparseLinear(512 * block.expansion, num_classes,N=N,M=M,search=search) # nn.Linear(512 * block.expansion, num_classes)

        if num_classes != 1000:
            self.init_fc = True
        else:
            self.init_fc = False

        self._set_sparse_layer_names()

        for m in self.modules():
            if isinstance(m, SparseConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, N = 2, M = 4,search=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SparseConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,  N=N, M=M,search=search),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, N=N, M=M,search=search))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, N=N, M=M,search=search))

        return nn.Sequential(*layers)

    def reset_classifier(self, num_classes, global_pool=''):
        if num_classes != 1000:
            print("reset fc linear layer..........")
            num_features = self.fc.in_features
            self.fc = SparseLinear(num_features, num_classes, N=self.N, M=self.M, search=True)
        self._set_sparse_layer_names()

    def set_weight_decay(self, weight_decay):
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) :
                mod.decay = weight_decay

    def _get_sparse_layer_names(self):
        layers = ""
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                layers = layers + "," + mod.name.replace(',', ' ')
        return layers
    
    def _set_sparse_layer_names(self):
        conv2d_idx = 0
        linear_idx = 0

        for mod in self.modules():
            if isinstance(mod, SparseConv):
                layer_name = 'SparseConv{}_{}-{}-{}'.format(
                    conv2d_idx, mod.in_channels, mod.out_channels,mod.kernel_size
                )
                
                mod.set_layer_name(layer_name)

                Cout = mod.weight.data.size()[0]
                C = mod.weight.data.size()[1]
                Kw = mod.weight.data.size()[2]
                Kh = mod.weight.data.size()[3]
                
                mod.layer_ind = conv2d_idx
                self.named_layers[layer_name] = list([Cout,C,Kw,Kh])

                conv2d_idx += 1
            # elif isinstance(mod, torch.nn.BatchNorm2d):
            #     layer_name = 'BatchNorm2D{}_{}'.format(
            #         batchnorm2d_idx, mod.num_features)
            #     named_layers[layer_name] = mod
            #     batchnorm2d_idx += 1
            elif isinstance(mod, SparseLinear):
                mod.out_features = self.num_new_classes
                layer_name = 'Linear{}_{}-{}'.format(
                    linear_idx, mod.in_features, mod.out_features
                )
                print("########################## layer_name " + layer_name)
                Cout = mod.weight.data.size()[0]
                C = mod.weight.data.size()[1]

                mod.set_layer_name(layer_name)
                mod.layer_ind = linear_idx

                self.named_layers[layer_name] = list([Cout,C])
                #self.dense_layers[layer_name] = list([Cout,C])

                linear_idx += 1

    def set_datalayout(self,layout):
        for mod in self.modules():
            if isinstance(mod, SparseConv): # for Linear Layer, data layout does not matter
                mod.change_layout(layout)

    def check_N_M(self):
        sparse_scheme = {}

        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                sparse_scheme[mod.get_name()] = list([mod.N,mod.M])
            #elif isinstance(mod, torch.nn.Linear): TODOs
            #    pass
        return sparse_scheme

    def get_overall_sparsity(self):
        dense_paras = 0
        sparse_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                dense_paras += mod.dense_parameters 
                sparse_paras += mod.get_sparse_parameters()    # number(M) of non-zeros
            # elif isinstance(mod, torch.nn.Linear): # at this moment we keep fully connected layer as dense, and does not account this layer
            #     dense_paras += mod.weight.data.size()[0] * mod.weight.data.size()[1]
            #     sparse_paras += 0
        
        return 1.0 - (sparse_paras/dense_paras)
    
    def Total_RMSI_ERROR (self):
        total_rms = 0.0
        nblayers = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                total_rms+=mod.RMSI_ERROR
                nblayers += 1
        #print ("nblayers = ", nblayers)
        total_rms /= nblayers
        return (total_rms )

    #*******************************************************************************
    def get_dense_parametrers(self):
        dense_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                dense_paras += mod.dense_parameters         
        return dense_paras

    def get_sparse_parametrers(self):
        sparse_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                sparse_paras += mod.get_sparse_parameters()         
        return sparse_paras
    #*******************************************************************************

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class ResNetV2(nn.Module):

    def __init__(self, block, layers, num_classes=10,num_new_classes=1000,  N=2, M=4,search=False):
        super(ResNetV2, self).__init__()

        self.N = N
        self.M = M

        self.num_new_classes = num_new_classes

        self.named_layers = {} # layers have parameters like conv2d and linear
        self.dense_layers = {} # layers which will be kept as dense

        self.inplanes = 16
        self.conv1 = SparseConv(3, 16, kernel_size=3, stride=1, padding=1, bias=False, N=self.N, M=self.M,search=search)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.linear = SparseLinear(64, num_classes,N=N,M=M,search=search)

        if num_classes != 10:
            self.init_fc = True
        else:
            self.init_fc = False
            
        self._set_sparse_layer_names()

        for m in self.modules():
            if isinstance(m, SparseConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def reset_classifier(self, num_classes, global_pool=''):
        if not self.init_fc:
            print("reset fc linear layer..........")
            num_features = self.linear.in_features
            self.linear = SparseLinear(num_features, num_classes, N=self.N, M=self.M, search=True)
        self._set_sparse_layer_names()

    def set_weight_decay(self, weight_decay):
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) :
                mod.decay = weight_decay

    def _get_sparse_layer_names(self):
        layers = ""
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                layers = layers + "," + mod.name.replace(',', ' ')
        return layers
    
    def _set_sparse_layer_names(self):
        conv2d_idx = 0
        linear_idx = 0

        for mod in self.modules():
            if isinstance(mod, SparseConv):
                layer_name = 'SparseConv{}_{}-{}-{}'.format(
                    conv2d_idx, mod.in_channels, mod.out_channels,mod.kernel_size
                )
                
                mod.set_layer_name(layer_name)

                Cout = mod.weight.data.size()[0]
                C = mod.weight.data.size()[1]
                Kw = mod.weight.data.size()[2]
                Kh = mod.weight.data.size()[3]
                
                mod.layer_ind = conv2d_idx
                self.named_layers[layer_name] = list([Cout,C,Kw,Kh])

                conv2d_idx += 1
            # elif isinstance(mod, torch.nn.BatchNorm2d):
            #     layer_name = 'BatchNorm2D{}_{}'.format(
            #         batchnorm2d_idx, mod.num_features)
            #     named_layers[layer_name] = mod
            #     batchnorm2d_idx += 1
            elif isinstance(mod, SparseLinear):
                mod.out_features = self.num_new_classes
                layer_name = 'Linear{}_{}-{}'.format(
                    linear_idx, mod.in_features, mod.out_features
                )
                print("########################## layer_name " + layer_name)
                Cout = mod.weight.data.size()[0]
                C = mod.weight.data.size()[1]

                mod.set_layer_name(layer_name)
                mod.layer_ind = linear_idx

                self.named_layers[layer_name] = list([Cout,C])
                #self.dense_layers[layer_name] = list([Cout,C])

                linear_idx += 1

    def set_datalayout(self,layout):
        for mod in self.modules():
            if isinstance(mod, SparseConv): # for Linear Layer, data layout does not matter
                mod.change_layout(layout)

    def check_N_M(self):
        sparse_scheme = {}

        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                sparse_scheme[mod.get_name()] = list([mod.N,mod.M])
            #elif isinstance(mod, torch.nn.Linear): TODOs
            #    pass
        return sparse_scheme

    def get_overall_sparsity(self):
        dense_paras = 0
        sparse_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                dense_paras += mod.dense_parameters 
                sparse_paras += mod.get_sparse_parameters()    # number(M) of non-zeros
            # elif isinstance(mod, torch.nn.Linear): # at this moment we keep fully connected layer as dense, and does not account this layer
            #     dense_paras += mod.weight.data.size()[0] * mod.weight.data.size()[1]
            #     sparse_paras += 0
        
        return 1.0 - (sparse_paras/dense_paras)
    
    def Total_RMSI_ERROR (self):
        total_rms = 0.0
        nblayers = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                total_rms+=mod.RMSI_ERROR
                nblayers += 1
        #print ("nblayers = ", nblayers)
        total_rms /= nblayers
        return (total_rms )

    #*******************************************************************************
    def get_dense_parametrers(self):
        dense_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                dense_paras += mod.dense_parameters         
        return dense_paras

    def get_sparse_parametrers(self):
        sparse_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                sparse_paras += mod.get_sparse_parameters()         
        return sparse_paras
    #*******************************************************************************

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet18_sparse(pretrained=False,progress=True,**kwargs):
    model = ResNetV1(BasicBlock, [2, 2, 2, 2],  **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18_sparse'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet34_sparse(pretrained=False,progress=True,**kwargs):
    model = ResNetV1(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34_sparse'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50_sparse(pretrained=False,progress=True,**kwargs):
    model = ResNetV1(Bottleneck, [3, 4, 6, 3],  **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_sparse'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet101_sparse(pretrained=False,progress=True,**kwargs):
    model = ResNetV1(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101_sparse'],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model

def resnet152_sparse(pretrained=False,progress=True,**kwargs):
    model = ResNetV1(Bottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152_sparse'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet56_sparse(pretrained=False,progress=True,**kwargs):
    
    model = ResNetV2(BasicBlockRes, [9, 9, 9], **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url_sort(model_urls['resnet56_sparse'],
                                              progress=progress)
        #checkpoint = load_state_dict_from_url(model_urls['resnet56_sparse'],
        #                                      progress=progress, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model

def resnet110_sparse(pretrained=False,progress=True,**kwargs):
    model = ResNetV2(BasicBlockRes, [18, 18, 18], **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url_sort(model_urls['resnet110_sparse'],
                                              progress=progress)
        #checkpoint = load_state_dict_from_url(model_urls['resnet56_sparse'],
        #                                      progress=progress, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model