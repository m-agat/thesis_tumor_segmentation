from torchvision.models.densenet import _load_state_dict
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

# https://github.com/stefano-malacrino/DenseUNet-pytorch/blob/master/dense_unet.py

class _DenseUNetEncoder3D(nn.Module):
    def __init__(self, skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample):
        super(_DenseUNetEncoder3D, self).__init__()

        self.skip_connections = skip_connections
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition3D(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        if not downsample:
            self.features.conv0.stride = 1
            del self.features.pool0

        for module in self.features.modules():
            if isinstance(module, nn.MaxPool3d):
                module.register_forward_hook(lambda _, input, output: self.skip_connections.append(input[0]))

    def forward(self, x):
        return self.features(x)
        
class _DenseUNetDecoder3D(nn.Module):
    def __init__(self, skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, upsample):
        super(_DenseUNetDecoder3D, self).__init__()
        
        self.skip_connections = skip_connections
        self.upsample = upsample
        
        # remove conv0, norm0, relu0, pool0, last denseblock, last norm, classifier
        features = list(self.features.named_children())[4:-2]
        delattr(self, 'classifier')

        num_features = num_init_features
        num_features_list = []
        for i, num_layers in enumerate(block_config):
            num_input_features = num_features + num_layers * growth_rate
            num_output_features = num_features // 2
            num_features_list.append((num_input_features, num_output_features))
            num_features = num_input_features // 2
        
        for i in range(len(features)):
            name, module = features[i]
            if isinstance(module, _Transition3D):
                num_input_features, num_output_features = num_features_list.pop(1)
                features[i] = (name, _TransitionUp3D(num_input_features, num_output_features, skip_connections))

        features.reverse()
        
        self.features = nn.Sequential(OrderedDict(features))
        
        num_input_features, _ = num_features_list.pop(0)
        
        if upsample:
            self.features.add_module('upsample0', nn.Upsample(scale_factor=4, mode='trilinear'))
        self.features.add_module('norm0', nn.BatchNorm3d(num_input_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('conv0', nn.Conv3d(num_input_features, num_init_features, kernel_size=1, stride=1, bias=False))
        self.features.add_module('norm1', nn.BatchNorm3d(num_init_features))

    def forward(self, x):
        return self.features(x)
          
class _Concatenate3D(nn.Module):
    def __init__(self, skip_connections):
        super(_Concatenate3D, self).__init__()
        self.skip_connections = skip_connections
        
    def forward(self, x):
        return torch.cat([x, self.skip_connections.pop()], 1)

class _DenseLayer3D(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate,
                                          kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer3D, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        
class _DenseBlock3D(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock3D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer3D(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
            
class _Transition3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.MaxPool3d(kernel_size=2, stride=2))
          
class _TransitionUp3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, skip_connections):
        super(_TransitionUp3D, self).__init__()
        
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, num_output_features * 2,
                                              kernel_size=1, stride=1, bias=False))
        
        self.add_module('upsample', nn.Upsample(scale_factor=2, mode='trilinear'))
        self.add_module('cat', _Concatenate3D(skip_connections))
        self.add_module('norm2', nn.BatchNorm3d(num_output_features * 4))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(num_output_features * 4, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class DenseUNet3D(nn.Module):
    def __init__(self, n_classes, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, downsample=False, pretrained_encoder_uri=None, progress=None):
        super(DenseUNet3D, self).__init__()
        self.skip_connections = []
        self.encoder = _DenseUNetEncoder3D(self.skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample)
        self.decoder = _DenseUNetDecoder3D(self.skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample)
        self.classifier = nn.Conv3d(num_init_features, n_classes, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        
        self.encoder._load_state_dict = self.encoder.load_state_dict
        self.encoder.load_state_dict = lambda state_dict : self.encoder._load_state_dict(state_dict, strict=False)
        if pretrained_encoder_uri:
            _load_state_dict(self.encoder, str(pretrained_encoder_uri), progress)
        self.encoder.load_state_dict = lambda state_dict : self.encoder._load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        y = self.classifier(x)
        return self.softmax(y)
