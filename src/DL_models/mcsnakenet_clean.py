from .u2D_2D import Unet2D_2D
from typing import Tuple
import torch.nn as nn

class MCSnakeNet(Unet2D_2D):

    def __init__(self, img_shape : Tuple[float, float], nb_control_points : int, nb_snake_layers : int, **kwargs):
        super().__init__(**kwargs)
        self.img_height, self.img_width = img_shape
        self.nb_snake_layers = nb_snake_layers
        self.nb_control_points = nb_control_points

        self.bottleneck_dim = (self.features_start*self.img_height*self.img_width)//2**(self.num_layers-1)

        self.FC_dim = [self.bottleneck_dim] + [4096 for _ in range(nb_snake_layers-1)] + [2*nb_control_points]


        self.snake_head = nn.ModuleDict({
            #"conv_1x1" : nn.Conv2d(in_channels=self.bottleneck_nb_features, out_channels=1, kernel_size=1),
            "FC" : nn.ModuleList([nn.Sequential(nn.Linear(self.FC_dim[i], self.FC_dim[i+1]),nn.BatchNorm1d(self.FC_dim[i+1]))\
                                  for i in range(self.nb_snake_layers)])
        })

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        

    
    def forward(self, x):
        
        bottleneck, skips = self.encode(x)
        decoder_features = self.decode(bottleneck, skips)

        mask = self.layers['final_layer'](decoder_features)

        #cp = self.flatten(self.snake_head["conv_1x1"](bottleneck))
        cp = self.flatten(bottleneck)

        for i in range(self.nb_snake_layers-1):
            cp = self.relu(self.snake_head["FC"][i](cp))
            
        cp = self.snake_head["FC"][-1](cp)

        return mask, cp
