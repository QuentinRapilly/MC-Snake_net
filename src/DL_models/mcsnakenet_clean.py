from .u2D_2D import Unet2D_2D
from typing import Tuple
import torch.nn as nn

class MCSnakeNet(Unet2D_2D):

    def __init__(self, img_shape : Tuple[float, float], nb_control_points : int, nb_snake_layers : int, **kwargs):
        super().__init__(**kwargs)
        self.img_height, self.img_width = img_shape
        self.nb_snake_layers = nb_snake_layers
        self.bottleneck_dim = (self.img_height*self.img_width)//4**(self.num_layers-1)
        self.bottleneck_nb_features = self.features_start*2**(self.num_layers-1)
        self.bottleneck_conv = nn.Conv2d(in_channels=self.bottleneck_nb_features, out_channels=1, kernel_size=1)
        self.snake_layers_dim = [self.bottleneck_dim] +[(2*nb_control_points)*2**(nb_snake_layers-i-1) for i in range(self.nb_snake_layers)]
        self.nb_control_points = nb_control_points

        self.snake_layers = nn.ModuleList([nn.Linear(self.snake_layers_dim[i], self.snake_layers_dim[i+1]) for i in range(self.nb_snake_layers)])
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        

    
    def forward(self, x):
        
        bottleneck, skips = self.encode(x)
        decoder_features = self.decode(bottleneck, skips)

        mask = self.layers['final_layer'](decoder_features)



        cp = self.flatten(self.bottleneck_conv(bottleneck))

        for i in range(self.nb_snake_layers-1):
            cp = self.relu(self.snake_layers[i](cp))
            
        cp = self.snake_layers[-1](cp)

        return mask, cp
