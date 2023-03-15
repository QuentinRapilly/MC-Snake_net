from .u2D_2D import Unet2D_2D
from typing import Tuple
import torch.nn as nn

class MCSnakeNet(Unet2D_2D):

    def __init__(self, img_shape : Tuple[float, float], nb_control_points : int = 10, **kwargs):
        super().__init__(**kwargs)
        self.img_height, self.img_width = img_shape
        self.bottleneck_dim = (self.features_start*self.img_height*self.img_width)//2**(self.num_layers-1)
        self.nb_control_points = nb_control_points
        self.snake_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.bottleneck_dim, out_features=2*nb_control_points)
        )

    
    def forward(self, x):
        
        bottleneck, skips = self.encode(x)
        decoder_features = self.decode(bottleneck, skips)

        mask = self.layers['final_layer'](decoder_features)

        control_points = self.snake_layer(bottleneck)


        return mask, control_points 
