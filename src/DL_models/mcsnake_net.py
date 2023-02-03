import torch
from torch import nn 


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, padding = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode="zeros")
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode="zeros")
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x


class MCSnakeNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, typeA="classic", typeB="snake", nb_control_points = 8):
        super().__init__()
        self.encoder = Encoder(enc_chs)

        self.decoderA = Decoder(dec_chs)
        self.decoderB = Decoder(dec_chs)

        self.typeA = typeA
        self.typeB = typeB

        self.nb_control_points = nb_control_points
        
        possible_types = ["classic", "snake"]

        assert (typeA in possible_types) and (typeB in possible_types)

        if typeA == "classic":
            self.headA = nn.Conv2d(dec_chs[-1], num_class, 1) # TODO definir correctement
        if typeA == "snake":
            self.headA = nn.Conv2d(dec_chs[-1], num_class, 1) # TODO definir correctement

        if typeB == "classic":
            self.headB = nn.Conv2d(dec_chs[-1], num_class, 1) # TODO definir correctement
        if typeB == "snake":
            self.headB = nn.Conv2d(dec_chs[-1], num_class, 1) # TODO definir correctement


    def forward(self, x):
        enc_ftrs = self.encoder(x)

        outA = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        outA = self.headA(outA)

        outB = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        outB = self.headB(outB)
        return outA, outB

    



