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
    
    def forward(self, x, verbose = False):
        ftrs = []
        for i,block in enumerate(self.enc_blocks):
            x = block(x)
            if verbose : 
                print("Encoder, step {}, shape = {}".format(i,x.shape))
            ftrs.append(x)
            x = self.pool(x)
        return ftrs



class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features, verbose = False):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
            if verbose : 
                print("Decoder, step {}, shape = {}".format(i,x.shape))
        return x


class MCSnakeNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1,\
                 typeA="classic", typeB="snake", nb_control_points = 8, img_shape = (256,256)):
        super().__init__()
        self.encoder = Encoder(enc_chs)

        self.decoderA = Decoder(dec_chs)
        self.decoderB = Decoder(dec_chs)

        self.typeA = typeA
        self.typeB = typeB

        self.img_shape = img_shape
        self.flatten_img_shape = img_shape[0]*img_shape[1]

        self.nb_control_points = nb_control_points
        
        possible_types = ["classic", "snake"]

        assert (typeA in possible_types) and (typeB in possible_types)

        if typeA == "classic":
            self.headA = nn.Conv2d(dec_chs[-1], num_class, 1)
        if typeA == "snake":
            self.headA = nn.Sequential(
                nn.Conv2d(dec_chs[-1], num_class, 1),
                nn.Flatten(),
                nn.Linear(in_features=self.flatten_img_shape,out_features=2*nb_control_points))

        if typeB == "classic":
            self.headB = nn.Conv2d(dec_chs[-1], num_class, 1)
        if typeB == "snake":
            self.headB = nn.Sequential(
                nn.Conv2d(dec_chs[-1], num_class, 1),
                nn.Flatten(),
                nn.Linear(in_features=self.flatten_img_shape,out_features=2*nb_control_points))


    def forward(self, x, verbose = False):
        enc_ftrs = self.encoder(x , verbose=verbose)

        outA = self.decoderA(enc_ftrs[::-1][0], enc_ftrs[::-1][1:], verbose = verbose)
        outA = self.headA(outA)

        #outB = self.decoderB(enc_ftrs[::-1][0], enc_ftrs[::-1][1:], verbose = verbose)
        #outB = self.headB(outB)

        return outA#, outB

    

if __name__ == "__main__":

    mcsnake = MCSnakeNet(typeA="classic", typeB="snake")

    x = torch.ones((1,3,256,256))

    yA, yB = mcsnake(x, verbose = True)

    print("Shape yA : {}, shape yB : {}".format(yA.shape,yB.shape))

