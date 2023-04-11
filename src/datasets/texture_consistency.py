from torch.utils.data import Dataset
import torch
from os import listdir
from os.path import join, splitext
from math import ceil, floor
from random import shuffle


from matplotlib.pyplot import imread

class TextureDataset(Dataset):

    def __init__(self, path, greyscale = True, subset = 1, no_GT_prop = 0, device = "cpu") -> None:
        super().__init__()
        self.path = path
        self.greyscale = greyscale
        self.subset = subset
        self.no_GT_prop = no_GT_prop

        if self.greyscale :
            self.imgs_dir = f"Database{self.subset}Greyscale"
        else :
            self.imgs_dir = f"Database{self.subset}"

        self.imgs_path = join(self.path, self.imgs_dir)
        self.masks_path = join(self.path, "Masks")

        self.imgs_names = listdir(self.imgs_path)

        n = len(self.imgs_names)

        self.have_GT = [True for _ in range(ceil(n*(1-self.no_GT_prop)))] + [False for _ in range(floor(n*self.no_GT_prop))]
        shuffle(self.have_GT)

        self.device = device

    def __getitem__(self, index):
        img_name = self.imgs_names[index]
        img_path = join(self.imgs_path, img_name)

        img = imread(img_path)

        mask_idx = splitext(img_name)[0].split("_")[-1]

        mask_path = join(self.masks_path, mask_idx+".bmp")
        mask = imread(mask_path)

        return torch.unsqueeze(torch.tensor(img)/255,0).to(self.device), (torch.tensor(mask)/255).to(device=self.device)

    def __len__(self):
        return len(self.imgs_names)