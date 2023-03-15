from torch.utils.data import Dataset
import torch
from os import listdir
from os.path import join, splitext

from matplotlib.pyplot import imread

class TextureDataset(Dataset):

    def __init__(self, path, greyscale = True, subset = 1, device = "cpu") -> None:
        super().__init__()
        self.path = path
        self.greyscale = greyscale
        self.subset = subset

        if self.greyscale :
            self.imgs_dir = f"Database{self.subset}Greyscale"
        else :
            self.imgs_dir = f"Database{self.subset}"

        self.imgs_path = join(self.path, self.imgs_dir)
        self.masks_path = join(self.path, "Masks")

        self.imgs_names = listdir(self.imgs_path)

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


if __name__ == "__main__":

    import sys
    from matplotlib.pyplot import imshow, show

    path = sys.argv[1]

    dataset = TextureDataset(path)

    img, mask = dataset[0]

    print(torch.max(img))

    imshow(img)

    show()