from torch.utils.data import Dataset
from os import listdir

class KaggleDataset(Dataset):

    def __init__(self, dataset_folder) -> None:
        super().__init__()

        self.dataset_folder = dataset_folder
        self.files_list = listdir(dataset_folder)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        pass
