from torch.utils.data import Dataset
from os import listdir

class MCSnakeDataset(Dataset):

    def __init__(self, dataset_folder, device) -> None:
        super().__init__()

        self.dataset_folder = dataset_folder
        self.files_list = listdir(dataset_folder)
        self.device = device

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        pass

