import torch
from torch.utils.data import DataLoader
import argparse
import json


from datasets.mcsnake_dataset import MCSnakeDataset 
from DL_models.mcsnake_net import MCSnakeNet


def train(model, optimizer, train_set, test_set):

    pass



if __name__ == "__main__" :

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type="str", help="Configuration file")
    args = parser.parse_args()

    config = args.config
    with open(config, "r") as f:
        config_dic = json.load(f)

    train_config = config_dic["train_set"]
    test_config = config_dic["test_set"]

    train_set = MCSnakeDataset(train_config["path_to_data"], device=device)
    test_set = MCSnakeDataset(test_config["path_to_data"], device=device)

    train_loader = DataLoader(train_set, batch_size=train_config["batchsize"])
    test_loader = DataLoader(test_set, batch_size=test_config["batchsize"])

    model = MCSnakeNet()

