import torch
from torch.utils.data import DataLoader
import argparse
import json


from datasets.mcsnake_dataset import MCSnakeDataset 
from DL_models.mcsnake_net import MCSnakeNet
from loss_functions.consistency_loss import MutualConsistency
from loss_functions.consistency_tools import contour_to_mask, mask_to_contour
from snake_representation.snake_tools import sample_contour


def train(model, optimizer, train_loader, criterion, M, W, H):

    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        imgs, GT_masks = batch

        optimizer.zero_grad()

        classic_mask, snake_cp = model(imgs)

        with torch.no_grad():
            GT_contour = [mask_to_contour(mask) for mask in GT_masks]
            classic_contour = [mask_to_contour(mask>0.5) for mask in classic_mask]

        snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], M=M) for i,cp in enumerate(snake_cp)]
        snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], M=M) for i,cp in enumerate(snake_cp)]

        with torch.no_grad():
            snake_mask = torch.stack([contour_to_mask(contour, W, H) for contour in snake_size_of_GT])


        loss = criterion(GT_masks, GT_contour, snake_size_of_GT, snake_size_of_classic, snake_mask, classic_contour, classic_mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss



if __name__ == "__main__" :

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type="str", help="Configuration file", default="/src/config/basic_config.json")
    args = parser.parse_args()

    config = args.config
    with open(config, "r") as f:
        config_dic = json.load(f)

    train_config = config_dic["train_set"]
    test_config = config_dic["test_set"]

    model_config = config_dic["model"]
    optimizer_config = config_dic["optimizer"]
    criterion_config = config_dic["criterion"]
    snake_config = config_dic["active_contour"]

    train_set = MCSnakeDataset(train_config["path_to_data"], device=device)
    test_set = MCSnakeDataset(test_config["path_to_data"], device=device)

    train_loader = DataLoader(train_set, batch_size=train_config["batchsize"])
    test_loader = DataLoader(test_set, batch_size=test_config["batchsize"])

    model = MCSnakeNet(typeA=model_config["typeA"], typeB=model_config["typeB"], nb_control_points=model_config["nb_control_points"])

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])

    criterion = MutualConsistency(gamma=criterion_config["gamma"])


    loss_list = list()

    for epoch in range(train_config["nb_epochs"]):

        loss = train(model, optimizer, train_loader, criterion, M=snake_config["M"])
        loss_list.append(loss)
        
        if epoch%10 == 9 :
            print("Epoch nb {} ; loss : {}".format(epoch, loss))
