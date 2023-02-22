import torch
from torch.utils.data import DataLoader
import argparse
import json
from time import time


from datasets.mcsnake_dataset import MCSnakeDataset 
from datasets.texture_dataset import TextureDataset

from DL_models.mcsnake_net import MCSnakeNet


from loss_functions.consistency_loss import MutualConsistency
from loss_functions.consistency_tools import contour_to_mask, mask_to_contour
from snake_representation.snake_tools import sample_contour


def train(model, optimizer, train_loader, criterion, M, W, H, verbose = False):

    running_loss = 0.0
    resizing_vect = torch.tensor([[[W, H]]])

    for _, batch in enumerate(train_loader):
        imgs, GT_masks = batch

        optimizer.zero_grad()
        
        tic = time()
        classic_mask, snake_cp = model(imgs)



        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], M, 2))*resizing_vect

        print("Min : {}, max : {}".format(torch.min(snake_cp), torch.max(snake_cp)))

        if verbose :
            print("Forward pass processed in {}s".format(time()-tic))
            tic = time()

        classic_mask = torch.squeeze(classic_mask)

        with torch.no_grad():
            GT_contour = [mask_to_contour(mask) for mask in GT_masks]
            classic_contour = [mask_to_contour(mask>0.5) for mask in classic_mask]

        if verbose :

            print("Contour computed in {}s".format(time()-tic))
            tic = time()

        snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], M=M) for i,cp in enumerate(reshaped_cp)]
        snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], M=M) for i,cp in enumerate(reshaped_cp)]

        if verbose :
            
            print("Contour sampled in {}s".format(time()-tic))
            tic = time()

        with torch.no_grad():
            snake_mask = torch.stack([contour_to_mask(contour, W, H) for contour in snake_size_of_GT])

        if verbose :
            print("Mask computed in {}s".format(time()-tic))
            tic = time()

        loss = criterion(GT_masks, GT_contour, snake_size_of_GT, snake_size_of_classic, snake_mask, classic_contour, classic_mask)
        loss.backward()
        optimizer.step()

        if verbose :
            print("Loss computed in {}s".format(time()-tic))
            tic = time()

        running_loss += loss.item()
    
    return running_loss



if __name__ == "__main__" :

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Configuration file", default="./src/config/basic_config.json")
    args = parser.parse_args()

    config = args.config
    with open(config, "r") as f:
        config_dic = json.load(f)

    train_config = config_dic["data"]["train_set"]
    test_config = config_dic["data"]["test_set"]

    model_config = config_dic["model"]
    optimizer_config = config_dic["optimizer"]
    criterion_config = config_dic["criterion"]
    snake_config = config_dic["active_contour"]

    W, H = config_dic["data"]["image_size"]

    train_set = TextureDataset(path=train_config["path_to_data"])
    test_set = TextureDataset(path=train_config["path_to_data"], subset=2)

    train_loader = DataLoader(train_set, batch_size=train_config["batchsize"])
    test_loader = DataLoader(test_set, batch_size=test_config["batchsize"])


    enc_chs=(config_dic["data"]["nb_channels"],64,128,256,512,1024)
    model = MCSnakeNet(enc_chs=enc_chs,typeA=model_config["typeA"], typeB=model_config["typeB"], nb_control_points=model_config["nb_control_points"], img_shape=(W,H))

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])

    criterion = MutualConsistency(gamma=criterion_config["gamma"], verbose = True)


    loss_list = list()

    epoch_modulo = train_config["print_every_nb_epochs"]

    for epoch in range(train_config["nb_epochs"]):

        loss = train(model, optimizer, train_loader, criterion, M=snake_config["M"], W=W, H=H, verbose= True)
        loss_list.append(loss)
        
        if epoch%epoch_modulo == epoch_modulo-1 :
            print("Epoch nb {} ; loss : {}".format(epoch, loss))
