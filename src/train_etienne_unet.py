import argparse
import json
from time import time
from datetime import datetime
from os.path import join

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch import sigmoid
from torch.optim.lr_scheduler import ExponentialLR

import wandb

import matplotlib.pyplot as plt # enlever quand le probleme est regle
from os.path import isdir       # enlever quand le probleme est regle
from os import mkdir            # enlever quand le probleme est regle

from datasets.texture_dataset import TextureDataset
from DL_models.mcsnakenet_clean import MCSnakeNet
from loss_functions.consistency_loss import DiceLoss, SnakeLoss
from loss_functions.consistency_tools import contour_to_mask, mask_to_contour
from snake_representation.snake_tools import sample_contour


def train(model, optimizer, train_loader, mask_loss, snake_loss, theta, gamma, W : int, H : int, M : int, epoch : int, apply_sigmoid : bool, verbose = False):

    tic_epoch = time()

    running_loss = 0.0

    running_reference_mask_loss = 0.0
    running_reference_snake_loss = 0.0

    running_consistency_mask_loss = 0.0
    running_consistency_snake_loss = 0.0
    
    rescaling_vect = torch.tensor([W, H]).to(device)
    rescaling_inv = torch.tensor([1/W, 1/H]).to(device)

    N = len(train_loader)


    ### To remove after debugging ###
    # From HERE
    plot_dir = "/net/serpico-fs2/qrapilly/model_storage/MC-snake/epoch_{}".format(epoch)
    if not isdir(plot_dir):
        mkdir(plot_dir)
    # To HERE

    for k, batch in enumerate(train_loader):

        print("Progress of the epoch : {}%          \r".format(round(k/len(train_loader)*100,ndigits=2)), end="")

        B = train_loader.batch_size
        imgs, GT_masks = batch

        optimizer.zero_grad()
        
        # Model applied to input
        tic_forward = time()
        classic_mask, snake_cp = model(imgs)
        tac_forward = time()

        # Some loss function (as BCEWithLogitsLoss) apply sigmoid so we don't need to in loss computation
        if apply_sigmoid :
            classic_mask = sigmoid(classic_mask)

        # we want our control points to be in [0;1] in a first time so we apply sigmoid, 
        # then we will be able to rescale them in WxH (our images shape)
        snake_cp = sigmoid(snake_cp)

        # Control points format (2M) -> (M,2)
        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], M, 2))

        print("control points : {}".format(reshaped_cp))

        classic_mask = torch.squeeze(classic_mask)

        # Transforming GT mask and predicted mask into contour for loss computation
        tic_contour = time()
        with torch.no_grad():
            GT_contour = [mask_to_contour(mask).to(device)*rescaling_inv for mask in GT_masks]
            classic_contour = [mask_to_contour((mask>0.5)).to(device)*rescaling_inv for mask in classic_mask]
        tac_contour = time()

        # Sampling the predicted snake to compute the snake loss
        tic_sample = time()
        snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], M=M, device = device) for i,cp in enumerate(reshaped_cp)]
        snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], M=M, device = device) for i,cp in enumerate(reshaped_cp)]
        tac_sample = time()
            
        # Creating mask form contour predicted by the snake part
        tic_mask = time()
        with torch.no_grad():
            snake_mask = torch.stack([contour_to_mask(contour*rescaling_vect, W, H, device = device) for contour in snake_size_of_GT])
        tac_mask = time() 

        # Computing the different part of the loss then the global loss
        tic_loss = time()
        reference_mask_loss = mask_loss(classic_mask, GT_masks)
        reference_snake_loss = snake_loss(snake_size_of_GT, GT_contour)

        consistency_mask_loss = mask_loss(classic_mask, snake_mask)
        consistency_snake_loss = snake_loss(snake_size_of_classic, classic_contour)
        tac_loss = time()


        loss = (1 - gamma)*(theta*reference_mask_loss + (1-theta)*reference_snake_loss) +\
            gamma*(theta*consistency_mask_loss + (1-theta)*consistency_snake_loss)


        # Backward gradient step
        tic_backward = time()
        loss.backward()
        optimizer.step()
        tac_backward = time()

        running_consistency_mask_loss += consistency_mask_loss.item()
        running_consistency_snake_loss += consistency_snake_loss.item()
        running_reference_mask_loss += reference_mask_loss.item()
        running_reference_snake_loss += reference_snake_loss.item()

        running_loss += loss.item()

        if k<N-1:
            plt.figure(figsize = (20,10))
            for i in range(B):
                plt.subplot(4,B,1+i)
                plt.imshow(torch.squeeze(imgs[i]).detach().cpu(), cmap="gray", vmin=0, vmax=1)

            for i in range(B):
                plt.subplot(4,B,1+B+i)
                plt.imshow(GT_masks[i].detach().cpu(), cmap="gray", vmin=0, vmax=1)
            
            for i in range(B):
                plt.subplot(4,B,1+2*B+i)
                plt.imshow(sigmoid(classic_mask[i]).detach().cpu(), cmap="gray", vmin=0, vmax=1)
                #plt.imshow(classic_mask[i].detach().cpu(), cmap="gray", vmin=0, vmax=1)

            for i in range(B):
                plt.subplot(4,B,1+3*B+i)
                plt.imshow(snake_mask[i].detach().cpu(), cmap="gray", vmin=0, vmax=1)

            plt.savefig(join(plot_dir,"batch_{}.png".format(k)))

            plt.close()

        # Choosing some image to plot in the WAndB recap
        if k == len(train_loader) - 1:
            plot_res = (GT_masks[0], sigmoid(classic_mask[0])) #, snake_mask[0]) if print_in_table else None
    

    tac_epoch = time()

    print("Epoch terminated in {}s".format(tac_epoch-tic_epoch))

    return running_loss / N, running_consistency_mask_loss / N, running_consistency_snake_loss / N,\
        running_reference_mask_loss / N, running_reference_snake_loss / N, plot_res



if __name__ == "__main__" :

    # TODO : creer une fonction de test

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Current device : {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Configuration file", default="./src/config/basic_config.json")
    args = parser.parse_args()

    config = args.config

    # Loading the config dic
    with open(config, "r") as f:
        config_dic = json.load(f)

    # Getting all the config information required into the config dic

    train_config = config_dic["data"]["train_set"]
    test_config = config_dic["data"]["test_set"]
    settings_config = config_dic["settings"]

    verbose = settings_config["verbose"]

    model_config = config_dic["model"]
    optimizer_config = config_dic["optimizer"]
    snake_config = config_dic["active_contour"]
    loss_config = config_dic["loss"]

    M = snake_config["M"]

    W, H = config_dic["data"]["image_size"]

    train_set_index = train_config["set_index"]
    test_set_index = test_config["set_index"]

    train_set = TextureDataset(path=train_config["path_to_data"], subset=train_set_index, device = device)
    test_set = TextureDataset(path=train_config["path_to_data"], subset=test_set_index, device = device)

    train_loader = DataLoader(train_set, batch_size=train_config["batchsize"])
    test_loader = DataLoader(test_set, batch_size=test_config["batchsize"])



    # Initializing the model
    #model = Unet2D_2D(num_classes =model_config["num_class"], input_channels=config_dic["data"]["nb_channels"],\
    #                  padding_mode="zeros", train_bn=False, inner_normalisation='BatchNorm').to(device)
    model = MCSnakeNet(num_classes =model_config["num_class"], input_channels=config_dic["data"]["nb_channels"],\
                       padding_mode="zeros", train_bn=False, inner_normalisation='BatchNorm', img_shape=(W,H),\
                        nb_control_points=M, nb_snake_layers=model_config["nb_snake_layers"]).to(device)
    # TODO : ajouter le nombre de couche snake


    # Initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"])

    scheduler_config = config_dic["scheduler"]
    scheduler_step = scheduler_config["step_every_nb_epoch"]
    scheduler = ExponentialLR(optimizer=optimizer, gamma=scheduler_config["gamma"])


    # Initializing the loss 
    mask_loss = {"bce": BCEWithLogitsLoss(), "dice": DiceLoss(), "mse" : MSELoss() }[loss_config["mask_loss"]]
    apply_sigmoid = loss_config["apply_sigmoid"]

    criterion = MSELoss()
    snake_loss = SnakeLoss(criterion=criterion)
    gamma = loss_config["gamma"]
    theta = loss_config["theta"]



    # Tracking of the loss and some images during the training
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="MC-snake_net",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": optimizer_config["lr"],
        "architecture": "UNET",
        "dataset": "Texture",
        "epochs": train_config["nb_epochs"]
        }
    )

    epoch_modulo = train_config["print_every_nb_epochs"]

    for epoch in range(train_config["nb_epochs"]):

        print(f"Starting epoch {epoch}")
        loss, consistency_mask_loss, consistency_snake_loss, reference_mask_loss, reference_snake_loss, plot_res = \
                train(model, optimizer, train_loader, mask_loss=mask_loss, apply_sigmoid=apply_sigmoid,\
                      snake_loss=snake_loss, gamma=gamma, theta=theta,\
                        M=M, W=W, H=H, epoch=epoch, verbose=verbose)

        
        gt, proba = plot_res #, snake
        gt = wandb.Image(gt, caption="GT")
        proba = wandb.Image(proba, caption="Probability map")
        #snake = wandb.Image(snake, caption="Snake mask")

        wandb.log({"loss": loss, "consistency_mask_loss" : consistency_mask_loss,\
                   "consistency_snake_loss" : consistency_snake_loss, "reference_mask_loss" : reference_mask_loss,\
                      "reference_snake_loss" : reference_snake_loss, "GT" : gt,\
                        "Probability map" : proba})#, "Snake mask" : snake})
        
        if (epoch + 1)%scheduler_step == 0:
            scheduler.step()
        

    wandb.finish()

    now = datetime.now()
    torch.save(model.state_dict(), join(model_config["save_path"],str(now).replace(" ","_").split(".")[0]+".pkl"))
