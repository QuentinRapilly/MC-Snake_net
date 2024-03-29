import argparse
import json
from datetime import datetime
from os.path import join
from random import shuffle
from math import ceil, floor

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch import sigmoid
from torch.optim.lr_scheduler import ExponentialLR

import wandb

from matplotlib.pyplot import close as close_plot

from plot.plot_batch import create_subplot_summary
from datasets.texture_dataset import TextureDataset
from DL_models.mcsnakenet_clean import MCSnakeNet
from loss_functions.consistency_loss import DiceLoss, SnakeLoss
from loss_functions.consistency_tools import contour_to_mask, mask_to_contour, limit_nb_points
from snake_representation.snake_tools import sample_contour, sample_circle

from time_management.time_management import time_manager, print_time_dict


#def tmp_name():





def train_consistency(model, unet_optimizer, mlp_optimizer, train_loader, have_GT, mask_loss, snake_loss, theta, gamma,\
          W : int, H : int, M : int, apply_sigmoid : bool, nb_polygon_edges :int = 100,\
            nb_batch_to_plot : int = 3, predict_dx_dy = False, device : str = "cpu", verbose = True):


    running_loss = 0.0

    running_reference_mask_loss = 0.0
    running_reference_snake_loss = 0.0

    running_consistency_mask_loss = 0.0
    running_consistency_snake_loss = 0.0
    
    rescaling_vect = torch.tensor([W, H]).to(device)
    rescaling_inv = torch.tensor([1/W, 1/H]).to(device)

    N = 0

    for k, batch in enumerate(train_loader):

        print("Progress of the epoch : {}%                           \r".format(round(k/len(train_loader)*100,ndigits=2)), end="")

        loss = 0
        
        imgs, GT_masks = batch

        unet_optimizer.zero_grad()
        mlp_optimizer.zero_grad()
        
        # Model applied to input
        classic_mask, snake_cp = model(imgs)
        
        # Some loss function (as BCEWithLogitsLoss) apply sigmoid so we don't need to in loss computation
        classic_mask = sigmoid(classic_mask)
        
        # we want our control points to be in [0;1] in a first time so we apply sigmoid, 
        # then we will be able to rescale them in WxH (our images shape)
        snake_cp = sigmoid(snake_cp)

        # Control points format (2M) -> (M,2)
        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], M, 2))

        classic_mask = torch.squeeze(classic_mask)

        if gamma > 0 :

            # Transforming predicted mask into contour for loss computation
            with torch.no_grad():
                classic_contour = [mask_to_contour((mask>0.5)).to(device)*rescaling_inv for mask in classic_mask]

            # Sampling the predicted snake to compute the snake loss
            snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]
            snake_for_mask = [sample_contour(cp, nb_samples = nb_polygon_edges, device = device) for cp in reshaped_cp]
            
            # Creating mask form contour predicted by the snake part
            with torch.no_grad():
                snake_mask = torch.stack([contour_to_mask(contour*rescaling_vect, W, H, device = device) for contour in snake_for_mask])

            consistency_mask_loss = mask_loss(classic_mask, snake_mask)
            consistency_snake_loss = snake_loss(snake_size_of_classic, classic_contour)

            running_consistency_mask_loss += consistency_mask_loss.item()
            running_consistency_snake_loss += consistency_snake_loss.item()

            loss = gamma*(theta*consistency_mask_loss + (1-theta)*consistency_snake_loss)


        if have_GT[k]:

            with torch.no_grad():
                GT_contour = [mask_to_contour(mask).to(device)*rescaling_inv for mask in GT_masks]

            snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]

            # Computing the different part of the loss then the global loss
            reference_mask_loss = mask_loss(classic_mask, GT_masks)
            reference_snake_loss = snake_loss(snake_size_of_GT, GT_contour)


            loss += (1 - gamma)*(theta*reference_mask_loss + (1-theta)*reference_snake_loss)
            
            running_reference_mask_loss += reference_mask_loss.item()
            running_reference_snake_loss += reference_snake_loss.item()

        if loss != 0:
            # Backward gradient step
            loss.backward()
            unet_optimizer.step()
            mlp_optimizer.step()
            running_loss += loss.item()
            N += 1

        

    return running_loss / N, running_consistency_mask_loss / N, running_consistency_snake_loss / N,\
        running_reference_mask_loss / N, running_reference_snake_loss / N


def test(model, test_loader, mask_loss, snake_loss, theta, gamma, W : int, H : int, M : int,\
         apply_sigmoid : bool, nb_polygon_edges :int = 100, nb_batch_to_plot : int = 3,\
              predict_dx_dy = False, device : str = "cpu"):
    
    running_loss = 0.0

    running_reference_mask_loss = 0.0
    running_reference_snake_loss = 0.0

    running_consistency_mask_loss = 0.0
    running_consistency_snake_loss = 0.0

    img_dict = {"images" : [], "GT" : [], "masks": [], "snakes" : []}
    
    rescaling_vect = torch.tensor([W, H]).to(device)
    rescaling_inv = torch.tensor([1/W, 1/H]).to(device)

    N = len(test_loader)

    for k, batch in enumerate(test_loader):

        print("Progress of the test step : {}%          \r".format(round(k/len(test_loader)*100,ndigits=2)), end="")

        B = test_loader.batch_size
        imgs, GT_masks = batch
        
        # Model applied to input
        classic_mask, snake_cp = model(imgs)

        # Some loss function (as BCEWithLogitsLoss) apply sigmoid so we don't need to in loss computation
        if apply_sigmoid :
            classic_mask = sigmoid(classic_mask)
        
        # we want our control points to be in [0;1] in a first time so we apply sigmoid, 
        # then we will be able to rescale them in WxH (our images shape)
        snake_cp = sigmoid(snake_cp)

        # Control points format (2M) -> (M,2)
        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], M, 2))

        if predict_dx_dy :
            init_cp = torch.unsqueeze(sample_circle(M = M, r = 0.35), dim=0).to(device=device)
            d_cp = 2*reshaped_cp - 1 
            reshaped_cp = init_cp + d_cp

        classic_mask = torch.squeeze(classic_mask)

        # Transforming GT mask and predicted mask into contour for loss computation
        with torch.no_grad():
            GT_contour = [mask_to_contour(mask).to(device)*rescaling_inv for mask in GT_masks]
            classic_contour = [mask_to_contour((mask>0.5)).to(device)*rescaling_inv for mask in classic_mask]

        # Sampling the predicted snake to compute the snake loss
        snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]
        snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]
        snake_for_mask = [sample_contour(cp, nb_samples = nb_polygon_edges, device = device) for cp in reshaped_cp]
            
        # Creating mask form contour predicted by the snake part
        with torch.no_grad():
            snake_mask = torch.stack([contour_to_mask(contour*rescaling_vect, W, H, device = device) for contour in snake_for_mask])

        # Computing the different part of the loss then the global loss
        reference_mask_loss = mask_loss(classic_mask, GT_masks)
        reference_snake_loss = snake_loss(snake_size_of_GT, GT_contour)

        consistency_mask_loss = mask_loss(classic_mask, snake_mask)
        consistency_snake_loss = snake_loss(snake_size_of_classic, classic_contour)

        loss = (1 - gamma)*(theta*reference_mask_loss + (1-theta)*reference_snake_loss)\
            + gamma*(theta*consistency_mask_loss + (1-theta)*consistency_snake_loss)

        running_consistency_mask_loss += consistency_mask_loss.item()
        running_consistency_snake_loss += consistency_snake_loss.item()
        running_reference_mask_loss += reference_mask_loss.item()
        running_reference_snake_loss += reference_snake_loss.item()

        running_loss += loss.item()

        # Choosing some image to plot in the WAndB recap
        if k < nb_batch_to_plot :
            img_dict["images"] += [torch.squeeze(imgs[i]).detach().cpu() for i in range(B)]
            img_dict["GT"] += [GT_masks[i].detach().cpu() for i in range(B)]
            img_dict["masks"] += [sigmoid(classic_mask[i]).detach().cpu() for i in range(B)]
            img_dict["snakes"] += [(GT_masks[i].detach().cpu(), (GT_contour[i]*rescaling_vect).detach().cpu(),(snake_for_mask[i]*rescaling_vect).detach().cpu(),\
                                    (reshaped_cp[i]*rescaling_vect).detach().cpu()) for i in range(B)]
    


    return running_loss / N, running_consistency_mask_loss / N, running_consistency_snake_loss / N,\
        running_reference_mask_loss / N, running_reference_snake_loss / N, img_dict



if __name__ == "__main__" :


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Current device : {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Configuration file", default="./src/config/consistency_config.json")
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
    test_every_nb_epochs = settings_config["test_every_nb_epochs"]
    no_GT_prop = settings_config["no_GT_prop"]

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

    nb_batch = len(train_loader)
    have_GT = [True for _ in range(ceil(nb_batch*(1-no_GT_prop)))] + [False for _ in range(floor(nb_batch*no_GT_prop))]
    shuffle(have_GT)

    print("Proportion of batch having a GT : {}%".format(100*torch.sum(torch.tensor(have_GT))/len(have_GT)))

    # Initializing the model
    model = MCSnakeNet(num_classes =model_config["num_class"], input_channels=config_dic["data"]["nb_channels"],\
                       padding_mode="zeros", train_bn=False, inner_normalisation='BatchNorm', img_shape=(W,H),\
                        nb_control_points=M, nb_snake_layers=model_config["nb_snake_layers"]).to(device)


    # Initializing the optimizer
    unet_optimizer = torch.optim.Adam(model.layers.parameters(), lr=optimizer_config["unet_lr"])
    mlp_optimizer = torch.optim.Adam(model.snake_head.parameters(), lr=optimizer_config["mlp_lr"])

    scheduler_config = config_dic["scheduler"]
    scheduler_step = scheduler_config["step_every_nb_epoch"]
    unet_scheduler = ExponentialLR(optimizer=unet_optimizer, gamma=scheduler_config["gamma"])
    mlp_scheduler = ExponentialLR(optimizer=mlp_optimizer, gamma=scheduler_config["gamma"])


    # Initializing the loss 
    mask_loss = {"bce": BCEWithLogitsLoss(), "dice": DiceLoss(), "mse" : MSELoss() }[loss_config["mask_loss"]]
    apply_sigmoid = loss_config["apply_sigmoid"]

    snake_loss = SnakeLoss()
    gamma_config = loss_config["gamma"]
    gamma_values = gamma_config["values"]
    gamma_epochs = gamma_config["epochs"]
    theta = loss_config["theta"]



    # Tracking of the loss and some images during the training
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=f"MC-snake_net_consistency-subset_{train_set_index}",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": optimizer_config["unet_lr"],
        "architecture": "UNET",
        "dataset": "Texture",
        "epochs": train_config["nb_epochs"]
        }
    )

    epoch_modulo = train_config["print_every_nb_epochs"]

    for epoch in range(train_config["nb_epochs"]):

        if epoch in gamma_epochs:
            gamma = gamma_values[gamma_epochs.index(epoch)]

        print(f"## Starting epoch {epoch}, current gamma : {gamma} ##")
        epoch_dict = {}
        with time_manager(epoch_dict, f"epoch {epoch}"):
            loss, consistency_mask_loss, consistency_snake_loss, reference_mask_loss, reference_snake_loss = \
                    train_consistency(model, unet_optimizer, mlp_optimizer, train_loader, have_GT=have_GT, mask_loss=mask_loss, apply_sigmoid=apply_sigmoid,\
                        snake_loss=snake_loss, gamma=gamma, theta=theta, M=M, W=W, H=H, device = device, verbose=verbose)

        print_time_dict(epoch_dict)

        log_dict = {"train/loss": loss, "train/consistency-mask_loss" : consistency_mask_loss,\
                   "train/consistency-snake_loss" : consistency_snake_loss, "train/reference-mask_loss" : reference_mask_loss,\
                      "train/reference-snake_loss" : reference_snake_loss}
        

        if (1 + epoch) % test_every_nb_epochs == 0:
            loss, consistency_mask_loss, consistency_snake_loss, reference_mask_loss, reference_snake_loss, img_dict = \
                test(model, test_loader, mask_loss=mask_loss, snake_loss=snake_loss, theta=theta, gamma=gamma,\
                     apply_sigmoid=apply_sigmoid, M=M, W=W, H=H, device = device)
            
            sum_plot = create_subplot_summary(images_dict=img_dict)

            test_dict = {"test/loss": loss, "test/consistency-mask_loss" : consistency_mask_loss,\
                    "test/consistency-snake_loss" : consistency_snake_loss, "test/reference-mask_loss" : reference_mask_loss,\
                        "test/reference-snake_loss" : reference_snake_loss, "test_samples" : sum_plot}
            
            log_dict = dict(log_dict, **test_dict)
            

        wandb.log(log_dict)
        close_plot()

        if (epoch + 1) % scheduler_step == 0:
            unet_scheduler.step()
            mlp_scheduler.step()
        

    wandb.finish()

    now = datetime.now()
    torch.save(model.state_dict(), join(model_config["save_path"],str(now).replace(" ","_").split(".")[0]+".pkl"))
