import argparse
import json
from datetime import datetime
from os.path import join

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
from snake_representation.snake_tools import sample_contour, sample_circle, polar_to_cartesian_cp

from time_management.time_management import time_manager, print_time_dict


def train(model, optimizer, train_loader, mask_loss, snake_loss, theta, gamma, W : int, H : int,\
          sigmoid_on_proba : bool, sigmoid_on_cp : bool, nb_polygon_edges :int = 100, nb_batch_to_plot : int = 3,\
            predict_dx_dy = False, device : str = "cpu", verbose = True, **kwargs):


    running_loss = 0.0
    running_reference_mask_loss = 0.0
    running_reference_snake_loss = 0.0
    running_consistency_mask_loss = 0.0
    running_consistency_snake_loss = 0.0

    img_dict = {"images" : [], "GT" : [], "masks": [], "snakes" : []}
    time_dict = {}
    
    rescaling_vect = torch.tensor([W, H]).to(device)
    rescaling_inv = torch.tensor([1/W, 1/H]).to(device)

    N = len(train_loader)

    for k, batch in enumerate(train_loader):

        print("Progress of the epoch : {}%          \r".format(round(k/len(train_loader)*100,ndigits=2)), end="")

        B = train_loader.batch_size
        imgs, GT_masks = batch

        optimizer.zero_grad()
        
        # Model applied to input
        with time_manager(time_dict, "forward pass"):
            classic_mask, snake_cp = model(imgs)
        
        # Some loss function (as BCEWithLogitsLoss) apply sigmoid so we don't need to in loss computation
        if sigmoid_on_proba :
            classic_mask = sigmoid(classic_mask)
        
        # if we want our control points to be in [0;1] we hate to apply sigmoid (easier to fit them in WxH)
        if sigmoid_on_cp:
            shift_cp = 0
            snake_cp = sigmoid(snake_cp)
        else :
            shift_cp = 0.5

        # Control points format (2M) -> (M,2)
        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], snake_cp.shape[1]//2, 2))

        # deprecated
        if predict_dx_dy :
            M = reshaped_cp.shape[1]
            init_cp = torch.unsqueeze(sample_circle(M = M, r = 0.35), dim=0).to(device=device)
            d_cp = 2*reshaped_cp - 1 
            reshaped_cp = init_cp + d_cp

        classic_mask = torch.squeeze(classic_mask)

        # Transforming GT mask and predicted mask into contour for loss computation
        with time_manager(time_dict, "masks to contours"):
            with torch.no_grad():
                GT_contour = [mask_to_contour(mask).to(device)*rescaling_inv - shift_cp for mask in GT_masks]
                classic_contour = [mask_to_contour((mask>0.5)).to(device)*rescaling_inv - shift_cp for mask in classic_mask]

        # Sampling the predicted snake to compute the snake loss
        with time_manager(time_dict, "sampling contours"):
            snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]
            snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]
            snake_for_mask = [sample_contour(cp, nb_samples = nb_polygon_edges, device = device) for cp in reshaped_cp]
            
        # Creating mask form contour predicted by the snake part
        with time_manager(time_dict, "contours to masks"):
            with torch.no_grad():
                snake_mask = torch.stack([contour_to_mask((contour+shift_cp)*rescaling_vect, W, H, device = device) for contour in snake_for_mask])

        # Computing the different part of the loss then the global loss
        with time_manager(time_dict, "loss computation"):
            reference_mask_loss = mask_loss(classic_mask, GT_masks)
            reference_snake_loss = snake_loss(snake_size_of_GT, GT_contour)

            consistency_mask_loss = mask_loss(classic_mask, snake_mask)
            consistency_snake_loss = snake_loss(snake_size_of_classic, classic_contour)

            loss = (1 - gamma)*(theta*reference_mask_loss + (1-theta)*reference_snake_loss)\
                + gamma*(theta*consistency_mask_loss + (1-theta)*consistency_snake_loss)


        # Backward gradient step
        with time_manager(time_dict, "backward pass"):
            loss.backward()
            optimizer.step()

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
            img_dict["snakes"] += [(GT_masks[i].detach().cpu(), ((GT_contour[i]+shift_cp)*rescaling_vect).detach().cpu(),((snake_for_mask[i]+shift_cp)*rescaling_vect).detach().cpu(),\
                                    ((reshaped_cp[i]+shift_cp)*rescaling_vect).detach().cpu()) for i in range(B)]
    
    if verbose :
        print_time_dict(time_dict)


    return running_loss / N, running_consistency_mask_loss / N, running_consistency_snake_loss / N,\
        running_reference_mask_loss / N, running_reference_snake_loss / N, img_dict


def test(model, test_loader, mask_loss, snake_loss, theta, gamma, W : int, H : int, sigmoid_on_proba : bool,\
         sigmoid_on_cp : bool, nb_polygon_edges :int = 100, nb_batch_to_plot : int = 3, predict_dx_dy = False,\
            device : str = "cpu"):
    
    # Train function commented with almost the same structure
    
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
        
        classic_mask, snake_cp = model(imgs)

        if sigmoid_on_proba :
            classic_mask = sigmoid(classic_mask)
        
        # if we want our control points to be in [0;1] we hate to apply sigmoid (easier to fit them in WxH)
        if sigmoid_on_cp:
            shift_cp = 0
            snake_cp = sigmoid(snake_cp)
        else :
            shift_cp = 0.5


        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], snake_cp.shape[1]//2, 2))

        if predict_dx_dy :
            M = reshaped_cp.shape[1]
            init_cp = torch.unsqueeze(sample_circle(M = M, r = 0.35), dim=0).to(device=device)
            d_cp = 2*reshaped_cp - 1 
            reshaped_cp = init_cp + d_cp

        classic_mask = torch.squeeze(classic_mask)

        GT_contour = [mask_to_contour(mask).to(device)*rescaling_inv - shift_cp for mask in GT_masks]
        classic_contour = [mask_to_contour((mask>0.5)).to(device)*rescaling_inv - shift_cp for mask in classic_mask]

        snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]
        snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], device = device) for i,cp in enumerate(reshaped_cp)]
        snake_for_mask = [sample_contour(cp, nb_samples = nb_polygon_edges, device = device) for cp in reshaped_cp]
            
        snake_mask = torch.stack([contour_to_mask((contour+shift_cp)*rescaling_vect, W, H, device = device) for contour in snake_for_mask])

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

        if k < nb_batch_to_plot :
            img_dict["images"] += [torch.squeeze(imgs[i]).detach().cpu() for i in range(B)]
            img_dict["GT"] += [GT_masks[i].detach().cpu() for i in range(B)]
            img_dict["masks"] += [sigmoid(classic_mask[i]).detach().cpu() for i in range(B)]
            img_dict["snakes"] += [(GT_masks[i].cpu(), ((GT_contour[i]+shift_cp)*rescaling_vect).cpu(),((snake_for_mask[i]+shift_cp)*rescaling_vect).cpu(),\
                                    ((reshaped_cp[i]+shift_cp)*rescaling_vect).cpu()) for i in range(B)]


    return running_loss / N, running_consistency_mask_loss / N, running_consistency_snake_loss / N,\
        running_reference_mask_loss / N, running_reference_snake_loss / N, img_dict



if __name__ == "__main__" :


    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" : 
        torch.cuda.empty_cache()

    print(f"Current device : {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Configuration file", default="./src/config/basic_config.json")
    args = parser.parse_args()

    config = args.config

    # Loading the config dic
    with open(config, "r") as f:
        config_dic = json.load(f)

    # Getting all the config information required into the config dic
    settings_config = config_dic["settings"]

    verbose = settings_config["verbose"]
    test_every_nb_epochs = settings_config["test_every_nb_epochs"]
    nb_batch_to_plot = settings_config["nb_batch_to_plot"]

    optimizer_config = config_dic["optimizer"]
    snake_config = config_dic["active_contour"]
    loss_config = config_dic["loss"]

    M = snake_config["M"]

    W, H = config_dic["data"]["image_size"]

    train_set = TextureDataset(**config_dic["data"]["train_set"], device = device)
    test_set = TextureDataset(**config_dic["data"]["test_set"], device = device)

    train_loader = DataLoader(train_set, **config_dic["data"]["train_loader"])
    test_loader = DataLoader(test_set, **config_dic["data"]["test_loader"])

    model_config = config_dic["model"]

    input_channels = config_dic["data"]["nb_channels"]

    # Initializing the model
    model = MCSnakeNet(**model_config, input_channels=input_channels, img_shape=(W,H), nb_control_points=M,\
                       padding_mode="zeros", inner_normalisation='BatchNorm', train_bn = True).to(device)


    # Initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)

    scheduler_config = config_dic["scheduler"]
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config)

    # Initializing the loss 
    mask_loss = {"bce": BCEWithLogitsLoss(), "dice": DiceLoss(), "mse" : MSELoss() }[loss_config["which_mask_loss"]]
    snake_loss = SnakeLoss()

    nb_epochs = settings_config["nb_epochs"]
    training_subset = config_dic["data"]["train_set"]["subset"]

    # Tracking of the loss and some images during the training
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=f"MC-snake_net-subset_{training_subset}",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": optimizer_config["lr"],
        "architecture": "UNET",
        "dataset": "Texture",
        "epochs": nb_epochs
        }
    )

    for epoch in range(nb_epochs):

        print(f"## Starting epoch {epoch} ##")
        epoch_dict = {}
        with time_manager(epoch_dict, f"epoch {epoch}"):
            loss, consistency_mask_loss, consistency_snake_loss, reference_mask_loss, reference_snake_loss, img_dict = \
                    train(model, optimizer, train_loader, mask_loss=mask_loss, snake_loss=snake_loss, **loss_config,\
                         W=W, H=H, device = device, nb_batch_to_plot=nb_batch_to_plot, verbose=verbose)

        print_time_dict(epoch_dict)

        sum_plot = create_subplot_summary(images_dict=img_dict)

        log_dict = {"train/loss": loss, "train/consistency-mask_loss" : consistency_mask_loss,\
                   "train/consistency-snake_loss" : consistency_snake_loss, "train/reference-mask_loss" : reference_mask_loss,\
                      "train/reference-snake_loss" : reference_snake_loss, "train_samples" : sum_plot}


        if (1 + epoch) % test_every_nb_epochs == 0:
            with torch.no_grad:
                loss, consistency_mask_loss, consistency_snake_loss, reference_mask_loss, reference_snake_loss, img_dict = \
                    test(model, test_loader, mask_loss=mask_loss, snake_loss=snake_loss, **loss_config, W=W, H=H, device = device)
            
            sum_plot = create_subplot_summary(images_dict=img_dict)

            test_dict = {"test/loss": loss, "test/consistency-mask_loss" : consistency_mask_loss,\
                    "test/consistency-snake_loss" : consistency_snake_loss, "test/reference-mask_loss" : reference_mask_loss,\
                        "test/reference-snake_loss" : reference_snake_loss, "test_samples" : sum_plot}
            
            log_dict = dict(log_dict, **test_dict)
            

        wandb.log(log_dict)
        close_plot()

        lr_scheduler.step()
        

    wandb.finish()

    now = datetime.now()
    torch.save(model.state_dict(), join(settings_config["save_path"],str(now).replace(" ","_").split(".")[0]+".pkl"))
