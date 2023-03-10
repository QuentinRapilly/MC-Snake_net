import torch
from torch.utils.data import DataLoader
import argparse
import json
from time import time
import wandb
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch import sigmoid

import matplotlib.pyplot as plt # enlever quand le probleme est regle


#from datasets.mcsnake_dataset import MCSnakeDataset 
from datasets.texture_dataset import TextureDataset

from DL_models.mcsnake_net import MCSnakeNet


from loss_functions.consistency_loss import MutualConsistency, DiceLoss, SnakeLoss
from loss_functions.consistency_tools import contour_to_mask, mask_to_contour
from snake_representation.snake_tools import sample_contour


def train(model, optimizer, train_loader, mask_loss, snake_loss, gamma, theta, M, W, H, verbose = False, device = "cpu", print_in_table = True):

    tic_epoch = time()

    running_loss = 0.0

    running_reference_mask_loss = 0
    running_reference_snake_loss = 0

    running_consistency_mask_loss = 0
    running_consistency_snake_loss = 0
    
    rescaling_vect = torch.tensor([W, H]).to(device)
    rescaling_inv = torch.tensor([1/W, 1/H]).to(device)

    for k, batch in enumerate(train_loader):

        print("Progress of the epoch : {}%          \r".format(round(k/len(train_loader)*100,ndigits=2)), end="")

        B = train_loader.batchsize
        imgs, GT_masks = batch

        optimizer.zero_grad()
        
        # Model applied to input
        tic_forward = time()
        classic_mask, snake_cp = model(imgs)
        tac_forward = time()

        #classic_mask = sigmoid(classic_mask)
        classic_mask = torch.squeeze(classic_mask)

        snake_cp = sigmoid(snake_cp)

        if B==1:
            GT_masks = torch.squeeze(GT_masks)
        

        """

        # Control points format (2M) -> (M,2)
        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], M, 2))


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

        #loss = (1 - gamma)*(theta*reference_mask_loss + (1-theta)*reference_snake_loss) +\
        #    gamma*(theta*consistency_mask_loss + (1-theta)*consistency_snake_loss)"""

        #print("Shape de classic_mask : {}, shape de GT mask : {}, extrema de classic_mask : {},{}".format(classic_mask.shape, GT_masks.shape,\
        #                                                                                                  torch.min(classic_mask),torch.max(classic_mask)))
        
        loss = mask_loss(classic_mask, GT_masks)


        # Backward gradient step
        tic_backward = time()
        loss.backward()
        optimizer.step()
        tac_backward = time()

        """
        # Printing time information if asked
        if verbose :
            print("Forward pass processed in {}s".format(tac_forward-tic_forward))
            print("Contour computed in {}s".format(tac_contour-tic_contour))
            print("Contour sampled in {}s".format(tac_sample-tic_sample))
            print("Mask computed in {}s".format(tac_mask-tic_mask))
            print("Loss computed in {}s".format(tac_loss-tic_loss))
            print("Backward pass processed in {}s".format(tac_backward-tic_backward))

        running_consistency_mask_loss += consistency_mask_loss.item()
        running_consistency_snake_loss += consistency_snake_loss.item()
        running_reference_mask_loss += reference_mask_loss.item()
        running_reference_snake_loss += reference_snake_loss.item()"""

        running_loss += loss.item()

        plt.figure(figsize = (20,10))
        for i in range(4):
            plt.subplot(3,B,1+i)
            plt.imshow(torch.squeeze(imgs[i]).detach().cpu(), cmap="gray", vmin=0, vmax=1)

        for i in range(4):
            plt.subplot(3,B,1+B+i)
            plt.imshow(GT_masks[i].detach().cpu(), cmap="gray", vmin=0, vmax=1)
        
        for i in range(4):
            plt.subplot(3,B,1+2*B+i)
            plt.imshow(sigmoid(classic_mask[i]).detach().cpu(), cmap="gray", vmin=0, vmax=1)
            #plt.imshow(classic_mask[i].detach().cpu(), cmap="gray", vmin=0, vmax=1)

        plt.savefig("/net/serpico-fs2/qrapilly/model_storage/MC-snake/batch_{}.png".format(k))

        plt.close()

        # Choosing some image to plot in the WAndB recap
        if k == len(train_loader) - 1:
            plot_res = (GT_masks[0], sigmoid(classic_mask[0])) #, snake_mask[0]) if print_in_table else None
    
    N = len(train_loader)

    tac_epoch = time()

    print("Epoch terminated in {}s".format(tac_epoch-tic_epoch))

    return running_loss / N, running_consistency_mask_loss / N, running_consistency_snake_loss / N,\
        running_reference_mask_loss / N, running_reference_snake_loss / N, plot_res



if __name__ == "__main__" :

    # TODO : sauvergarder le modele a la fin du training, creer une fonction de test

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
    criterion_config = config_dic["criterion"]
    snake_config = config_dic["active_contour"]

    W, H = config_dic["data"]["image_size"]

    train_set_index = train_config["set_index"]
    test_set_index = test_config["set_index"]

    train_set = TextureDataset(path=train_config["path_to_data"], subset=train_set_index, device = device)
    test_set = TextureDataset(path=train_config["path_to_data"], subset=test_set_index, device = device)

    train_loader = DataLoader(train_set, batch_size=train_config["batchsize"])
    test_loader = DataLoader(test_set, batch_size=test_config["batchsize"])


    # Initializing the model
    enc_chs=(config_dic["data"]["nb_channels"],64,128,256,512,1024)
    model = MCSnakeNet(enc_chs=enc_chs,typeA=model_config["typeA"], typeB=model_config["typeB"], nb_control_points=model_config["nb_control_points"], img_shape=(W,H)).to(device)

    # Initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])

    # Initializing the loss
    #mask_loss = DiceLoss() 
    #mask_loss = MSELoss() 
    mask_loss = BCEWithLogitsLoss() 

    criterion = MSELoss()
    snake_loss = SnakeLoss(criterion=criterion)
    gamma=criterion_config["gamma"]
    theta=criterion_config["theta"]
    #criterion = MutualConsistency(gamma=criterion_config["gamma"], device=device, verbose = verbose)



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
                train(model, optimizer, train_loader, mask_loss=mask_loss, snake_loss=snake_loss, gamma=gamma, theta=theta,\
                      M=snake_config["M"], W=W, H=H, verbose=verbose, device = device)

        
        gt, proba = plot_res #, snake
        gt = wandb.Image(gt, caption="GT")
        proba = wandb.Image(proba, caption="Probability map")
        #snake = wandb.Image(snake, caption="Snake mask")

        wandb.log({"loss": loss, "consistency_mask_loss" : consistency_mask_loss,\
                   "consistency_snake_loss" : consistency_snake_loss, "reference_mask_loss" : reference_mask_loss,\
                      "reference_snake_loss" : reference_snake_loss, "GT" : gt,\
                        "Probability map" : proba})#, "Snake mask" : snake})
        

    wandb.finish()

    torch.save(model.state_dict(), model_config["save_path"])
