import torch
from torch.utils.data import DataLoader
import argparse
import json
from time import time
import wandb
from torch.nn import MSELoss


#from datasets.mcsnake_dataset import MCSnakeDataset 
from datasets.texture_dataset import TextureDataset

from DL_models.mcsnake_net import MCSnakeNet


from loss_functions.consistency_loss import MutualConsistency, DiceLoss, SnakeLoss
from loss_functions.consistency_tools import contour_to_mask, mask_to_contour
from snake_representation.snake_tools import sample_contour


def train(model, optimizer, train_loader, mask_loss, snake_loss, gamma, M, W, H, verbose = False, device = "cpu", print_in_table = True):

    tic_epoch = time()

    running_loss = 0.0

    running_reference_mask_loss = 0
    running_reference_snake_loss = 0

    running_consistency_mask_loss = 0
    running_consistency_snake_loss = 0
    
    rescaling_vect = torch.tensor([1/W, 1/H]).to(device)

    for k, batch in enumerate(train_loader):

        print("Progress of the epoch : {}%          \r".format(round(k/len(train_loader)*100,ndigits=2)), end="")

        imgs, GT_masks = batch

        optimizer.zero_grad()
        
        tic_forward = time()
        classic_mask, snake_cp = model(imgs)
        tac_forward = time()

        reshaped_cp = torch.reshape(snake_cp, (snake_cp.shape[0], M, 2))
        reshaped_cp = reshaped_cp*rescaling_vect

        classic_mask = torch.squeeze(classic_mask)

        tic_contour = time()
        with torch.no_grad():
            GT_contour = [mask_to_contour(mask).to(device)*rescaling_vect for mask in GT_masks]
            classic_contour = [mask_to_contour((mask>0.5)).to(device)*rescaling_vect for mask in classic_mask]
        tac_contour = time()


        tic_sample = time()
        snake_size_of_GT = [sample_contour(cp, nb_samples = GT_contour[i].shape[0], M=M, device = device) for i,cp in enumerate(reshaped_cp)]
        snake_size_of_classic = [sample_contour(cp, nb_samples = classic_contour[i].shape[0], M=M, device = device) for i,cp in enumerate(reshaped_cp)]
        tac_sample = time()
            

        tic_mask = time()
        with torch.no_grad():
            snake_mask = torch.stack([contour_to_mask(contour, W, H, device = device) for contour in snake_size_of_GT])
        tac_mask = time()   

        tic_loss = time()
        reference_mask_loss = mask_loss(classic_mask, GT_masks)
        reference_snake_loss = snake_loss(snake_size_of_GT, GT_contour)

        consistency_mask_loss = mask_loss(classic_mask, snake_mask)
        consistency_snake_loss = snake_loss(snake_size_of_classic, classic_contour)
        tac_loss = time()

        loss = (1 - gamma)*(reference_mask_loss + reference_snake_loss) + gamma*(consistency_mask_loss + consistency_snake_loss)


        tic_backward = time()
        loss.backward()
        optimizer.step()
        tac_backward = time()

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
        running_reference_snake_loss += reference_snake_loss.item()

        running_loss += loss.item()

        if print_in_table and (k==0):


            plot_res = (wandb.Image(GT_masks[0], caption="ground truth mask"), wandb.Image(classic_mask[0], caption="probability map"),\
                         wandb.Image(snake_mask[0], caption="snake mask"))

        else : 
            plot_res = None
    
    N = len(train_loader)

    tac_epoch = time

    print("Epoch terminated in {}s".format(tac_epoch-tic_epoch))

    return running_loss / N, running_consistency_mask_loss / N, running_consistency_snake_loss / N,\
        running_reference_mask_loss / N, running_reference_snake_loss / N, plot_res



if __name__ == "__main__" :

    # TODO : gerer partout l'utilisation de CUDA, logger les infos sur la loss etc avec wandb, 
    # sauvergarder le modele a la fin du training, creer une fonction de test

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Current device : {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Configuration file", default="./src/config/basic_config.json")
    args = parser.parse_args()

    config = args.config
    with open(config, "r") as f:
        config_dic = json.load(f)

    train_config = config_dic["data"]["train_set"]
    test_config = config_dic["data"]["test_set"]
    settings_config = config_dic["settings"]

    verbose = settings_config["verbose"]

    model_config = config_dic["model"]
    optimizer_config = config_dic["optimizer"]
    criterion_config = config_dic["criterion"]
    snake_config = config_dic["active_contour"]

    W, H = config_dic["data"]["image_size"]

    train_set = TextureDataset(path=train_config["path_to_data"], device = device)
    test_set = TextureDataset(path=train_config["path_to_data"], subset=2, device = device)

    train_loader = DataLoader(train_set, batch_size=train_config["batchsize"])
    test_loader = DataLoader(test_set, batch_size=test_config["batchsize"])


    enc_chs=(config_dic["data"]["nb_channels"],64,128,256,512,1024)
    model = MCSnakeNet(enc_chs=enc_chs,typeA=model_config["typeA"], typeB=model_config["typeB"], nb_control_points=model_config["nb_control_points"], img_shape=(W,H)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])

    mask_loss = DiceLoss()
    criterion = MSELoss()
    snake_loss = SnakeLoss(criterion=criterion)
    gamma=criterion_config["gamma"]
    #criterion = MutualConsistency(gamma=criterion_config["gamma"], device=device, verbose = verbose)



    # Tracking of the loss
    wandb.init(
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

    wandb_table = wandb.Table(columns=["epoch", "GT", "proba", "snake"])

    epoch_modulo = train_config["print_every_nb_epochs"]

    for epoch in range(train_config["nb_epochs"]):

        print(f"Starting epoch {epoch}")
        loss, consistency_mask_loss, consistency_snake_loss, reference_mask_loss, reference_snake_loss, plot_res = \
                train(model, optimizer, train_loader, mask_loss=mask_loss, snake_loss=snake_loss, gamma=gamma, M=snake_config["M"], W=W, H=H, verbose=verbose, device = device)

        wandb.log({"loss": loss, "consistency_mask_loss" : consistency_mask_loss,\
                   "consistency_snake_loss" : consistency_snake_loss, "reference_mask_loss" : reference_mask_loss,\
                      "reference_snake_loss" : reference_snake_loss})
        
        gt, proba, snake = plot_res
        wandb_table.add_data(epoch, gt, proba, snake)
        wandb.log({"Image table" : wandb_table})
        

    wandb.finish()

    torch.save(model.state_dict(), model_config["save_path"])
