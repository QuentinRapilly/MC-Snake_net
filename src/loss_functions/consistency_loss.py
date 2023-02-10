import torch.nn as nn
import torch.nn.functional as F
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class MutualConsistency(nn.Module):

    def __init__(self, gamma, device = "cpu") -> None:
        super().__init__()

        self.dice = DiceLoss()
        self.snake_loss = nn.MSELoss() #nn.L1Loss()
        self.gamma = gamma
        self.device = device


    def forward(self, ground_truth_mask, ground_truth_contour, snake_GT_size, snake_classic_size, snake_mask, classic_contour, classic_mask):
        # Dans un premier temps on part du principe que les contours sont des listes des differents contours

        nb_cp, _ = snake_GT_size[0].shape
        permut_matrix = torch.eye(nb_cp, requires_grad=False)
        permut_list = [nb_cp-1] + [k for k in range(nb_cp-1)]
        permut_matrix = permut_matrix[permut_list].to(self.device)

        seg_tot = 0
        
        for i in range(len(snake_GT_size)):
            current_ref_snake = ground_truth_contour[i]

            tmp_seg_loss = self.snake_loss(snake_classic_size[i], current_ref_snake)
            
            for _ in range(nb_cp-1):
                current_ref_snake = permut_matrix @ current_ref_snake
                seg = self.snake_loss(snake_GT_size[i], current_ref_snake)
                if seg.item() < tmp_seg_loss.item():
                    tmp_seg_loss = seg

            seg_tot += tmp_seg_loss


        nb_cp, _ = snake_classic_size[0].shape
        permut_matrix = torch.eye(nb_cp, requires_grad=False)
        permut_list = [nb_cp-1] + [k for k in range(nb_cp-1)]
        permut_matrix = permut_matrix[permut_list].to(self.device)

        consistency_tot = 0

        for i in range(len(snake_classic_size)):
            current_classic_snake = classic_contour[i]
            tmp_consistency_loss = self.snake_loss(snake_classic_size[i], current_classic_snake)
            
            for _ in range(nb_cp-1):
                current_classic_snake = permut_matrix @ current_classic_snake
                consistency = self.snake_loss(snake_classic_size[i], current_classic_snake)
                if consistency.item() < tmp_consistency_loss.item():
                    tmp_consistency_loss = consistency

            consistency_tot += tmp_consistency_loss
        

        # TODO : /!\ Bien verifier le bon fonctionnement de cette loss



        return (self.dice(classic_mask, ground_truth_mask) + seg_tot) + self.gamma * (self.dice(classic_mask, snake_mask) + consistency_tot)