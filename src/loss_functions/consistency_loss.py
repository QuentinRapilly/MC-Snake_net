import torch.nn as nn
import torch.nn.functional as F
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)     

        assert inputs.shape == targets.shape  
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class SnakeLoss(nn.Module):

    def __init__(self, criterion) -> None:
        super().__init__()
        
        self.criterion = criterion

    def forward(self, input, target):
        loss_tot = 0

        for i in range(len(target)):
            ref_snake = target[i].float()
            tmp_loss = self.criterion(input[i], ref_snake)

            nb_cp, _ = ref_snake.shape
            
            for _ in range(nb_cp-1):
                ref_snake = torch.roll(ref_snake, shifts=1, dims = 0)
                seg = self.criterion(input[i], ref_snake)
                if seg.item() < tmp_loss.item():
                    tmp_loss = seg
            
            loss_tot += tmp_loss
        
        return loss_tot/len(target)


class MutualConsistency(nn.Module):

    def __init__(self, gamma, device = "cpu", verbose = False) -> None:
        super().__init__()

        self.dice = DiceLoss()
        self.snake_loss = nn.MSELoss() #nn.L1Loss()
        self.gamma = gamma
        self.device = device

        self.verbose = verbose


    def forward(self, ground_truth_mask, ground_truth_contour, snake_GT_size, snake_classic_size, snake_mask, classic_contour, classic_mask):
        # Dans un premier temps on part du principe que les contours sont des listes des differents contours

        seg_tot = 0
        
        for i in range(len(snake_GT_size)):

            current_ref_snake = ground_truth_contour[i].float()
            tmp_seg_loss = self.snake_loss(snake_GT_size[i], current_ref_snake)

            nb_cp, _ = snake_GT_size[i].shape
            
            for _ in range(nb_cp-1):
                current_ref_snake = torch.roll(current_ref_snake, shifts=1, dims = 0)
                seg = self.snake_loss(snake_GT_size[i], current_ref_snake)
                if seg.item() < tmp_seg_loss.item():
                    tmp_seg_loss = seg

            seg_tot += tmp_seg_loss

        consistency_tot = 0

        for i in range(len(snake_classic_size)):

            current_classic_snake = classic_contour[i].float()
            tmp_consistency_loss = self.snake_loss(snake_classic_size[i], current_classic_snake)

            nb_cp, _ = snake_classic_size[i].shape
            
            for _ in range(nb_cp-1):
                current_classic_snake = torch.roll(current_classic_snake, shifts=1, dims=0)
                consistency = self.snake_loss(snake_classic_size[i], current_classic_snake)
                if consistency.item() < tmp_consistency_loss.item():
                    tmp_consistency_loss = consistency

            consistency_tot += tmp_consistency_loss

        # TODO : /!\ Bien verifier le bon fonctionnement de cette loss

        return (1 - self.gamma)*(self.dice(classic_mask, ground_truth_mask) + seg_tot/len(snake_GT_size)) + self.gamma * (self.dice(classic_mask, snake_mask) + consistency_tot/len(snake_classic_size))
