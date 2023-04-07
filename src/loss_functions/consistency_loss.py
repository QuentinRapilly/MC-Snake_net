import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)     

        assert inputs.shape == targets.shape  
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class _SnakeLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, target):
        loss_tot = 0.0

        for i in range(len(target)):
            loss_list = list()
            ref_snake = target[i].float()
            pred_snake = x[i]

            nb_cp = ref_snake.shape[0]

            for _ in range(nb_cp):
                diff = torch.linalg.vector_norm(pred_snake-ref_snake, dim=-1)
                diff = torch.sum(diff, dim=-1)

                loss_list.append(diff/nb_cp)
                ref_snake = torch.roll(ref_snake, shifts=1, dims=0)
            
            if nb_cp>0:
                to_add, _ = torch.min(torch.tensor(loss_list, requires_grad=True), dim=0)
                loss_tot += to_add
            
        return loss_tot/len(target)


class SnakeLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, target):
        loss_tot = 0.0

        for i in range(len(target)):

            ref_snake = target[i].float()
            pred_snake = x[i]

            nb_cp = ref_snake.shape[0]

            if nb_cp > 0 :
                diff = torch.linalg.vector_norm(pred_snake-ref_snake, dim=-1)
                min_loss = torch.sum(diff, dim=-1)/nb_cp


                for _ in range(nb_cp - 1):
                    ref_snake = torch.roll(ref_snake, shifts=1, dims=0)

                    diff = torch.linalg.vector_norm(pred_snake-ref_snake, dim=-1)
                    diff = torch.sum(diff, dim=-1)/nb_cp

                    if diff < min_loss :
                        min_loss = diff
                
                loss_tot += min_loss
                    
            
        return loss_tot/len(target)

