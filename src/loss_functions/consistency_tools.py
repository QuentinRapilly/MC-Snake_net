# This file regroup the functions dedicated to create the pseudo label for one decoder, using the output of the other.

from shapely.geometry import Polygon, Point
import torch
from torch.nn.functional import pad
#import matplotlib.pyplot as plt
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE
import numpy as np

# Changer les deux fonctions pour passer en multiobjet

def contour_to_mask(contour_samples : torch.tensor, W : int, H : int):
    poly = Polygon(contour_samples)
    mask = torch.zeros((W,H))

    for i in range(W):
        for j in range(H):
            p = Point(i,j)
            if p.within(poly): mask[i,j] = 1

    return mask


def mask_to_contour(mask : torch.tensor, only_one = True , add_last = False, verbose = False):
    np_mask = mask.cpu().numpy()
    contours, _ = findContours(np_mask.astype(np.uint8), mode=RETR_EXTERNAL, method=CHAIN_APPROX_NONE)

    if len(contours)==0:
        return torch.zeros((0,2))
    
    if only_one :
        contour = contours[np.argmax([len(contour) for contour in contours])]
        return torch.squeeze(torch.tensor(contour))
    
    return contours



if __name__ == "__main__":

    W, H = 50, 50

    img = torch.zeros((W,H))
    img[10:20,20:30] = torch.ones((10,10))

    contour = mask_to_contour(img, verbose=False, add_last=True)
    mask = contour_to_mask(contour, W, H)


    #plt.imshow(mask.T)
    #plt.plot(contour[0,:],contour[1,:])

    #plt.show()


