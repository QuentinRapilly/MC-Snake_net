# This file regroup the functions dedicated to create the pseudo label for one decoder, using the output of the other.

from shapely.geometry import Polygon, Point
import torch
from torch.nn.functional import pad
#import matplotlib.pyplot as plt
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE
import numpy as np

# Changer les deux fonctions pour passer en multiobjet

"""def contour_to_mask(contour_samples : torch.tensor, W : int, H : int, device = "cpu"):
    poly = Polygon(contour_samples)
    mask = torch.zeros((W,H))

    for i in range(W):
        for j in range(H):
            p = Point(i,j)
            if p.within(poly): mask[i,j] = 1

    return mask"""

def contour_to_mask(contour_samples : torch.tensor, W : int, H : int, device = "cpu"):

    v1_tmp = contour_samples
    v2_tmp = torch.roll(contour_samples, -1, 0)
    eps = 10e-5

    v1 = v1_tmp * (v1_tmp >= v2_tmp) + v2_tmp * (v1_tmp < v2_tmp)
    v2 = v1_tmp * (v1_tmp < v2_tmp) + v2_tmp * (v1_tmp >= v2_tmp)

    a = torch.unsqueeze((v1[:,1]+eps)*v2[:,1], 0)
    b = torch.unsqueeze(v1[:,1] + eps + v2[:,1], 0)
    c = torch.unsqueeze(v1[:,0] - v2[:,0],0)
    d = torch.unsqueeze(v1[:,1] - v2[:,1],0)
    e = torch.unsqueeze(v1[:,1]*v2[:,0] - v1[:,0]*v2[:,1],0)

    u_y = torch.unsqueeze(torch.arange(0, H, 1),1).to(device)
    u_x = torch.unsqueeze(torch.arange(0, W, 1),1).to(device)

    cdt1 =  (a + u_y*u_y - b*u_y < 0)*1.
    cdt2 = ((torch.unsqueeze(c*u_y,0) - torch.unsqueeze(d*u_x,1) + torch.unsqueeze(e, 0)) < 0)*1.

    S = torch.sum(torch.unsqueeze(cdt1,0)*cdt2, dim=2)

    return ((S % 2) == 1)*1.



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

    contour = torch.tensor([[10,10],[30,10],[30,20],[20,20],[20,30],[30,40],[10,40]])

    #contour = mask_to_contour(img, verbose=False, add_last=True)
    mask = contour_to_mask(contour, W, H)


    #plt.imshow(mask)
    #plt.plot(contour[0,:],contour[1,:])

    #plt.show()


