import torch
import matplotlib.pyplot as plt
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE
import numpy as np
from math import ceil

# This file regroup the functions dedicated to create the pseudo label for one decoder, using the output of the other.


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
            # Pas optimal si le contour le plus long n'est composé que d'un point
            # car ça va squeezer une dimension de trop TODO : trouver un meilleur moyen
    
    return contours


def limit_nb_points(contour : torch.tensor, max_points : int):

    l = contour.shape[0]
    r = max(1,ceil(l/max_points))

    return contour[::r]



if __name__ == "__main__":

    #TEST = "CONTOUR"
    TEST = "MASK"

    ### To test contour_to_mask feature ###
    if TEST == "CONTOUR":
        W, H = 50, 50

        contour = torch.tensor([[10,10],[30,10],[30,20],[20,20],[20,30],[30,40],[10,40]])
        mask = contour_to_mask(contour, W, H)

        print(mask.shape)

        plt.imshow(mask)
        plt.plot(contour[:,1],contour[:,0])

        plt.show()


    
    ### To test mask_to_contour feature ###
    if TEST == "MASK":
        mask = plt.imread("/net/serpico-fs2/qrapilly/data/Texture/Masks/mask0.bmp")
        mask = torch.tensor(mask)
        contour = mask_to_contour(mask)
        print(mask.shape)

        plt.imshow(mask)
        plt.plot(contour[::32,0], contour[::32,1])
        plt.show()


