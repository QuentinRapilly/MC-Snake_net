# This file regroup the functions dedicated to create the pseudo label for one decoder, using the output of the other.

from shapely.geometry import Polygon, Point
import torch
from torch.nn.functional import pad

def contour_to_mask(contour_samples : torch.tensor, W : int, H : int):
    poly = Polygon(contour_samples)
    mask = torch.zeros((W,H))

    for i in range(W):
        for j in range(H):
            p = Point(i,j)
            if p.within(poly): mask[i,j] = 1

    return mask

def mask_to_contour(mask : torch.tensor, add_last = False, verbose = False):

    W, H = mask.shape

    mask = pad(mask, (1,1,1,1), mode="constant", value=0)

    print(mask.shape)

    moore_pixel_to_ind = [[0,1,2],[7,-1,3],[6,5,4]]
    moore_ind_to_pixel = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]

    ### Moore Neighborhood ###
    # 
    #    # 0, 1, 2 #
    #    # 7, x, 3 #
    #    # 6, 5, 4 #
    #
    ##########################
    
    # find the first non black pixel
    i,j = 0,0
    while i<W and mask[i,j]==0:
        j = j+1
        if j == H:
            i = i+1
            j = 0

    current = [i,j]

    if verbose : print("First pixel : {}".format(current))

    contour = [current]
    
    moore_ind = 0 # we came in the first contour pixel by increasing j (pos 7 in Moore neighborhood, so the following one is 0)

    while mask[i+moore_ind_to_pixel[moore_ind][0],j+moore_ind_to_pixel[moore_ind][1]]==0:
        moore_ind = (moore_ind + 1) % 8

    following = [i+moore_ind_to_pixel[moore_ind][0],j+moore_ind_to_pixel[moore_ind][1]]


    while following != contour[0]:
        contour.append(following)

        previous = current
        current = following 

        i, j = current[0], current[1]
        di, dj = previous[0] - i, previous[1] - j 


        moore_ind = (moore_pixel_to_ind[di+1][dj+1] + 1) % 8


        while mask[i+moore_ind_to_pixel[moore_ind][0],j+moore_ind_to_pixel[moore_ind][1]]==0:
            moore_ind = (moore_ind + 1) % 8

        following = [i+moore_ind_to_pixel[moore_ind][0],j+moore_ind_to_pixel[moore_ind][1]]

    if add_last : contour.append(following)

    if verbose : print("Contour : {}".format(contour))

    return torch.tensor(contour) - 1


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    W, H = 50, 50

    img = torch.zeros((W,H))
    img[10:20,20:30] = torch.ones((10,10))

    contour = mask_to_contour(img, verbose=False, add_last=True)
    mask = contour_to_mask(contour, W, H)


    plt.imshow(mask.T)
    #plt.plot(contour[0,:],contour[1,:])

    plt.show()


