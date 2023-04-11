import matplotlib.pyplot as plt

def create_subplot_summary(images_dict : dict):
    fig = plt.figure(figsize = (20,10))
    n = len(images_dict)
    for i,key in enumerate(images_dict):
        row = images_dict.get(key)

        batch_size = len(row)
        for j,img in enumerate(row) :
            plt.subplot(n,batch_size,1+i*batch_size+j) 
            if type(img) == tuple:
                GT_img, GT_contour, contour, cp = img
                plt.imshow(GT_img, cmap="gray", vmin=0, vmax=1)
                #plt.scatter(GT_contour[:,0], GT_contour[:,1], c=[ind for ind in range(len(GT_contour))], marker=".", cmap="Greens")
                plt.scatter(contour[:,0],contour[:,1], marker=".", cmap="Blues", s=5)
                plt.scatter(cp[:,0], cp[:,1], marker="x", c="red", s=5)
                #for k in range(len(cp)):
                #    plt.text(cp[k,0], cp[k,1], str(k), c="red")
            else:
                plt.imshow(img, cmap="gray", vmin=0, vmax=1)

    return fig