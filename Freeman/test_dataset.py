import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':
    f = h5py.File('temp.h5','r')

    dataCS = f.get('CS')
    dataHR = f.get('HR')


    test_vec = dataCS[8]
    test_vec[-1] = 90
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(dataCS)

    dis, ind = nbrs.kneighbors([test_vec])

    proxHR = dataHR[ind[0]]
    proxHR = proxHR.reshape(5,5,3)
    proxCS = dataCS[ind[0]]
    proxCS = proxCS.reshape(7,7,3)
    test = test_vec.reshape(7,7,3)

    f.close()

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(test)
    ax[1].imshow(proxCS)
    ax[2].imshow(proxHR)

    plt.draw()
    plt.pause(1e-3)