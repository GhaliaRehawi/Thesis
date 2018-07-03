import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

def Gabor(img):

    kernels = compute_kernels()
    return compute_feats(img, kernels)


#Compute features using the kernels
def compute_feats(image, kernels):
    feats= np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k,0] = filtered.mean()
        feats[k,1] = filtered.var()
    return np.ravel(feats, order='F')


# prepare filter bank kernels
def compute_kernels():
    kernels = []
    for theta in range(4):  #direction (0,1,2,3)
        theta = theta / 4. * np.pi
        for sigman in range (1,3)
        kernel = np.real(gabor_kernel(frequency = 0.1, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel)
    return kernels
