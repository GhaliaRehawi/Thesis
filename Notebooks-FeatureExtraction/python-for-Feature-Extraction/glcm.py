import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops

def Glcm(x):

    #Input: Integer typed input image
    ##Output: 2-dimensional array.
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'correlation', 'homogeneity']

    glcm = greycomatrix(x,
                    distances=distances,
                    angles=angles)

    #Output has the form : for one prop [[1,2] , [3,4]] --> ravel --> [1,2,3,4]
    ##For all prop : --> hstack --> [1,2,3,4,5,6,7,8,...]

    return np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
