import numpy as np
import gdal
from skimage import io
from matplotlib import pyplot as plt 
import os 


def make_display_image(img, bgPixels):
    img = img.astype(np.float)
    img[bgPixels] = np.nan
    imgMean = np.nanmean(img)
    imgStd = np.nanstd(img)
    imgMinOrig = np.nanmin(img)
    imgMinCalc = imgMean-(imgStd*2)
    if imgMinCalc < imgMinOrig:
        imgMin = imgMinOrig
    else:
        imgMin = imgMinCalc
    img[bgPixels] = imgMin
    imgMax = imgMean+(imgStd*2)
    img[img < imgMin] = imgMin
    img[img > imgMax] = imgMax
    img = np.round(((img-imgMin) / (imgMax-imgMin+0.0)) * 255).astype(np.uint8)
    return img
    