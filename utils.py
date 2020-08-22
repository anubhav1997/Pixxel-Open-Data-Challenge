import numpy as np
import gdal
from skimage import io
from matplotlib import pyplot as plt 
import os 
import folium


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
    

def plot(img, data, band):

	if data == 'A':
		IMAGE_UL_CORNER_LAT = 51.025447
		IMAGE_UL_CORNER_LON = -103.500179
		IMAGE_UR_CORNER_LAT = 51.007606
		IMAGE_UR_CORNER_LON = -103.395644
		IMAGE_LL_CORNER_LAT = 50.104951
		IMAGE_LL_CORNER_LON = -103.812492
		IMAGE_LR_CORNER_LAT = 50.122141
		IMAGE_LR_CORNER_LON = -103.915221

		PRODUCT_UL_CORNER_LAT = 51.033217
		PRODUCT_UL_CORNER_LON = -103.894729
		PRODUCT_UR_CORNER_LAT = 51.027294
		PRODUCT_UR_CORNER_LON = -103.385774
		PRODUCT_LL_CORNER_LAT = 50.105292
		PRODUCT_LL_CORNER_LON = -103.916231
		PRODUCT_LR_CORNER_LAT = 50.099560
		PRODUCT_LR_CORNER_LON = -103.417168

	elif data == 'B':
		IMAGE_UL_CORNER_LAT = 22.513800
		IMAGE_UL_CORNER_LON = 80.785039 
		IMAGE_UR_CORNER_LAT = 22.499792
		IMAGE_UR_CORNER_LON = 80.857910 
		IMAGE_LL_CORNER_LAT = 21.579904
		IMAGE_LL_CORNER_LON = 80.557728 
		IMAGE_LR_CORNER_LAT = 21.565990
		IMAGE_LR_CORNER_LON = 80.630142 
		PRODUCT_UL_CORNER_LAT = 22.515894
		PRODUCT_UL_CORNER_LON = 80.551749
		PRODUCT_UR_CORNER_LAT = 22.516457
		PRODUCT_UR_CORNER_LON = 80.860953
		PRODUCT_LL_CORNER_LAT = 21.561877
		PRODUCT_LL_CORNER_LON = 80.554743
		PRODUCT_LR_CORNER_LAT = 21.562415
		PRODUCT_LR_CORNER_LON = 80.861882

	elif data == 'c':
		IMAGE_UL_CORNER_LAT = 19.565218
		IMAGE_UL_CORNER_LON = 72.968500 
		IMAGE_UR_CORNER_LAT = 19.552701
		IMAGE_UR_CORNER_LON = 73.036910 
		IMAGE_LL_CORNER_LAT = 18.641216
		IMAGE_LL_CORNER_LON = 72.761644 
		IMAGE_LR_CORNER_LAT = 18.628743
		IMAGE_LR_CORNER_LON = 72.829959 
		PRODUCT_UL_CORNER_LAT = 19.566794
		PRODUCT_UL_CORNER_LON = 72.747583
		PRODUCT_UR_CORNER_LAT = 19.570199
		PRODUCT_UR_CORNER_LON = 73.039126
		PRODUCT_LL_CORNER_LAT = 18.623998
		PRODUCT_LL_CORNER_LON = 72.760308
		PRODUCT_LR_CORNER_LAT = 18.627227
		PRODUCT_LR_CORNER_LON = 73.050207



	m = folium.Map([IMAGE_LR_CORNER_LAT, IMAGE_LR_CORNER_LON], zoom_start=8, tiles='Stamen Toner')
	folium.raster_layers.ImageOverlay(
	    image=img,
	    bounds=[[IMAGE_UL_CORNER_LAT, IMAGE_UL_CORNER_LON], [IMAGE_LR_CORNER_LAT, IMAGE_LR_CORNER_LON]],
	    # colormap=lambda x: (1, 0, 0, x),
	).add_to(m)


	m.save(os.path.join('Results', str(data), str(band) + '.html'))
