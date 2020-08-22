import numpy as np
from skimage import segmentation, color, io
from skimage.future import graph
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
# from osgeo import gdal
import gdal
import os



def weight_mean_color(graph, src, dst, n):
  diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
  diff = np.linalg.norm(diff)
  return {'weight': diff}

def merge_mean_color(graph, src, dst):
  graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
  graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
  graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                   graph.nodes[dst]['pixel count'])


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

inFile = './EO1H0350252010205110KU_B024_L1T.TIF'
outFile = './out.tif'

bgValue = -9999
io.use_plugin('gdal')

img = io.imread(inFile)


bgPixels = np.where(img == bgValue)

displayImg = make_display_image(img, bgPixels)
plt.imshow(displayImg, cmap='gray')
plt.show()


img[bgPixels] = -100
img = np.round(((img-np.min(img)) / (np.max(img)-np.min(img)+0.0)) * 600)




imgKmeans = segmentation.slic(img, compactness=110, n_segments=4000)

kMeansLabelsAve = color.label2rgb(imgKmeans, img, kind='avg')
fig, (ax0, ax1) = plt.subplots(1, 2 ,figsize=(20,10), dpi=72)
ax0.imshow(img, cmap='gray')
ax1.imshow(kMeansLabelsAve)
ax0.axis('off')
ax1.axis('off')
plt.show()


rag = graph.rag_mean_color(img, imgKmeans, mode='distance')

imgLabels = graph.merge_hierarchical(imgKmeans, rag, thresh=75, rag_copy=True,
                                     in_place_merge=True,
                                     merge_func=merge_mean_color,
                                     weight_func=weight_mean_color)

imgLabels[bgPixels] = imgLabels[0,0]

plt.figure(1, figsize=(10,10), dpi=72)
plt.imshow(mark_boundaries(displayImg, imgLabels))
plt.show()


src = gdal.Open(inFile)
driver = gdal.GetDriverByName('GTiff')
labelMax = np.max(imgLabels)

if labelMax <= 255:
  dataType = 1
elif labelMax <= 65535:
  dataType = 2
elif labelMax <= 4294967295:
  dataType = 4

outFile = os.path.splitext(outFile)[0]+'.tif'
outImg = driver.Create(outFile, src.RasterXSize, src.RasterYSize, 1, dataType)
outImg.SetGeoTransform(src.GetGeoTransform())
outImg.SetProjection(src.GetProjection())
outBand = outImg.GetRasterBand(1) 
outBand.WriteArray(imgLabels)
outImg = None
