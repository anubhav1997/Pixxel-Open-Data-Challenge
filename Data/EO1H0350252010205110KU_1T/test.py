# from PIL import Image
import os
import gdal
import glob 
import traceback
import numpy as np 
import cv2 
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


import rasterio
import rasterio.plot


def dimensionality_reduction(MB_img):
	# Convert 2d band array in 1-d to make them as feature vectors and Standardization
	
	print(MB_img[:,:,0].size)

	img_shape = MB_img.shape

	n_bands = MB_img.shape[2]

	MB_matrix = np.zeros((MB_img[:,:,0].size,n_bands))

	for i in range(n_bands):
	    MB_array = MB_img[:,:,i].flatten()  # covert 2d to 1d array 
	    MB_arrayStd = (MB_array - MB_array.mean())/MB_array.std()  
	    MB_matrix[:,i] = MB_arrayStd
	
	print(MB_matrix.shape)

	# Rearranging 1-d arrays to 2-d arrays of image size

	PC_2d = np.zeros((img_shape[0],img_shape[1],n_bands))
	for i in range(n_bands):
	    PC_2d[:,:,i] = PC[:,i].reshape(-1,img_shape[1])

	# normalizing between 0 to 255
	PC_2d_Norm = np.zeros((img_shape[0],img_shape[1],n_bands))
	for i in range(n_bands):
	    PC_2d_Norm[:,:,i] = cv2.normalize(PC_2d[:,:,i], np.zeros(img_shape),0,255 ,cv2.NORM_MINMAX)


	fig,axes = plt.subplots(2,4,figsize=(50,23),sharex='all',
	                        sharey='all')
	fig.subplots_adjust(wspace=0.1, hspace=0.15)
	fig.suptitle('Intensities of Principal Components ', fontsize=30)

	axes = axes.ravel()
	for i in range(n_bands):
	    axes[i].imshow(PC_2d_Norm[:,:,i],cmap='gray', vmin=0, vmax=255)
	    axes[i].set_title('PC '+str(i+1),fontsize=25)
	    axes[i].axis('off')
	fig.delaxes(axes[-1])

	return PC_2d_Norm





def main():

	img = []
	for i in range(1,243):
		filename = './EO1H0350252010205110KU_B' + str(i).zfill(3) + '_L1T.TIF'


		# with rasterio.open(filename) as src:
		#     print(src.profile)

		# x = input()

		ds = gdal.Open(filename)
		band = ds.GetRasterBand(1)
		arr = band.ReadAsArray()		
		img.append(arr)

	img = np.asarray(img)
	img = img.reshape((img.shape[1], img.shape[2], img.shape[0]))
	
	output_image = np.zeros(img.shape)
	

	img = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

	pca = PCA(0.95)
	
	img_red = pca.fit_transform(img)

	kmeans = KMeans(n_clusters=5, random_state=0).fit(img_red)
	
	clustered = kmeans.cluster_centers_[kmeans.labels_]

	for i in range(img.shape[2]):
		plt.imshow(output_image[:,:, i])
		plt.title('Clustered Image band ' + str(i+1) )
		plt.show()






	# for i in range(10):
	# 	for j in range(10):

	# 		patch = img[img.shape[0]//10*i: img.shape[0]//10*(i+1), img.shape[1]//10*j:img.shape[1]//10*(j+1), :]
	# 		print(patch.shape)

	# 		patch_flat = patch.reshape((patch.shape[0]*patch.shape[1], patch.shape[2]))

	# 		# pca = PCA(0.95)

	# 		# patch_flat = pca.fit_transform(patch_flat)

	# 		nmf = NMF(n_components=12, init='random', random_state=0)

	# 		patch_flat = nmf.fit_transform(patch_flat)

	# 		kmeans = KMeans(n_clusters=5, random_state=0).fit(patch_flat)
			
	# 		clustered = kmeans.cluster_centers_[kmeans.labels_]


	# 		### add x,y information as well - show cluster in original 242 band image 

	# 		### or reconstruct this reduced image to back to original band value 

			
	# 		output_clustered = nmf.inverse_transform(clustered)

	# 		output_clustered = output_clustered.reshape((patch.shape[0], patch.shape[1], -1))

	# 		print(output_clustered.shape)

	# 		output_image[img.shape[0]//10*i: img.shape[0]//10*(i+1), img.shape[1]//10*j:img.shape[1]//10*(j+1), :] = output_clustered


	# for i in range(img.shape[2]):
	# 	if(i==23):
	# 		plt.imshow(output_image[:,:, i])
	# 		plt.title('Clustered Image band ' + str(i+1) )
	# 		plt.show()




# 	img = dimensionality_reduction(img)

# 	print(img.shape) 

# 	x = input()
	
# 	print(img.shape)
# 	vectorized = img.reshape((-1,3))
# 	vectorized = np.float32(vectorized)
# 	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 	K = 3
# 	attempts=10
# 	ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

# 	center = np.uint8(center)
# 	res = center[label.flatten()]
# 	result_image = res.reshape((img.shape))
	
# 	figure_size = 15
# 	plt.figure(figsize=(figure_size,figure_size))
# 	plt.subplot(1,2,1),plt.imshow(img)
# 	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# 	plt.subplot(1,2,2),plt.imshow(result_image)
# 	plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
# 	plt.show()

if __name__ == '__main__':
    try:
        main()
    except:
        print(traceback.format_exc())



