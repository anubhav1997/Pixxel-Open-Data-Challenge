import numpy as np
# import gdal
from skimage import io
from matplotlib import pyplot as plt 
import os 
from utils import make_display_image, plot
from argparse import ArgumentParser
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import traceback 
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from spectral import imshow, save_rgb
io.use_plugin('gdal')

def parse_args():

	parser = ArgumentParser(description='Segmentation of Hyperspectral Images using unsupervised clustering')
	parser.add_argument(
	    '-classifier', '--classifier',
	    type=str, default='MeanShift',
	    help='Select which clustering algorithm to use: MeanShift, DBSCAN or KMeans'
		)
	parser.add_argument(
	    '-data', '--data',
	    type=str, default='A',
	    help='Select which data to use'
		)
	
	return parser.parse_args()


def main():
	
	args = parse_args()

	X = []
	bgValue = -9999

	for i in range(1,243):

		if args.data == 'A':
		    filename = './Data/EO1H0350252010205110KU_1T/EO1H0350252010205110KU_B' + str(i).zfill(3) + '_L1T.TIF'
		elif args.data == 'B':
			filename = './Data/EO1H1430452010208110Kt/EO1H1430452010208110Kt_B' + str(i).zfill(3) + '_L1GST.TIF'
		elif args.data == 'C': 
			filename = './Data/EO1H1480472016328110PZ/EO1H1480472016328110PZ_B' + str(i).zfill(3) + '_L1GST.TIF'
	    
		img = io.imread(filename)
		X.append(img)

	X = np.asarray(X)
	X = X.transpose()

	# fig = plt.figure(figsize = (12, 6))
	# for i in range(1, 1+6):
	# 	fig.add_subplot(2,3, i)
	# 	q = np.random.randint(X.shape[2])
	# 	img = X[:,:,q]
	# 	img = img.transpose()
	# 	bgPixels = np.where(img==bgValue)
	# 	img_show = make_display_image(img, bgPixels)
	# 	plt.imshow(img, cmap='gray')
	# 	plt.axis('off')
	# 	plt.title(f'Band - {q}')


	X_flat = X.reshape((-1, X.shape[2]))

	print('here1')
	nmf = NMF(n_components=12, init='random', random_state=0)
	X_red = nmf.fit_transform(X_flat)

	print('here2')
	kmeans = KMeans(n_clusters=15, random_state=0).fit(X_red) 
	# Number of clusters found to be 13 using Elbow method for the central patch
	# This method isn't preferred with the patch based approach as each patch has different number of clusters. 

	print('here3')
	clustered = kmeans.cluster_centers_[kmeans.labels_]


	clustered = clustered.reshape((X.shape[0], X.shape[1], -1))


	# print("Silhouette Score: ", np.mean(score))


	save_rgb('Output.jpg', clustered, (30, 20, 10))

	print('final')


	# fig = plt.figure(figsize = (12, 6))
	# for i in range(1, 1+6):
	# 	fig.add_subplot(2,3, i)
	# 	q = np.random.randint(X.shape[2])
	# 	img = output_image[:,:,q]
	# 	img = img.transpose()
	# 	bgPixels = np.where(img==bgValue)
	# 	img_show = make_display_image(img, bgPixels)
	# 	plt.imshow(img, cmap='gray')
	# 	plt.axis('off')
	# 	plt.title(f'Band - {q}')

	for i in range(clustered.shape[2]):
		img = clustered[:,:,i]
		img = img.transpose()
		bgPixels = np.where(img==bgValue)
		img_show = make_display_image(img, bgPixels)
		plot(img_show, args.data, i)



if __name__ == '__main__':
	try:
	    main()
	except:
	    print(traceback.format_exc())
	    

