import numpy as np
import gdal
from skimage import io
from matplotlib import pyplot as plt 
import os 
from utils import make_display_image
from argparse import ArgumentParser
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import traceback 
from sklearn.cluster import MeanShift

io.use_plugin('gdal')

def parse_args():

	parser = ArgumentParser(description='Segmentation of Hyperspectral Images using unsupervised clustering')
	parser.add_argument(
	    '-classifier', '--classifier',
	    type=str, default='KMeans',
	    help='Select which clustering algorithm to use: KMeans, MeanShift, Seeding or Thresholding'
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

	fig = plt.figure(figsize = (12, 6))
	for i in range(1, 1+6):
		fig.add_subplot(2,3, i)
		q = np.random.randint(X.shape[2])
		img = X[:,:,q]
		img = img.transpose()
		bgPixels = np.where(img==bgValue)
		img_show = make_display_image(img, bgPixels)
		plt.imshow(img, cmap='gray')
		plt.axis('off')
		plt.title(f'Band - {q}')


	output_image = np.zeros(X.shape)

	for i in range(10):
		for j in range(10):
			patch = X[X.shape[0]*i//10: X.shape[0]*(i+1)//10, X.shape[1]*j//10: X.shape[1]*(j+1)//10, :]

			patch_flat = patch.reshape((-1, patch.shape[2]))

			nmf = NMF(n_components=12, init='random', random_state=0)

			patch_flat = nmf.fit_transform(patch_flat)

			if args.classifier == 'KMeans':

				kmeans = KMeans(n_clusters=10, random_state=0).fit(patch_flat)

				clustered = kmeans.cluster_centers_[kmeans.labels_]
			else:
				meanshift = MeanShift(bandwidth=2, bin_seeding=True).fit(patch_flat)

				clustered = meanshift.cluster_centers_[meanshift.labels_]

			output_clustered = nmf.inverse_transform(clustered)

			output_clustered = output_clustered.reshape((patch.shape[0], patch.shape[1], -1))

			output_image[X.shape[0]*i//10: X.shape[0]*(i+1)//10, X.shape[1]*j//10: X.shape[1]*(j+1)//10, :] = output_clustered


	fig = plt.figure(figsize = (12, 6))
	for i in range(1, 1+6):
		fig.add_subplot(2,3, i)
		q = np.random.randint(X.shape[2])
		img = output_image[:,:,q]
		img = img.transpose()
		bgPixels = np.where(img==bgValue)
		img_show = make_display_image(img, bgPixels)
		plt.imshow(img, cmap='gray')
		plt.axis('off')
		plt.title(f'Band - {q}')
	    

if __name__ == '__main__':
	try:
	    main()
	except:
	    print(traceback.format_exc())
	    

