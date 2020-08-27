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
	labels = []
	centers = []
	score = []
	index = []
	ind = 0
	for i in range(10):
		for j in range(10):

			patch = X[X.shape[0]*i//10: X.shape[0]*(i+1)//10, X.shape[1]*j//10: X.shape[1]*(j+1)//10, :]
			patch_flat = patch.reshape((-1, patch.shape[2]))

			nmf = NMF(n_components=12, init='random', random_state=0)
			patch_flat = nmf.fit_transform(patch_flat)

			if args.classifier == 'KMeans':

				kmeans = KMeans(n_clusters=20, random_state=0).fit(patch_flat) 
				clustered = kmeans.cluster_centers_[kmeans.labels_]

				if(len(centers)==0):
					centers = nmf.inverse_transform(kmeans.cluster_centers_)
				else:
					centers = np.append(centers, nmf.inverse_transform(kmeans.cluster_centers_), axis=0)
				index = np.append(index, np.ones(len(kmeans.cluster_centers_))*ind)
				ind+=1 

				labels.append(np.expand_dims(kmeans.labels_, axis =0))
			
			elif args.classifier == 'MeanShift':

				meanshift = MeanShift(bin_seeding=True).fit(patch_flat) 
				clustered = meanshift.cluster_centers_[meanshift.labels_]
				
				if(len(centers)==0):
					centers = nmf.inverse_transform(meanshift.cluster_centers_)
				else:
					centers = np.append(centers, nmf.inverse_transform(meanshift.cluster_centers_), axis=0)
				index = np.append(index, np.ones(len(meanshift.cluster_centers_))*ind)
				ind+=1 
				print(ind)

				labels.append(np.expand_dims(meanshift.labels_, axis =0))

				if(np.max(meanshift.labels_)>1):
					score.append(silhouette_score(patch_flat, meanshift.labels_))

			elif args.classifier == 'DBSCAN':
				
				patch = patch_flat.reshape((patch.shape[0], patch.shape[1], -1))

				indices = np.dstack(np.indices(patch.shape[:2]))
				xycolors = np.concatenate((patch, indices), axis=-1) 
				feature_image = np.reshape(xycolors, [-1,3])
				db = DBSCAN(eps=3, min_samples=10).fit(feature_image)

				labels = db.labels_
				clustered = labels.reshape((labels2.shape[0], 1))

				if(np.max(clf.labels_)>1):
					score.append(silhouette_score(patch_flat, db.labels_))


			output_clustered = nmf.inverse_transform(clustered)
			output_clustered = output_clustered.reshape((patch.shape[0], patch.shape[1], -1))
			output_image[X.shape[0]*i//10: X.shape[0]*(i+1)//10, X.shape[1]*j//10: X.shape[1]*(j+1)//10, :] = output_clustered

	print("Silhouette Score: ", np.mean(score))


	del output_image, output_clustered, meanshift, clustered

	centers = np.asarray(centers)

	nmf = NMF(n_components=12, init='random', random_state=0)
	centers = nmf.fit_transform(centers)

	kmeans = KMeans(n_clusters=10, random_state=0).fit(centers) 
	l = 0

	output_image2 = np.zeros(X.shape)

	centers = nmf.inverse_transform(centers)
	cluster_centers = nmf.inverse_transform(kmeans.cluster_centers_)

	for i in range(10):
		for j in range(10):
			k = np.where(index==l)
			c = centers[k]
			new_labels_of_old_centers = kmeans.labels_[k]
			new_labels = new_labels_of_old_centers[np.array(labels[l])]
			clustered = np.array(cluster_centers)[np.array(new_labels)]
			l+=1 

			try:
				clustered = clustered.reshape((X.shape[0]//10, -1 , 242))
			except:
				try:
					clustered = clustered.reshape((-1, X.shape[1]//10 , 242))
				except:
					clustered = clustered.reshape((X.shape[0]//10+1, X.shape[1]//10+1, 242))

			output_image2[X.shape[0]*i//10: X.shape[0]*(i+1)//10, X.shape[1]*j//10: X.shape[1]*(j+1)//10, :] = clustered


	save_rgb('Output.jpg', output_image2, (30, 20, 10))



	fig = plt.figure(figsize = (12, 6))
	for i in range(1, 1+6):
		fig.add_subplot(2,3, i)
		q = np.random.randint(X.shape[2])
		img = output_image2[:,:,q]
		img = img.transpose()
		bgPixels = np.where(img==bgValue)
		img_show = make_display_image(img, bgPixels)
		plt.imshow(img, cmap='gray')
		plt.axis('off')
		plt.title(f'Band - {q}')

	for i in range(output_image2.shape[2]):
		img = output_image2[:,:,i]
		img = img.transpose()
		bgPixels = np.where(img==bgValue)
		img_show = make_display_image(img, bgPixels)
		plot(img_show, args.data, i)



if __name__ == '__main__':
	try:
	    main()
	except:
	    print(traceback.format_exc())
	    

