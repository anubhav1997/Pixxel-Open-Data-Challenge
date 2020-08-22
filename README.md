# Pixxel Challenge

### Implementation Details

The hyperspectral data contains 242 bands. To reduce the computational complexity of the algorithm I have followed a patch/ window based approach. The algorithm divides the image in 100 non-overlapping patches. For each of these patch the dimenions (number of bands) is reduced using NMF algorithm. The reduction is done such that ~95% (13 components) of the energy is preserved. The resultant image is then clustered using one of the clustering algorithms. The image is transformed back to 242 bands for final visualization. All the patches are stitched together to from the final clustered 242 band hyperspectral image. 


### Repository Structure
- `main.py` conains the implementation of hyperspectral image segmentation. It takes in two parameters: 
  
  i. -data - which refers to which hyperspectral image you want to utilize. 
  
  ii. -classifier which clustering algorithm do you want to utilize - MeanShift or DBSCAN or KMeans

- `utils.py` contains helper functions for plotting and visualizing the results

## Code

To test the code you run the command below. The code saves clustered images which are saved in `./Results/A/` 

```bash
$ python3 main.py -data A -classifier MeanShift
```
Please note that the implmentation makes use of the following libraries: Gdal, Sklearn, Skimage, Numpy, Matplotlib, folium. 



#### Results

##### Silhouette score for different values of Bandwidth for MeanShift Algorithm

| Bandwidth  | Score |
| ------------- | ------------- |
|  1 | 0.7892 |
| 2  | 0.7902  |
| 3 | 0.7942 |
| 4 | 0.7976 |
| 5 | 0.466 |

##### Analysis

I followed grid search for hyperparameter tuning. Out of the three clustering algorithms the best performing one is MeanShift based on the Silhouette score. 
