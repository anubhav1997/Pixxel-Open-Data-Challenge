# Pixxel Challenge

### Repository Structure
- `main.py` conains the implementation of hyperspectral image segmentation. It takes in two parameters: 
  
  i. -data - which refers to which hyperspectral image you want to utilize. 
  
  ii. -classifier which clustering algorithm do you want to utilize - MeanShift or DBSCAN or KMeans



## Code

To test the code you run the command below. The code saves clustered images which are saved in `./Results/A/` 

```bash
$ python3 main.py -data A -classifier MeanShift
```




#### Results

##### Silhouette score for different values of Bandwidth for MeanShift Algorithm

| Bandwidth  | Score |
| ------------- | ------------- |
|  1 | 0.7892 |
| 2  | 0.7902  |
| 3 | 0.7942 |
| 4 | 0.7976 |
| 5 | 0.8102 |
| 6 |  0.7987 |

##### Analysis

I followed grid search for hyperparameter tuning. Out of the three clustering algorithms the best performing one is MeanShift based on the Silhouette score. 
