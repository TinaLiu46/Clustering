# Clustering
Clustering is a type of unsupervised machine learning, which means the actual "y" (target variable) is not given when traning the model. By seperating the datapoints based on their structures or similarities, we're able to create several groups, called cluster.

## Implemente k-means Algorithm
- Step1: Select k unique points from datapoints as initial centroids (k is the number of clusters we want to generate)
- Step2: Put each datapoints into the k clusters based on the minimum distance to each centroids
- Step3: Calculate the mean of each cluster
- Step4: Repeat Step2 and Step3 so that the mean of each cluster = previously chosen centroids
## Apply "kmeans++" initialization
- Step1: Randomly select the first centroid from the data points.
- Step2: For each data point compute its distance from the nearest, previously chosen centroid.
- Step3: Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most likely to be selected next as a centroid)
- Step4: Repeat steps 2 and 3 until k centroids have been sampled
## Try the algorithm on 1-D, 2-D and Multi-dimensional dataset

- 1-D    
![Goal2](https://github.com/TinaLiu46/k_means/blob/main/images/one_dim.png?raw=true "Title")
- 2-D：
      
![Goal2](https://github.com/TinaLiu46/k_means/blob/main/images/pic.png?raw=true "Title")
- Multi-dimensional:
  Use breast_cancer data from sklearn, generate the following confusion matrix
![Goal2](https://github.com/TinaLiu46/k_means/blob/main/images/confusion_matrix.png?raw=true "Title")
## Implemente the algorirhm on image processing
K-means can generate some interesting artificial effect of the picture by clustering the color of the image. Rather than use millions of colors, we can usually get away with 256 or even 64 colors by setting different k in our k-means function. Below is an example of only showing 30 colors.
```python
  pixel_values = image.reshape((-1, 30))
  pixel_values = np.float32(pixel_values)
  k=30
  centroids = select_centroids(pixel_values, k=k)
  centroids, labels = kmeans(pixel_values, k=k, centroids=centroids, tolerance=.01)
  centroids = centroids.astype(np.uint8)
  segmented_image = centroids[labels.flatten()]
  segmented_image = segmented_image.reshape(image.shape)
  plt.imshow(segmented_image)
  plt.show()
```
![Goal2](https://github.com/TinaLiu46/k_means/blob/main/images/pearl_.png?raw=true "Title")

## Spectral clustering
Algorithm:
- Take our graph and built an adjacency matrix
- Create the Graph Laplacian by subtracting the adjacency matrix from the degree matrix and calculate the eigenvalues of the Laplacian. The vectors associated with those eigenvalues contain information on how to segment the nodes
- Performe K-Means on those vectors in order to get the labels for the nodes
## Conclusion

  Clustering is a very important method in Machine Learning. This report discussed one types of clustering - k-means and its initialization - kmeans++. K-means is good at group things that are continuous and visually seperated since it divides datapoints based on their distances. This report also mentions Spectral clustering, which group datapoints based on the graph structure.
