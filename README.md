### Haisu 配す
*Hierarchical Supervised Nonlinear Dimensionality Reduction*

## Minimum Prerequisites
Python   : 3.7+ 
Libraries: networkx(2+), sklearn

Installation:
Create and activate a local anaconda environment with additional libraries for debugging and visualization:
```sh
conda env create -f environment.yaml --prefix haisu
conda activate ./haisu
```

## Usage
1. Initialize HAISU object w/ set of labels and an adjacency matrix defining the hierarchy graph. Optionally set disconnected_dist to a value > 1 to specify the normalized distance of disconnected nodes where non-disconnected nodes are between values [0,1]

- disconnected_dist: The path distance before normalization between two disconnected graph nodes.
- avoid_self: The list of labels to apply the parameter self_dist to.
- self_dist: The distance from a node to itself [0,1] applied after normalization. 
- edge_weights: Weights for nodes using tuple graph node indices matching the adjanceny matrix indices - applied before normalization (1 by default).
               ex: edge_weights = {(1, 3): 0.2, (3, 0): 2}

```sh
haisu = HAISU(label_set, adjacency_matrix)
```
2. Get Pairwise distance matrix for application to dimensionality reduction
```sh
X = haisu.get_pairwise_matrix(X, X_labels, factor=strength, metric='euclidean', n_jobs=1)
```
3. Apply as 'pre-computed' distance function to any application technique:
```sh
# Example: t-SNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, metric='precomputed').fit_transform(X)
```

*See Example.py or Example.ipynb for adjacency matrix definition & application to UMAP, t-SNE, PHATE, and PCA*
*See Random.ipynb for examples of extra Haisu parameters including disconnected_dist, avoid_self, and self_dist*

## Authors
* Kevin VanHorn
* Murat Can Cobanoglu

## Copyright
Copyright © 2022, University of Texas Southwestern Medical Center. All rights reserved. Contributors: Kevin C. VanHorn, Murat Can Cobanoglu. Department: Lyda Hill Department of Bioinformatics.

## License
MIT License (see LICENSE.txt)
