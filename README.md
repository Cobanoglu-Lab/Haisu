### Haisu 配す
*Hierarchical Supervised Nonlinear Dimensionality Reduction*

## Minimum Prerequisites
Python   : 3.7+ 
Libraries: networkx, sklearn

## Usage
1. Initialize HAISU object w/ set of labels and an adjacency matrix defining the hierarchy graph.
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

*See Example.py for adjacency matrix definition & application to UMAP, t-SNE, PHATE, and PCA*

## Authors
* Kevin VanHorn
* Murat Can Cobanoglu

## Copyright
Copyright © 2020, University of Texas Southwestern Medical Center. All rights reserved. Contributors: Kevin VanHorn, Murat Can Cobanoglu. Department: Lyda Hill Department of Bioinformatics.

## License
MIT License (see LICENSE.txt)
