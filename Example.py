import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(".") # To include Haisu module from local dir
from Haisu import HAISU 

# Parameters:
strength = 0.15
bTranspose = False
bNormalize = False
ncols = 1000
mpi = 1

# Create Data (Demonstrational):
submatrix = np.random.rand(1000, ncols)

#Input Hierarchy Adjacency Matrix
label_set = ['Lymphoid','Myeloid','T Cells','CD4 T','Conventional T','Effector/Memory T',
                'CD8+ Cytotoxic T', 'CD8+/CD45RA+ Naive Cytotoxic', 'CD4+/CD25 T Reg', 'CD19+ B', 
                'CD4+/CD45RO+ Memory', 'Dendritic','CD56+ NK', 'CD34+', 'CD4+/CD45RA+/CD25- Naive T',
                'CD14+ Monocyte', 'CD4+ T Helper2', 'Granulocyte/Macrophage','Dummy']

labels = [label_set[x] for x in np.random.randint(low=0, high=len(label_set), size=ncols)]

ajmatrix = np.zeros(shape=(len(label_set),len(label_set)))
ajmatrix[13] = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # 13 to 0 and 1
ajmatrix[1]  = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0] # 1 to 11 and 17
ajmatrix[17] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0] # 17 to 15
ajmatrix[0]  = [0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0] # 0 to 2,9,12
ajmatrix[2] =  [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0] # 2 to 3 and 6
ajmatrix[3] =  [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0] # 3 to 4 and 8
ajmatrix[4] =  [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0] # 4 to 14 5
ajmatrix[5] =  [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0] # 5 to 16 and 10
ajmatrix[6] =  [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0] # 6 to 7

# Init HAISU
print('Init')
haisu = HAISU(label_set, ajmatrix)

# Show graph
haisu.show_graph_info()
plt.show()
haisu.labeldict

# Run HAISU
print('Get pairwise matrix...')
X = haisu.get_pairwise_matrix(submatrix, labels, factor=strength, transpose=bTranspose, normalize=bNormalize, metric='euclidean', n_jobs=1)

sns.set(rc={'figure.figsize':(10,8)})

#TSNE:
from sklearn.manifold import TSNE
print('t-SNE')
X_embedded = TSNE(n_components=2, metric='precomputed',learning_rate=800, random_state=32, perplexity=100, n_jobs=mpi).fit_transform(X)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full', alpha=1,hue=labels)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show();

#UMAP
import umap
print('UMAP')
reducer = umap.UMAP(metric='precomputed',init='random', n_neighbors=100,min_dist=0.8)
X_embedded = reducer.fit(X).embedding_
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full', alpha=1,hue=labels)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show();
    
#PHATE
import phate
print('PHATE')
phate_op = phate.PHATE(knn_dist='precomputed',n_jobs=1,knn=50,decay=50)
X_embedded = phate_op.fit_transform(X)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full', alpha=1,hue=labels)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show();

#PCA
print('PCA')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_embedded = pca.fit_transform(X.copy())
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full', alpha=1,hue=labels)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show();