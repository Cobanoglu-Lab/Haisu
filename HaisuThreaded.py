import numpy as np
import sys
import networkx as nx
from sklearn.metrics import pairwise_distances
from scipy import stats
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import multiprocessing as mp
import ctypes

#HELPERS:
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
def concat_onehot(X, labels, global_weight = 1):
    values = np.array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return np.concatenate((X.to_numpy(),onehot_encoded*global_weight),axis=1)
# Ex Usage: X2 = concat_onehot(X.transpose(), np.array(labels[0:ncols]), 1); tsne.fit_transform(X2)

'''
An Expanded version of Haisu to support additional configurations
'''
class HAISU:
    def __init__(self, graph_labels, ajmatrix):
        self.X_embedded = None
        self.labels = None
        self.labeldict = {}
        self.graph = None
        self.max_shortestpath = 0
        self.pathcache = None
        self.labelvalues = None
        self.label_probs = []
        self._init_graph(graph_labels, ajmatrix)
        
        # AUTO:
        self._X = None
        self._ylabels = None
        self._probs = None
        self._factor = None
        self._ylabels_probs = None
        self._transpose = None
        self._normalize = None
        self._metric = None
        self._n_job = None
        
    def _init_graph(self, graph_labels, ajmatrix):
        # Dictionary from labels:
        cnt = 0
        for label in graph_labels:
            #print(str(cnt) + " " + label);
            self.labeldict[sys.intern(label)] = cnt; cnt+=1
            
        # Make graph & find maximum shortest path:
        self.graph = nx.from_numpy_matrix(ajmatrix)
        for i in range(self.graph.size()+1):
            for j in range(self.graph.size()+1):
                path_len = nx.shortest_path_length(self.graph, i, j)
                if path_len > self.max_shortestpath:
                    self.max_shortestpath = path_len
                    
        #self.pathcache = np.zeros((self.graph.size()+1, self.graph.size()+1))
        self.pathcache = np.ones((len(graph_labels)+1, len(graph_labels)+1)) # max dist (1) for disconnected graphs
        for i in range(self.graph.size()+1):
            for j in range(self.graph.size()+1):
                if(i == j):
                    self.pathcache[i,j] = 0
                elif (nx.shortest_path_length(self.graph, i, j) < 1):
                    self.pathcache[i,j] = 0
                else:
                    self.pathcache[i,j] = (nx.shortest_path_length(self.graph, i, j))/self.max_shortestpath
                    
    def show_graph_info(self):
        cnt = 0; glabels={}
        for label in self.labeldict:
            glabels[cnt] = str(cnt)
            cnt+=1
        nx.draw(self.graph, labels = glabels)
        
    def get_overlaps(self, data, labels):
        indices = np.array(labels)
        overlaps = []; shapes = []; alldists = []
        for label in np.unique(labels):
            shape_overlaps = []; dists = [float('inf')]
            for label2 in np.unique(labels):
                shape = None; 
                if(label == label2):
                    shape_overlaps.append(0)
                else:
                    points1 = data[(indices == label)[0:data.shape[0]]]
                    points2 = data[(indices == label2)[0:data.shape[0]]]
                    c1 = points1.mean(axis=0); c2 = points2.mean(axis=0); 
                    dists.append((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) # centroid eucledian distance
                    # remove verts by z-value mean (aggressive)
                    z1 = np.abs(stats.zscore(points1))
                    z2 = np.abs(stats.zscore(points2))
                    points1_ = points1[(z1 < (z1.mean() + (z1.max()-z1.mean())/4)).all(axis=1)]
                    points2_ = points2[(z2 < (z2.mean() + (z2.max()-z2.mean())/4)).all(axis=1)]
                    if(points1_.shape[0] > 3): points1 = points1_;
                    if(points2_.shape[0] > 3): points2 = points2_;
                    shape1 = Polygon(points1[ConvexHull(points1).vertices])
                    shape2 = Polygon(points2[ConvexHull(points2).vertices])
                    shape_overlaps.append(shape1.intersection(shape2).area/shape1.area)
                    shape = shape1
            shapes.append(shape)
            overlaps.append(shape_overlaps)
            alldists.append(dists)
        return overlaps, shapes, alldists
    
    def get_overlap_score(self, data, labels):
        overlaps, shapes, dists = self.get_overlaps(data, labels)
        # For each cluster, get the shape overlap of closest cluster by centroid. Take the mean of all of that to get the score
        return np.array([overlaps[i][dists[i].index(min(dists[i]))] for i in range(len(overlaps))]).mean(), shapes
        
    def init(self, X, ylabels, factor, ylabel_probs=[], transpose=False, normalize=True, metric='euclidean',n_jobs=1):
        self._X = X
        self._ylabels = ylabels
        self._probs = ylabel_probs
        self._factor = factor
        self._ylabels_probs = ylabel_probs
        self._transpose = transpose
        self._normalize = normalize
        self._metric = metric
        self._n_jobs = n_jobs
    
    def get_pairwise_matrix(self, X, ylabels, factor, ylabel_probs=[], transpose=False, squared=True,normalize=True, metric='euclidean',n_jobs=1):
        if transpose:
            X = X.transpose()
        
        if len(ylabel_probs) == len(ylabels):
            self.label_probs = ylabel_probs
        
        if(len(ylabels) != X.shape[1]):
            print('%d labels provided for %d columns/samples.' %(len(ylabels),X.shape[1]))
            print('Please provide the input matrix with axis=1 as the labeled samples, or specify transpose arg.')
            sys.exit("Columns do not match labels.")

        self.labels = ylabels;
        X = X.transpose()
        
        #Considers the rows of X (and Y=X) as vectors, & computes the distance between each pair of vectors.
        distances = pairwise_distances(X, metric=metric, squared=squared,n_jobs=1) 
        X = X.transpose()
        pathpairwise = self.path_pairwise(X, factor, n_jobs = n_jobs)
        np.fill_diagonal(pathpairwise, 0)
        distances = np.multiply(distances,pathpairwise)

        # Normalize distances
        if normalize:
            distances=(distances-distances.min())/(distances.max()-distances.min())
        
        return distances
    
    def get_pairwise_silo(self, X, ylabels, factor, ylabel_probs=[], transpose=False, normalize=True, metric='euclidean',n_jobs=1):
        # Remove restriction within class:
        for i in range(self.graph.size()+1):
            for j in range(self.graph.size()+1):
                if(i == j):
                	self.pathcache[i,j] = 1/(self.max_shortestpath)
        print('WARNING: destructively modified haisu graph.')
        if transpose:
            X = X.transpose()
        
        if len(ylabel_probs) == len(ylabels):
            self.label_probs = ylabel_probs
        
        if(len(ylabels) != X.shape[1]):
            print('%d labels provided for %d columns/samples.' %(len(ylabels),X.shape[1]))
            print('Please provide the input matrix with axis=1 as the labeled samples, or specify transpose arg.')
            sys.exit("Columns do not match labels.")
        self.labels = ylabels;
        X = X.transpose()
        distances = pairwise_distances(X, metric=metric, squared=False,n_jobs=n_jobs) 
        X = X.transpose()
        pathpairwise = self.path_pairwise(X, factor, n_jobs = n_jobs)

        np.fill_diagonal(pathpairwise, 0)
        distances = np.multiply(distances,pathpairwise)

        # Normalize distances
        if normalize:
            distances=(distances-distances.min())/(distances.max()-distances.min())
        
        return distances

    def path_dist(self, label1, label2):
    	return self.pathcache[self.labeldict.get(self.labels[label1]),self.labeldict.get(self.labels[label2])]

    def path_multi(self, i):
        dists = np.zeros(len(self.labels))
        for j in range(len(self.labels)):
            dists[j] = (1-self.factor)+self.pathcache[self.labeldict[self.labels[i]],self.labeldict[self.labels[j]]]*self.factor
        return dists

    def path_pairwise(self, x, factor = 1, squared=True, n_jobs = 1):
        shape = len(self.labels)
        dists = None#= np.zeros((shape, shape))

        if(len(self.label_probs) > 0):
            dists = np.zeros((shape, shape))
            for i in range(shape):     # loops over rows of `x`
                for j in range(shape): # loops over rows of `y`
                    ijp = min(self.label_probs[i],self.label_probs[j])
                    dists[i, j] = (1-(factor*ijp))+self.path_dist(i,j)*(factor*ijp) 
            dists = dists
        else:
            if(n_jobs == 1):
                dists = np.zeros((shape, shape),dtype=np.float16)
                for i in range(1,shape):
                    for j in range(i):
                        dists[i,j] = (1-factor)+self.pathcache[self.labeldict[self.labels[i]],self.labeldict[self.labels[j]]]*factor
                        dists[j,i] = dists[i,j]
                dists = dists
            elif(n_jobs > 1):
                self.factor = factor
                with mp.Pool(n_jobs) as p: #mp.cpu_count()
                    dists = np.array(p.map(self.path_multi, range(shape)))#.reshape((shape, shape))
        return dists