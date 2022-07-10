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

disconnected_dist: the graph distance between disconnected components (before normalization [0,1])
avoid_self: a list of labels for which intra-class labels are weighted (after normalization) with min(self_dist,1)
'''
class HAISU:
    def __init__(self, graph_labels, ajmatrix, disconnected_dist=1, avoid_self=[], self_dist=1, edge_weights={}):
        self.X_embedded = None
        self.labels = None
        self.labeldict = {}
        self.graph = None
        self.max_shortestpath = 0
        self.pathcache = None
        self.labelvalues = None
        self.label_probs = []
        self.edge_weights = {}
        self._init_graph(graph_labels, ajmatrix,disconnected_dist, avoid_self, self_dist, edge_weights)

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
        
    def _init_graph(self, graph_labels, ajmatrix,disconnected_dist,avoid_self,self_dist, edge_weights):
        # Dictionary from labels:
        cnt = 0
        for label in graph_labels:
            #print(str(cnt) + " " + label);
            self.labeldict[sys.intern(label)] = cnt; cnt+=1
            
        # Make graph & find maximum shortest path:
        self.edge_weights = edge_weights
        self.graph = nx.from_numpy_matrix(ajmatrix)
        nx.set_edge_attributes(self.graph, self.edge_weights, 'weight')

        def weight_func(i,j,ddict):
            return ddict['weight']

        for i in range(self.graph.size()+1):
            for j in range(self.graph.size()+1):
                #path_len = 0 # disconnected default
                if not nx.has_path(self.graph,i,j): continue#
                path_len = nx.shortest_path_length(self.graph, i, j,weight=weight_func)
                if path_len > self.max_shortestpath:
                    self.max_shortestpath = path_len
                    
        disconnected = []
        #self.pathcache = np.zeros((self.graph.size()+1, self.graph.size()+1))
        self.pathcache = np.ones((len(graph_labels), len(graph_labels))) # max dist (1) for disconnected graphs
        for i in range(len(graph_labels)):
            for j in range(len(graph_labels)):
                if i == j:
                    self.pathcache[i,j] = 1
                if not nx.has_path(self.graph,i,j):
                   self.pathcache[i,j] = disconnected_dist
                #elif i == j:
                #   self.pathcache[i,j] = 0
                elif (nx.shortest_path_length(self.graph, i, j, weight=weight_func) < 1):
                    self.pathcache[i,j] = 0
                else:
                    self.pathcache[i,j] = (nx.shortest_path_length(self.graph, i, j, weight=weight_func))/self.max_shortestpath
        self.pathcache=(self.pathcache -np.min(self.pathcache)) / (np.max(self.pathcache)-np.min(self.pathcache)) # ensure 0,1
        for a in avoid_self:
            self.pathcache[self.labeldict[a],self.labeldict[a]] = min(self_dist, 1) # set self_dist to self_dist or 1
        #if avoid_self: 
        #    for d in range(len(graph_labels)): self.pathcache[d,d] = min(self_dist, disconnected_dist) # set self_dist to disconnected_dist or 1         
    
    def show_graph_info(self,seed=-1):
        cnt = 0; glabels={}
        for label in self.labeldict:
            glabels[cnt] = str(cnt)
            cnt+=1
        if seed == -1: pos=nx.spring_layout(self.graph) # default use random
        else: pos=nx.spring_layout(self.graph,seed=seed)
        nx.draw_networkx_edge_labels(self.graph,pos=pos,edge_labels=self.edge_weights)
        nx.draw(self.graph, pos=pos,labels = glabels)
        
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
                    # remove verts by z-value mean (aggressive
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
        #return 0, shapes, dists
        # for each cluster, get the shape overlap of closest cluster by centroid. Take the mean of all of that to get the score
        return np.array([overlaps[i][dists[i].index(min(dists[i]))] for i in range(len(overlaps))]).mean(), shapes
        #return np.array(overlaps).mean(),shapes # Ideal is 1/#labels?
        #return np.median(np.array(overlaps),axis=1).mean(), shapes # Ideal is 0.5?
        #nonzeros = [np.nonzero(t)[0] for t in np.array(overlaps)]
        #for n in range(len(nonzeros)):
        #    if(len(nonzeros[n]) == 0):
        #        nonzeros[n] = np.append(nonzeros[n],0)
        #return np.nanmean(np.array([np.array(overlaps)[i][r].min() for i,r in enumerate(nonzeros)])), shapes
        #return np.nanmin(np.array(overlaps), axis=1).mean(), shapes
        
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
    
    def get_pairwise_matrix(self, X, ylabels, factor, ylabel_probs=[], transpose=False, normalize=True, squared=True, metric='euclidean',n_jobs=1):
        if transpose:
            X = X.transpose()
        
        if len(ylabel_probs) == len(ylabels):
            self.label_probs = ylabel_probs
        
        if(len(ylabels) != X.shape[1]):
            print('%d labels provided for %d columns/samples.' %(len(ylabels),X.shape[1]))
            print('Please provide the input matrix with axis=1 as the labeled samples, or specify transpose arg.')
            sys.exit("Columns do not match labels.")

        self.labels = ylabels;
        
        #print('X.shape',X.shape);
        #print('len(ylabels)', len(ylabels));

        # Compute euclidean distance based on the axis with labels (1):
        #if(len(ylabels) > X.shape[0]):
        #    print('transpose')
        X = X.transpose()
        #Considers the rows of X (and Y=X) as vectors, & computes the distance between each pair of vectors.
        distances = pairwise_distances(X, metric=metric, squared=True, n_jobs=n_jobs)
        #if(len(ylabels) > X.shape[1]):
        #print('transpose return')
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
        dists = np.zeros((shape, shape))

        def get_ranges(size, mpi):
            arr = []
            llength = int(size/mpi)
            r = int(size%mpi)-1
            for i in range(mpi):
                if i == mpi-1: arr.append(list(range(1+i*llength,r+llength+1+i*llength)));
                else: arr.append(list(range(1+i*llength,llength+1+i*llength)));
            return arr

        if(len(self.label_probs) > 0):
            for i in range(shape):     # loops over rows of `x`
                for j in range(shape): # loops over rows of `y`
                    ijp = min(self.label_probs[i],self.label_probs[j])
                    dists[i, j] = (1-(factor*ijp))+self.path_dist(i,j)*(factor*ijp) 
            dists = dists
        else:
            if(n_jobs == 1):
                print('single thread...')
                for i in range(1,shape):
                    for j in range(i):
                        dists[i,j] = (1-factor)+self.pathcache[self.labeldict[self.labels[i]],self.labeldict[self.labels[j]]]*factor
                        dists[j,i] = dists[i,j]
                dists = dists
            elif(n_jobs > 1):
                print('multi thread...')
                self.factor = factor
                with mp.Pool(n_jobs) as p:
                    dists = np.array(p.map(self.path_multi, range(shape)))
        return dists
