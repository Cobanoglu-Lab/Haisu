import numpy as np
import sys
import networkx as nx
from sklearn.metrics import pairwise_distances

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
        
    def _init_graph(self, graph_labels, ajmatrix):
        # Dictionary from labels:
        cnt = 0
        for label in graph_labels:
            #print(str(cnt) + " " + label);
            self.labeldict[label] = cnt; cnt+=1
            
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
        
    def get_pairwise_matrix(self, X, ylabels, factor, ylabel_probs=[], transpose=False, normalize=True, metric='euclidean',n_jobs=1):
        if transpose:
            X = X.transpose()
        
        if len(ylabel_probs) == len(ylabels):
            self.label_probs = ylabel_probs
        
        if(len(ylabels) != X.shape[1]):
            print('%d labels provided for %d columns/samples.' %(len(ylabels),X.shape[1]))
            print('Please provide the input matrix with axis=1 as the labeled samples, or specify transpose arg.')
            sys.exit("Columns do not match labels.")

        self.labels = ylabels;
        
        print('X.shape',X.shape);
        print('len(ylabels)', len(ylabels));

        # Compute euclidean distance based on the axis with labels (1):
        #if(len(ylabels) > X.shape[0]):
        #    print('transpose')
        X = X.transpose()
        #Considers the rows of X (and Y=X) as vectors, & computes the distance between each pair of vectors.
        distances = pairwise_distances(X, metric=metric, squared=True, n_jobs=n_jobs)
        #if(len(ylabels) > X.shape[1]):
        #print('transpose return')
        X = X.transpose()
        pathpairwise = self.path_pairwise(X, factor)
        np.fill_diagonal(pathpairwise, 0)
        distances = np.multiply(distances,pathpairwise)

        # Normalize distances
        if normalize:
            distances=(distances-distances.min())/(distances.max()-distances.min())
        
        return distances
    
    def path_dist(self, label1, label2):
        i = self.labeldict.get(self.labels[label1])
        j = self.labeldict.get(self.labels[label2])
        return self.pathcache[i,j]
        #return (nx.shortest_path_length(self.graph, i, j)-1)/self.max_shortestpath

    def path_pairwise(self, x, factor = 1, squared=True):
        shape = len(self.labels)
        dists = np.zeros((shape, shape))
        if(len(self.label_probs) > 0):
            for i in range(shape):     # loops over rows of `x`
                for j in range(shape): # loops over rows of `y`
                    ijp = min(self.label_probs[i],self.label_probs[j])
                    dists[i, j] = (1-(factor*ijp))+self.path_dist(i,j)*(factor*ijp) 
        else:
            for i in range(shape):     # loops over rows of `x`
                for j in range(shape): # loops over rows of `y`
                    dists[i, j] = (1-factor)+self.path_dist(i,j)*factor
        return dists