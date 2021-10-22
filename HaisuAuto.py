import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(".") # To include Haisu module from local dir
from Haisu import HAISU 
import random

def _getHaisuScore(tsne, haisu, strength):
    print('trying strength ' + str(strength))
    X = haisu.get_pairwise_matrix(haisu._X, haisu._ylabels, strength, haisu._probs, haisu._transpose, haisu._normalize, haisu._metric, haisu._n_jobs)
    X_embedded = tsne.fit_transform(X)
    # Calculate Mean of Max convex overlaps:
    score, shapes =  haisu.get_overlap_score(X_embedded, haisu._ylabels)
    return score, shapes, X_embedded

def HaisuAuto(tsne, haisu, iterations):
    if(iterations <= 1):
        print('Too few iterations for autorunner')
        return;
    i = 1.0 - 1.0/(iterations+1)
    j = 1.0; cnt =1
    score, shapes, X = _getHaisuScore(tsne, haisu, i)
    print('score: ' + str(score))
    while score < 0.5 and cnt < iterations:
        i -= 1.0/(iterations+1)
        score, shapes, X = _getHaisuScore(tsne, haisu, i)
        print('score: ' + str(score))
        if(score >= 0.4 and score <= 0.6): return score, shapes, X
        cnt+=1
    
    return score,shapes, X