import numpy as np
from scipy.spatial import distance
from itertools import combinations
from time import time

def jensen_shannon_dist(x1,x2):
    pdf1, bin_edges = np.histogram(x1, bins=50, density=True)
    pdfn1 = pdf1/pdf1.sum()
    
    pdf2, bin_edges = np.histogram(x2, bins=50, density=True)
    pdfn2 = pdf2/pdf2.sum()
    
    return distance.jensenshannon(pdfn1, pdfn2, 2)

def get_jsd(X,y,verbose=False):
    classes = np.unique(y)
    data_by_class = [X[y==c] for c in classes]
    n_classes=len(classes)
    
    # Get all combinations of 2-classes
    dist_mat = np.zeros((n_classes,n_classes))
    t0 = time()
    
    comb=combinations(range(n_classes),2)
    for i,j in comb:       
        x_i=data_by_class[i]
        x_j=data_by_class[j]
        
        dist_=jensen_shannon_dist(x_i,x_j)             
                    
        dist_mat[i,j] = np.round(dist_/100,4)
        dist_mat[j,i] = dist_mat[i,j]
        
    dur_=time()-t0    
    if verbose: print(f'time for distance matrix generation {dur_:.2f} secs')
    return dist_mat

if __name__=='__main__':
    from utils import prep_data
    ts_name='Beef'
    repo='ucr'
    rs = 42 
    n_te = 5
    
    dset = prep_data(ts_name,
                     repo=repo,
                     sel_class=None,
                     reorder=False,
                     orig_split=False, 
                     stratify=True, 
                     k=n_te, 
                     kth=1,
                     rs=rs,
                     shuffle_=True)
    x_tr, y_tr, x_te, y_te = dset
    X = np.r_[x_tr, x_te]
    y = np.r_[y_tr, y_te]
    classes = np.unique(y)
    n_classes = len(classes)
    n_sample = len(y)
    
    dist_matrix = generate_dist_mat(X,y)    
    
    
    