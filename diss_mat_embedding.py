import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding as lle
from sklearn.manifold import Isomap
import sklearn.cluster as skc
import sklearn.metrics as sm
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy import linalg
from scipy.linalg import fractional_matrix_power as mp
from utils import dim_reduce

def run_k_means(X, n_clusters):
    """K-means clustering."""
    k_means = skc.KMeans(random_state=25, n_clusters=n_clusters)
    k_means.fit(X)
    y_pred = k_means.predict(X)
    return y_pred

def generate_graph_laplacian(x, nn):
    """Generate graph Laplacian from data."""
    # Adjacency Matrix.
    connectivity = kneighbors_graph(x, n_neighbors=nn, mode='connectivity')
    adjacency_matrix_s = (1/2)*(connectivity + connectivity.T)
    # Graph Laplacian.
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
    graph_laplacian = graph_laplacian_s.toarray()
    return graph_laplacian 

def compute_spectrum_graph_laplacian(graph_laplacian):
    """Compute eigenvalues and eigenvectors and project 
    them onto the real numbers.
    """
    eigenvals, eigenvcts = linalg.eig(graph_laplacian)
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    return eigenvals, eigenvcts

def project_and_transpose(eigenvals, eigenvcts, num_ev):
    """Select the eigenvectors corresponding to the first 
    (sorted) num_ev eigenvalues as columns in a data frame.
    """
    eigenvals_sorted_indices = np.argsort(eigenvals)
    indices = eigenvals_sorted_indices[: num_ev]

    # proj_df = pd.DataFrame(eigenvcts[:, indices.squeeze()])
    # proj_df.columns = ['v_' + str(c) for c in proj_df.columns]
    return eigenvcts[:, indices.squeeze()]

def decomp(M, dim, dim_auto=False, methd='eig'):
    m, n = M.shape  
    #SVD
    hf=False # check if Hermitian
    if np.allclose(M,np.conj(M.T)): hf=True
    fm=True # Fulll matrices?
    u,s,vh=np.linalg.svd(M,full_matrices=fm,hermitian=hf)
    
    # automatically assign n_dimension
    if dim_auto:
        s_sum = s.sum()
        sum_=0
        for d_,s_ in enumerate(s):
            sum_ += s_
            if sum_/s_sum >= 0.95:
                dim=d_+1
                print(dim,'out of',M.shape[1],' components selected')
                break
    
    if methd=='eig':     
        #Eigen decomposition
        s, U = np.linalg.eig(M)
        #M2=U@S@U.T #Check reconstruction 
        X=U@np.diag(s)[:,:dim]        
    
    elif methd=='svd':      
        X = M @ vh[:dim,:].T
    
    elif methd in ['lle','isomap','pca']:
        # if m<5:
        #     n_neighbor = m-1
        # elif m<=10:
        #     n_neighbor = m-2
        # else:
        #     n_neighbor = 12
           
        X = dim_reduce(M,
                       ndim=dim,
                       n_neighbor=0,                       
                       redu_meth=methd,
                       verbose=0)                      
    return X,s

def renorm(X):    
    #Renormalizing each of X's rows to have unit length
    Y=X.copy()
    m=X.shape[0]
    rr=np.sqrt(np.sum(X**2,axis=1))
    for i in range(m):
        Y[i,:] = Y[i,:]/rr[i]
    return Y    

def dist_to_euclidean(mat_,dim,dmeth='lle',renrm=True):
    #Torgerson, W.S. Multidimensional scaling: I. Theory and method. Psychometrika 17, 401–419 (1952).
    n = mat_.shape[0]
    # Gram matrix
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j] = .5*(mat_[0,j]**2+mat_[i,0]**2-mat_[i,j]**2)    
    X,Lambda = decomp(M,dim,methd=dmeth)
    if renrm:X=renorm(X)
    return X, Lambda

def spectral_embedding(mat_,dim=3,sigma=0.25,nrm='symm_div',dmeth='isomap',renrm=True):
    #Normalized spectral clustering from Ng (2002)
    n = mat_.shape[0]
    
    #Affinity matrix A with free scale parameter sigma
    A = np.exp(-mat_/(2*sigma**2))
    
    #Diagonal matrix D by lumping A    
    D = np.diag(A.sum(axis=1))
    
    #Normalization
    if nrm=='symm_div': #Symmetric Divisive
        L = mp(D,-.5) @ A @ mp(D,-.5)
    elif nrm=='div': #Divisive
        L = mp(D,-1) @ A
    elif nrm=='maxrow': #Maximum rowsum
        L = (A+D.max()*np.eye(n)-D)/D.max()
    
    X,Lambda = decomp(L,dim,methd=dmeth)
    if renrm:X=renorm(X)
    return X, Lambda 

def unnormalized_spectral_clustering(x, n_neighbors, dim):
    """Spectral Clustering Algorithm."""
    graph_laplacian = generate_graph_laplacian(x, n_neighbors)
    eigenvals, eigenvcts = compute_spectrum_graph_laplacian(graph_laplacian)
    proj_x = project_and_transpose(eigenvals, eigenvcts, dim)  
    
    return proj_x, eigenvals

def kmeans_score(x,y_true,eval_metric='acc'):
    kmeans = skc.KMeans(n_clusters=2, random_state=0, n_init=10).fit(x)
    if eval_metric=='acc':
        eval_score = sm.accuracy_score(y_true, kmeans.labels_)
    elif eval_metric=='jac':
        eval_score = sm.jaccard_score(y_true, kmeans.labels_, average='weighted')
    return eval_score