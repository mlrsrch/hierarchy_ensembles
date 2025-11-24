# built-in libraries
import numpy as np
import pandas as pd
import math
from time import time
import re
from itertools import islice
import matplotlib.pyplot as plt
from scipy import stats
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import (TSNE,
                              LocallyLinearEmbedding as lle,
                              Isomap,)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             roc_auc_score,
                             log_loss,)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.spatial import distance

# libraries to be installed
import prince
import openml

def loadOpenMLdata(dset_id=None,
                   dset_name=None,
                   clean_nan=True,
                   nan_percent_thres=10,
                   min_class_size=5,
                   verbose=1):

    if dset_id == None or dset_name == None:
        if dset_name == None and dset_id == None:
            raise ValueError('Either dataset name or id should be provided.')
        if dset_id == None:
            token_=dset_name
        if dset_name == None:
            token_=int(dset_id)
    else:
        token_=int(dset_id)
            
    t_load = time()    
    dset = openml.datasets.get_dataset(token_,
                                       download_data=False,
                                       error_if_multiple=True,
                                       download_qualities=False,
                                       download_features_meta_data=False)
    dset_name = dset.name
    if verbose: print(f'{dset_name} with id [{dset_id}] is being downloaded... ', end = '')
    X_df, y_df, _, _ = dset.get_data(dataset_format="dataframe", target=dset.default_target_attribute)
    # X_, ys_, _, _ = dset.get_data(dataset_format="to_numpy", target=dset.default_target_attribute)
       
    dur_load = time()-t_load
    if verbose: print(f'in {dur_load:.1f} secs')
            
    X_num_cols = X_df.select_dtypes(exclude=['category']).columns
    X_sym_cols = X_df.select_dtypes(include=['category']).columns
    n_feat_num=len(X_num_cols)
    n_feat_sym=len(X_sym_cols)
    if verbose==2: print(f'Feature numbers without cleaning: numeric:{n_feat_num}, symbolic:{n_feat_sym}')
    
    # remove few populated classes
    classes_ini, class_counts_ini = np.unique(y_df, return_counts=True)
    if verbose==2: print(f'Class counts without cleaning:{class_counts_ini}')
    classes_to_remove=classes_ini[class_counts_ini<min_class_size]
    for c_ in classes_to_remove:
        X_df.drop(y_df[y_df==c_].index, inplace=True)
        y_df.drop(y_df[y_df==c_].index, inplace=True)
    
    if clean_nan:
        n_sample_ini, n_feat_ini = X_df.shape
        # drop columns (features) if they have nan percentage above threshold
        perc_=X_df.isna().sum()/n_feat_ini*100>nan_percent_thres
        X_df=X_df.drop(X_df.columns[perc_],axis=1)
        
        # drop rows (samples) if they include nan
        Xy_df = pd.concat([X_df,y_df],axis=1)
        Xy_df_dropped = Xy_df.dropna()
        
        X_df = Xy_df_dropped[Xy_df_dropped.columns[:-1]]
        y_df = Xy_df_dropped[Xy_df_dropped.columns[-1]]
        
        X_num_cols = X_df.select_dtypes(exclude=['category']).columns
        X_sym_cols = X_df.select_dtypes(include=['category']).columns
        n_feat_num=len(X_num_cols)
        n_feat_sym=len(X_sym_cols)
        if verbose==2: print(f'Feature numbers after cleaning nans: numeric:{n_feat_num}, symbolic:{n_feat_sym}')
    
    # encode categorical features
    for col_ in X_sym_cols:
        slice_=X_df[col_]
        col_encoded=slice_.cat.codes
        col_encoded[col_encoded==-1]=np.nan
        
        X_df_copy=X_df.copy()
        X_df_copy[col_]=col_encoded
        X_df=X_df_copy
    
    # separate numerical and categorical features
    X_num_df = X_df[X_num_cols]
    X_sym_df = X_df[X_sym_cols]
    X_num = np.array(X_num_df)
    X_sym = np.array(X_sym_df)
    
    # encode class labels
    le = LabelEncoder()
    y_ = pd.Series(le.fit_transform(y_df),name='class')    
    if verbose==2:
        classes, class_counts = np.unique(y_, return_counts=True)
        print(f'Class counts after cleaning:{class_counts}')
    
    return (X_num,X_sym), y_


def sort_class_id(y):
    classes=np.unique(y)
    Nclass = len(classes)
    # Standard order of class ids: 0,1,..,N-1
    if np.any(classes!=np.arange(Nclass)):
        for i in range(Nclass):
            y[y==classes[i]]=i         
        # classes=np.unique(y_train)
    return y

def encode_super_labels(super_classes, y, x=None, reorder=True):
    sel_inds = np.array([],dtype=int)
    y_sel = y.copy()
    for i,g in enumerate(super_classes):
        # find where indices for selected memberships occur
        if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
            for j in g:
                loc_ = np.where(y==j)[0].astype(int)
                sel_inds = np.r_[sel_inds,loc_]
                if reorder:
                    y_sel[y==j]=i
                else:
                    y_sel[y==j]=g[0]
        else:
            loc_ = np.where(y==g)[0].astype(int)
            sel_inds = np.r_[sel_inds,loc_]
            if reorder:
                y_sel[y==g]=i
            else:
                y_sel[y==g]=g
    y_sel = y_sel[sel_inds]
    if x is None:
        return y_sel
    else:        
        x_sel = x[sel_inds,:].copy()
        return y_sel, x_sel

def get_indices_for_selected_groups(super_classes, y_train, y_test):
    # example:super_classes=[0,[10,13],8]
    sel_inds_train=np.array([],dtype=int)
    sel_inds_test=np.array([],dtype=int)
    
    y_train_new = y_train.copy()
    y_test_new = y_test.copy()
    for i,g in enumerate(super_classes):
        # find where indices for selected memberships occur
        if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
            for j in g:
                loc_tr = np.where(y_train==j)[0].astype(int)
                loc_te = np.where(y_test==j)[0].astype(int)
                sel_inds_train= np.r_[sel_inds_train,loc_tr]
                sel_inds_test = np.r_[sel_inds_test,loc_te]
                
                y_train_new[y_train==j]=i
                y_test_new[y_test==j]=i
        else:
            loc_tr = np.where(y_train==g)[0].astype(int)
            loc_te = np.where(y_test==g)[0].astype(int)
            sel_inds_train= np.r_[sel_inds_train,loc_tr]
            sel_inds_test = np.r_[sel_inds_test,loc_te]
            
            y_train_new[y_train==g]=i
            y_test_new[y_test==g]=i
            
    y_train_new= y_train_new[sel_inds_train]
    y_test_new = y_test_new[sel_inds_test]
    return sel_inds_train, sel_inds_test, y_train_new, y_test_new


def class_labels_sanity_check(y_train,y_test):
    classes_train = np.unique(y_train)
    classes_test = np.unique(y_test)
    if np.all(classes_train==classes_test):
        classes=classes_train
    else:
        raise ValueError("Discrepancy bw train and test labels. Class labels don't match.")
    return classes
    
def plotData(data,labels, close_all=True):
    if close_all:plt.close('all')
    classes = np.unique(labels)
    plt.figure()
    for c in classes:
        c_x_train = data[labels == c]
        plt.plot(c_x_train[0], label="class " + str(c))
    plt.legend(loc="best")

   
def monotonize_rescale_(Z,swap=False):
    # Monotonize and rescale the tree linkage
    Zh = Z[:,2]
    n = len(Zh)
    
    start_ = Zh.max()/(n)
    stop_ = Zh.max()
    Zh[:] = np.round(np.linspace(start_,stop_,n),2)
    
    if Zh.ptp()/(n-1) < 0.05:
        scale_=0.05*(n-1)/Zh.ptp()
        Zh *= scale_
    
    # Swap column 0 and 1 for compatiblity with dendogram
    if swap:
        z0=Z[:,0].copy()
        Z[:,0]=Z[:,1]
        Z[:,1]=z0
    
    return Z

def nCk(n,k):
    f = math.factorial
    return f(n) // f(k) // f(n-k)

def batched(iterable, n):
      "Batch data into lists of length n. The last batch may be shorter."
      # batched('ABCDEFG', 3) --> ABC DEF G
      if n < 1:
          raise ValueError('n must be >= 1')
      it = iter(iterable)
      while (batch := list(islice(it, n))):
          yield batch

def C_n(n):
    """All divisions of an n-element cluster into two non-empty 
       subsets: 2**(n-1)-1
    """
    sum_=0
    for k in range(1,round(n/2)+1):
        if n%2==0 and k==n/2:
            sum_ += int(nCk(n,k)/2)
        else:
            sum_ += nCk(n,k)
    print(sum_)
    return sum_

def T_n(n):
    "Estimate total number of trees given number of classes"    
    if n==2:        
        return 1
    elif n==3:
        return 3
    elif n>3:
        sum_=0
        for i in range(1,round(n/2)+1):
            if n%2==0 and i==n/2:
                sum_ += int(nCk(n,n-i)/2)*T_n(n-i)
            else:
                sum_ += nCk(n,n-i)*T_n(n-i)
        return sum_

def T_n_look(n):
    "Estimate total number of trees given number of classes"
    table_=[0,0]
    if n==2:
        table_.append(1)
        return table_
    elif n==3:
        table_.append(1)
        table_.append(3)
        return table_
    elif n>3:
        table_.append(1)
        table_.append(3)
        for n_i in range(4,n+1):
            sum_=0           
            for k in range(1,round(n_i/2)+1):
                if n_i%2==0 and k==n_i/2:
                    sum_ += int(nCk(n_i,n_i-k)/2)*table_[n_i-k]
                else:
                    sum_ += nCk(n_i,n_i-k)*table_[n_i-k]
            table_.append(sum_)
        return table_
    
def compare_(root1, root2, sol):
    if root1 is not None and root2 is not None:
        if root1.subsets[0]==root2.subsets[0]:
            sol.append(1)
            compare_(root1.left, root2.left, sol)
            compare_(root1.right, root2.right, sol)
        elif root1.subsets[0]==root2.subsets[1]:
            sol.append(2)
            compare_(root1.left, root2.right, sol)
            compare_(root1.right, root2.left, sol)
        else:            
            sol.append(0)
    else:
        sol.append(-1)

def isEqual(root1, root2):
    eq_=[]
    compare_(root1,root2,eq_)
    if 0 in eq_:
        return False
    else:
        return True

def compare_tree(n_i,Yi,n_j,Yj):
    if np.all(Yi==Yj):
        return True
    elif isEqual(n_i[0],n_j[0]):
        return True
    else:
        return False

def plot_dendrogram(Z, close_all=0, orient="top", leafFont=9, title_=False, class_list=None):
    if close_all:plt.close('all')
    if title_: title_text= "Hierarchical Clustering Dendrogram"
    if class_list is not None:
        sch.dendrogram(Z,orientation=orient,leaf_font_size=leafFont,labels=[txt for txt in class_list])
    else:
        sch.dendrogram(Z,orientation=orient,leaf_font_size=leafFont)
    if title_: plt.title(title_text)

def select_initial_clusters(x,y):
    indexes = np.unique(y, return_index=True)[1]
    classes = [y[index] for index in sorted(indexes)]
    init_=np.array([x[y==c_][0] for c_ in classes])
    return init_

def get_significance(scores_hc,scores_fc,les_=None):
    le_=np.mean(scores_hc)/np.mean(scores_fc)
    if scores_fc==scores_hc:
        sig_=np.nan
    else:
        t_stat, p_val = stats.wilcoxon(scores_hc, scores_fc)
        if p_val<0.05:
            sig_=np.sign(le_-1)
        else:
            sig_=np.nan
    if les_ is not None: les_.append(le_)
    return le_,p_val,sig_

def get_score(y_true, y_pred=None, pred_proba=None, eval_metric='f1'):
    if eval_metric=='acc':
        score_ = accuracy_score(y_true, y_pred)
    elif eval_metric=='bac':        
        score_ = balanced_accuracy_score(y_true, y_pred)
    elif eval_metric=='f1':
        score_ = f1_score(y_true, y_pred, average='macro')
    elif eval_metric=='auc':
        if pred_proba.shape[-1]>2:
            score_ = roc_auc_score(y_true, pred_proba, average='weighted', multi_class='ovr')
        else:
            score_ = roc_auc_score(y_true, pred_proba[:,1])
    elif eval_metric=='nll':
        score_ = log_loss(y_true, pred_proba)
    return score_

def reduction_model(data, ndim=0, n_neighbor=5, redu_meth='lle', rseed=None, verbose=False):
    if data is tuple and len(data)==2:
        X, y = data
    else:
        X = data
    n_object, dim_prior = X.shape     
        
    if n_neighbor==0:     
        n_neighbor = min(n_object-1, 4+int(n_object/136))
        if verbose: print(f'n_neighbor:{n_neighbor}')
    
    if redu_meth=='lle':
        try:
            model = lle(n_neighbors=n_neighbor, 
                        n_components=ndim, 
                        random_state=rseed,
                        n_jobs=-1)
            X_reduced = model.fit_transform(X)
        except:
            model = lle(n_neighbors=n_neighbor, 
                        n_components=ndim, 
                        random_state=rseed,
                        eigen_solver='dense',
                        n_jobs=-1)
            X_reduced = model.fit_transform(X)
                
    elif redu_meth=='isomap':
        try:            
            model = Isomap(n_neighbors=n_neighbor,
                           n_components=ndim,
                           n_jobs=-1)
            X_reduced = model.fit_transform(X)
        except:
            model = Isomap(n_neighbors=n_neighbor,
                           n_components=ndim,
                           eigen_solver='dense',
                           n_jobs=-1)
            X_reduced = model.fit_transform(X)
    
    elif redu_meth=='tsne':
        try:
            model = TSNE(n_components=ndim,
                         random_state=rseed,
                         n_jobs=-1)
            X_reduced = model.fit_transform(X)
        except:
            model = TSNE(n_components=ndim,
                         perplexity = data.shape[0]-5,
                         random_state=rseed,
                         n_jobs=-1)
            X_reduced = model.fit_transform(X)
        
    return X_reduced

"""
Automatically setting number of dimesions after reduction
ndim= 0: set using PCA
ndim=-1: set using LDA
ndim=-2: set using MCA
"""
def dim_reduce(data,**kwargs):
    if type(data) is tuple and len(data)==2:
        X, y = data        
    else:
        X = data
    n_sample, n_feat = X.shape
    
    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose = 0
        
    # Scaling Option
    if 'scale_' in kwargs.keys():
        if kwargs['scale_']=='min_max':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif kwargs['scale_']=='std':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
    
    if 'ndim' in kwargs.keys():
        ndim = kwargs['ndim']
    else:
        ndim = 0
    if ndim in [0,-1,-2]:        
        if ndim==-2:
            # Multiple correspondence analysis
            mca_ = prince.MCA(n_components=n_feat)
            Xdf=pd.DataFrame(data=X)
            mca_.fit(Xdf)
            cum_sum_=mca_.cumulative_percentage_of_variance_
        else:    
            if ndim==-1: # supervised dim_reduce
                mdl=LinearDiscriminantAnalysis()
                mdl.fit(X,y)
            elif ndim==0:
                mdl=PCA() # unsupervised dim_reduce
                mdl.fit(X)
            sum_=100*mdl.explained_variance_ratio_
            cum_sum_=np.cumsum(sum_)
        if verbose: print(f'Cumulative percentage of variance:{cum_sum_}')
        # automatically choose ndim
        if 'cum_var_thresh' in kwargs.keys(): 
            var_thresh=kwargs['cum_var_thresh']
        else:
            var_thresh=95
        ndim=len(cum_sum_[cum_sum_<var_thresh])
        if verbose: print(f'{ndim} out of {n_feat} components selected.')
        if ndim==n_feat:
            if verbose==2: print('No need to reduce dimensionality.')
            return X
                       
    if 'model' in kwargs.keys():
        model = kwargs['model']
    else:
        model = 'pca'
    if model in ['lda','nca']:
        models = {'lda': LinearDiscriminantAnalysis(n_components=ndim),
                     'nca': NeighborhoodComponentsAnalysis(n_components=ndim)}
        models[model].fit(X,y)
        Xr = models[model].transform(X)        
    elif model=='pca':
        pca_ = PCA(n_components=ndim)
        Xr = pca_.fit_transform(X)    
    elif model == 'mca':
        mca_ = prince.MCA(n_components=ndim)
        Xdf=pd.DataFrame(data=X)
        Xr = np.array(mca_.fit_transform(Xdf))
    else:
        # print('dim reduction with',model)
        nn = min(n_sample-1, 4+int(n_sample/136))        
        # print('nearest_neighbors:',nn)
        Xr = reduction_model(X, ndim=ndim, n_neighbor=nn, redu_meth=model)
        
    return Xr

def preprocess_diss_mat(D,scale_=True):
    D[np.isnan(D)]=np.nanmax(D)
    m=D.shape[0]
    inds_i=[i for i in range(m) for j in range(m) if i!=j]
    inds_j=[j for i in range(m) for j in range(m) if i!=j]
    if len(D[inds_i,inds_j][D[inds_i,inds_j]==0])>0:
        D[inds_i,inds_j]=D[inds_i,inds_j]+(D[inds_i,inds_j][D[inds_i,inds_j]!=0]).min()
    if scale_: D = (D - np.min(D)) / (np.max(D) - np.min(D))
    return D

def jensen_shannon_dist(x1,x2):
    pdf1, bin_edges = np.histogram(x1, bins=50, density=True)
    pdfn1 = pdf1/pdf1.sum()
    
    pdf2, bin_edges = np.histogram(x2, bins=50, density=True)
    pdfn2 = pdf2/pdf2.sum()
    
    return distance.jensenshannon(pdfn1, pdfn2, 2)
    
def parse_higec_string(s):
    """
    Parse a HiGEC scheme string into its components.
    Always returns: (diss_type, build_type, build_fun, he_type, clf_base, clf_pf)
    clf_pf is None if not present.
    """
    results = []

    # First part (diss_type, build_type, build_fun)
    first_match = re.match(r'^([A-Z0-9]+)\[([A-Z0-9\|]+)\]', s)
    if not first_match:
        raise ValueError(f"Invalid HiGEC string format: {s}")
    results.append(first_match.group(1))                 # diss_type
    results.extend(first_match.group(2).split('|'))      # build_type, build_fun

    # HE type and classifier base
    he_match = re.search(r'-([A-Z0-9]+)\[([A-Z0-9\|]+)\](\+F)?', s)
    if not he_match:
        raise ValueError(f"HE section not found in HiGEC string: {s}")
    he_base = he_match.group(1)
    he_suffix = he_match.group(3) or '+'                  # +F or +
    he_type = he_base + he_suffix
    results.append(he_type)
    results.extend(he_match.group(2).split('|'))         # clf_base

    # Optional +F classifier post-filter
    f_match = re.search(r'\+F\[([A-Z0-9\|]+)\]', s)
    if f_match:
        results.append(f_match.group(1))                  # clf_pf
    else:
        results.append(None)                              # clf_pf missing

    # Sanity check
    if len(results) != 6:
        raise ValueError(f"Unexpected parsed result length for HiGEC string: {results}")

    return tuple(results)