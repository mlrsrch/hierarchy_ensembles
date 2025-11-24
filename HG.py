"""
Automated class hierarchy generation has two phases:
    - Class dissimilarity assessment
    - Hierarchy build
    
Class dissimilarity parameters:
    - dissimilarity_type: Class centroid based (ccm) or classifier based (cmd or fpd)
    - dissimilarity_output_type: pairwise dissimilarity matrix or vector

Hierarchy build parameters:
    - build_type: hierarchical agglomerative clustering (hac) or 
    hierarchical divisive clustering (hdc)
    - dist_hac: (for hac) group pair distance method
    - split_fun: (for hdc) group pair distance method

"""

import numpy as np
from time import time
from itertools import combinations
import scipy.cluster.hierarchy as sch
from sklearn.metrics import DistanceMetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from diss_mat_embedding import spectral_embedding
from utils import monotonize_rescale_, jensen_shannon_dist, preprocess_diss_mat
from hdc import linkHDC

class HiGen:
    """Automated Class Hierarchy Generation"""
    def __init__(self,
                 X,
                 y,
                 dissimilarity_type='ccm',
                 dissimilarity_output_type='diss_mat', # or obs_vec
                 metric_cc='euclidean',
                 
                 precomputed_pred=False,
                 y_pred_proba=None,
                 y_pred=None,
                 conf_mat=None,
                 clf_cbd=None,
                 cbd_val_size=0.25,
                 
                 build_type='hac',
                 dist_hac = 'complete', # group pair distance method
                 split_fun = 'kmeans', # for HDC
                 ):
        self.X=X
        self.y=y
        self.dissimilarity_type=dissimilarity_type
        self.dissimilarity_output_type=dissimilarity_output_type
        self.metric_cc=metric_cc
        
        self.precomputed_pred=precomputed_pred
        self.y_pred_proba=y_pred_proba
        self.y_pred=y_pred
        self.conf_mat=conf_mat
        self.clf_cbd=clf_cbd
        self.cbd_val_size=cbd_val_size
                
        self.build_type=build_type
        self.dist_hac=dist_hac.lower()
        self.split_fun=split_fun.lower()
        
    def obtain_dissimilarity(self):
        diss_type=self.dissimilarity_type
        out_type=self.dissimilarity_output_type
        
        if diss_type.lower() in ['ccm']:
            cc = self.get_cc()
            if out_type.lower() in ['diss_mat']:
                distf = DistanceMetric.get_metric(self.metric_cc)
                mat = distf.pairwise(cc)
                return mat
            else:
                return cc
        else:
            if diss_type.lower() in ['cmd','fpd']:
                mat_ = self.get_cbd()
            elif diss_type.lower() in ['jsd']:
                mat_ = self.get_jsd()
            elif diss_type.lower() in ['tsd']:
                mat_ = self.get_tsd()
            else:
                raise ValueError(f'{diss_type} is not recognized as a dissimilarity type')
            
            mat = preprocess_diss_mat(mat_)
            if out_type.lower()=='obs_vec':
                vec,_ = spectral_embedding(mat_,dim=0,sigma=0.85,dmeth='pca')
                return vec
            else:
                return mat
    
    def get_cc(self): 
        classes = np.unique(self.y)
        # class conditional means
        cc=np.array([self.X[self.y == i].mean(axis=0) for i in classes])
        return cc    
    
    # Classifier Based Dissimilarity (CBD)
    def get_cbd(self):
        
        if not self.precomputed_pred or (self.dissimilarity_type == 'cmd' and self.y_pred is None):
            x_tr, x_va, y_tr, y_va = train_test_split(self.X,
                                                      self.y,
                                                      test_size=self.cbd_val_size,
                                                      random_state=0,
                                                      stratify=self.y)
                
        if self.precomputed_pred:
            if np.any(np.isnan(self.y_pred_proba)):
                raise ValueError('Prediction probabilities need to have numerical values. They have Nan.')
            if self.dissimilarity_type in ['fpd'] and self.y_pred_proba is None:
                raise ValueError('Prediction probabilities have to be provided.')
            if self.dissimilarity_type in ['cmd'] and self.y_pred_proba is None and self.y_pred is None:
                raise ValueError('Either prediction probabilities or predicted labels have to be provided.')
        else:
            clf=self.clf_cbd       
            clf.fit(x_tr,y_tr)
            
            self.y_pred = clf.predict(x_va)
            self.y_pred_proba = clf.predict_proba(x_va)
            if self.dissimilarity_type == 'cmd':
                conf_mat = confusion_matrix(y_va, self.y_pred)
            
        classes, class_counts = np.unique(self.y, return_counts=True)
        n_class = len(classes)
        
        diss_mat = np.zeros((n_class,n_class))
         
        """
        Confusion Matrix-based Dissimilarity – CMD
        see Silva-Palacios et al. 2017 "Improving Performance of Multiclass Classification by Inducing Class Hierarchies"
        """
        if self.dissimilarity_type == 'cmd':
            if self.conf_mat is None:
                if self.y_pred is None:
                    self.y_pred = classes[np.argmax(self.y_pred_proba,axis=1)]
                conf_mat = confusion_matrix(y_va, self.y_pred)
                        
            for r_,row_ in enumerate(conf_mat):
                for c_,col_ in enumerate(conf_mat):
                    if r_ != c_:
                        diss_mat[r_,c_]=(conf_mat[r_,r_]+conf_mat[c_,c_])/(conf_mat[r_,c_]+conf_mat[c_,r_]+conf_mat[r_,r_]+conf_mat[c_,c_])
        
        
        """
        Fractionated Probability-based Dissimilarity – FPD
        see del Moral et al. 2023 "Pitfalls of Assessing Extracted Hierarchies for Multi-class Classification"
        """
        if self.dissimilarity_type == 'fpd':
            for j_,cj in enumerate(classes):
                for k_,ck in enumerate(classes[j_+1:],j_+1):
                    
                    # binary classifier probability output carved from multiclass classifier probability output 
                    hprime = self.y_pred_proba[:,[j_,k_]]
                    
                    diss_mat[j_,k_]=np.sum(np.max(hprime,axis=1))/np.sum(hprime)
                    diss_mat[k_,j_]=diss_mat[j_,k_]
                    
        return diss_mat
        
    def get_jsd(self, verbose=False):
        classes = np.unique(self.y)
        data_by_class = [self.X[self.y==c] for c in classes]
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
        if verbose: print(f'time for distance matrix generation using JSD {dur_:.2f} secs')
        return dist_mat
    
    def get_tsd(self, verbose=False):
        from tsd import get_TSD
        t0 = time()
        dist_mat = get_TSD(self.X, self.y)
        dur_=time()-t0
        if verbose: print(f'time for distance matrix generation using TSD {dur_:.2f} secs')
        return dist_mat
    
    def build_hierarchy(self):
        input_build = self.obtain_dissimilarity()
        if self.build_type.lower() in ['hac']:
            Z0= sch.linkage(input_build,
                            method=self.dist_hac,)
            PNs = None
        elif self.build_type.lower() in ['hdc']:
            model_link = linkHDC(y=input_build,
                                 input_type= self.dissimilarity_output_type,
                                 split_fun = self.split_fun)
            Z0, PNs = model_link.fit()
            
        Z = monotonize_rescale_(Z0)
        return Z, PNs
    
if __name__=='__main__':
    from utils import loadOpenMLdata, class_labels_sanity_check, plot_dendrogram
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier # pip install xgboost
    from time import time
    import copy
        
    did_= 41
    dname_= 'Glass'
    test_size = 0.2
    
    hc_type = 'lcl+'
    rseed = 0
    
    # load dataset
    (X_num,X_cat),y = loadOpenMLdata(dset_id=did_,
                                     dset_name=dname_,
                                     verbose=0)
    X = np.c_[X_num,X_cat]
    y = np.array(y)
    
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rseed, stratify=y)
    
    classes = class_labels_sanity_check(y_tr,y_te)
    n_class = len(classes)
    
    clf_cbd=copy.deepcopy(XGBClassifier())
    
    model_HG = HiGen(X,
                     y,
                     dissimilarity_type='tsd',
                     dissimilarity_output_type='diss_mat', 
                     metric_cc='euclidean',
                     
                     precomputed_pred=False,
                     y_pred_proba=None,
                     y_pred=None,
                     conf_mat=None,
                     clf_cbd=clf_cbd,
                     cbd_val_size=0.25,
                      
                     build_type='hdc',
                     dist_hac = 'complete', 
                     split_fun= 'kmed', 
                     )
    
    Z, PNs = model_HG.build_hierarchy()
    plot_dendrogram(Z,close_all=1)
            
        
        