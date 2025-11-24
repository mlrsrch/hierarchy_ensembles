import numpy as np
from utils import loadOpenMLdata, class_labels_sanity_check, plot_dendrogram, get_score, parse_higec_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier,                              
                              ExtraTreesClassifier)
from xgboost import XGBClassifier  # pip install xgboost
from lightgbm import LGBMClassifier # pip install lightgbm
from time import time
from HG import HiGen
from HE import hier_binary_tree
import copy

import warnings
warnings.filterwarnings("ignore")

PLOT_HIERARCHY=True
DISPLAY_LINKAGE_TABLE=True

# -----------------------------
# CHOOSE DATASET
# -----------------------------
DID = 46264       # OpenML Dataset ID
DNAME = ''        # Dataset name (optional)

# -----------------------------
# HiGEC PARAMETERS
# -----------------------------
HiGEC = 'CCM[HAC|COMPLETE]-LCPN[ETC]+F[XGB]'
DISS_TYPE, BUILD_TYPE, BUILD_FUN, HE_TYPE, CLF_NAME_BASE, CLF_NAME_PF = parse_higec_string(HiGEC)

# -----------------------------
# ASSIGN CLASSIFIERS
# -----------------------------
CLF_NAME_FC = 'RF'
CLF_NAME_CBD = CLF_NAME_BASE

# -----------------------------
# EVALUATION PARAMETERS
# -----------------------------
TEST_SIZE = 0.2         # Train/Test split ratio
EVAL_METRIC = 'f1'      # Evaluation metric
RSEED = 0               # Random seed

# -----------------------------
# "OUT OF BOX" CLASSIFIERS
# -----------------------------
CLFs={
    'RF'  : RandomForestClassifier(),
    'XGB' : XGBClassifier(),
    'ETC' : ExtraTreesClassifier(),
    'LGB' : LGBMClassifier(verbose=-1),
}

# -----------------------------
# LOAD DATA
# -----------------------------
(X_num, X_cat), y = loadOpenMLdata(
    dset_id=DID,
    dset_name=DNAME,
    verbose=0
)
X = np.c_[X_num, X_cat]     # Combine numeric and categorical features
y = np.array(y)

# Split into training and testing sets
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RSEED, stratify=y)

# Sanity check for class labels
classes = class_labels_sanity_check(y_tr, y_te)
n_class = len(classes)

# -----------------------------
# FLAT CLASSIFICATION (FC)
# -----------------------------
clf_fc=copy.deepcopy(CLFs[CLF_NAME_FC])

t0 = time()
clf_fc.fit(x_tr, y_tr)
dur_fc_fit = time() - t0

t0 = time()
y_pred = clf_fc.predict(x_te)
y_pred_proba_fc = clf_fc.predict_proba(x_te)
dur_fc_pred = time() - t0

dur_fc = dur_fc_fit + dur_fc_pred

score_fc = get_score(y_te, y_pred=y_pred, pred_proba=y_pred_proba_fc, eval_metric=EVAL_METRIC)

# -----------------------------
# HIERARCHY GENERATION (HG)
# -----------------------------
t0 = time()
model_hg = HiGen(
    X, y,
    dissimilarity_type=DISS_TYPE,
    dissimilarity_output_type='diss_mat',
    metric_cc='euclidean',
    precomputed_pred=False,
    y_pred_proba=None,
    y_pred=None,
    conf_mat=None,
    clf_cbd=copy.deepcopy(CLFs[CLF_NAME_CBD]),
    cbd_val_size=0.25,
    build_type=BUILD_TYPE,
    dist_hac=BUILD_FUN,
    split_fun=BUILD_FUN
)
dur_hg = time() - t0

Z, PNs = model_hg.build_hierarchy()

if PLOT_HIERARCHY:
    plot_dendrogram(Z, close_all=True)

# -----------------------------
# HIERARCHY EXPLOITATION (HE)
# -----------------------------
tree = hier_binary_tree(
    pnodes=PNs,
    y_train=y_tr,
    y_test=y_te,
    link_mat=Z,
    pred_proba_fc=y_pred_proba_fc  # for HE+F variants, pass FC predictions
)

if DISPLAY_LINKAGE_TABLE:
    # Display Extended Linkage Table
    tree.display_extended_linkage(he_type=HE_TYPE)

# Train HE
t0 = time()
tree.fit(
    copy.deepcopy(CLFs[CLF_NAME_BASE]), 
    x_tr, 
    he_type=HE_TYPE, 
    multi_process=True
)
dur_he_fit = time() - t0

# Predict with HE
t0 = time()
tree.predict_proba(x_te, he_type=HE_TYPE)
dur_he_pred = time() - t0

dur_higec = dur_hg + dur_he_fit + dur_he_pred

# Evaluate HC
score_hc = tree.score(eval_metric=EVAL_METRIC)

# -----------------------------
# FINAL RESULTS
# -----------------------------
print('\nPerformance Comparison:')
print(f'- Flat Classification ({CLF_NAME_FC}) ({EVAL_METRIC}): {score_fc:.4f} in {dur_fc:.4f} seconds')
print(f'- HiGEC: {HiGEC} ({EVAL_METRIC}): {score_hc:.4f} in {dur_higec:.4f} seconds')
