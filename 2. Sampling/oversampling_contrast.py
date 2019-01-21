# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from coverage import weighted_coverage
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.datasets import make_classification
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_val_score


# 构造数据集，并划分训练集与测试集
X, y = make_classification(n_samples=12000, n_classes=2, n_features=20, n_informative=20, class_sep=0.75,
                           n_redundant=0, n_clusters_per_class=3, weights=[0.8, 0.2], random_state=2019)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2019)

# 构建XGBoost分类器
xgb = XGBClassifier(n_estimators=350, max_depth=6, reg_lambda=1.15, learning_rate=0.15, n_jobs=-1)

# 构建过采样函数
def over_sample(X, y, oversample_mode, random_state=2018):
    if oversample_mode == 'Original':
        X_rebalanced, y_rebalanced = X, y 
    elif oversample_mode == 'ADASYN':
        ada = ADASYN(random_state=random_state)
        X_rebalanced, y_rebalanced = ada.fit_resample(X, y)
    elif oversample_mode == 'SMOTE':
        sm = SMOTE(kind='regular', random_state=random_state)
        X_rebalanced, y_rebalanced = sm.fit_resample(X, y)
    elif oversample_mode == 'Bordline_1':
        bl1 = SMOTE(kind='borderline1', random_state=random_state)
        X_rebalanced, y_rebalanced = bl1.fit_resample(X, y)
    elif oversample_mode == 'Bordline_2':
        bl2 = SMOTE(kind='borderline2', random_state=random_state)
        X_rebalanced, y_rebalanced = bl2.fit_resample(X, y) 
    elif oversample_mode == 'BOS_SVM':
        bos_svm = SMOTE(kind='svm', random_state=random_state)
        X_rebalanced, y_rebalanced = bos_svm.fit_resample(X, y)      
    return X_rebalanced, y_rebalanced

def method_peformance(mode, clf=xgb):
    X_train_new, y_train_new = over_sample(X_train, y_train, oversample_mode=mode)
    clf.fit(X_train_new, y_train_new)
    y_pred, y_prob = clf.predict(X_test), clf.predict_proba(X_test)[:,-1]
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    ap_score = average_precision_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    coverage = weighted_coverage(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (fp + tn)    
    gmean = np.sqrt(recall * specificity) 
    acc = accuracy_score(y_test, y_pred)
    return auc, f1, gmean, coverage, ap_score, recall, acc

modes = ['Original', 'ADASYN', 'SMOTE', 'Bordline_1', 'Bordline_2', 'BOS_SVM']
metrics = ['AUC', 'F1_Score', 'G_Mean', 'Coverage', 'AP_Score', 'Recall', 'ACC']
performance = np.array(list(map(method_peformance, modes)))
model_performance = pd.DataFrame(performance, index=modes, columns=metrics)
model_performance.index.name = 'OverSampling'
print(model_performance)

# 对最高的分数予以标黄，仅适用于Jupyter
def highlight_max(s):
    is_max = s == s.max() 
    bg = ['background-color: yellow' if v else '' for v in is_max]
    return bg

model_performance.style.apply(highlight_max, axis=0)
