# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.metrics import confusion_matrix


def gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    return np.sqrt(recall * specificity)