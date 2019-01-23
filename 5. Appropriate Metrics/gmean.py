# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer


def gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    return np.sqrt(recall * specificity)

# 通过工厂函数make_scorer生成gmean_score，可直接用于网格搜索进行调参
# 形如：sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=gmean_score）
gmean_score = make_scorer(gmean, greater_is_better=True)
