# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix


X, y = make_classification(n_samples=15000, n_classes=2, n_features=10, n_informative=8, 
                           weights=[0.8, 0.2], random_state=2018)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)

# 构造gmeans指标，等价于recall与specificity的几何平均
def gmean_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (fp + tn)
    return np.sqrt(recall * specificity)

# 构造样本权重生成函数，fn_cost，fp_cost分别表示FN，FP的代价
# 在风控领域，FN是指没有检测出欺诈交易，FP是指将正常交易误判为欺诈交易，因此FN的代价应大于FP的代价
# 实际应用场景中，正样本为少数，因此应赋予更大的权重，即正负样本的权重应与它们被误分引致的代价成一定的比例关系
# the weights are in proportion to their corresponding misclassification costs
def get_weight(fn_cost, fp_cost, y_train=y_train):
    weight = [fn_cost if i == 1 else fp_cost for i in y_train]
    return np.array(weight)

xgb = XGBClassifier(n_estimators=150, learning_rate=0.15)
# 构造模型评估函数，暂取AUC、F1、Recall、Gmeans作为模型评估指标
def model_perfomance(train_weight, model=xgb):
    model.fit(X_train, y_train, sample_weight=train_weight)
    y_pred, y_prob = model.predict(X_train), model.predict_proba(X_train)[:,-1]
    auc = roc_auc_score(y_train, y_prob)
    f1 = f1_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    gmeans = gmean_score(y_train, y_pred)
    return auc, f1, recall, gmeans


weights = [get_weight(1, 1), get_weight(2, 1), get_weight(4, 1), get_weight(6, 1)]
performance = pd.DataFrame(list(map(model_perfomance, weights)))
performance.columns = ['AUC', 'F1_Score', 'Recall', 'G_Means']
performance.index = ['1 : 1', '2 : 1', '4 : 1', '6 : 1']
performance.index.name = 'P : N'
print(performance)

# 对最高的分数予以标黄，仅适用于Jupyter
def highlight_max(s):
    is_max = s == s.max() 
    bg = ['background-color: yellow' if v else '' for v in is_max]
    return bg
performance.T.style.apply(highlight_max, axis=1)