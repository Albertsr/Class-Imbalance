# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

X, y = make_classification(n_samples=15000, n_classes=2, n_features=10, 
                           n_informative=8, weights=[0.8, 0.2], random_state=2018)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)

# ratio为训练集中负样本数与正样本数之间的比率
ratio = sum(y_train==0) / sum(y_train==1)


# 构造gmeans指标，等价于recall与specificity的几何平均
def gmean_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (fp + tn)
    return np.sqrt(recall * specificity)

# 构造样本权重生成函数，fn_cost，fp_cost分别表示FN，FP的代价
# 在风控领域，FN是指没有检测出欺诈交易，FP是指将正常交易误判为欺诈交易，因此FN的代价应大于FP的代价
# 实际应用场景中，正样本为少数，因此用赋予更大的权重，即正负样本的权重应与它们被误分引致的代价成一定的比例关系
# the weights are in proportion to their corresponding misclassification costs
def get_weight(fn_cost, fp_cost, y_train=y_train):
    weight = [fn_cost if i == 1 else fp_cost for i in y_train]
    return np.array(weight)

# 构造模型评估函数，暂取AUC,F1,Recall,Gmeans作为模型评估指标
def model_perfomance(model, train_weight):
    model.fit(X_train, y_train, sample_weight=train_weight)
    y_pred, y_prob = model.predict(X_train), model.predict_proba(X_train)[:,-1]
    auc = roc_auc_score(y_train, y_prob)
    f1 = f1_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    gmeans = gmean_score(y_train, y_pred)
    # print('AUC:{:.6f}, F_Score:{:.6f}, Recall:{:.6f}, GMeans:{:.6f}'.format(auc, f1, recall, gmeans))
    return np.array([auc, f1, recall, gmeans])

# 在分类器clf_1中，设置参数scale_pos_weight的参数为ratio
# train_weight设置为get_weight(1, 1)，表示正负训练样本的权重都设置为1，不进行区分
clf_1 = XGBClassifier(n_estimators=50, scale_pos_weight=ratio)
perfomance_1 = model_perfomance(model=clf_1, train_weight=get_weight(1, 1))

# 在分类器clf_2中，未设置参数scale_pos_weight的参数
# train_weight设置为get_weight(ratio, 1)，表示正负训练样本的权重之比为ratio
clf_2 = XGBClassifier(n_estimators=50)
perfomance_2 = model_perfomance(clf_2, get_weight(ratio, 1))

contrast = np.allclose(perfomance_1, perfomance_2)
decription = 'The parameter scale_pos_weight can be equivalent to the sample_weight parameter in the fit method.'
assert contrast, decription
if contrast: print(decription)