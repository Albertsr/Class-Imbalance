# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com

# 出处：蚂蚁金服-风险大脑-支付风险识别大赛(第一赛季) 
# https://dc.cloud.alipay.com/index#/topic/data?id=4
# TPR1：当FPR等于0.001时的TPR; TPR2：当FPR等于0.005时的TPR; TPR3：当FPR等于0.01时的TPR
# 模型成绩 = 0.4 * TPR1 + 0.3 * TPR2 + 0.3 * TPR3

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer
def weighted_coverage(y_true, y_prob):    
    fprs, tprs, thresholds = roc_curve(y_true, y_prob)
    
    # min_delta_index返回与target_fpr最接近的阈值索引
    def min_delta_index(target_fpr):
        delta = np.array([np.abs(fpr - target_fpr) for fpr in fprs])
        return np.argmin(delta)
    
    target_tprs = [tprs[min_delta_index(target_fpr)] for target_fpr in (0.001, 0.005, 0.01)]
    weights = [0.4, 0.3, 0.3]
    return np.dot(weights, target_tprs)
