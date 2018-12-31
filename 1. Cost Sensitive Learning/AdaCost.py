# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.ensemble import AdaBoostClassifier


class AdaCostClassifier(AdaBoostClassifier): 
    # n_estimators的默认值由50升高至250
    # learning_rate的默认值由1.0降低至0.85
    # 新增了两个属性：FNcost、FPcost；分别表示FN、FP的代价
    def __init__(self, base_estimator=None, n_estimators=250, learning_rate=0.5, 
                 FNcost='auto', FPcost=1, algorithm='SAMME.R', random_state=None):
        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,n_estimators=n_estimators,
            learning_rate=learning_rate, random_state=random_state)
        
        self.FPcost = FPcost
        self.FNcost = FNcost
        self.algorithm = algorithm
    
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)
        incorrect = y_predict != y
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        proba = y_predict_proba  # alias for readability
        
        # 对proba进行截断操作
        # 等价于proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        
        estimator_weight = (-1. * self.learning_rate * ((n_classes - 1.) / n_classes)
                            * (y_coding * np.log(y_predict_proba)).sum(axis=1))
        
        # 样本更新的公式，只需要改写这里
        if not iboost == self.n_estimators - 1:
            criteria = ((sample_weight > 0) | (estimator_weight < 0))
            # 在原来的基础上乘以self._beta(y, y_predict)，即代价调整函数
            sample_weight *= np.exp(estimator_weight * criteria * self._beta(y, y_predict))  
        return sample_weight, 1., estimator_error

    #  新定义的代价调整函数
    def _beta(self, y, y_hat):
        res = []
        ratio = sum(y==0) / sum(y==1)
        if self.FNcost == 'auto':
            self.FNcost = ratio
        
        for i in zip(y, y_hat):
            # FN错误:将风险客户误判为正常用户
            if  i[0] == 1 and i[1] == 0:
                res.append(self.FNcost)  
            # FP错误：将正常客户误判为风险用户
            elif i[0] == 0 and i[1] == 1:
                res.append(self.FPcost)  
            # 正确分类，系数保持不变，按原来的比例减少    
            else:
                res.append(1)
        return np.array(res)