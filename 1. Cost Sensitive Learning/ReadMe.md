- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr
---

## 1. 代价敏感与非代价敏感算法的区别

#### 1.1 关键区别：
- cost-sensitive Learning对不同的分类错误赋予不同的代价
- cost-insensitive Learning不区分不同分类的错误的代价

#### 1.2 算法目标不同
- cost-sensitive Learning以最小的代价为目标
- cost-insensitive Learning以最小的分类误差为目标

![关键区别](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/CSL%E7%9A%84CIL%E7%9A%84%E5%8C%BA%E5%88%AB.jpg)

---
## 2. 基于代价矩阵的分类决策
#### 2.1 在欺诈识别等领域：FN的代价大于FP的代价
- 在欺诈检测等应用场景中，FN是指没有识别出真实的欺诈交易，FP是指将正常交误判为欺诈交易。显然FN的代价大于FP的代价

#### 2.2 代价矩阵
![代价矩阵](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/cost%20matrix.jpg)


#### 2.3 样本x被预测为正类的充要条件
- **期望代价**

  ![代价矩阵](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E6%9C%9F%E6%9C%9B%E4%BB%A3%E4%BB%B7.jpg)

- **样本x被预测为正类的充要条件：被分类为正样本的期望代价更小**
 
 ![阈值推导](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E9%98%88%E5%80%BC%E6%8E%A8%E5%AF%BC.jpg)

- **阈值的设定**
 
  ![阈值设定](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E9%98%88%E5%80%BC%E8%AE%BE%E5%AE%9A.jpg)
---

## 3. 代价敏感算法的分类
#### 3.1 Direct methods（直接法） 
- 算法构建过程中就已考虑不同分类错误的代价
- incorporates the misclassification costs into the learning algorithm, to design classifiers that are cost-sensitive in themselves 

#### 3.2 Wrapper-based methods (封装法)
- 在不改变算法本身的情况下，将非代价敏感算法转化为代价敏感算法，也称为meta-learning method

- **主要分为两大类**
  - Thresholding(阈值法)
  - Sampling(采样法)
    - Weighting(权重法)
    - Costing
    
#### 3.3 分类结构图
     
   ![CSL结构图](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/CSL%E7%BB%93%E6%9E%84%E5%9B%BE.jpg)

---

## 4. 经典论文《The Foundations of Cost-Sensitive Learning》总结

- [《The Foundations of Cost-Sensitive Learning》解读](https://github.com/Albertsr/Class-Imbalance#第三部分论文解读与reference)
