- **Author：** MaXiao
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：** https://github.com/Albertsr

---
## 若图片无法正常显示， 请点击链接：

[Class-Imbalance](https://note.youdao.com/ynoteshare1/index.html?id=bfd1d597b41c3645cda19efaed10b3d4&type=note)
---

# 第一部分：代价敏感综述
## 1. 代价敏感与非代价敏感算法的区别

### 1.1 关键区别：
- cost-sensitive Learning对不同的分类错误赋予不同的代价
- cost-insensitive Learning不区分不同分类的错误的代价

### 1.2 算法目标不同
- cost-sensitive Learning以最小的代价为目标
- cost-insensitive Learning以最小的分类误差为目标

![关键区别](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/CSL%E7%9A%84CIL%E7%9A%84%E5%8C%BA%E5%88%AB.jpg)

---

## 2. 基于代价矩阵的分类决策
### 2.1 代价矩阵的定义
![代价矩阵](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/cost%20matrix.jpg)

### 2.2 在欺诈识别等业务场景：FN的代价大于FP的代价
- FN是指没有识别出真实的欺诈交易，FP是指将正常交误判为欺诈交易。显然FN的代价大于FP的代价

### 2.3 样本x被预测为正类的充要条件
- **期望代价**

  ![代价矩阵](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E6%9C%9F%E6%9C%9B%E4%BB%A3%E4%BB%B7.jpg)

- **样本x被预测为正类的充要条件：被分类为正样本的期望代价更小**
 
 ![阈值推导](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E9%98%88%E5%80%BC%E6%8E%A8%E5%AF%BC.jpg)

- **阈值的设定**
 
  ![阈值设定](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E9%98%88%E5%80%BC%E8%AE%BE%E5%AE%9A.jpg)
---

## 3. 代价敏感算法的分类
### 3.1 Direct methods（直接法） 
- **算法构建过程中就已考虑不同分类错误的代价**
- incorporates the misclassification costs into the learning algorithm, to design classifiers that are cost-sensitive in themselves 

### 3.2 Wrapper-based methods (封装法)
- **在不改变算法本身的情况下，将非代价敏感算法转化为代价敏感算法，也称为meta-learning method**

- **主要分为两大类**
  - Thresholding(阈值法)
  - Sampling(采样法)
    - Weighting(权重法)
    - Costing
    
### 3.3 分类结构图
     
   ![CSL结构图](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/CSL%E7%BB%93%E6%9E%84%E5%9B%BE.jpg)

---

# 第二部分：Thresholding、Sampling与Weighting

## 1. Thresholding(阈值法)
- **多数情况下样本被正确分类的代价为0，因此阈值默认取**
  
  ![阈值默认值](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E9%98%88%E5%80%BC%E9%BB%98%E8%AE%A4%E5%80%BC.jpg)
  
- **非代价敏感算法的阈值一般设定为0.5，改变此阈值则间接实现了代价敏感学习**
  
  ![阈值与误分代价](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/threshold_cost.jpg)
    
- **Thresholding的相关理论阐述**
  - 论文：[Cost-Sensitive Learning and the Class Imbalance Problem](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/2.%20Cost-Sensitive%20Learning%20and%20the%20Class%20Imbalance%20Problem.pdf)
  
  ![阈值法](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/%E9%98%88%E5%80%BC%E6%B3%95.jpg) 

---

## 2. Sampling(采样法)
### 2.1 Sampling(采样法)与Thresholding(阈值法)之间可相互转换
  
- **Charles Elkan在论文[The Foundations of Cost-Sensitive Learning](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/1.%20The%20Foundations%20of%20Cost-Sensitive%20Learning.pdf)中明确地指出了阈值与负样本比例之间的转换关系**

 ![定理一](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/%E5%AE%9A%E7%90%86%E4%B8%80.jpg)
 
- **Theorem 1解读：** 设算法将样本推断为正样本的初始阈值为p'，为了达到目标阈值p，则训练集中负样本的数量应从初始值n'变为n，且满足以下比例关系：
  
  ![theorem1_math](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/theorem1_math.jpg)

- **阈值的变化趋势与负样本数成正比关系：为了达到更小的目标阈值，则应对负样本进行欠采样(Under Sampling)**

- **Theorem 1为Sampling(采样法)提供了坚实理论基础:** 严格论述了在二分类问题中，如何改变训练集中负样本的比例，使得非代价敏感算法学习到代价最小化的决策边界
   
     ![定理一阐述](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/%E5%AE%9A%E7%90%86%E4%B8%80%E9%98%90%E8%BF%B0.jpg)
 

### 2.2 过采样与欠采样
- **常见的过采样方法**
    - [SMOTE](https://github.com/Albertsr/Class-Imbalance/tree/master/2.%20Sampling#1-smote)
    - [Borderline_SMOTE](https://github.com/Albertsr/Class-Imbalance/tree/master/2.%20Sampling#2-borderline_smote)
    - [ADASYN](https://github.com/Albertsr/Class-Imbalance/tree/master/2.%20Sampling#3-adasyn)
    - [BOS：Borderline Over_Sampling](https://github.com/Albertsr/Class-Imbalance/tree/master/2.%20Sampling#4-bosborderline-over_sampling) **(基于正类支持向量的线性外插与内插)**
    
- **欠采样及其缺点：** 
  - [UnderSampling(欠采样综述)](https://github.com/Albertsr/Class-Imbalance/tree/master/2.%20Sampling#二-undersampling欠采样)
  - 欠抽样的缺点
    - [欠采样可能会丢失有价值的样本](https://github.com/Albertsr/Class-Imbalance/tree/master/2.%20Sampling#22-欠采样可能会丢失有价值的样本)
    - [模型学习到的决策边界与理想边界之间角度偏离较大](https://github.com/Albertsr/Class-Imbalance/tree/master/2.%20Sampling#23-模型学习到的决策边界与理想边界之间角度偏离较大)
    
    
### 2.3 实证分析：过采样方法性能对比
- **对比代码：** [oversampling_contrast.py](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/oversampling_contrast.py)

- **对比结果：**
  - **Coverage、G_Mean的定义：** [模型评估指标](https://github.com/Albertsr/Anomaly-Detection/blob/master/README.md#22-模型评估指标)
  
  ![oversample_contrast](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/over_sample_contrast.jpg) 


- **对比结论**
  - **过采样算法的表现与具体的数据集、模型评估指标等多方面因素有关，不存在持续占优的过采样算法**
  - **通常将少类样本视为正样本，所有过采样算法均有效地增加了正样本的数量，对Recall均有明显提升作用**
  - **BOS_SVM能灵活地选择内插或外插方式，生成的新样本更具合理性，有助于提升分类器的综合表现，AUC、G_Mean等指标也证明了此结论**
  
---

## 3. Weighting(权重法)
### 3.1 **Weighting等价于Sampling**

- **高样本权重(大于1)可以视为对样本的复制，即为Over_Sampling；低样本权重(小于1)可以视为对样本的删减，即为Under_Sampling**

  ![weighting3](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/weighting.jpg)

- **样本的权重应与其被误分的代价成比：** 少类样本一般为正样本，FN的代价高于FP的代价；应用Weighting(权重法)时，应对正样本赋予更高的权重

### 3.2 Weighting的实现
- **运用fit方法中的sample_weight参数实现** 
  - sklearn实现的监督算法、XGBoost、LightGBM均提供了fit方法，均含有sample_weight参数
  - **以sklearn文档为例：** [fit方法中的参数sample_weightt](https://github.com/Albertsr/Class-Imbalance/tree/master/3.%20Weighting#1-运用fit方法中的参数sample_weight)

     ![Weighting_fit](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/Weighting_fit.jpg)

- **scale_pos_weight参数：** 
  - XGBoost与LightGBM不仅提供了sample_weight参数，还提供了[scale_pos_weight参数](https://github.com/Albertsr/Class-Imbalance/tree/master/3.%20Weighting#2-运用scale_pos_weight参数)
  
  
  - **XGBOOST相关文档**

![xgb-scale](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/xgb-scale.jpg)

- **LightGBM相关文档**

![lgb scale_pos](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/lgb%20scale_pos.jpg)
  
  - 若scale_pos_weight的取值设定为某常数ratio，则等价于将正样本权重设置为ratio，负样本的权重设置为1；
  - **代码验证：** [scale_pos_weight.py](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/scale_pos_weight.py)

  
---

## 4. **Charles Elkan、Chris Drummond为Sampling、Weighting提供了理论基础**

### 4.1 Charles Elkan明确指出Theroem 1既适用于Weighting，也适用于Sampling

 ![定理1适用于权重与采样](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/%E5%AE%9A%E7%90%861%E9%80%82%E7%94%A8%E4%BA%8E%E6%9D%83%E9%87%8D%E4%B8%8E%E9%87%87%E6%A0%B7.jpg)

### 4.2 Chris Drummond明确指出，在二分类问题中，各类别的先验概率与误分代价可相互转换
- **正样本的先验概率加倍，等价于FN的代价加倍或FP的代价减半**
   
  ![加倍减半](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/interchangble.jpg)
 
- 论文原文

  ![先验概率与误分代价的转换](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/prior_cost_interchage.jpg)
 
### 4.3 理论小结
- **Charles Elkan提出的Theroem 1论述了负样本在训练集中的占比与阈值之间的转换关系，并给出了严格的转换公式**
- **在TP、TN的代价为0时，非代价敏感算法判定为正样本的后验概率阈值为0.5；若此阈值变更为[0, 1]区间内的其他值，则非代价敏感学习转化为代价敏感学习**
- **Sampling（或Weighting）通过改变正、负样本在训练集中的占比，内生地改变了阈值，从而间接实现了代价敏感学习**

---

## 5. Ensemble(集成法)

### 5.1 AdaCost
- **代码实现：** [adacost.py](https://github.com/Albertsr/Class-Imbalance/blob/master/4.%20Ensemble%20Methods/AdaCost/adacost.py)

### 5.1 EasyEnsemble & BalanceCascade
- **论文：** [Exploratory Undersampling for Class-Imbalance Learning](https://github.com/Albertsr/Class-Imbalance/blob/master/4.%20Ensemble%20Methods/Exploratory%20Undersampling%20for%20Class-Imbalance%20Learning.pdf)

- **EasyEnsemble**

   ![easyensemble](https://github.com/Albertsr/Class-Imbalance/blob/master/4.%20Ensemble%20Methods/Pics/easyensemble.jpg)


- **BalanceCascade**

   ![balancecascade](https://github.com/Albertsr/Class-Imbalance/blob/master/4.%20Ensemble%20Methods/Pics/balancecascade.jpg)

---

# 第三部分：论文解读与Reference

## 1. 经典论文《The Foundations of Cost-Sensitive Learning》解读
### 1.1 主要结论
- **Theorem 1：** 设算法将样本推断为正样本的初始阈值为p'，为了达到目标阈值p，则训练集中负样本的数量应从初始值n'变为n，且满足以下比例关系：
  
  ![theorem1_math](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/theorem1_math.jpg)

  - 若p<p'，则n<n'；若p>p'，则n>n'
  - **为了达到更小的目标阈值，则应对负样本进行欠采样(Under Sampling)**

- **Theorem 2：** 若训练集中正类样本的先验概率由b变为b'，则从训练集中随机抽取的样本x为正样本的后验概率由p变为p'，其中
  
  ![theorem2_math](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/theorem2_math.jpg)


- **Theorem 3：** 对于二分类问题，若决策树的生成过程中以p(1-p)的平方根形式衡量数据集的不纯度，则训练集中正、负样本的先验概率变化不影响决策树的结构；其中p为数据集中正样本的先验概率

 ![theorem3](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/theorem3.jpg)


### 1.2 定理的证明
- **先证明Theorem 2**

![theorem2_proof](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/theorem2_proof.jpg) 


- **再证明Theorem 1** 
  
![theorem1_proof!](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/theorem1_proof.jpg)

- **Theorem 3的证明**
  - 略，请查阅原论文；

---

## 2. References

- [The Foundations of Cost-Sensitive Learning](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/1.%20The%20Foundations%20of%20Cost-Sensitive%20Learning.pdf)

- [Cost-Sensitive Learning and the Class Imbalance Problem](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/2.%20Cost-Sensitive%20Learning%20and%20the%20Class%20Imbalance%20Problem.pdf)

- [Exploiting the Cost (In)sensitivity of Decision Tree Splitting Criteria](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/3.%20Exploiting%20the%20Cost%20(In)sensitivity%20of%20Decision%20Tree%20Splitting%20Criteria%20(Drummond2000).pdf)

- [Cost-Sensitive Learning vs. Sampling: Which is Best for Handling Unbalanced
Classes with Unequal Error Costs?](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/4.%20Cost-Sensitive%20Learning%20vs.%20Sampling.pdf)

- [Analysis and Visualization of Classifier Performance_ Comparison under Imprecise Class and Cost Distributions](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/5.%20Analysis%20and%20Visualization%20of%20Classifier%20Performance_%20Comparison%20under%20Imprecise%20Class%20and%20Cost%20Distributions.pdf)
