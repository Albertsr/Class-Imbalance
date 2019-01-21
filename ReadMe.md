- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：** https://github.com/Albertsr

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
### 2.1 在欺诈识别等业务场景：FN的代价大于FP的代价
- FN是指没有识别出真实的欺诈交易，FP是指将正常交误判为欺诈交易。显然FN的代价大于FP的代价

### 2.2 代价矩阵的定义
![代价矩阵](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/cost%20matrix.jpg)


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
 
    ![比例关系1](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/threom1_math.jpg)

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

- **初步结论：BOS_SVM性能相对较佳**
  - **BOS_SVM能根据正类支持向量周边的正样本密度来决定是以内插(Interpolation)还是外插(Extrapolation)来合成新的样本**，而其他合成方法只采用了内插的方式来生成新的样本
  - **在实际运用中，需要根据数据集的特点、模型性能评估指标等多方面因素灵活选择过采样方法**
    
    ![Oversample_contrast](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/Oversample_contrast.jpg) 

---

## 3. Weighting(权重法)
### 3.1 **Weighting等价于Sampling**

- **高样本权重(大于1)可以视为对样本的复制，即为Over_Sampling；低样本权重(小于1)可以视为对样本的删减，即为Under_Sampling**

  ![weighting3](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/weighting.jpg)

- **样本的权重应与其被误分的代价成比：** 少类样本一般为正样本，FN的代价高于FP的代价；应用Weighting(权重法)时，应对正样本赋予更高的权重

### 3.2 Weighting的实现
- **sklearn为常见监督算法均提供了fit方法，利用其参数sample_weight实现：** [sample_weight方法](https://github.com/Albertsr/Class-Imbalance/tree/master/3.%20Weighting#1-运用fit方法)

- **scale_pos_weight参数：** XGBoost与LightGBM不仅在fit方法中提供了sample_weight参数，还提供了[scale_pos_weight参数](https://github.com/Albertsr/Class-Imbalance/tree/master/3.%20Weighting#2-运用scale_pos_weight参数)
  - **代码示例：** [scale_pos_weight.py](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/scale_pos_weight.py)

---

## 4. **Charles Elkan、Chris Drummond为Sampling、Weighting提供了理论基础**

### 4.1 Charles Elkan明确指出Theroem 1既适用于Weighting，也适用于Sampling

 ![定理1适用于权重与采样](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/%E5%AE%9A%E7%90%861%E9%80%82%E7%94%A8%E4%BA%8E%E6%9D%83%E9%87%8D%E4%B8%8E%E9%87%87%E6%A0%B7.jpg)

### 4.2 Chris Drummond明确指出，在二分类问题中，各类别的先验概率与误分代价可相互转换
- **正样本的先验概率加倍，等价于FN的代价加倍或FP的代价减半**
   
  ![加倍减半](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/interchangble.jpg)
 
- 论文原文

  ![先验概率与误分代价的转换](https://github.com/Albertsr/Class-Imbalance/blob/master/1.%20Cost%20Sensitive%20Learning/Pics/SubPics/prior_cost_interchage.jpg)
 
### 4.3 理论分析
- Charles Elkan 提出的Theroem 1证明了负样本在训练集中的占比与阈值之间的转换关系，并给出了严格的转换公式

- 由前面的分析可知，改变样本被判定为正样本的阈值，等价于改变FN与FP之间的代价比例关系

- 在TP、TN的代价均为零的情况下，若阈值不再等于0.5，则非代价敏感学习将转化为代价敏感学习

- 所以，**Sampling（或Weighting）通过改变负样本在训练集中的占比，间接地改变了阈值，从而间接实现了代价敏感学习**

---

## 5. 经典论文《The Foundations of Cost-Sensitive Learning》的定理证明过程
- 作者与发表年份：
- 主要结论：
   - 定理一：
   - 定理二：
   - 定理三：
- 证明过程：
   - 
   - 
   -  
   
---

## 6. Reference

- [The Foundations of Cost-Sensitive Learning](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/1.%20The%20Foundations%20of%20Cost-Sensitive%20Learning.pdf)

- [Cost-Sensitive Learning and the Class Imbalance Problem](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/2.%20Cost-Sensitive%20Learning%20and%20the%20Class%20Imbalance%20Problem.pdf)

- [Exploiting the Cost (In)sensitivity of Decision Tree Splitting Criteria](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/3.%20Exploiting%20the%20Cost%20(In)sensitivity%20of%20Decision%20Tree%20Splitting%20Criteria%20(Drummond2000).pdf)

- [Cost-Sensitive Learning vs. Sampling: Which is Best for Handling Unbalanced
Classes with Unequal Error Costs?](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/4.%20Cost-Sensitive%20Learning%20vs.%20Sampling.pdf)

- [Analysis and Visualization of Classifier Performance_ Comparison under Imprecise Class and Cost Distributions](https://github.com/Albertsr/Class-Imbalance/blob/master/Papers/5.%20Analysis%20and%20Visualization%20of%20Classifier%20Performance_%20Comparison%20under%20Imprecise%20Class%20and%20Cost%20Distributions.pdf)
