- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr
---

## 一. Over-Sampling（过采样）
### 1. SMOTE
- **论文：** [SMOTE：Synthetic Minority Over-sampling Technique](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Papers/SMOTE%EF%BC%9ASynthetic%20Minority%20Over-sampling%20Technique.pdf)
- **算法流程**

  [SMOTE](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/SMOTE.jpg)

- **缺点：** 对所有少类样本一视同仁，没有重点关注邻近边界线的样本，使得分类器的性能增长有限

- **API**
  ```
  from imblearn.over_sampling import SMOTE
  sm = SMOTE(kind='regular', random_state=random_state)
  X_train, y_train = sm.fit_sample(X_train, y_train)
  ```
--- 

### 2. Borderline_SMOTE
#### 2.1 Borderline_SMOTE1
- **论文：** [Borderline-SMOTE A New Over-Sampling Method in Imbalanced Data Sets Learning](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Papers/Borderline-SMOTE%20A%20New%20Over-Sampling%20Method%20in%20Imbalanced%20Data%20Sets%20Learning.pdf)

- **算法流程**

   ![BordlineSMOTE1](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/Bordline_SMOTE1.jpg)
    
- **API**
  ```
  from imblearn.over_sampling import SMOTE
  sm = SMOTE(kind='borderline1', random_state=random_state)
  X_train, y_train = sm.fit_sample(X_train, y_train)
  ```
  
#### 2.2 Borderline_SMOTE2
- **算法流程**
  
  ![BordlineSMOTE2](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/Bordline_SMOTE2.jpg)

- **API**
  ```
  from imblearn.over_sampling import SMOTE
  sm = SMOTE(kind='borderline2', random_state=random_state)
  X_train, y_train = sm.fit_sample(X_train, y_train)
  ```
---

### 3. ADASYN
- **论文：** [ADASYN：Adaptive Synthetic Sampling Approach for Imbalanced Learning](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Papers/ADASYN%EF%BC%9AAdaptive%20Synthetic%20Sampling%20Approach%20for%20Imbalanced%20Learning.pdf)

- **算法流程**

    ![ADASYN](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/ADASYN.jpg)

- **算法特点**
  - 以少数类的密度作为标准来决定每个少数类样本需要合成的样本数，密度越大，需要生成的样本数越多
  - 优势在于使得算法集中在更难以学习的样本上

- **API**
  ```
  from imblearn.over_sampling import ADASYN
  ada = ADASYN(random_state=random_state)
  X_train, y_train = ada.fit_sample(X_train, y_train)
  ```
---

### 4. BOS：Borderline Over_Sampling
- **论文：** [Borderline Over-sampling for Imbalanced Data Classification](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Papers/Borderline%20Over-sampling%20for%20Imbalanced%20Data%20Classification.pdf)

- **算法流程**
  - 设T为训练集，通过SVM找到其正类(即少数类)支持向量，记为`$sv^+$`
  - 对每个支持向量`$sv_i^+$`在P集上找到其k近邻样本集，构成k维向量nn[i]
  - 以**线性外插**或**线性内插**的方式合成新样本

- **线性外插**
  - **线性内插的应用条件：** `$sv_i^+$`在整个训练集T上的m邻域中**正样本占比高于0.5**
  - **线性外插的作用：** 朝负样本区域方向扩展正样本区域，推动决策边界更接近于理想位置
  - **外插公式：**
  
    ![extrapolation](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/extrapolation.jpg)
    
- **线性内插**
  - **线性内插的应用条件：** `$sv_i^+$`在整个训练集T上的m邻域中**正样本占比不高于0.5**
  - **线性外插的作用：** 巩固现有决策边界
  - **内插公式：**
  
      ![interpolation](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/interpolation.jpg)
      
  - **与SMOTE的区别：** SMOTE随机挑选k近邻正样本进行内插，本方法根据向量nn[i]中距离sv由近及远的顺序依次生成新样本

- **线性外插与内插示意图**
    
    ![orientation_hyperplane](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/orientation_%20hyperplane.jpg)
  
---

## 二. UnderSampling(欠采样)
### 1. 常见的欠采样技术
#### 1.1 ENN
- 如果某样本的大部分的k近邻样本都与其本身的类不一样，则将其删除

#### 1.2 Tomek Link Removal
- A、B为不同类别的样本，且互为最近邻，则A,B就是Tomek link
- 删除所有Tomek link
---

### 2. 欠抽样的缺点
#### 2.1 论文：[Applying Support Vector Machines to Imbalanced Datasets](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Papers/Applying%20Support%20Vector%20Machines%20to%20Imbalanced%20Datasets.pdf)

#### 2.2 欠采样可能会丢失有价值的样本
- we are throwing away valid instances, which contain valuable information

#### 2.3 模型学习到的决策边界与理想边界之间角度偏离较大
- **Fig. 1与Fig. 2的解读：**  欠采样改进了未采样时分离超平面更靠近正样本集的缺点，但是模型从负样本中学习到的关于分离超平面方向的信息减少了，使得分离超平面与理想超平面之间的方向偏离度变得更大；

  ![orientation](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/orientation_%20hyperplane.jpg)

- **Fig. 3的解读：** 
- 随着的欠采样的进行，样本的Imblace Ratio逐步下降，而模型生成的分离超平面与理想分离超平面的偏离程度衡量指标Angel却呈上升趋势
- 图像从右至左的方向看，更易理解
 
  ![imbalance_angle](https://github.com/Albertsr/Class-Imbalance/blob/master/2.%20Sampling/Pics/imbalance%20ratio%26angle.jpg)
