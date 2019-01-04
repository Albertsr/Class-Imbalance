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

## 4. Thresholding
#### 4.1 **Thresholding**
- 要求算法能返回后验概率预测值`$P(1|x)$`
- 一般情况下，阈值设定为0.5，当样本取正的后验概率大于0.5时判定为正样本，否则判定为负样本
 
#### 4.2 理论基础：

```math
p^{*} = \frac{C_{FP}}{C_{FN}+C_{FP}}  \text{；此时$C_{TN}=C_{TP}=0$}

\text{当设置$p^*<0.5$时, $C_{FN} > C_{FP}$, 从而实现了代价敏感算法}
```


---
## 5. Sampling

#### 5.1 理论基础
- 通过**Sampling方法**再平衡训练集，可以使得threshold为0.5的非代价敏感算法等价于threshold=p*(p*小于0.5)的代价敏感算法。

- **具体步骤**：保留所有正样本，对负样本进行欠采样，使得正负样本的比例保持为：
```math
P(1) C_{FN} : P(0) C_{FP}

\text{其中$P(1)$<$P(0)$，分别为正负样本的先验概率，即各自在训练集中的占比}
```
![TH](70CD01DCF472430D8AF4279FF0BB9515)

- **备注：**

```math
\frac{p^{*}}{1-p^{*}}=\frac{C_{FP}}{C_{FN}},\ 
\text{而} p_0 = 0.5\text{时，} \frac{1-p_0}{p_0}=1
\text{，因此}P(0) * \frac{p^{*}}{1-p^{*}} * \frac{1-p_0}{p_0} 
= P(0) * \frac{C_{FP}}{C_{FN}} 

\text{从而对负样本集进行欠采样后，正负样本的比例为：}
P(1) C_{FN} : P(0) C_{FP}
```


- **调整正负样本的比例与调整误分代价比例是可置换的：**
  - 正样本的占比加倍等价于`$C_{FN}$`加倍或`$C_{FP}$`减半
  -  the prior probabilities and the costs are interchangeable: doubling p(1) has the same effect as doubling FN, or halving FP (Drummond and Holte, 2000). 

---
#### 5.2 Weighting：权重法
- 对样本设置高或低的权重分别可视为样本的复制与删减，分别类似于过采样与欠采样，因此**weighting也属于Sampling的范畴**
---

#### 5.3 Costing：代价法

```math
\text{对正样本以概率} \frac{C(j, i)}{Z} \text{进行抽样，其中}
Z \geq max(C(j, i)) \text{ ，若} Z = max(C(j, i)) \text{则保留所有正样本；}

\text{对负样本以概率} \frac{C_{FP}}{C_{FN}} \text{进行不放回的欠采样，再训练bagging算法的基学习器}
```
- 很显然，**Costing也属于Sampling的范畴**
---
