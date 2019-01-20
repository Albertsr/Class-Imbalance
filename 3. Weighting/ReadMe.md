- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

## Weighting的实现方法
### 1. 运用fit方法
- sklearn为常见监督算法均提供了fit方法，fit方法自带参数sample_weight
- 如下图所示

  ![Weighting_fit](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/Weighting_fit.jpg)

## 2. 运用scale_pos_weight参数
### 2.1 代码示例
- [scale_pos_weight.py](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/scale_pos_weight.py)

### 2.2 XGBOOST

![xgb-scale](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/xgb-scale.jpg)

### 2.3 LightGBM

![lgb scale_pos](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/lgb%20scale_pos.jpg)

