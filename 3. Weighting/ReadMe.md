- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

## Weighting的实现方法
### 1. 运用fit方法中的参数sample_weight
- sklearn为常见监督算法均提供了fit方法，fit方法自带参数sample_weight
- 如下图所示

  ![Weighting_fit](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/Weighting_fit.jpg)

## 2. 运用scale_pos_weight参数
### 2.1 XGBoost与LightGBM不仅提供了sample_weight参数，还提供了scale_pos_weight参数
- **XGBOOST相关文档**

![xgb-scale](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/xgb-scale.jpg)

- **LightGBM相关文档**

![lgb scale_pos](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/Pics/lgb%20scale_pos.jpg)

### 2.2 若scale_pos_weight的取值设定为某常数ratio，则等价于将正样本权重设置为ratio，负样本的权重设置为1；

- **代码验证：** [scale_pos_weight.py](https://github.com/Albertsr/Class-Imbalance/blob/master/3.%20Weighting/scale_pos_weight.py)
