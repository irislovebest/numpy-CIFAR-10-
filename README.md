# 神经网络训练与测试流程说明

## 1. 数据准备

### 数据集信息
- ​**​数据集​**​: CIFAR-10（包含10类物体图片）
- ​**​样本数量​**​:
  - 原始训练集: 50,000
  - 测试集: 10,000
  - 验证集: 5,000（从训练集划分10%）

### 预处理步骤
```python
# 数据加载与预处理代码
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(50000, 3072)  # 展平为向量
x_train = x_train.astype('float32') / 255.0  # 归一化
y_train = to_categorical(y_train, 10)     # One-Hot编码
```
### 数据形状
| 数据集 | 特征维度 | 标签维度 |
|-------|-------|-------|
| 训练集 | (3072, 45000) | (10, 45000) |
| 验证集 | (3072, 5000) | (10, 5000) |
|测试集|	(3072, 10000)	|(10, 10000)
## 2. 超参数搜索
### 搜索空间配置
```python
param_grid = {
    'learning_rate': [0.1, 0.01],
    'hidden_size': [512, 256],
    'reg_lambda': [0.01, 0.001]
}
```
### 搜索结果表格
|组合	|学习率	|隐藏层	|正则化λ|	验证准确率|	验证损失|
|-------|-------|-------|-------|-------|-------|
|1	|0.1	|512|	0.01|	0.5240|	7.0118|
|2	|0.1	|512	|0.001	|0.5226	|2.0129|
|3	|0.1	|256	|0.01	|0.5150|	4.4670|
|4	|0.1	|256	|0.001	|0.5126|	1.7710|
|5	|0.01	|512|	0.01	|0.4784	|6.8391|
|6	|0.01	|512	|0.001|	0.4768	|2.0602|
|7	|0.01	|256	|0.01|	0.4708	|4.3061|
|8  |0.01	|256	|0.001	|0.4720|	1.8186|
### ​​最佳参数组合​​:

    yaml
    learning_rate: 0.1
    hidden_size: 512
    reg_lambda: 0.01
## 3. 模型训练
### 训练配置
```python
final_trainer = EnhancedTrainer(
    lr=0.1,
    hidden_size=512,
    reg_lambda=0.01,
    checkpoint_dir='best_model_checkpoints'
)
final_trainer.train(epochs=100, batch_size=128)
```
### 训练过程输出示例
Epoch 1/100 | Train Loss: 1.9523 | Val Loss: 1.8432 | Val Acc: 0.4120  
Epoch 50/100 | Train Loss: 1.1024 | Val Loss: 1.4021 | Val Acc: 0.5080  
Epoch 100/100 | Train Loss: 0.8765 | Val Loss: 1.2356 | Val Acc: 0.5320
## 4. 测试评估
### 测试结果
测试准确率	52.40%
混淆矩阵路径	confusion_matrix.png
混淆矩阵示例:
```python
cifar10_classes = ['飞机', '汽车', '鸟', '猫', '鹿',
                   '狗', '青蛙', '马', '船', '卡车']
EnhancedVisualizer.plot_confusion_matrix(test_true, test_pred, cifar10_classes)
```
## 5. 模型管理
### 模型保存
```python
final_model.save("final_model.pkl")  # 保存完整模型
```
### 文件结构
final_model.pkl  
├── params        # 权重参数  
├── config        # 网络结构配置    
└── metadata      # 训练时间等元信息  
### 模型加载
```python
loaded_model = ModelIO.load_model("final_model.pkl")
test_acc = loaded_model.evaluate(X_test, y_test)
print(f"加载模型测试准确率: {test_acc:.4f}")
```
## 6. 可视化输出
### 生成的可视化文件
training_metrics.png - 训练曲线  
params_visualization.png - 参数分布  
confusion_matrix.png - 分类结果矩阵  
### 可视化代码示例
```python
# 绘制训练曲线
EnhancedVisualizer.plot_training_metrics(final_trainer)

# 查看权重分布
EnhancedVisualizer.visualize_parameters(final_model)
```
