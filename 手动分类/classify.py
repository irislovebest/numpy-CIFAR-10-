import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import datetime
import pickle

# 数据预处理模块
class DataLoader:
    @staticmethod
    def load_data(val_ratio=0.1):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio)
        # 修复这里的大小写不一致问题
        return (x_train.T, y_train.T), (x_val.T, y_val.T), (x_test.T, y_test.T)
        # 原错误代码：return (x_train.T, y_train.T), (X_val.T, y_val.T), (X_test.T, y_test.T)

# 修改 DataLoader.load_data() 方法
@staticmethod
def load_data(val_ratio=0.1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)  # 形状变为 (50000, 3072)
    x_test = x_test.reshape(x_test.shape[0], -1)    # 形状变为 (10000, 3072)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio)
    
    # 转置数据，使形状变为 (特征维度, 样本数)
    return (x_train.T, y_train.T), (x_val.T, y_val.T), (x_test.T, y_test.T)

# 激活函数模块
class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=0))
        return exps / np.sum(exps, axis=0)

# 模型保存与加载模块
class ModelIO:
    @staticmethod
    def save_model(model, filepath):
        save_data = {
            'params': model.params,
            'config': {
                'input_size': model.params['W1'].shape[1],
                'hidden_size': model.params['W1'].shape[0],
                'output_size': model.params['W2'].shape[0],
                'activation': 'relu'
            },
            'metadata': model.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = ThreeLayerNN(
            data['config']['input_size'],
            data['config']['hidden_size'],
            data['config']['output_size'],
            activation=data['config']['activation']
        )
        model.params = data['params']
        model.metadata = data['metadata']
        print(f"Model loaded from {filepath}")
        return model

# 神经网络模型模块
class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.params = {
            'W1': np.random.randn(hidden_size, input_size) * np.sqrt(2./input_size),
            'b1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(output_size, hidden_size) * np.sqrt(2./hidden_size),
            'b2': np.zeros((output_size, 1))
        }
        self.activation = getattr(Activation, activation)
        self.activation_deriv = getattr(Activation, f"{activation}_derivative")
        self.cache = {}
        self.metadata = {
            'created_time': datetime.datetime.now().isoformat(),
            'last_modified': None
        }

    def forward(self, X):
        
        
        Z1 = np.dot(self.params['W1'], X) + self.params['b1']
        A1 = self.activation(Z1)
        Z2 = np.dot(self.params['W2'], A1) + self.params['b2']
        A2 = Activation.softmax(Z2)
        self.cache = {'A0': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2
    
    def backward(self, X, y, reg_lambda=0.0):
        m = X.shape[1]
        dZ2 = self.cache['A2'] - y
        dW2 = (1/m) * np.dot(dZ2, self.cache['A1'].T) + (reg_lambda/m) * self.params['W2']
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = np.dot(self.params['W2'].T, dZ2)
        dZ1 = dA1 * self.activation_deriv(self.cache['Z1'])
        dW1 = (1/m) * np.dot(dZ1, X.T) + (reg_lambda/m) * self.params['W1']
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def save(self, filepath):
        self.metadata['last_modified'] = datetime.datetime.now().isoformat()
        ModelIO.save_model(self, filepath)

# 增强训练模块
class EnhancedTrainer:
    def __init__(self, model, lr=0.01, reg_lambda=0.01, lr_decay=0.95, checkpoint_dir='checkpoints'):
        self.model = model
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.lr_decay = lr_decay
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_params = None
        self.best_val_acc = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[1]
        log_probs = -np.log(y_pred[y_true.argmax(axis=0), np.arange(m)])
        data_loss = np.sum(log_probs) / m
        
        reg_loss = 0.5*self.reg_lambda*(np.sum(self.model.params['W1']**2) + np.sum(self.model.params['W2']**2))
        return data_loss + reg_loss
    
    def evaluate(self, X, y):
        y_pred = self.model.forward(X)
        accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y, axis=0))
        return accuracy
    def test_eval(self,y_pred,y):
        accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y, axis=0))
        return accuracy
    def update_learning_rate(self):
        self.lr *= self.lr_decay
        
    def sgd_step(self, grads):
        self.model.params['W1'] -= self.lr * grads['dW1']
        self.model.params['b1'] -= self.lr * grads['db1']
        self.model.params['W2'] -= self.lr * grads['dW2']
        self.model.params['b2'] -= self.lr * grads['db2']
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, save_every=10):
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[1])
            X_shuffled = X_train[:, permutation]
            y_shuffled = y_train[:, permutation]
            
            for i in range(0, X_train.shape[1], batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]
                y_pred = self.model.forward(X_batch)
                grads = self.model.backward(X_batch, y_batch, self.reg_lambda)
                self.sgd_step(grads)
            
            val_acc = self.evaluate(X_val, y_val)
            val_loss = self.compute_loss(self.model.forward(X_val), y_val)
            train_loss = self.compute_loss(self.model.forward(X_train), y_train)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {k: v.copy() for k, v in self.model.params.items()}
                self.model.save(f"{self.checkpoint_dir}/best_model.pkl")
            
            if (epoch + 1) % save_every == 0:
                self.model.save(f"{self.checkpoint_dir}/epoch_{epoch+1}.pkl")
            
            self.update_learning_rate()
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# 超参数搜索模块
class HyperparameterTuner:
    @staticmethod
    def grid_search(param_grid, X_train, y_train, X_val, y_val, max_epochs=50):
        best_params = None
        best_acc = 0
        results = []
        
        # 遍历所有参数组合
        for lr in param_grid['learning_rate']:
            for hidden_size in param_grid['hidden_size']:
                for reg in param_grid['reg_lambda']:
                    print(f"\nTesting lr={lr}, hidden={hidden_size}, reg={reg}")
                    
                    # 初始化模型和训练器
                    model = ThreeLayerNN(X_train.shape[0], hidden_size, y_train.shape[0])
                    trainer = EnhancedTrainer(model, lr=lr, reg_lambda=reg)
                    
                    # 训练并验证
                    trainer.train(X_train, y_train, X_val, y_val, epochs=max_epochs)
                    
                    # 记录结果
                    current_result = {
                        'params': {
                            'learning_rate': lr,
                            'hidden_size': hidden_size,
                            'reg_lambda': reg
                        },
                        'val_acc': trainer.best_val_acc,
                        'val_loss': min(trainer.val_losses)
                    }
                    results.append(current_result)
                    
                    # 更新最佳参数
                    if trainer.best_val_acc > best_acc:
                        best_acc = trainer.best_val_acc
                        best_params = {
                            'learning_rate': lr,
                            'hidden_size': hidden_size,
                            'reg_lambda': reg
                        }
        
        # 打印搜索结果
        print("\n=== Grid Search Results ===")
        for i, res in enumerate(results):
            print(f"Combination {i+1}: {res['params']} => Val Acc: {res['val_acc']:.4f}, Val Loss: {res['val_loss']:.4f}")
        
        return best_params, best_acc, results

# 可视化模块
class EnhancedVisualizer:
    @staticmethod
    def plot_training_metrics(trainer, save_path="training_metrics.png"):
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(trainer.train_losses, label='Train Loss', color='blue')
        plt.plot(trainer.val_losses, label='Val Loss', color='orange')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(trainer.val_accuracies, color='green')
        plt.title("Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        
        plt.subplot(1, 3, 3)
        
        
        lr_history = [trainer.lr*(trainer.lr_decay**i)for i in range(len(trainer.train_losses))]
        plt.plot(lr_history, color='red')
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def visualize_parameters(model, save_prefix="params_"):
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        sns.histplot(model.params['W1'].flatten(), bins=100, kde=True)
        plt.title("W1 Weight Distribution")
        
        plt.subplot(1, 3, 2)
        sample_weights = model.params['W1'][:50, :50]
        sns.heatmap(sample_weights, cmap='coolwarm', center=0)
        plt.title("W1 Weight Matrix (Partial)")
        
        plt.subplot(1, 3, 3)
        sns.histplot(model.params['W2'].flatten(), bins=100, kde=True)
        plt.title("W2 Weight Distribution")
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}visualization.png")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.close()

# 主程序
if __name__ == "__main__":
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataLoader.load_data()
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"验证集形状: X={X_val.shape}, y={y_val.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
    # 定义搜索空间
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'hidden_size': [512, 256],
        'reg_lambda': [0.01, 0.001]
    }
    
    # 执行网格搜索
    best_params, best_acc, search_results = HyperparameterTuner.grid_search(
        param_grid, X_train, y_train, X_val, y_val, max_epochs=30
    )
    
    print(f"\nBest params: {best_params} with val_acc: {best_acc:.4f}")
    
    # 使用最佳参数训练最终模型
    final_model = ThreeLayerNN(X_train.shape[0], best_params['hidden_size'], y_train.shape[0])
    final_trainer = EnhancedTrainer(
        final_model,
        lr=best_params['learning_rate'],
        reg_lambda=best_params['reg_lambda'],
        checkpoint_dir='best_model_checkpoints'
    )
    
    # 完整训练
    final_trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=128)
    
    # 可视化
    EnhancedVisualizer.plot_training_metrics(final_trainer)
    EnhancedVisualizer.visualize_parameters(final_model)
    
    # 测试集评估
    test_pred = np.argmax(final_model.forward(X_test), axis=0)
    test_true = np.argmax(y_test, axis=0)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    EnhancedVisualizer.plot_confusion_matrix(test_true, test_pred, cifar10_classes)
    
    # 保存最终模型
    final_model.save("final_model.pkl")
    
    # 加载测试
    loaded_model = ModelIO.load_model("final_model.pkl")
    
    test_acc = final_trainer.test_eval(loaded_model.forward(X_test), y_test)
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")