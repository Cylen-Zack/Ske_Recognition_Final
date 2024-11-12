import pickle
import numpy as np

# 用'rb'模式打开文件：读模式和二进制模式
with open('/root/autodl-tmp/Ensemble/Test_Score/InfoGCN_loss_3d/best_score.pkl', 'rb') as file:
    data = pickle.load(file)

# 检查数据类型
print(type(data))

# 如果数据是 NumPy 数组，打印它的形状
if isinstance(data, np.ndarray):
    print(data.shape)
elif isinstance(data, list):
    print(len(data))  # 列表没有“形状”，但我们可以查看其长度
elif isinstance(data, dict):
    print(data.keys())  # 对于字典，查看其键
# 可以根据需要添加更多的类型检查和处理
