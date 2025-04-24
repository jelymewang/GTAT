import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载Tensor数据
benign_activations = torch.load('/data_B/renjie/CSA/act4.pt')  # 替换为实际文件名
adversarial_activations = torch.load('/data_B/renjie/CSA/act4_adv.pt')  # 替换为实际文件名

# 确保Tensor在CPU上（若你在GPU上运行）
benign_activations = benign_activations.cpu().numpy()
adversarial_activations = adversarial_activations.cpu().numpy()

# 确保数据的形状正确（假设是(2048,)的通道强度）
benign_activations = benign_activations.flatten()
adversarial_activations = adversarial_activations.flatten()

# 排序数据（根据攻击前的激活强度降序排序）
sorted_indices = np.argsort(-benign_activations)  # 按照降序排序

# 绘制图像
plt.figure(figsize=(10, 4))
plt.plot(benign_activations[sorted_indices], label='Benign examples', color='blue')
plt.plot(adversarial_activations[sorted_indices], label='Adversarial examples', color='orange')
plt.xlabel('Channel')
plt.ylabel('Magnitude of activation')
plt.legend()
plt.title('Channel-wise Activation Magnitude')

# 保存图像而不显示
plt.savefig('activation_magnitude_comparison.png', bbox_inches='tight')  # 保存为PNG格式文件
plt.close()  # 关闭图像，避免在后台显示
