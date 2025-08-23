import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FixedLocator, FixedFormatter

# --- 1. 数据准备 ---
# 数据保持不变
data = {
    'mix2': {
        'scGen': (1.0000, 0.0000),
        'scDiff': (0.7894, 0.0000),
        'Squidiff': (0.0306, 0.0020),
        'scDiffusion': (0.0199, 0.0000),
        'DDPM': (0.3089, 0.0360),
        'DDPM+MLP': (0.0070, 0.0016)
    },
    'mix3': {
        'scGen': (1.0000, 0.0000),
        'scDiff': (0.9674, 0.0000),
        'Squidiff': (0.7753, 0.0013),
        'scDiffusion': (0.0395, 0.0016),
        'DDPM': (0.3257, 0.0482),
        'DDPM+MLP': (0.0352, 0.0028)
    },
    'mix4': {
        'scGen': (1.0000, 0.0000),
        'scDiff': (0.9742, 0.0000),
        'Squidiff': (0.0242, 0.0386),
        'scDiffusion': (0.0629, 0.0017),
        'DDPM': (0.3415, 0.0362),
        'DDPM+MLP': (0.0227, 0.0003)  
    },
    'mix5': {
        'scGen': (1.0000, 0.0000), 
        'scDiff': (0.9829, 0.0000), 
        'Squidiff': (0.2877, 0.0055),
        'scDiffusion': (0.1280, 0.0071), 
        'DDPM': (0.4018, 0.1329), 
        'DDPM+MLP': (0.0182, 0.0019)
    },
    'mix6': {
        'scGen': (1.0000, 0.0000), 
        'scDiff': (0.9907, 0.0000), 
        'Squidiff': (0.2442, 0.0071),
        'scDiffusion': (0.0694, 0.0012), 
        'DDPM': (0.4154, 0.0951), 
        'DDPM+MLP': (0.0164, 0.0007)
    },
    'mix7': {
        'scGen': (1.0000, 0.0000), 
        'scDiff': (0.9918, 0.0000), 
        'Squidiff': (0.0074, 0.0151),
        'scDiffusion': (0.0856, 0.0025), 
        'DDPM': (0.4490, 0.1301), 
        'DDPM+MLP': (0.0097, 0.0032)
    },
}

# --- 2. 设置绘图顺序和颜色 ---
# x轴上细胞簇的顺序
x_labels = ['mix2', 'mix3', 'mix4', 'mix5', 'mix6', 'mix7']
# 每个簇内，不同方法的顺序
method_order = ['scGen', 'scDiff', 'Squidiff', 'scDiffusion', 'DDPM', 'DDPM+MLP']

# 颜色映射保持不变
color_map = {
    'scGen': '#ff6f00ff',      
    'scDiff': '#c71000ff',      
    'Squidiff': '#008ea0ff',    
    'scDiffusion': '#8a4198ff', 
    'DDPM': '#5a9599ff',        
    'DDPM+MLP': '#ff6348ff'     
}

# --- 3. 绘图逻辑 ---
# 创建图和坐标轴
fig, ax = plt.subplots(figsize=(10, 7))
x = np.arange(len(x_labels)) # x轴的位置

# 循环绘制每个方法的折线
for method in method_order:
    # 提取当前方法在所有细胞簇上的均值和误差
    means = np.array([data[label][method][0] for label in x_labels])
    errors = np.array([data[label][method][1] for label in x_labels])

    # 绘制折线图
    ax.plot(x, means, marker='o', linestyle='-', label=method, color=color_map[method])
    
    # 填充误差范围
    ax.fill_between(x, means - errors, means + errors, color=color_map[method], alpha=0.2)

# --- 4. 美化图表 ---
ax.set_yscale('symlog', linthresh=0.1)
tick_locations = [-0.01, 0, 0.01, 0.1, 1]
ax.set_yticks(tick_locations)
# 对于symlog，有时需要手动设置标签以确保格式正确
ax.yaxis.set_major_formatter(FixedFormatter([str(tick) for tick in tick_locations]))
ax.set_ylabel('Pearson Correlation', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=15)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.axhline(0, color='grey', linewidth=0.8)
ax.tick_params(axis='y', labelsize=12)

# 调整图例位置，使其不与图表重叠
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#           ncol=3, fancybox=True, shadow=False, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局为图例留出空间

# --- 5. 保存图像 ---
# 确保目录存在
os.makedirs('figs/fig1', exist_ok=True)
# 保存图像
plt.savefig('figs/fig1/fig1_task3.svg', dpi=300, bbox_inches='tight')