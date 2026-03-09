import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FixedLocator, FixedFormatter

# --- 1. 数据准备 ---
# 数据保持不变
data = {
    '0.10': {
        'scGen': (0.8956, 0.0007),
        'scDiff': (0.9269, 0.0000),
        'Squidiff': (0.0163, 0.0132),
        'scDiffusion': (0.7247, 0.0031),
        'DDPM': (0.0172, 0.0147),
        'DDPM+MLP': (0.0146, 0.0032)
    },
    '0.25': {
        'scGen': (0.8873, 0.0021),
        'scDiff': (0.9258, 0.0000),
        'Squidiff': (0.0158, 0.0219),
        'scDiffusion': (0.6026, 0.0011),
        'DDPM': (0.0175, 0.0014),
        'DDPM+MLP': (0.0071, 0.0017)
    },
    '0.50': {
        'scGen': (0.7638, 0.0006),
        'scDiff': (0.9212, 0.0000),
        'Squidiff': (0.0096, 0.0041),
        'scDiffusion': (0.4885, 0.0011),
        'DDPM': (0.0122, 0.0049),
        'DDPM+MLP': (0.0011, 0.0092)  
    },
    '1.00': {
        'scGen': (0.7714, 0.0000), 
        'scDiff': (0.8922, 0.0000), 
        'Squidiff': (0.0021, 0.0076),
        'scDiffusion': (0.3223, 0.0062), 
        'DDPM': (0.0052, 0.0166), 
        'DDPM+MLP': (0.0112, 0.0104)
    },
    '1.50': {
        'scGen': (0.3878, 0.0004), 
        'scDiff': (0.2877, 0.0000), 
        'Squidiff': (0.0037, 0.0084),
        'scDiffusion': (0.2307, 0.0004), 
        'DDPM': (0.0083, 0.0079), 
        'DDPM+MLP': (0.0033, 0.0087)
    },
}

# --- 2. 设置绘图顺序和颜色 ---
# x轴上细胞簇的顺序
x_labels = ['0.10', '0.25', '0.50', '1.00', '1.50']
# 每个簇内，不同方法的顺序
method_order = ['scGen', 'scDiff', 'Squidiff', 'scDiffusion', 'DDPM+MLP', 'DDPM']

# 颜色映射保持不变
color_map = {
    'scGen': '#ff6f00ff',      
    'scDiff': '#c71000ff',      
    'Squidiff': '#008ea0ff',    
    'DDPM': '#8a4198ff', 
    'scDiffusion': '#5a9599ff',        
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
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
          ncol=5, fancybox=True, shadow=False, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局为图例留出空间

# --- 5. 保存图像 ---
# 确保目录存在
os.makedirs('figs/fig_plus', exist_ok=True)
# 保存图像
plt.savefig('figs/fig_plus/gaus_plot.svg', dpi=300, bbox_inches='tight')