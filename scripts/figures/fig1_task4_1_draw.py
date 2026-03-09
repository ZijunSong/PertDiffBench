import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

# --- 1. 数据准备 ---
# 请在此处填入您的完整数据
# 格式为: '方法': (均值, 误差)
data = {
    'ACTA2_control_coculture': {
        'scGen': (0.9918, 0.0002),
        'scDiff': (0.5830, 0.0000),
        'Squidiff': (0.1799, 0.0188),
        'scDiffusion': (0.7314, 0.0228),
        'DDPM': (0.0024, 0.0204),
        'DDPM+MLP': (0.0014, 0.0008)
    },
    'ACTA2_control_ifn': {
        'scGen': (0.9887, 0.0002),
        'scDiff': (0.6314, 0.0000),
        'Squidiff': (0.1450, 0.0106),
        'scDiffusion': (0.6977, 0.0933),
        'DDPM': (0.0040, 0.0175),
        'DDPM+MLP': (0.0059, 0.0017)
    },
    # 'B2M_control_coculture': {
    #     'scGen': (0.0083, 0.0134),
    #     'scDiff': (0.5582, 0.0000),
    #     'Squidiff': (0.1810, 0.0070),
    #     'scDiffusion': (0.8989, 0.0141),
    #     'DDPM': (0.0083, 0.0134),
    #     'DDPM+MLP': (0.0122, 0.0044)  
    # },
    # 'B2M_control_ifn': {
    #     'scGen': (0.9910, 0.0006), 
    #     'scDiff': (0.5888, 0.0000), 
    #     'Squidiff': (0.1814, 0.0051),
    #     'scDiffusion': (0.2857, 0.1730), 
    #     'DDPM': (0.0104, 0.0063), 
    #     'DDPM+MLP': (0.0270, 0.0036)
    # },
}

# --- 2. 设置绘图顺序和颜色 ---
# x轴上细胞簇的顺序
x_labels = ['ACTA2_control_coculture', 'ACTA2_control_ifn']
# 每个簇内，不同方法的顺序
method_order = ['scGen', 'scDiff', 'Squidiff', 'scDiffusion', 'DDPM', 'DDPM+MLP']
method_order = method_order[::-1]

color_map = {
    'scGen': '#ff6f00ff',       
    'scDiff': '#c71000ff',      
    'Squidiff': '#008ea0ff',    
    'scDiffusion': '#8a4198ff', 
    'DDPM': '#5a9599ff',        
    'DDPM+MLP': '#ff6348ff'     
}

# --- 3. 绘图逻辑 ---
space_per_group_inches = 2.2 
margin_inches = 2.0 
n_labels = len(x_labels)
fig_width = n_labels * space_per_group_inches + margin_inches

fig, ax = plt.subplots(figsize=(fig_width, 8))

# 计算条形的位置
n_methods = len(method_order)
x = np.arange(n_labels)
width = 0.8 / n_methods  # 每个条形的宽度

# 循环绘制每个方法的条形
for i, method in enumerate(method_order):
    # 提取当前方法在所有细胞簇上的均值和误差
    means = [data[label][method][0] for label in x_labels]
    errors = [data[label][method][1] for label in x_labels]

    ax.bar(x - (n_methods/2 - 0.5 - i) * width, means, width,
           yerr=errors, 
           capsize=3, 
           label=method, 
           color=color_map[method], # 从颜色映射中获取颜色
           alpha=0.85)

# --- 4. 美化图表 ---
ax.set_yscale('symlog', linthresh=0.1)
tick_locations = [-0.01, 0, 0.01, 0.1, 1]
ax.set_yticks(tick_locations)
# 对于symlog，有时需要手动设置标签以确保格式正确
ax.yaxis.set_major_formatter(FixedFormatter([str(tick) for tick in tick_locations]))
ax.set_ylabel('Pearson Correlation', fontsize=15)
# ax.set_title('各方法在不同细胞簇上的Pearson相关性比较', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=15)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.axhline(0, color='grey', linewidth=0.8)
ax.tick_params(axis='y', labelsize=12) # 调整Y轴刻度字体大小
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#           ncol=len(method_order), fancybox=True, shadow=False, frameon=False, fontsize=12)
plt.tight_layout()

import os
os.makedirs('figs/fig1', exist_ok=True)
plt.savefig('figs/fig1/fig1_task4_1.png', dpi=300, bbox_inches='tight')

