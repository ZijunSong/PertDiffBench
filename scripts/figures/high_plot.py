import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FixedLocator, FixedFormatter

# --- 1. 数据准备 ---
# 数据保持不变
data = {
    '6998': {
        'scGen': (0.9919, 0.0007),
        'scDiff': (0.0022, 0.0000),
        'Squidiff': (0.0440, 0.0081),
        'scDiffusion': (0.9248, 0.0015),
        'DDPM': (0.3663, 0.0559),
        'DDPM+MLP': (0.0003, 0.0011)
    },
    '6000': {
        'scGen': (0.4129, 0.0065),
        'scDiff': (0.0087, 0.0000),
        'Squidiff': (0.0582, 0.0007),
        'scDiffusion': (0.3465, 0.0006),
        'DDPM': (0.0058, 0.0090),
        'DDPM+MLP': (0.0091, 0.0002)
    },
    '5000': {
        'scGen': (0.6747, 0.0005),
        'scDiff': (0.0172, 0.0000),
        'Squidiff': (0.0516, 0.0052),
        'scDiffusion': (0.4778, 0.0034),
        'DDPM': (0.0124, 0.0107),
        'DDPM+MLP': (0.0021, 0.0017)  
    },
    '4000': {
        'scGen': (0.6958, 0.0136), 
        'scDiff': (0.0063, 0.0000), 
        'Squidiff': (0.3089, 0.0017),
        'scDiffusion': (0.4947, 0.0006), 
        'DDPM': (0.0193, 0.0014), 
        'DDPM+MLP': (0.0348, 0.0008)
    },
    '3000': {
        'scGen': (0.7323, 0.0149), 
        'scDiff': (0.0087, 0.0000), 
        'Squidiff': (0.0001, 0.0043),
        'scDiffusion': (0.5785, 0.0015), 
        'DDPM': (0.0017, 0.0246), 
        'DDPM+MLP': (0.0101, 0.0034)
    },
    '2000': {
        'scGen': (0.7676, 0.0102), 
        'scDiff': (0.0187, 0.0000), 
        'Squidiff': (0.2262, 0.0079),
        'scDiffusion': (0.5990, 0.0014), 
        'DDPM': (0.0086, 0.0116), 
        'DDPM+MLP': (0.0152, 0.0016)
    },
    '1000': {
        'scGen': (0.8330, 0.0021), 
        'scDiff': (0.0563, 0.0000), 
        'Squidiff': (0.0440, 0.0091),
        'scDiffusion': (0.6469, 0.0011), 
        'DDPM': (0.3663, 0.0559), 
        'DDPM+MLP': (0.0183, 0.0015)
    },
}

# --- 2. 设置绘图顺序和颜色 ---
# x轴上细胞簇的顺序
x_labels = ['6998', '6000', '5000', '4000', '3000', '2000', '1000']
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
plt.savefig('figs/fig_plus/high_plot.svg', dpi=300, bbox_inches='tight')