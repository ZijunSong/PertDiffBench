import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 ---
# 请在此处填入您的完整数据
# 格式为: '方法': (均值, 误差)
data = {
'task1': { # CD14
        'scGen': (0.9740, 0.0003), 
        'scDiff': (0.7770, 0.0000), 
        'Squidiff': (0.4188, 0.0323),
        'scDiffusion': (0.4388, 0.0046), 
        'DDPM': (0.0443, 0.0210), 
        'DDPM+MLP': (0.0013, 0.0002)
    },
    'task2': { # random1
        'scGen': (0.8480, 0.0362),
        'scDiff': (0.7723, 0.0000),
        'Squidiff': (0.0942, 0.0072),
        'scDiffusion': (0.6898, 0.0616),
        'DDPM': (0.0299, 0.0131),
        'DDPM+MLP': (0.0258, 0.0010)
    },
    'task3': { # mix2
        'scGen': (1.0000, 0.0000),
        'scDiff': (0.77894, 0.0000),
        'Squidiff': (0.0306, 0.0020),
        'scDiffusion': (0.0199, 0.0000),
        'DDPM': (0.3089, 0.0360),
        'DDPM+MLP': (0.0070, 0.0016)  
    },
    'task4': {
        'scGen': (0.9910, 0.0006), 
        'scDiff': (0.5888, 0.0000), 
        'Squidiff': (0.1814, 0.0051),
        'scDiffusion': (0.2857, 0.1730), 
        'DDPM': (0.0104, 0.0063), 
        'DDPM+MLP': (0.0270, 0.0036)
    }
}

# --- 2. 设置绘图顺序和颜色 ---
# x轴上细胞簇的顺序
x_labels = ['task1', 'task2', 'task3', 'task4']
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
# 创建画布和子图
fig, ax = plt.subplots(figsize=(16, 8))

# 计算条形的位置
n_methods = len(method_order)
n_labels = len(x_labels)
x = np.arange(n_labels)  # x轴上每个大组的位置
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
ax.set_yscale('symlog', linthresh=0.01)
ax.set_ylabel('Pearson Correlation', fontsize=15)
# ax.set_title('各方法在不同细胞簇上的Pearson相关性比较', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=15)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.axhline(0, color='grey', linewidth=0.8)
ax.tick_params(axis='y', labelsize=12) # 调整Y轴刻度字体大小
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=len(method_order), fancybox=True, shadow=False, frameon=False, fontsize=12)
plt.tight_layout()

import os
os.makedirs('figs/fig1', exist_ok=True)
plt.savefig('figs/fig1/fig1.png', dpi=300, bbox_inches='tight')

