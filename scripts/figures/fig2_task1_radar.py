import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- 1. 为三个图表准备数据 ---
data1 = {
    'scGen': {
        'Pearson': 0.9990, 'Pearson Delta': 0.5709, 'Pearson Delta DEG 20': 0.6705,
        'Pearson Delta DEG 50': 0.7548, 'Pearson Delta DEG 100': 0.7075
    },
    'Squidiff': {
        'Pearson': 0.3058, 'Pearson Delta': 0.0682, 'Pearson Delta DEG 20': 0.1589,
        'Pearson Delta DEG 50': 0.0147, 'Pearson Delta DEG 100': 0.0077
    },
    'scDiffusion': {
        'Pearson': 0.9077, 'Pearson Delta': 0.2618, 'Pearson Delta DEG 20': 0.1562,
        'Pearson Delta DEG 50': 0.2177, 'Pearson Delta DEG 100': 0.0360
    },
    'DDPM': {
        'Pearson': 0.0086, 'Pearson Delta': 0.2565, 'Pearson Delta DEG 20': 0.4682,
        'Pearson Delta DEG 50': 0.4539, 'Pearson Delta DEG 100': 0.3386
    },
    'DDPM+MLP': {
        'Pearson': 0.0146, 'Pearson Delta': 0.2091, 'Pearson Delta DEG 20': 0.5742,
        'Pearson Delta DEG 50': 0.5696, 'Pearson Delta DEG 100': 0.5145
    }
}

data2 = {
    'scGen': {
        'Pearson': 0.9979, 'Pearson Delta': 0.6161, 'Pearson Delta DEG 20': 0.8422,
        'Pearson Delta DEG 50': 0.7683, 'Pearson Delta DEG 100': 0.7416
    },
    'Squidiff': {
        'Pearson': 0.3419, 'Pearson Delta': 0.0378, 'Pearson Delta DEG 20': 0.0980,
        'Pearson Delta DEG 50': 0.1910, 'Pearson Delta DEG 100': 0.1952
    },
    'scDiffusion': {
        'Pearson': 0.9610, 'Pearson Delta': 0.5218, 'Pearson Delta DEG 20': 0.6119,
        'Pearson Delta DEG 50': 0.4088, 'Pearson Delta DEG 100': 0.5083
    },
    'DDPM': {
        'Pearson': 0.0052, 'Pearson Delta': 0.5251, 'Pearson Delta DEG 20': 0.8928,
        'Pearson Delta DEG 50': 0.8211, 'Pearson Delta DEG 100': 0.7033
    },
    'DDPM+MLP': {
        'Pearson': 0.0659, 'Pearson Delta': 0.5835, 'Pearson Delta DEG 20': 0.9258,
        'Pearson Delta DEG 50': 0.8241, 'Pearson Delta DEG 100': 0.7439
    }
}

data3 = {
    'scGen': {
        'Pearson': 0.9978, 'Pearson Delta': 0.7873, 'Pearson Delta DEG 20': 0.8695,
        'Pearson Delta DEG 50': 0.8557, 'Pearson Delta DEG 100': 0.8303
    },
    'Squidiff': {
        'Pearson': 0.3574, 'Pearson Delta': 0.0453, 'Pearson Delta DEG 20': 0.1836,
        'Pearson Delta DEG 50': 0.1473, 'Pearson Delta DEG 100': 0.1507
    },
    'scDiffusion': {
        'Pearson': 0.9587, 'Pearson Delta': 0.3862, 'Pearson Delta DEG 20': 0.1562,
        'Pearson Delta DEG 50': 0.7443, 'Pearson Delta DEG 100': 0.6483
    },
    'DDPM': {
        'Pearson': 0.0055, 'Pearson Delta': 0.6243, 'Pearson Delta DEG 20': 0.8768,
        'Pearson Delta DEG 50': 0.8118, 'Pearson Delta DEG 100': 0.7235
    },
    'DDPM+MLP': {
        'Pearson': 0.0059, 'Pearson Delta': 0.5910, 'Pearson Delta DEG 20': 0.8918,
        'Pearson Delta DEG 50': 0.8021, 'Pearson Delta DEG 100': 0.7142
    }
}

# 将所有数据和标题组合起来以便循环处理
all_data = [data1, data2, data3]
dfs = [pd.DataFrame(d).T for d in all_data]

# --- 2. 设置通用绘图参数 ---
# 从第一个DataFrame中获取方法和指标的名称
methods = dfs[0].index.tolist()
metrics = dfs[0].columns.tolist()

# 每个指标一个角度
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1] # 闭合雷达图

# 颜色映射
color_map = {
    'scGen': '#ff6f00ff',
    'Squidiff': '#008ea0ff',
    'scDiffusion': '#8a4198ff',
    'DDPM': '#c71000ff',
    'DDPM+MLP': '#3d3b25ff'
}

# --- 3. 绘图逻辑 ---
# 创建一个包含1行3列子图的图表
fig, axes = plt.subplots(figsize=(24, 8), nrows=1, ncols=3, subplot_kw=dict(polar=True))

# 循环绘制每个子图
for i, ax in enumerate(axes):
    df = dfs[i]
    # 循环绘制每个方法的雷达图
    for method in methods:
        values = df.loc[method].values.flatten().tolist()
        values += values[:1] # 闭合
        ax.plot(angles, values, color=color_map[method], linewidth=2, linestyle='solid', label=method)
        ax.fill(angles, values, color=color_map[method], alpha=0.2)

    # --- 4. 美化每个子图 ---
    ax.set_rscale('symlog', linthresh=0.01)
    
    tick_values = [0, 0.01, 0.1, 1.0]
    ax.set_yticks(tick_values)
    ax.set_yticklabels([str(val) for val in tick_values], color="grey", size=14)
    
    ax.set_ylim(0, 1.2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=16)
    
    ax.grid(True, color="grey", linestyle="--", linewidth=0.5)

# --- 5. 添加共享图例 ---
# 从第一个子图中获取图例句柄和标签
handles, labels = axes[0].get_legend_handles_labels()
# 在图表下方创建一个共享图例
fig.legend(handles, labels, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, -0.05), fontsize=16)

# 调整布局以防止图例重叠
plt.tight_layout(rect=[0, 0.05, 1, 1])

# --- 6. 保存图像 ---
os.makedirs('figs/fig2', exist_ok=True)
plt.savefig('figs/fig2/fig2_task1_radar_combined.svg', dpi=300, bbox_inches='tight')

plt.show()
