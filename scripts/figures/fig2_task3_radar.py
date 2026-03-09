import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- 1. 为三个图表准备数据 ---
data1 = {
    'scGen': {
        'Pearson': 0.7360, 'Pearson Delta': 0.4216, 'Pearson Delta DEG 20': 0.6156,
        'Pearson Delta DEG 50': 0.8017, 'Pearson Delta DEG 100': 0.7435
    },
    'Squidiff': {
        'Pearson': 0.1644, 'Pearson Delta': 0.0115, 'Pearson Delta DEG 20': 0.1620,
        'Pearson Delta DEG 50': 0.1045, 'Pearson Delta DEG 100': 0.0327
    },
    'scDiffusion': {
        'Pearson': 0.6563, 'Pearson Delta': 0.0263, 'Pearson Delta DEG 20': 0.0180,
        'Pearson Delta DEG 50': 0.1276, 'Pearson Delta DEG 100': 0.0859
    },
    'DDPM': {
        'Pearson': 0.0200, 'Pearson Delta': 0.2452, 'Pearson Delta DEG 20': 0.4661,
        'Pearson Delta DEG 50': 0.6056, 'Pearson Delta DEG 100': 0.5801
    },
    'DDPM+MLP': {
        'Pearson': 0.0083, 'Pearson Delta': 0.2358, 'Pearson Delta DEG 20': 0.4625,
        'Pearson Delta DEG 50': 0.5911, 'Pearson Delta DEG 100': 0.5804
    }
}

data2 = {
    'scGen': {
        'Pearson': 0.6557, 'Pearson Delta': 0.2024, 'Pearson Delta DEG 20': 0.6321,
        'Pearson Delta DEG 50': 0.5982, 'Pearson Delta DEG 100': 0.5830
    },
    'Squidiff': {
        'Pearson': 0.1418, 'Pearson Delta': 0.0330, 'Pearson Delta DEG 20': 0.3200,
        'Pearson Delta DEG 50': 0.1123, 'Pearson Delta DEG 100': 0.0185
    },
    'scDiffusion': {
        'Pearson': 0.5768, 'Pearson Delta': 0.0897, 'Pearson Delta DEG 20': 0.4896,
        'Pearson Delta DEG 50': 0.2649, 'Pearson Delta DEG 100': 0.1057
    },
    'DDPM': {
        'Pearson': 0.0067, 'Pearson Delta': 0.1082, 'Pearson Delta DEG 20': 0.0582,
        'Pearson Delta DEG 50': 0.3928, 'Pearson Delta DEG 100': 0.4575
    },
    'DDPM+MLP': {
        'Pearson': 0.0037, 'Pearson Delta': 0.1107, 'Pearson Delta DEG 20': 0.0840,
        'Pearson Delta DEG 50': 0.4137, 'Pearson Delta DEG 100': 0.4715
    }
}

data3 = {
    'scGen': {
        'Pearson': 0.8023, 'Pearson Delta': 0.3490, 'Pearson Delta DEG 20': 0.5352,
        'Pearson Delta DEG 50': 0.5645, 'Pearson Delta DEG 100': 0.6019
    },
    'Squidiff': {
        'Pearson': 0.1936, 'Pearson Delta': 0.0336, 'Pearson Delta DEG 20': 0.3381,
        'Pearson Delta DEG 50': 0.2259, 'Pearson Delta DEG 100': 0.0724
    },
    'scDiffusion': {
        'Pearson': 0.7475, 'Pearson Delta': 0.0105, 'Pearson Delta DEG 20': 0.0738,
        'Pearson Delta DEG 50': 0.1826, 'Pearson Delta DEG 100': 0.0911
    },
    'DDPM': {
        'Pearson': 0.0161, 'Pearson Delta': 0.1725, 'Pearson Delta DEG 20': 0.1788,
        'Pearson Delta DEG 50': 0.0046, 'Pearson Delta DEG 100': 0.3163
    },
    'DDPM+MLP': {
        'Pearson': 0.0018, 'Pearson Delta': 0.1622, 'Pearson Delta DEG 20': 0.1249,
        'Pearson Delta DEG 50': 0.0027, 'Pearson Delta DEG 100': 0.3219
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
plt.savefig('figs/fig2/fig2_task3_radar_combined.svg', dpi=300, bbox_inches='tight')

plt.show()
