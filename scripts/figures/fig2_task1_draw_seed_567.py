import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. 数据准备 ---
full_data = {
    'scGen': {
        'Pearson': 0.9978,
        'Pearson Delta': 0.7873, 
        'Pearson Delta DEG 20': 0.8695,
        'Pearson Delta DEG 50': 0.8557,
        'Pearson Delta DEG 100': 0.8303  
    },
    'Squidiff': {
        'Pearson': 0.3574,
        'Pearson Delta': 0.0453, 
        'Pearson Delta DEG 20': 0.1836,
        'Pearson Delta DEG 50': 0.1473,
        'Pearson Delta DEG 100': 0.1507  
    },
    'scDiffusion': {
        'Pearson': 0.9587,
        'Pearson Delta': 0.3862, 
        'Pearson Delta DEG 20': 0.1562,
        'Pearson Delta DEG 50': 0.7443,
        'Pearson Delta DEG 100': 0.6483  
    },
    'DDPM': {
        'Pearson': 0.0055,
        'Pearson Delta': 0.6243, 
        'Pearson Delta DEG 20': 0.8768,
        'Pearson Delta DEG 50': 0.8118,
        'Pearson Delta DEG 100': 0.7235  
    },
    'DDPM+MLP': {
        'Pearson': 0.0059,
        'Pearson Delta': 0.5910, 
        'Pearson Delta DEG 20': 0.8918,
        'Pearson Delta DEG 50': 0.8021,
        'Pearson Delta DEG 100': 0.7142  
    }
}

# 将数据转换为DataFrame，方便绘图
df = pd.DataFrame(full_data).T
df_log = df.apply(lambda x: np.log10(x.clip(lower=1e-5)))

# 从DataFrame中自动获取方法和指标的名称
methods = df.index.tolist()
metrics = df.columns.tolist()


# --- 2. 设置绘图参数 ---
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
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 循环绘制每个方法的雷达图
for method in methods:
    values = df.loc[method].values.flatten().tolist()
    values += values[:1] # 闭合
    ax.plot(angles, values, color=color_map[method], linewidth=2, linestyle='solid', label=method)
    ax.fill(angles, values, color=color_map[method], alpha=0.2)

# --- 4. 美化图表 ---
ax.set_rscale('symlog', linthresh=0.01)

tick_values = [0, 0.01, 0.1, 1.0]
ax.set_yticks(tick_values)
ax.set_yticklabels([str(val) for val in tick_values], color="grey", size=16)

# 设置Y轴的范围
ax.set_ylim(0, 1.2)

# 设置X轴（角度轴）的标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, size=18)

# 调整网格线
ax.grid(True, color="grey", linestyle="--", linewidth=0.5)

# # 添加图例
# ax.legend(loc='lower center', bbox_to_anchor=(1.3, 1.1), fontsize=12)

ax.set_title('Unseen Perturbation: Seed 567', size=22, color='black', y=1.15)

# --- 5. 保存图像 ---
import os
os.makedirs('figs/fig2', exist_ok=True)
plt.savefig('figs/fig2/fig2_task1_radar_567.svg', dpi=300, bbox_inches='tight')

plt.show()
