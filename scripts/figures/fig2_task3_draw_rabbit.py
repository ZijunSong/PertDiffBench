import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. Data prep ---
full_data = {
    'scGen': {
        'Pearson': 0.6557,
        'Pearson Delta': 0.2024, 
        'Pearson Delta DEG 20': 0.6321,
        'Pearson Delta DEG 50': 0.5982,
        'Pearson Delta DEG 100': 0.5830 
    },
    'Squidiff': {
        'Pearson': 0.1418,
        'Pearson Delta': 0.0330, 
        'Pearson Delta DEG 20': 0.3200,
        'Pearson Delta DEG 50': 0.1123,
        'Pearson Delta DEG 100': 0.0185  
    },
    'scDiffusion': {
        'Pearson': 0.5768,
        'Pearson Delta': 0.0897, 
        'Pearson Delta DEG 20': 0.4896,
        'Pearson Delta DEG 50': 0.2649,
        'Pearson Delta DEG 100': 0.1057  
    },
    'DDPM': {
        'Pearson': 0.0067,
        'Pearson Delta': 0.1082, 
        'Pearson Delta DEG 20': 0.0582,
        'Pearson Delta DEG 50': 0.3928,
        'Pearson Delta DEG 100': 0.4575  
    },
    'DDPM+MLP': {
        'Pearson': 0.0037,
        'Pearson Delta': 0.1107, 
        'Pearson Delta DEG 20': 0.0840,
        'Pearson Delta DEG 50': 0.4137,
        'Pearson Delta DEG 100': 0.4715  
    }
}

# DataFrame for plotting
df = pd.DataFrame(full_data).T
df_log = df.apply(lambda x: np.log10(x.clip(lower=1e-5)))

# method/metric names from DataFrame
methods = df.index.tolist()
metrics = df.columns.tolist()


# --- 2. Plot params ---
# one angle per metric
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # close radar

# colormap
color_map = {
    'scGen': '#ff6f00ff',
    'Squidiff': '#008ea0ff',
    'scDiffusion': '#8a4198ff',
    'DDPM': '#c71000ff',
    'DDPM+MLP': '#3d3b25ff'
}

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# radar per method
for method in methods:
    values = df.loc[method].values.flatten().tolist()
    values += values[:1]  # close
    ax.plot(angles, values, color=color_map[method], linewidth=2, linestyle='solid', label=method)
    ax.fill(angles, values, color=color_map[method], alpha=0.2)

# --- 4. Style plots ---
ax.set_rscale('symlog', linthresh=0.01)

tick_values = [0, 0.01, 0.1, 1.0]
ax.set_yticks(tick_values)
ax.set_yticklabels([str(val) for val in tick_values], color="grey", size=16)

# Y 
ax.set_ylim(0, 1.2)

# angular axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, size=18)

# 
ax.grid(True, color="grey", linestyle="--", linewidth=0.5)

# add 
ax.set_title('Cross Species Prediction (Mouse→Rabbit)', size=22, color='black', y=1.15)

# --- 5. Save figures ---
import os
os.makedirs('figs/fig2', exist_ok=True)
plt.savefig('figs/fig2/fig2_task3_radar_rabbit.svg', dpi=300, bbox_inches='tight')

plt.show()
