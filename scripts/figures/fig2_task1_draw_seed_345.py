import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. Data prep ---
full_data = {
    'scGen': {
        'Pearson': 0.9979,
        'Pearson Delta': 0.6161, 
        'Pearson Delta DEG 20': 0.8422,
        'Pearson Delta DEG 50': 0.7683,
        'Pearson Delta DEG 100': 0.7416 
    },
    'Squidiff': {
        'Pearson': 0.3419,
        'Pearson Delta': 0.0378, 
        'Pearson Delta DEG 20': 0.0980,
        'Pearson Delta DEG 50': 0.1910,
        'Pearson Delta DEG 100': 0.1952 
    },
    'scDiffusion': {
        'Pearson': 0.9610,
        'Pearson Delta': 0.5218, 
        'Pearson Delta DEG 20': 0.6119,
        'Pearson Delta DEG 50': 0.4088,
        'Pearson Delta DEG 100': 0.5083  
    },
    'DDPM': {
        'Pearson': 0.0052,
        'Pearson Delta': 0.5251, 
        'Pearson Delta DEG 20': 0.8928,
        'Pearson Delta DEG 50': 0.8211,
        'Pearson Delta DEG 100': 0.7033 
    },
    'DDPM+MLP': {
        'Pearson': 0.0659,
        'Pearson Delta': 0.5835, 
        'Pearson Delta DEG 20': 0.9258,
        'Pearson Delta DEG 50': 0.8241,
        'Pearson Delta DEG 100': 0.7439  
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

# add legend
# ax.legend(loc='lower center', bbox_to_anchor=(1.3, 1.1), fontsize=12)

ax.set_title('Unseen Perturbation: Seed 345', size=22, color='black', y=1.15)

# --- 5. Save figures ---
import os
os.makedirs('figs/fig2', exist_ok=True)
plt.savefig('figs/fig2/fig2_task1_radar_345.svg', dpi=300, bbox_inches='tight')

plt.show()
