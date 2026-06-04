import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

# --- 1. Data prep ---
# fill in your data here
# format: 'method': (mean, error)
data = {
    'B': {
        'scGen': (0.9924, 0.0002),
        'scDiff': (0.8978, 0.0000),
        'Squidiff': (0.0213, 0.0061),
        'scDiffusion': (0.4019, 0.0054),
        'DDPM': (0.0358, 0.0565),
        'DDPM+MLP': (0.1069, 0.0347)
    },
    'CD4T': {
        'scGen': (0.9892, 0.0004),
        'scDiff': (0.9265, 0.0000),
        'Squidiff': (0.5057, 0.0053),
        'scDiffusion': (0.4531, 0.0017),
        'DDPM': (0.0148, 0.0426),
        'DDPM+MLP': (0.0116, 0.0032)
    },
    'CD8T': {
        'scGen': (0.8818, 0.0006),
        'scDiff': (0.8997, 0.0000),
        'Squidiff': (0.5148, 0.0582),
        'scDiffusion': (0.4003, 0.0052),
        'DDPM': (0.0495, 0.0387),
        'DDPM+MLP': (0.0080, 0.0018)  
    },
    'FCGR3A+Mono': {
        'scGen': (0.9622, 0.0009), 
        'scDiff': (0.8082, 0.0000), 
        'Squidiff': (0.5051, 0.0728),
        'scDiffusion': (0.4636, 0.0064), 
        'DDPM': (0.0303, 0.0421), 
        'DDPM+MLP': (0.0150, 0.0010)
    },
    'CD14+Mono': {
        'scGen': (0.9740, 0.0003), 
        'scDiff': (0.7770, 0.0000), 
        'Squidiff': (0.4188, 0.0323),
        'scDiffusion': (0.4388, 0.0046), 
        'DDPM': (0.0443, 0.0210), 
        'DDPM+MLP': (0.0013, 0.0002)
    },
    'NK': {
        'scGen': (0.8539, 0.0011), 
        'scDiff': (0.8578, 0.0000), 
        'Squidiff': (0.2974, 0.0090),
        'scDiffusion': (0.4373, 0.0103), 
        'DDPM': (0.0357, 0.0535), 
        'DDPM+MLP': (0.0234, 0.0030)
    },
    'Dendritic': {
        'scGen': (0.9606, 0.0019), 
        'scDiff': (0.8326, 0.0000), 
        'Squidiff': (0.3282, 0.0152),
        'scDiffusion': (0.3479, 0.0113), 
        'DDPM': (0.0108, 0.0269), 
        'DDPM+MLP': (0.0015, 0.0059)
    }
}

# --- 2. Plot order and colors ---
# cluster order on x-axis
x_labels = ['CD4T', 'FCGR3A+Mono', 'CD14+Mono', 'B', 'NK', 'CD8T', 'Dendritic']
# method order within cluster
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

# --- 3. Plotting ---
space_per_group_inches = 2.2 
margin_inches = 2.0 
n_labels = len(x_labels)
fig_width = n_labels * space_per_group_inches + margin_inches

fig, ax = plt.subplots(figsize=(fig_width, 8))

# bar positions
n_methods = len(method_order)
x = np.arange(n_labels)
width = 0.8 / n_methods  # bar width

# bar plot per method
for i, method in enumerate(method_order):
    # per-cluster mean and error
    means = [data[label][method][0] for label in x_labels]
    errors = [data[label][method][1] for label in x_labels]

    ax.bar(x - (n_methods/2 - 0.5 - i) * width, means, width,
           yerr=errors, 
           capsize=3, 
           label=method, 
           color=color_map[method], # color from colormap
           alpha=0.85)

# --- 4. Style plots ---
ax.set_yscale('symlog', linthresh=0.1)
tick_locations = [-0.01, 0, 0.01, 0.1, 1]
ax.set_yticks(tick_locations)
# symlog: set tick labels manually if needed
ax.yaxis.set_major_formatter(FixedFormatter([str(tick) for tick in tick_locations]))
ax.set_ylabel('Pearson Correlation', fontsize=15)
# ax.set_title('methods across clustersPearsoncorrelation comparison', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=15)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.axhline(0, color='grey', linewidth=0.8)
ax.tick_params(axis='y', labelsize=12) # y-axis tick font size
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#           ncol=len(method_order), fancybox=True, shadow=False, frameon=False, fontsize=12)
plt.tight_layout()

import os
os.makedirs('figs/fig1', exist_ok=True)
plt.savefig('figs/fig1/fig1_task1.png', dpi=300, bbox_inches='tight')

