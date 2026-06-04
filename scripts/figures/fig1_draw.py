import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data prep ---
# fill in your data here
# format: 'method': (mean, error)
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

# --- 2. Plot order and colors ---
# cluster order on x-axis
x_labels = ['task1', 'task2', 'task3', 'task4']
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
# fig
fig, ax = plt.subplots(figsize=(16, 8))

# bar positions
n_methods = len(method_order)
n_labels = len(x_labels)
x = np.arange(n_labels) # x oneach 
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
ax.set_yscale('symlog', linthresh=0.01)
ax.set_ylabel('Pearson Correlation', fontsize=15)
# ax.set_title('methods across clustersPearsoncorrelation comparison', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=15)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.axhline(0, color='grey', linewidth=0.8)
ax.tick_params(axis='y', labelsize=12) # y-axis tick font size
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=len(method_order), fancybox=True, shadow=False, frameon=False, fontsize=12)
plt.tight_layout()

import os
os.makedirs('figs/fig1', exist_ok=True)
plt.savefig('figs/fig1/fig1.png', dpi=300, bbox_inches='tight')

