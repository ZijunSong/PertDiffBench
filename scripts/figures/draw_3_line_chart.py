import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FixedLocator, FixedFormatter

# --- 1. Prepare all data ---

data2 = {
    'mix2': {
        'scGen': (1.0000, 0.0000), 'scDiff': (0.7894, 0.0000), 'Squidiff': (0.0306, 0.0020),
        'scDiffusion': (0.0199, 0.0000), 'DDPM': (0.3089, 0.0360), 'DDPM+MLP': (0.0070, 0.0016)
    },
    'mix3': {
        'scGen': (1.0000, 0.0000), 'scDiff': (0.9674, 0.0000), 'Squidiff': (0.7753, 0.0013),
        'scDiffusion': (0.0395, 0.0016), 'DDPM': (0.3257, 0.0482), 'DDPM+MLP': (0.0352, 0.0028)
    },
    'mix4': {
        'scGen': (1.0000, 0.0000), 'scDiff': (0.9742, 0.0000), 'Squidiff': (0.0242, 0.0386),
        'scDiffusion': (0.0629, 0.0017), 'DDPM': (0.3415, 0.0362), 'DDPM+MLP': (0.0227, 0.0003)
    },
    'mix5': {
        'scGen': (1.0000, 0.0000), 'scDiff': (0.9829, 0.0000), 'Squidiff': (0.2877, 0.0055),
        'scDiffusion': (0.1280, 0.0071), 'DDPM': (0.4018, 0.1329), 'DDPM+MLP': (0.0182, 0.0019)
    },
    'mix6': {
        'scGen': (1.0000, 0.0000), 'scDiff': (0.9907, 0.0000), 'Squidiff': (0.2442, 0.0071),
        'scDiffusion': (0.0694, 0.0012), 'DDPM': (0.4154, 0.0951), 'DDPM+MLP': (0.0164, 0.0007)
    },
    'mix7': {
        'scGen': (1.0000, 0.0000), 'scDiff': (0.9918, 0.0000), 'Squidiff': (0.0074, 0.0151),
        'scDiffusion': (0.0856, 0.0025), 'DDPM': (0.4490, 0.1301), 'DDPM+MLP': (0.0097, 0.0032)
    },
}
x_labels2 = ['mix2', 'mix3', 'mix4', 'mix5', 'mix6', 'mix7']
method_order2 = ['scGen', 'scDiff', 'Squidiff', 'scDiffusion', 'DDPM', 'DDPM+MLP']

data3 = {
    '0.10': {
        'scGen': (0.8956, 0.0007), 'scDiff': (0.9269, 0.0000), 'Squidiff': (0.0163, 0.0132),
        'scDiffusion': (0.7247, 0.0031), 'DDPM': (0.0172, 0.0147), 'DDPM+MLP': (0.0146, 0.0032)
    },
    '0.25': {
        'scGen': (0.8873, 0.0021), 'scDiff': (0.9258, 0.0000), 'Squidiff': (0.0158, 0.0219),
        'scDiffusion': (0.6026, 0.0011), 'DDPM': (0.0175, 0.0014), 'DDPM+MLP': (0.0071, 0.0017)
    },
    '0.50': {
        'scGen': (0.7638, 0.0006), 'scDiff': (0.9212, 0.0000), 'Squidiff': (0.0096, 0.0041),
        'scDiffusion': (0.4885, 0.0011), 'DDPM': (0.0122, 0.0049), 'DDPM+MLP': (0.0011, 0.0092)
    },
    '1.00': {
        'scGen': (0.7714, 0.0000), 'scDiff': (0.8922, 0.0000), 'Squidiff': (0.0021, 0.0076),
        'scDiffusion': (0.3223, 0.0062), 'DDPM': (0.0052, 0.0166), 'DDPM+MLP': (0.0112, 0.0104)
    },
    '1.50': {
        'scGen': (0.3878, 0.0004), 'scDiff': (0.2877, 0.0000), 'Squidiff': (0.0037, 0.0084),
        'scDiffusion': (0.2307, 0.0004), 'DDPM': (0.0083, 0.0079), 'DDPM+MLP': (0.0033, 0.0087)
    },
}
x_labels3 = ['0.10', '0.25', '0.50', '1.00', '1.50']
method_order3 = ['scGen', 'scDiff', 'Squidiff', 'scDiffusion', 'DDPM+MLP', 'DDPM']

data1 = {
    '6998': {
        'scGen': (0.9919, 0.0007), 'scDiff': (0.0022, 0.0000), 'Squidiff': (0.0440, 0.0081),
        'scDiffusion': (0.9248, 0.0015), 'DDPM': (0.3663, 0.0559), 'DDPM+MLP': (0.0003, 0.0011)
    },
    '6000': {
        'scGen': (0.4129, 0.0065), 'scDiff': (0.0087, 0.0000), 'Squidiff': (0.0582, 0.0007),
        'scDiffusion': (0.3465, 0.0006), 'DDPM': (0.0058, 0.0090), 'DDPM+MLP': (0.0091, 0.0002)
    },
    '5000': {
        'scGen': (0.6747, 0.0005), 'scDiff': (0.0172, 0.0000), 'Squidiff': (0.0516, 0.0052),
        'scDiffusion': (0.4778, 0.0034), 'DDPM': (0.0124, 0.0107), 'DDPM+MLP': (0.0021, 0.0017)
    },
    '4000': {
        'scGen': (0.6958, 0.0136), 'scDiff': (0.0063, 0.0000), 'Squidiff': (0.3089, 0.0017),
        'scDiffusion': (0.4947, 0.0006), 'DDPM': (0.0193, 0.0014), 'DDPM+MLP': (0.0348, 0.0008)
    },
    '3000': {
        'scGen': (0.7323, 0.0149), 'scDiff': (0.0087, 0.0000), 'Squidiff': (0.0001, 0.0043),
        'scDiffusion': (0.5785, 0.0015), 'DDPM': (0.0017, 0.0246), 'DDPM+MLP': (0.0101, 0.0034)
    },
    '2000': {
        'scGen': (0.7676, 0.0102), 'scDiff': (0.0187, 0.0000), 'Squidiff': (0.2262, 0.0079),
        'scDiffusion': (0.5990, 0.0014), 'DDPM': (0.0086, 0.0116), 'DDPM+MLP': (0.0152, 0.0016)
    },
    '1000': {
        'scGen': (0.8330, 0.0021), 'scDiff': (0.0563, 0.0000), 'Squidiff': (0.0440, 0.0091),
        'scDiffusion': (0.6469, 0.0011), 'DDPM': (0.3663, 0.0559), 'DDPM+MLP': (0.0183, 0.0015)
    },
}
x_labels1 = ['6998', '6000', '5000', '4000', '3000', '2000', '1000']
method_order1 = ['scGen', 'scDiff', 'Squidiff', 'scDiffusion', 'DDPM+MLP', 'DDPM']


# --- 2. Unified plot order/colors ---
# : Ensureallfig name colormap 
method_order = ['scGen', 'scDiff', 'Squidiff', 'scDiffusion', 'DDPM', 'DDPM+MLP']
color_map = {
    'scGen': '#ff6f00ff',
    'scDiff': '#c71000ff',
    'Squidiff': '#0099a6ff',
    'scDiffusion': '#8a4198ff', # fig1 
    'DDPM': '#6bc7d5ff', # fig1 
    'DDPM+MLP': '#ff95a8ff'
}

# --- 3. Plot helpers ---
def plot_subplot(ax, data, x_labels, method_order, color_map, title, is_first_plot=False):
    """
    inspecify on fig.
    
    args:
    ax: Matplotlib object
    data: figdata
    x_labels: x labels
    method_order: 
    color_map: colormap
    title: fig 
    is_first_plot: whetherasfirst fig (for whether y )
    """
    x = np.arange(len(x_labels))
    
    for method in method_order:
        means = np.array([data[label][method][0] for label in x_labels])
        errors = np.array([data[label][method][1] for label in x_labels])
        
        ax.plot(x, means, marker='o', linestyle='-', label=method, color=color_map[method])
        ax.fill_between(x, means - errors, means + errors, color=color_map[method], alpha=0.2)
        
    # --- style plots ---
    ax.set_yscale('symlog', linthresh=0.1)
    tick_locations = [-0.01, 0, 0.01, 0.1, 1]
    ax.set_yticks(tick_locations)
    ax.yaxis.set_major_formatter(FixedFormatter([str(tick) for tick in tick_locations]))
    
    if is_first_plot:
        ax.set_ylabel('Pearson Correlation', fontsize=15)
        
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=15) # x to 
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(title, fontsize=16)


# --- 4. Plotting ---
# 1x3 fig , and fig 
# to three fig
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True) # sharey=True y 

# three fig
plot_subplot(axes[0], data1, x_labels1, method_order1, color_map, title="Task 3 Performance", is_first_plot=True)
plot_subplot(axes[1], data2, x_labels2, method_order2, color_map, title="Gaussian Noise Performance")
plot_subplot(axes[2], data3, x_labels3, method_order3, color_map, title="High Variable Genes Performance")

# --- 5. Shared legend ---
# fromfirst figgetfig 
handles, labels = axes[0].get_legend_handles_labels()
# infigon fig 
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=6, fancybox=True, shadow=False, frameon=False, fontsize=14)

# --- 6. Layout and save ---
# to fig 
plt.tight_layout(rect=[0, 0, 1, 0.92]) # rect=[left, bottom, right, top] asfig empty 

# Ensure directoryexist
os.makedirs('figs/fig_plus', exist_ok=True)
# andafter fig
plt.savefig('figs/fig_plus/combined_figure.svg', dpi=300, bbox_inches='tight')

plt.show() # in on fig 