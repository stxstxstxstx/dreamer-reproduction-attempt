import json
import matplotlib.pyplot as plt
import numpy as np

def format_task_name(task_name):
    """Format task name from 'atari_pong' to 'Pong' or similar."""
    # Remove common prefixes
    for prefix in ['atari_', 'dmc_', 'procgen_']:
        if task_name.startswith(prefix):
            task_name = task_name[len(prefix):]
    
    # Capitalize first letter of each word separated by underscore
    task_name = ' '.join(word.capitalize() for word in task_name.split('_'))
    
    return task_name

# Read main JSON data file
file_path = 'dreamerv2-main/scores/repro_atari.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Read baseline data
baseline_path = 'dreamerv2-main/scores/baselines.json'
try:
    with open(baseline_path, 'r') as file:
        baseline_data = json.load(file)
except FileNotFoundError:
    print(f"Warning: Baseline file '{baseline_path}' not found, baselines will not be displayed.")
    baseline_data = {}

# Extract and deduplicate all task names
tasks = list(set(item['task'] for item in data))

# Calculate subplot layout (maximum 3 columns)
n_tasks = len(tasks)
n_cols = min(3, n_tasks)
n_rows = (n_tasks + n_cols - 1) // n_cols

# Create figure and subplots with appropriate size (add extra space at bottom for legend)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows + 1.5))  # Added 1.5 inches at bottom

# Convert axes to array if there's only one subplot
if n_tasks == 1:
    axes = np.array([axes])
else:
    axes = axes.flatten()

# Define color cycle with better aesthetics
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', "#2C2A2A", '#bcbd22', '#17becf'
]

# Define baseline styles (only keep iqn, rainbow, random)
baseline_styles = {
    'random': {'color': '#2C2A2A', 'linestyle': ':', 'linewidth': 1.5, 'zorder': 1},
    'iqn': {'color': 'purple', 'linestyle': '--', 'linewidth': 2, 'zorder': 2},
    'rainbow': {'color': 'orange', 'linestyle': '-.', 'linewidth': 2, 'zorder': 2},
}

# Baseline names to include (case-insensitive)
BASELINE_WHITELIST = {'iqn', 'rainbow', 'random'}

# Function to check if a baseline name should be included
def should_include_baseline(name):
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in BASELINE_WHITELIST)

# Function to get standardized baseline key
def get_standardized_key(name):
    name_lower = name.lower()
    if 'iqn' in name_lower:
        return 'iqn'
    elif 'rainbow' in name_lower:
        return 'rainbow'
    elif 'random' in name_lower:
        return 'random'
    return None

# Collect all legend handles and labels
all_handles = []
all_labels = []

# Iterate over each task
for i, task in enumerate(tasks):
    # Get current subplot
    ax = axes[i]
    
    # Filter data for current task
    task_data = [item for item in data if item['task'] == task]
    
    # Get baseline data for this task and filter
    task_baselines = {}
    for baseline_name, value in baseline_data.get(task, {}).items():
        if should_include_baseline(baseline_name):
            std_key = get_standardized_key(baseline_name)
            if std_key:
                # Use the first occurrence if multiple matches (e.g., iqn_sticky_2e8 and iqn_determ_2e8)
                if std_key not in task_baselines:
                    task_baselines[std_key] = value
    
    # Plot baselines (if any)
    for baseline_name, baseline_value in task_baselines.items():
        # Get style or use default
        style = baseline_styles.get(baseline_name, {'color': 'gray', 'linestyle': '--', 'linewidth': 1.5, 'zorder': 1})
        handle = ax.axhline(y=baseline_value, **style)
        # Add to legend collection if not already present
        label = baseline_name.capitalize()
        if label not in all_labels:
            all_handles.append(handle)
            all_labels.append(label)
    
    # Iterate over each method
    for j, item in enumerate(task_data):
        method = item['method']
        # Filter data up to 17 million steps
        filtered_xs = []
        filtered_ys = []
        for x, y in zip(item['xs'], item['ys']):
            if x <= 17000000:
                filtered_xs.append(x)
                filtered_ys.append(y)
        
        # Plot line with color cycling
        color = colors[j % len(colors)]
        handle, = ax.plot(filtered_xs, filtered_ys, color=color, linewidth=2, alpha=0.8)
        # Add to legend collection if not already present
        if method not in all_labels:
            all_handles.append(handle)
            all_labels.append(method)
    
    # Set subplot title with formatted task name
    formatted_task = format_task_name(task)
    ax.set_title(formatted_task, fontsize=12)
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('Return', fontsize=10)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis to millions
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6))

# Hide empty subplots
for i in range(n_tasks, len(axes)):
    axes[i].axis('off')

# Create a single legend at the bottom of the figure
fig.legend(all_handles, all_labels, 
           loc='lower center', 
           ncol=5,  # Adjust number of columns as needed
           fontsize=10,
           frameon=True,
           framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))  # Adjust vertical position as needed

# Adjust layout with padding (add more space at bottom for legend)
plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 1])  # rect=[left, bottom, right, top]

# Save figure (optional)
plt.savefig('results.png', dpi=300, bbox_inches='tight')

# Display figure
plt.show()