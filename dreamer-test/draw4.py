import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(file_path):
    """Load jsonl file and return list of dictionaries"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_data(method_dirs, max_step=1e6):
    """Extract step and test/return data from method directories"""
    method_data = {}
    for method_dir in method_dirs:
        method_name = os.path.basename(os.path.dirname(method_dir))
        jsonl_path = os.path.join(method_dir, "metrics.jsonl")
        if os.path.exists(jsonl_path):
            data = load_jsonl(jsonl_path)
            steps = []
            returns = []
            for item in data:
                if "step" in item and "test/return" in item:
                    step = item["step"]
                    if step <= max_step:
                        steps.append(step)
                        returns.append(item["test/return"])
            method_data[method_name] = {
                "steps": steps,
                "returns": returns
            }
    return method_data


def load_baselines(baseline_path):
    """Load baselines data"""
    try:
        with open(baseline_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Baseline file '{baseline_path}' not found")
        return {}


def format_task_name(task_name):
    """Format task name from 'dmc_hopper_hop' to 'Hopper Hop'"""
    # Remove prefix
    for prefix in ['dmc_', 'atari_', 'procgen_']:
        if task_name.startswith(prefix):
            task_name = task_name[len(prefix):]
    
    # Capitalize each word
    task_name = ' '.join(word.capitalize() for word in task_name.split('_'))
    return task_name


def format_baseline_name(baseline_name):
    """Format baseline names for display in legend"""
    name_map = {
        'd4pg_100m': 'D4PG (1e8 steps)',
        'a3c_100m_proprio': 'A3C (1e8 steps, proprio)',
        # Add more mappings as needed
    }
    return name_map.get(baseline_name, baseline_name)


def plot_comparison(method_data, baselines, task_name, save_path=None):
    """Plot comparison with baselines"""
    plt.figure(figsize=(8, 8))
    colors = ['#377eb8', '#4daf4a', '#984ea3']  # Main method colors
    baseline_colors = ["#ff7f00", "#e41a1c", "#8c564b"]  # Baseline colors
    
    # Method name mapping
    name_mapping = {
        "dreamer": "Dreamer",
        "dreamer_no_value": "No value",
    }
    
    # Plot main methods
    for i, (method_name, data) in enumerate(method_data.items()):
        display_name = name_mapping.get(method_name, method_name)
        plt.plot(data["steps"], data["returns"], 
                label=display_name, 
                color=colors[i % len(colors)], 
                linewidth=2)

    # Plot baselines
    task_baselines = baselines.get(task_name, {})
    if task_baselines:
        for i, (baseline_name, baseline_value) in enumerate(task_baselines.items()):
            display_name = format_baseline_name(baseline_name)
            plt.axhline(y=baseline_value, 
                       color=baseline_colors[i % len(baseline_colors)],
                       linestyle='--',
                       linewidth=1.5,
                       alpha=0.7,
                       label=display_name)

    # Format plot
    formatted_task = format_task_name(task_name)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Test Return", fontsize=12)
    plt.title(f"{formatted_task}", fontsize=14)
    
    # Legend below plot
    plt.legend(bbox_to_anchor=(0.5, -0.1), 
              loc='upper center', 
              ncol=5,
              frameon=True,
              fontsize=10)
    
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save or show
    if save_path is None:
        save_path = f"{task_name}_comparison2.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Path configuration
    base_dir = "/root/shared-nvme/logdir2/dmc_hopper_hop"
    method_dirs = [
        os.path.join(base_dir, "dreamer/1"),
        os.path.join(base_dir, "dreamer_no_value/1"),
    ]
    
    # Load data
    baseline_path = "dreamer-master/scores/baselines.json"
    baselines = load_baselines(baseline_path)
    task_name = os.path.basename(base_dir)
    method_data = extract_data(method_dirs, max_step=4100000)
    
    # Generate plot
    plot_comparison(method_data, baselines, task_name)