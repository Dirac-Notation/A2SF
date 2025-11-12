#!/usr/bin/env python3
"""
Script to generate accuracy plots from result.json files.
X-axis: snap parameter (n in snap(n), h2o=8192, snap=16)
Y-axis: accuracy from result.json files
"""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_snap_parameter(folder_name):
    """Extract snap parameter from folder name"""
    if "128" in folder_name:
        if 'h2o' in folder_name:
            return 8192
        elif 'snap' in folder_name:
            # Extract number after 'snap'
            match = re.search(r'snap(\d+)', folder_name)
            return int(match.group(1))
    return None

def extract_budget(folder_name):
    """Extract budget from folder name"""
    match = re.search(r'_(\d+)$', folder_name)
    if match:
        return int(match.group(1))
    return None

def load_result_data(result_dir):
    """Load all result data from the directory"""
    data = {}
    
    for folder in os.listdir(result_dir):
        folder_path = os.path.join(result_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        result_file = os.path.join(folder_path, 'result.json')
        if not os.path.exists(result_file):
            continue
            
        snap_param = extract_snap_parameter(folder)
        budget = extract_budget(folder)
        
        if snap_param is None or budget is None:
            continue
            
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                
            if budget not in data:
                data[budget] = {}
                
            data[budget][snap_param] = {
                'group_averages': result_data.get('group_averages', {}),
                'overall_average': result_data.get('overall_average', 0)
            }
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            
    return data

def create_plots(data, output_dir):
    """Create plots for each budget"""
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 23,
        'axes.labelsize': 23,
        'xtick.labelsize': 23,
        'ytick.labelsize': 23,
        'legend.fontsize': 23,
        'figure.titlesize': 23
    })
    
    for budget in sorted(data.keys()):
        budget_data = data[budget]
        
        # Sort snap parameters
        snap_params = sorted(budget_data.keys())
        
        # Get all group names and sort by snap1 values (descending)
        all_groups = set()
        for snap_data in budget_data.values():
            all_groups.update(snap_data['group_averages'].keys())
        all_groups = list(all_groups)
        
        # Sort groups by their values at snap1 (first snap parameter)
        if snap_params:
            snap1_values = {}
            for group in all_groups:
                # Find the first available snap parameter that has this group
                for snap_param in snap_params:
                    if group in budget_data[snap_param]['group_averages']:
                        snap1_values[group] = budget_data[snap_param]['group_averages'][group]
                        break
                # If group not found in any snap parameter, set to 0
                if group not in snap1_values:
                    snap1_values[group] = 0
            
            all_groups = sorted(all_groups, key=lambda x: snap1_values.get(x, 0), reverse=True)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Group averages
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_groups)))
        
        for i, group in enumerate(all_groups):
            group_values = []
            valid_x_positions = []
            
            for j, snap_param in enumerate(snap_params):
                if group in budget_data[snap_param]['group_averages']:
                    group_values.append(budget_data[snap_param]['group_averages'][group])
                    valid_x_positions.append(j)
            
            if group_values:
                # Plot the line and regular points
                plt.plot(valid_x_positions, group_values, 'o-', 
                        label=group, color=colors[i], linewidth=2, markersize=6)
                
                # Find and highlight the maximum value
                max_idx = group_values.index(max(group_values))
                max_x = valid_x_positions[max_idx]
                max_y = group_values[max_idx]
                
                # Highlight the maximum point with a larger, different marker
                plt.scatter(max_x, max_y, s=150, color=colors[i], 
                           marker='*', edgecolors='black', linewidth=1, zorder=5)
        
        plt.xlabel('Observation Window Size')
        plt.ylabel('Average Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Set x-axis to equal spacing with snap parameter labels
        x_positions = range(len(snap_params))
        plt.xticks(x_positions, ["TOVA", "SnapKV", "256", "512", "1024", "2048", "3072", "4096", "5120", "6144", "7168", "H2O"], rotation=90)
        
        # Adjust layout
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        
        # Save plot
        output_file = os.path.join(output_dir, f'accuracy_budget_{budget}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        
        plt.close()

def main():
    result_dir = '/home/smp9898/A2SF/result_txt/pred'
    output_dir = '/home/smp9898/A2SF/plots/acc'
    
    print("Loading result data...")
    data = load_result_data(result_dir)
    
    if not data:
        print("No data found!")
        return
    
    print(f"Found data for budgets: {sorted(data.keys())}")
    for budget in sorted(data.keys()):
        snap_params = sorted(data[budget].keys())
        print(f"  Budget {budget}: snap parameters {snap_params}")
    
    print("Creating plots...")
    create_plots(data, output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
