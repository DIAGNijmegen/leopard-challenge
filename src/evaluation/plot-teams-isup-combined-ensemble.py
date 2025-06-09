import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_dataset(ax, dataset, dataset_name, team_scores, isup_scores, isup_pathologist_score, ensemble_score, title):
    # Plot team predictions (transparent blue squares)
    ax.scatter(team_scores, [1] * len(team_scores), color='blue', alpha=0.3, s=150, marker='s', label='AI Models C-index')
    
    # Plot average team prediction with std error bar
    mean_team = np.mean(team_scores)
    std_team = np.std(team_scores)
    ax.errorbar(mean_team, 0.8, xerr=std_team, fmt='s', color='blue', alpha=0.5, markersize=11, capsize=10, label='Avg AI Model C-index ± Std Dev')
    
    # Plot team combined ISUP predictions (transparent red squares)
    ax.scatter(isup_scores, [0.6] * len(isup_scores), color='red', alpha=0.3, s=150, marker='s', label='AI Models Combined with ISUP C-index')
    
    # Plot average team combined ISUP with std error bar
    mean_isup = np.mean(isup_scores)
    std_isup = np.std(isup_scores)
    ax.errorbar(mean_isup, 0.4, xerr=std_isup, fmt='s', color='red', alpha=0.5, markersize=11, capsize=10, label='Avg AI Models Combined with ISUP ± Std Dev')
    
    # Plot ISUP pathologist c-index as vertical grey dashed line
    ax.axvline(isup_pathologist_score, color='grey', linestyle='--', label='ISUP C-index', linewidth=3)
    
    # Plot Ensemble c-index as grey solid vertical line
    ax.axvline(ensemble_score, color='grey', linestyle='-', label='AI Models Ensemble C-index', linewidth=3)
    
    ax.set_yticks([])
    ax.set_xlim(0.5, 0.8)
    ax.set_title(title, fontsize=18)  # Increased title font size
    ax.set_xlabel("C-index", fontsize=15)  # Increased x-axis label font size
    ax.tick_params(axis='x', labelsize=12)  # Increased x-tick label size


def main(config_path):
    # Load data from CSV files
    config = load_config(config_path)
    
    team_predictions = pd.read_csv(os.path.join(config["output_dir"], "c_index_results.csv"))
    team_combined_isup = pd.read_csv(os.path.join(config["output_dir"], "c_index_model_isup_combined_results.csv"))
    pathologist_isup = pd.read_csv(os.path.join(config["output_dir"], "c_index_isup_results.csv"))
    team_ensemble = pd.read_csv(os.path.join(config["output_dir"], "c_index_ensemble_results.csv"))

    datasets = ["radboud", "plco", "brazil", "cologne"]
    dataset_names = config["dataset_names"]
    
    # Set global font size (double the default, e.g., if default is ~10, set to 20)
    plt.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 18,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Create 4 plots, one for each dataset
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))  # Increased figure size slightly for better spacing
    axs = axs.flatten()
    
    for i, dataset in enumerate(datasets):
        dataset_name = dataset_names[dataset]
        
        # Extract scores
        team_scores = team_predictions[dataset_name + '_c_index'].dropna().values
        isup_scores = team_combined_isup[dataset_name].dropna().values
        isup_pathologist_score = pathologist_isup[pathologist_isup["dataset"] == dataset]["c_index"].item()
        ensemble_score = team_ensemble[team_ensemble["dataset"] == dataset]["ensemble_c_index"].item()
        
        # Plot each dataset in separate subplot
        plot_dataset(
            axs[i],
            dataset,
            dataset_name,
            team_scores,
            isup_scores,
            isup_pathologist_score,
            ensemble_score,
            dataset_name
        )
    
    # Collect handles and labels from the first axis (assuming all axes have the same labels)
    handles, labels = axs[0].get_legend_handles_labels()

    # Adjust layout so legend fits
    fig.tight_layout(rect=[0, 0.05, 1, 0.75])  # Adjusted to leave more space at bottom for legend

    # Add shared legend below all subplots with larger font and double spacing
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=3,
        fontsize=15,  # Doubled legend font size
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),  # Adjusted for better positioning
        labelspacing=1.5  # Double line spacing
    )

    # Save and show plot
    plt.savefig(os.path.join(config["output_dir"], "detailed_c_index_plots.png"), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main("/data/temporary/leopard/source/evaluation/pathology-leopard-evaluation/config/config.yaml")
