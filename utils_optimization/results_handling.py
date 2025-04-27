import json
import os
from datetime import datetime
import torch
from utils_optimization.functions_to_construct_objective_function import lmfit_to_torch_values
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def save_results(params, residuals, run_id,results_dir,dtype,device):
    """
    Function to save the results of the optimization using lmfit package

    Args:
        params (dict): parameters to save
        residuals (list): history of residuals
        run_id (str): run id
        results_dir (str): results directory
        dtype (torch.dtype): data type
        device (torch.device): device

    """
    try:
        # Assigning base filename
        base_filename = os.path.join(results_dir, f"run_{run_id}")

        # Save parameters as torch.Tensor
        param_file = f"{base_filename}_params.pt"
        torch.save(lmfit_to_torch_values(params,dtype,device), param_file)

        # Save residuals history
        residual_file = f"{base_filename}_residuals.npy"
        np.save(residual_file, np.array(residuals))

        # Save metadata
        metadata = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'num_params': len(params),
            'final_residual': float(residuals[-1]) if residuals else None
        }

        with open(f"{base_filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False


def load_best_run(results_dir):
    """
    Load the best run from all saved results from given directory.

    Args:
        results_dir (str): results directory which contains all runs from specific configuration

    Returns:
        params (torch.Tensor): optimal parameters
        residuals (numpy.list): history of residuals
        best_run (str): best run id in given directory

    """
    # Check whether directory exist
    if not os.path.exists(results_dir):
        return None

    # Place-holder for the best objective value and id of best run
    best_residual = float('inf')
    best_run = None

    # Going through all files in the directory and checking residual value at the end of the run
    for filename in os.listdir(results_dir):
        if filename.endswith("_metadata.json"):
            run_id = filename.split('_')[1]  # Extract timestamp
            try:
                with open(os.path.join(results_dir, filename)) as f:
                    meta = json.load(f)
                if meta['final_residual'] < best_residual:
                    best_residual = meta['final_residual']
                    best_run = meta['run_id']
            except:
                continue

    # After completing the search we load optimal parameters and history of objective function values
    if best_run:
        params = torch.load(os.path.join(results_dir, f"run_{best_run}_params.pt"))
        residuals = np.load(os.path.join(results_dir, f"run_{best_run}_residuals.npy"))
        return params, residuals, best_run

    return None

def extract_final_residuals(results_dir):
    """
    Load all *.npy files and extract final residuals

    Args:
        results_dir (str): results directory which contains all runs from specific configuration

    Returns:
        residuals (numpy.list): returns list which contains final objective values after optimization from specific configuration of optimization run

    """
    final_residuals = []

    # Walk through all subdirectories
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".npy"):
                try:
                    residuals = np.load(os.path.join(root, file))
                    if len(residuals) > 0:
                        final_residuals.append(residuals[-1])  # We save last value of objective function from all runs in specific optimization configuration
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")

    return np.array(final_residuals)



def plot_kde_overlays(residual_lists, labels=None, colors=None, bw=0.3, alpha=0.5):
    """
    Plot KDE overlays for multiple lists of residuals.

    Args:
        residual_lists: List of lists or arrays, each containing residuals.
        labels: Optional list of labels for the plots.
        colors: Optional list of colors for each distribution.
        bw: Bandwidth method for KDE.
        alpha: Transparency level for fills.
    """
    plt.figure(figsize=(10, 6))

    for idx, residuals in enumerate(residual_lists):
        label = labels[idx] if labels and idx < len(labels) else f"Set {idx+1}"
        color = colors[idx] if colors and idx < len(colors) else None

        sns.kdeplot(
            residuals, bw_method=bw, fill=True,
            label=label, color=color, alpha=alpha, linewidth=2
        )

    plt.title("Kernel Density Estimate of Objective function final value", pad=20)
    plt.xlabel("Objective function value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.8)
    sns.despine()
    plt.tight_layout()
    plt.show()


