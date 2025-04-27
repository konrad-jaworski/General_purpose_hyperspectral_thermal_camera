from lmfit import Parameters,Minimizer
import random
import torch
from datetime import datetime
import os
from tqdm import tqdm
from utils_optimization.Set_up_dispersion_and_materials import set_up_dispersion_and_materials
from utils_optimization.functions_to_construct_objective_function import *
from utils_optimization.results_handling import save_results


dtype=torch.float64
device = torch.device("cpu")

# Initialization of initial conditions
n=10 # Number of slots for filters in each of the wheel
# Range of wavelengths points
wavelengths=torch.linspace(8000,14000,n**2-1)*1e-9 # Expressed in [m] as tmm_torch requires
# Angles of incidence
angles = torch.linspace(0, 89, 90)

# Setting up material and dispersion models for simulation
Ti_mat,Au_mat,aSi_mat,env,subs=set_up_dispersion_and_materials(dtype,device)

# Definition of wrapper function for lmfit optimizer
def Objective_wrapper(params):
    """
    Lmfit wrapper of torch objective function.

     Args:
         params (lmfit.Parameters): parameters of thickness in lmfit compatible format

     Returns:
         residual_value (float): value obtained from torch defined objective function which was detached and projected to numpy format

     """
    # Converting lmfit.Parameters to torch.nn.Parameter
    param_tensor = lmfit_to_torch_values(params,dtype,device)

    # Global variables used for capturing value of objective function with each iteration
    global  Residuals_over#,Jacobian, Hessian

    # Jacobian and Hessian calculations
    #Jacobian=torch.autograd.functional.jacobian(Objective_expanded,param_tensor).detach().numpy()
    #Hessian=torch.autograd.functional.hessian(Objective_expanded,param_tensor).detach().numpy()

    # Residual from true objective function
    residual_value=objective_bandwidth(param_tensor[:(n-1)*6],wavelengths,n,Au_mat,Ti_mat,aSi_mat,env,subs,angles,dtype,device,param_tensor[(n-1)*6:])

    # Detaching residual from computational graph and appending it to history of residuals
    residual_value = residual_value.detach().numpy()
    Residuals_over.append(residual_value)

    return residual_value

# Main loop for investigating the bandwidth effect
for k in tqdm(range(300)):
    results_dir = os.path.join(os.getcwd(), f"optimization_{(n-1)*2}_with_same_initial_bandwidth_lsq")
    os.makedirs(results_dir, exist_ok=True)
    run_id = f"{k+1}"

    # Defining parameters for the lmfit optimizer
    params = Parameters()
    # Thickness of the Ti
    for i in range((n - 1) * 2):
        params.add(f'Ti_filter_{i + 1}', value=random.uniform(1e-9, 10e-9), min=1e-9, max=10e-9)

    # Thickness of the Au
    for i in range((n - 1) * 2):
        params.add(f'Au_filter_{i + 1}', value=random.uniform(1e-9, 10e-9), min=1e-9, max=10e-9)

    # Thickness of the aSi
    for i in range((n - 1) * 2):
        params.add(f'aSi_filter_{i + 1}', value=random.uniform(1000e-9, 10000e-9), min=1000e-9, max=10000e-9)

    # Bandwidth of Lorentzian
    for i in range(n**2-1):
        params.add(f'FWHM_{i+1}', value=0.6, min=0.2, max=20)

    # List which stores objective function value over the iterations
    Residuals_over = []

    minimizer = Minimizer(Objective_wrapper, params)
    result = minimizer.minimize(method='least_squares')

    save_success = save_results(result.params, Residuals_over, run_id,results_dir,dtype,device)

    if not save_success:
        print("Warning: Failed to save optimization results")