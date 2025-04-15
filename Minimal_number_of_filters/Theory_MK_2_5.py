from lmfit import Parameters,Minimizer
import random
import torch
from tqdm import tqdm
from datetime import datetime
import os
from tqdm import tqdm
from utils_optimization.Set_up_dispersion_and_materials import set_up_dispersion_and_materials
from utils_optimization.functions_to_construct_objective_function import *
from utils_optimization.results_handling import save_results

dtype=torch.float64
device=torch.device('cpu')

angles = torch.linspace(0, 89, 90)
Ti_mat,Au_mat,aSi_mat,env,subs=set_up_dispersion_and_materials(dtype,device)


def Objective_wrapper(params):
    """ Lmfit wrapper of the Pythorch objective function.
    It takes as input parameters object of lmfit and outputs scalar value of numpy type.
    All autograd functionality is preserved and calculated inside as well as Jacobian and Hessian.

     Parameters : lmfit.Parameters object

     Returns : numpy scalar value

     """
    param_tensor = lmfit_to_torch_values(params,dtype,device) # Converting to Pytorch tensor

    global Jacobian, Hessian, Residuals_over

    # Jacobian and Hessian calculations
    #Jacobian=torch.autograd.functional.jacobian(Objective_expanded,param_tensor).detach().numpy()
    #Hessian=torch.autograd.functional.hessian(Objective_expanded,param_tensor).detach().numpy()

    # Residual from true objective function
    residual_value=Objective_expanded(param_tensor,wavelengths,n,Au_mat,Ti_mat,aSi_mat,env,subs,angles,dtype,device)

    # Detaching residual from computational graph and appending it to history of residuals
    residual_value = residual_value.detach().numpy()
    Residuals_over.append(residual_value)

    return residual_value


# loop for each of the number of filter slots
for k in tqdm(range(8,11),desc="Optimization process"):
    # loop for each of the configuration 1000 tries
    for j in range(1500):

        # number of filter slots on each wheel
        n=k
        # Total number if combinations of filters
        N=n**2-1
        wavelengths = torch.linspace(8000, 14000, N)*1e-9 # in [m] as in torch_tmm library

        results_dir = os.path.join(os.getcwd(), f"optimization_runs_with_{k}_number_of_filter_slots")
        os.makedirs(results_dir, exist_ok=True)
        run_id = datetime.now().strftime(f"_{j+1}")

        # Setting up initial parameters for the geometry parameters
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

        Jacobian = []
        Hessian = []
        Residuals_over = []

        minimizer = Minimizer(Objective_wrapper, params)
        result = minimizer.minimize(method='least_squares')

        save_success = save_results(result.params, Residuals_over, run_id,results_dir,dtype,device)

        if not save_success:
            print("Warning: Failed to save optimization results")
        # else:
        #      print(f"Optimization complete. Results saved with ID: {run_id} |||| Optimization counter: {j}")