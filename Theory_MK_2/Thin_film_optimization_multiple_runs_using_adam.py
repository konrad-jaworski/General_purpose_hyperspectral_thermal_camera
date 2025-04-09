from torch_tmm import Model, BaseLayer, BaseMaterial
from torch_tmm.dispersion import Constant_epsilon, Lorentz,Drude,LorentzComplete
import torch
from lmfit import Parameters
import random
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

dtype=torch.complex64
device=torch.device('cpu')

n=10 #number of filter slots on each wheel
N=n**2-1 #Total number if combinations of filters

wavelengths = torch.linspace(8000, 14000, 99)*1e-9 # in [m]
angles = torch.linspace(0, 89, 90)

#Defining dispersion models
env_disp = [Constant_epsilon(torch.nn.Parameter(torch.tensor(1,dtype=dtype,device=device)), dtype, device)]
subs_disp = [Constant_epsilon(torch.nn.Parameter(torch.tensor(3.41**2,dtype=dtype,device=device)), dtype, device)]

Au_disp=[Drude(A=torch.nn.Parameter(torch.tensor(1,dtype=dtype,device=device)),E0=torch.nn.Parameter(torch.tensor(80.92,dtype=dtype,device=device)),C=torch.nn.Parameter(torch.tensor(0.25,dtype=dtype,device=device)),dtype=dtype,device=device)]
Ti_disp=[Drude(A=torch.nn.Parameter(torch.tensor(1,dtype=dtype,device=device)),E0=torch.nn.Parameter(torch.tensor(10.06881,dtype=dtype,device=device)),C=torch.nn.Parameter(torch.tensor(0.15458,dtype=dtype,device=device)),device=device,dtype=dtype)]
aSi_disp=[LorentzComplete(A=torch.nn.Parameter(torch.tensor(0.002,dtype=dtype,device=device)),E0=torch.nn.Parameter(torch.tensor(0.08,dtype=dtype,device=device)),C=torch.nn.Parameter(torch.tensor(0.014,dtype=dtype,device=device)),Con=torch.nn.Parameter(torch.tensor(12,dtype=dtype,device=device)),dtype=dtype,device=device)]

# Defining materials
env_mat = BaseMaterial(env_disp,name = 'air',dtype= dtype,device= device)
subs_mat = BaseMaterial(subs_disp,name = 'c_Si',dtype= dtype, device=device)

Ti_mat=BaseMaterial(Ti_disp,name = 'Ti',dtype= dtype,device= device)
Au_mat=BaseMaterial(Au_disp,name = 'Au',dtype= dtype,device= device)
aSi_mat=BaseMaterial(aSi_disp,name = 'aSi',dtype= dtype,device= device)

# Defining layers of env and substrate
env=BaseLayer(env_mat,thickness=torch.nn.Parameter(torch.tensor(0,dtype=dtype,device=device)),LayerType='env')
subs=BaseLayer(subs_mat,thickness=torch.nn.Parameter(torch.tensor(0,dtype=dtype,device=device)),LayerType='subs')

def Dirac():
    """ Creates Dirac delta matrix.

    Parameters:
    ----------

    Returns :
    ---------
    Pytorch tensor
        Matrix contains w x M matrix with Dirac delta functions at different positions
    """

    w = len(wavelengths)  # Number of wavelength points (rows)
    M = n**2 - 1  # Number of Dirac delta positions (columns)

    row_indices = torch.arange(w)
    col_indices = torch.arange(M)

    # Create zeros matrix (leaf tensor with requires_grad=True)
    D_m = torch.zeros(w, M, dtype=torch.float32)  # Initially all zeros

    # Use of functional assignment without in-place modification
    D_m = D_m + torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=torch.ones(row_indices.size(0), dtype=torch.float32),
        size=(w, M)
    ).to_dense()

    return D_m

def lmfit_to_torch_values(Parameters, dtype=torch.float32, device='cpu'):
    """
    Convert a lmfit Parameters object into a single torch nn.Parameter object.

    Args:
        lmfit_params (Parameters): An lmfit.Parameters object containing parameter values.
        dtype (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
        device (str or torch.device, optional): Device where the tensors will be stored. Default is 'cpu'.

    Returns:
        torch.Tensor: A single tensor containing all parameters, with requires_grad set to True.
    """
    return torch.nn.Parameter(torch.tensor([param.value for param in Parameters.values()], dtype=dtype, device=device, requires_grad=True))

def generate_single_filter_expanded(aSi_thickness,Au_thickness,Ti_thickness):

    """Generate individual filter response.

    Parameters:
    ----------
    aSi_thickness :  torch.nn.Parameter
        Thickness of amorphous silicon

    Au_thickness :  torch.nn.Parameter
        Thickness of gold

    Ti_thickness :  torch.nn.Parameter
        Thickness of titanium

    Returns:
    --------
    Pytorch tensor : Transmission of one filter response in shape 1 x w

    """
    Au=BaseLayer(Au_mat,thickness=Au_thickness,LayerType='coh')
    Ti=BaseLayer(Ti_mat,thickness=Ti_thickness,LayerType='coh')
    aSi = BaseLayer(aSi_mat, thickness=aSi_thickness, LayerType='coh')

    # Model of the thin film geometry
    model = Model(env, [aSi,Au,Ti], subs, dtype, device)
    results= model.evaluate(wavelengths, angles[45].reshape(1,))

    # Model of the substrate
    model_Si=Model(env,[],subs,dtype,device)
    results_Si=model_Si.evaluate(wavelengths,angles[45].reshape(1,))

    # Transmissions and reflections of the substrate
    Reflection_Si=(results_Si.reflection('s')+results_Si.reflection('p'))/2
    Transmission_Si=(results_Si.transmission('s')+results_Si.transmission('p'))/2

    # Transmission and reflection of the filter
    Reflection=(results.reflection('s')+results.reflection('p'))/2
    Transmission=(results.transmission('s')+results.transmission('p'))/2

    output=(Transmission*Transmission_Si)/(1-Reflection*Reflection_Si)
    return  output.T

def filter_wheel_expanded(params):

    """ Generate geometry matrix of filters responses. First nine entries represents individual responses on one wheel and next nine entries represents second wheel. Rest of the entries are combinations of them.

    Parameters:
    ----------
    params : Pythorch tensor
        Consist of thickness values for each layer obtained from params2tensor() function

    Returns:
    -------
    G_m : Pytorch tensor
        Corresponds to Pytorch tensor which holds all possible combinations of filter responses in format of  N x w

     """

    num_filters = n - 1  # Number of individual filters per wheel
    total_transmissions = num_filters * 2  # First and second sets of transmissions

    # Generate single filter responses and remove extra dim to make them [99]
    transmission_list = [generate_single_filter_expanded(params[i+(3*(n-1))],params[i+(2*(n-1))],params[i]).squeeze(0) for i in range(num_filters)]
    transmission_list += [generate_single_filter_expanded(params[num_filters + i + (3*(n-1))],params[num_filters + i + (2*(n-1))],params[num_filters + i]).squeeze(0) for i in range(num_filters)]

    # Stack transmissions -> Shape [18, 99]
    transmissions = torch.stack(transmission_list)

    # Initialize G_m with individual transmissions
    G_m = transmissions.clone()  # Shape [18, 99]

    # Pre-allocate space for combinations
    new_combinations = []

    for i in range(num_filters):
        for j in range(num_filters, total_transmissions):
            if G_m.shape[0] < n ** 2 - 1:
                combination = transmissions[i] * transmissions[j]  # Element-wise multiplication
                new_combinations.append(combination)

    if new_combinations:
        # Stack and ensure correct shape
        new_combinations = torch.stack(new_combinations)
        G_m = torch.cat([G_m, new_combinations], dim=0)

    return G_m

def Objective_expanded(params):

    """Computes objective function residual.

    Objective function accepts Pytorch tensor and returns a scalar value also as a Pytorch tensor. This method allows for autograd functionality.

    1. Generate geometry matrix
    2. Generate Dirac delta matrix
    3. Calculate coefficient matrix
    4. Approximation of Dirac delta matrix from Coefficient matrix and Geometry matrix
    5. Frobenius norm of residual of Dirac matrices

    Parameters:
    ----------
        params : Pytorch tensor obtained from lmfit_to_torch_values function

    Returns:
    ---------
        Pytorch tensor
            Scalar value representing Frobenius norm

    """

    # 1. From params -> Geometry matrix
    G_m=filter_wheel_expanded(params) # Already transposed inside of method

    # 2. Construction of Dirac delta matrix
    D_m=Dirac()

    # 3. Coefficient matrix formulation
    C_m=torch.matmul(G_m,D_m)

    # 4. Approximating the Dirac delta matrix
    D_app=torch.linalg.lstsq(G_m,C_m,driver='gelsd')

    # 5. Calculating norm of residual of Dirac matrices
    r=torch.norm(D_app[0]-D_m,p='fro')

    return r

overall_best_loss = float('inf')
overall_best_params = None

for trial in tqdm(range(2000), desc='Total Trials', position=0):
    # Initialize NEW parameters for each trial
    params = Parameters()

    # Random parameter initialization for this trial
    for i in range((n - 1) * 2):
        params.add(f'Ti_filter_{i + 1}', value=random.uniform(1e-9, 1e-8), min=1e-9, max=1e-8)

    for i in range((n - 1) * 2):
        params.add(f'Au_filter_{i + 1}', value=random.uniform(1e-9, 1e-8), min=1e-9, max=1e-8)

    for i in range((n - 1) * 2):
        params.add(f'aSi_filter_{i + 1}', value=random.uniform(1e-6, 1e-5), min=1e-6, max=1e-5)

    # Convert to torch parameters
    torch_params = lmfit_to_torch_values(params)
    optimizer = optim.Adam([torch_params], lr=0.0001)

    trial_best_loss = float('inf')
    trial_best_params = None
    trial_losses = []

    # 500-epoch optimization loop
    for epoch in tqdm(range(500), desc=f'Trial {trial+1}', position=1, leave=False):
        loss = Objective_expanded(torch_params)
        current_loss = loss.item()
        trial_losses.append(current_loss)

        # Update trial best
        if current_loss < trial_best_loss:
            trial_best_loss = current_loss
            trial_best_params = torch_params.data.clone()

        # Standard optimization steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply constraints
        with torch.no_grad():
            torch_params.data[:len(torch_params)-18] = torch.clamp(
                torch_params.data[:len(torch_params)-18],
                1e-12, 1e-7
            )
            torch_params.data[-18:] = torch.clamp(
                torch_params.data[-18:],
                1000e-9, 10000e-9
            )

    # Save results for this trial
    # 1. Save final loss of this trial
    with open('trial_final_losses.txt', 'a') as f:
        f.write(f"{trial_losses[-1]}\n")

    # 2. Save trial's best parameters
    torch.save(trial_best_params, f'best_params_trial_{trial}.pt')

    # 3. Update overall best
    if trial_best_loss < overall_best_loss:
        overall_best_loss = trial_best_loss
        overall_best_params = trial_best_params.clone()

# Save the absolute best parameters from all trials
torch.save(overall_best_params, 'best_params_all_trials.pt')