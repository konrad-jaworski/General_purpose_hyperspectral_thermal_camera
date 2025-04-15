import torch
from torch_tmm import Model, BaseLayer, BaseMaterial

def Delta(wavelengths,n,dtype):

    """ Creates Dirac delta matrix.

    Parameters:
    ----------
    wavelengths : torch linspace of wavelengths we operate
    n : int which depicts number of slots on the filter wheel

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
    D_m = torch.zeros(w, M, dtype=dtype)  # Initially all zeros

    # Use of functional assignment without in-place modification
    D_m = D_m + torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=torch.ones(row_indices.size(0), dtype=dtype),
        size=(w, M)
    ).to_dense()

    return D_m

def lmfit_to_torch_values(Parameters, dtype, device):
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

def generate_single_filter_expanded(aSi_thickness,Au_thickness,Ti_thickness,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device):

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
    # Formulating layers for geometry
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

    # Correction of transmission based on the substrate transmission and reflection
    output=(Transmission*Transmission_Si)/(1-Reflection*Reflection_Si)
    return  output.T

def filter_wheel_expanded(params,n,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device):

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

    # Number of individual filters per wheel
    num_filters = n - 1
    # First and second sets of transmissions
    total_transmissions = num_filters * 2

    # Generate single filter responses
    transmission_list = [generate_single_filter_expanded(params[i+(3*(n-1))],params[i+(2*(n-1))],params[i],Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device).squeeze(0) for i in range(num_filters)]
    transmission_list += [generate_single_filter_expanded(params[num_filters + i + (3*(n-1))],params[num_filters + i + (2*(n-1))],params[num_filters + i],Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device).squeeze(0) for i in range(num_filters)]

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

def Objective_expanded(params,wavelengths,n,Au_mat,Ti_mat,aSi_mat,env,subs,angles,dtype,device):

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
    G_m=filter_wheel_expanded(params,n,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device) # Already transposed inside of method

    # 2. Construction of Dirac delta matrix
    D_m=Delta(wavelengths,n,dtype)

    # 3. Coefficient matrix formulation
    C_m=torch.matmul(G_m,D_m)

    # 4. Approximating the Dirac delta matrix
    D_app=torch.linalg.lstsq(G_m,C_m,driver='gelsd')

    # 4.5 Normalization of the Delta function approximation

    D_app_normalized=(D_app[0]-D_app[0].min())/(D_app[0].max()-D_app[0].min())

    # 5. Calculating norm of residual of Delta matrices
    r=torch.norm(D_app_normalized-D_m,p='fro')

    return r