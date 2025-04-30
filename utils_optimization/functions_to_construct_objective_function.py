import torch

from torch_tmm import Model, BaseLayer, BaseMaterial

def delta(wavelengths,dtype,device):
    """
    Creates Dirac delta matrix.

    Args:
        wavelengths (torch.Tensor): tensor containing wavelengths
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors

    Returns :
        torch.Tensor: Contains w x w matrix with Dirac delta at diagonal
    """
    # Number of wavelength points
    w = len(wavelengths)

    # Construction of the Delta matrix
    D_m=torch.eye(w,requires_grad=True,dtype=dtype,device=device)

    return D_m

def lorentzian(wavelengths,FWHM,dtype,device):
    """
    Create a square matrix with Lorentzian functions centered on its diagonal.

    Args:
        wavelengths (torch.Tensor): tensor containing wavelengths
        FWHM (float or torch.Tensor): Full width half maximum of Lorentzian distribution
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors

    Returns:
        torch.Tensor: Matrix with shape (size, size) with Lorentzian max value on diagonal
    """
    # Obtaining size from number of wavelengths points
    size = len(wavelengths)
    # Calculating eta parameters from full width half maximum of Lorentzian
    eta=FWHM/2

    # Create x coordinates (column positions)
    x = torch.arange(size, dtype=dtype,device=device)

    # Create x0 positions (row centers)
    x0 = torch.arange(size, dtype=dtype,device=device)

    # Making sure that tensors can be broadcasted to square matrix (size, 1) - (1, size) = (size, size)
    x = x.view(1, -1)  # shape (1, size)
    x0 = x0.view(-1, 1)  # shape (size, 1)

    # Compute Lorentzian function for all elements
    L_m = 1/torch.pi * (eta / ((x - x0)**2 + eta**2))

    # Normalizing the Lorentzian
    L_m=(L_m-L_m.min())/(L_m.max()-L_m.min())

    return L_m

def lorentzian_diff_eta(wavelengths,FWHM,dtype,device):
    """
    Create a square matrix with Lorentzian functions centered on its diagonal.
    This version of Lorentzian basis allows for tunable FWHM for specific function.

    Args:
        wavelengths (torch.Tensor): tensor containing wavelengths
        FWHM (float or torch.Tensor): FWHM parameters for Lorentzian distribution in [nm]
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors

    Returns:
        torch.Tensor: Matrix with shape (size, size) with Lorentzian max value on diagonal
    """

    # x spans number of wavelength points
    # x=torch.arange(len(wavelengths))
    # Reassigning for representing lorentzian in terms of [Nm]
    x=wavelengths
    # Calculating eta parameters from full width half maximum of Lorentzian
    #eta=FWHM/2
    # Now eta parameter would be expressed in [Nm]
    eta=(FWHM/2)*1e-9

    lorentzian=1/torch.pi*(eta[0]/((x-x[0])**2+eta[0]**2))
    lorentzian=(lorentzian-lorentzian.min())/(lorentzian.max()-lorentzian.min())
    L=lorentzian.reshape(-1,1)

    for i in range(1,len(x)):
        lorentzian=1/torch.pi*(eta[i]/((x-x[i])**2+eta[i]**2))
        lorentzian=(lorentzian-lorentzian.min())/(lorentzian.max()-lorentzian.min())
        lorentzian=lorentzian.reshape(-1,1)
        L=torch.cat((L,lorentzian),dim=1)

    return L

def lorentzian_sweep(wavelengths,alpha,n,dtype,device):
    """
    Create a square matrix with Lorentzian functions centered on its diagonal.
    This version of Lorentzian basis produce FWHM in a sweep fashion where alpha dictates .

    Args:
        wavelengths (torch.Tensor): tensor containing wavelengths
        alpha (float or torch.Tensor): parameter determining sweep of FWHM
        n (int): number of filter sockets in camera
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors

    Returns:
        torch.Tensor: Matrix with shape (size, size) with Lorentzian max value on diagonal
    """

    # x spans number of wavelength points
    x=wavelengths.to(dtype=dtype,device=device)

    # Sweep of the FWHM parameters
    FWHM=torch.linspace(50.0,alpha*50.0,n**2-1)#,dtype=dtype,device=device,requires_grad=True)

    # Calculating eta parameters from full width half maximum of Lorentzian
    # Now eta parameter would be expressed in [Nm]
    eta=(FWHM/2)*1e-9

    lorentzian=1/torch.pi*(eta[0]/((x-x[0])**2+eta[0]**2))
    lorentzian=(lorentzian-lorentzian.min())/(lorentzian.max()-lorentzian.min())
    L=lorentzian.reshape(-1,1)

    for i in range(1,len(x)):
        lorentzian=1/torch.pi*(eta[i]/((x-x[i])**2+eta[i]**2))
        lorentzian=(lorentzian-lorentzian.min())/(lorentzian.max()-lorentzian.min())
        lorentzian=lorentzian.reshape(-1,1)
        L=torch.cat((L,lorentzian),dim=1)

    return L

def lmfit_to_torch_values(parameters, dtype, device):
    """
    Convert a lmfit Parameters object into a single torch nn.Parameter object.

    Args:
        lmfit_params (Parameters): An lmfit.Parameters object containing parameter values
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors

    Returns:
        torch.nn.Parameter: A single tensor containing all parameters, with requires_grad set to True
    """
    # Assignment of lmfit_parameters to the torch.nn.Parameter
    return torch.nn.Parameter(torch.tensor([param.value for param in parameters.values()], dtype=dtype, device=device, requires_grad=True))

def generate_single_filter(aSi_thickness,Au_thickness,Ti_thickness,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device):
    """
    Generate individual filter response based on torch_tmm library.

    Args:
        aSi_thickness (torch.nn.Parameter): Thickness of amorphous silicon in [m]
        Au_thickness (torch.nn.Parameter): Thickness of gold in [m]
        Ti_thickness (torch.nn.Parameter): Thickness of tissue in [m]
        Au_mat (torch_tmm.BaseMaterial): Object determining gold material
        Ti_mat (torch_tmm.BaseMaterial): Object determining titanium material
        aSi_mat (torch_tmm.BaseMaterial): Object determining amorphous silicon material
        env (torch_tmm.BaseLayer): Object determining environment for simulation
        subs (torch_tmm.BaseLayer): Object determining substrate for simulation
        wavelengths (torch.Tensor): tensor containing wavelengths
        angles (torch.Tensor): tensor containing angles of incidence used in simulation
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors

    Returns:
        torch.Tensor: Transmission of one filter response in shape 1 x w

    """

    # Material layers creation with their specific thicknesses
    Au=BaseLayer(Au_mat,thickness=Au_thickness,LayerType='coh')
    Ti=BaseLayer(Ti_mat,thickness=Ti_thickness,LayerType='coh')
    aSi = BaseLayer(aSi_mat, thickness=aSi_thickness, LayerType='coh')

    # Model of the thin film geometry which looks like env -> aSi -> Au -> Ti -> substrate
    model = Model(env, [aSi,Au,Ti], subs, dtype, device)

    # Retrieving results from geometry simulation at 45 [deg] light incident
    results= model.evaluate(wavelengths, angles[45].reshape(1,))

    # Model of the substrate which will be used for correction of transmission results
    model_Si=Model(env,[],subs,dtype,device)
    results_Si=model_Si.evaluate(wavelengths,angles[45].reshape(1,))

    # Retrieving Transmission and Reflection of the substrate
    Reflection_Si=(results_Si.reflection('s')+results_Si.reflection('p'))/2
    Transmission_Si=(results_Si.transmission('s')+results_Si.transmission('p'))/2

    # Retrieving Transmission and Reflection of the geometry
    Reflection=(results.reflection('s')+results.reflection('p'))/2
    Transmission=(results.transmission('s')+results.transmission('p'))/2

    # Correction of transmission with respect to substrate behaviour
    output=(Transmission*Transmission_Si)/(1-Reflection*Reflection_Si)

    return  output.T

def filter_wheel(params,n,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device):

    """
    Generate matrix of encoder spectra.
    First 2*n entries represents individual responses on first and second wheel.
    Rest (n)**2-2*n-1 entries are combinations of them.

    Args:
        params (torch.Tensor): Consist of thickness values for each layer obtained from lmfit_to_torch_values()
        n (int): Number of slots in each filter wheel
        Au_mat (torch_tmm.BaseMaterial): Object determining material model of gold
        Ti_mat (torch_tmm.BaseMaterial): Object determining titanium material
        aSi_mat (torch_tmm.BaseMaterial): Object determining amorphous silicon material
        env (torch_tmm.BaseLayer): Object determining environment for simulation
        subs (torch_tmm.BaseLayer): Object determining substrate for simulation
        wavelengths (torch.Tensor): tensor containing wavelengths
        angles (torch.Tensor): tensor containing angles of incidence used in simulation
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors

    Returns:
        torch.Tensor : Psi matrix of encoder spectra of size N x w where N=n**2-1
     """

    # Number of filters per wheel
    num_filters = n - 1
    # First and second sets of transmissions
    total_transmissions = num_filters * 2

    # Creation of Psi matrix which contains at this point only one encoder transmission
    Psi=generate_single_filter(params[num_filters*4],params[num_filters*2],params[0],Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device)

    # Concatenating Psi matrix with new individual encoder spectra
    for i in range(1,total_transmissions):
        transmission=generate_single_filter(params[num_filters*4+i],params[num_filters*2+i],params[i],Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device)
        Psi=torch.cat([Psi,transmission],dim=0)

    # Concatenating Psi matrix with products of encoder transmission
    for j in range(num_filters):
        for k in range(num_filters):
            transmission_product=Psi[j,:]*Psi[k+num_filters,:]
            new_transmission=torch.reshape(transmission_product,(1,-1))
            Psi=torch.cat([Psi,new_transmission],dim=0)

    # Normalization of Psi matrix
    P_norm=(Psi-Psi.min())/(Psi.max()-Psi.min())

    return P_norm

def objective(params,wavelengths,n,Au_mat,Ti_mat,aSi_mat,env,subs,angles,dtype,device,FWHM):

    """
    Computes objective function in the following way:
        1. Generate matrix P N x w where N=n**2-1. Matrix contains products of encoder spectra.
        2. Generate Delta/Lorentzian matrix w x M where M is number of function
        3. Calculate coefficient matrix C=P@Lorentzian
        4. Solve linear least square problem with Lorentzian unknown
        5. Frobenius Norm of difference between Lorentzian_estimated_from_lstq - Lorentzian

    Args:
        params (torch.Tensor): Pytorch tensor obtained from lmfit_to_torch_values function containing all parameter thicknesses
        wavelengths (torch.Tensor): tensor containing wavelengths
        n (int): Number of slots in each filter wheel
        Au_mat (torch_tmm.BaseMaterial): Object determining material model of gold
        Ti_mat (torch_tmm.BaseMaterial): Object determining material model of titanium
        aSi_mat (torch_tmm.BaseMaterial): Object determining amorphous silicon material
        env (torch_tmm.BaseLayer): Object determining environment for simulation
        subs (torch_tmm.BaseLayer): Object determining substrate for simulation
        angles (torch.Tensor): tensor containing angles of incidence used in simulation
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors
        FWHM (float): FWHM of Lorentzian

    Returns:
        torch.Tensor: Frobenius Norm of difference between estimated Lorentzian and Lorentzian

    """

    # 1. Generation of the P matrix of dimensions N x w containing products of encoder spectra
    P=filter_wheel(params,n,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device)

    # 2. Construction of Lorentzian
    Lorentzian=lorentzian(wavelengths,FWHM,dtype,device)

    # 3. Coefficient matrix
    C=P@Lorentzian

    # 4. Approximating the Lorentzian
    Lorentzian_app=torch.linalg.lstsq(P,C,driver='gelsd')

    # 4.5 Normalization of the Lorentzian approximation
    Lorentzian_app_normalized=(Lorentzian_app[0]-Lorentzian_app[0].min())/(Lorentzian_app[0].max()-Lorentzian_app[0].min())

    # 5. Calculating Frobenius Norm of difference between approximated Lorentzian and original Lorentzian
    result=torch.norm(Lorentzian_app_normalized-Lorentzian,p='fro')

    return result

def objective_bandwidth(params,wavelengths,n,Au_mat,Ti_mat,aSi_mat,env,subs,angles,dtype,device,FWHM):

    """
    Computes objective function in the following way:
        1. Generate matrix P N x w where N=n**2-1. Matrix contains products of encoder spectra.
        2. Generate Delta/Lorentzian matrix w x M where M is number of function
        3. Calculate coefficient matrix C=P@Lorentzian
        4. Solve linear least square problem with Lorentzian unknown
        5. Frobenius Norm of difference between Lorentzian_estimated_from_lstq - Lorentzian

    Args:
        params (torch.Tensor): Pytorch tensor obtained from lmfit_to_torch_values function containing all parameter thicknesses
        wavelengths (torch.Tensor): tensor containing wavelengths
        n (int): Number of slots in each filter wheel
        Au_mat (torch_tmm.BaseMaterial): Object determining material model of gold
        Ti_mat (torch_tmm.BaseMaterial): Object determining material model of titanium
        aSi_mat (torch_tmm.BaseMaterial): Object determining amorphous silicon material
        env (torch_tmm.BaseLayer): Object determining environment for simulation
        subs (torch_tmm.BaseLayer): Object determining substrate for simulation
        angles (torch.Tensor): tensor containing angles of incidence used in simulation
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors
        FWHM (float): FWHM of Lorentzian

    Returns:
        torch.Tensor: Frobenius Norm of difference between estimated Lorentzian and Lorentzian

    """

    # 1. Generation of the P matrix of dimensions N x w containing products of encoder spectra
    P=filter_wheel(params,n,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device)

    # 2. Construction of Lorentzian
    Lorentzian=lorentzian_diff_eta(wavelengths,FWHM,dtype,device)

    # 3. Coefficient matrix
    C=P@Lorentzian

    # 4. Approximating the Lorentzian
    Lorentzian_app=torch.linalg.lstsq(P,C,driver='gelsd')

    # 4.5 Normalization of the Lorentzian approximation
    Lorentzian_app_normalized=(Lorentzian_app[0]-Lorentzian_app[0].min())/(Lorentzian_app[0].max()-Lorentzian_app[0].min())

    # 5. Calculating Frobenius Norm of difference between approximated Lorentzian and original Lorentzian
    result=torch.norm(Lorentzian_app_normalized-Lorentzian,p='fro')

    return result

def objective_bandwidth_sweep(params,wavelengths,n,Au_mat,Ti_mat,aSi_mat,env,subs,angles,dtype,device,alpha):

    """
    Computes objective function in the following way:
        1. Generate matrix P N x w where N=n**2-1. Matrix contains products of encoder spectra.
        2. Generate Delta/Lorentzian matrix w x M where M is number of function
        3. Calculate coefficient matrix C=P@Lorentzian
        4. Solve linear least square problem with Lorentzian unknown
        5. Frobenius Norm of difference between Lorentzian_estimated_from_lstq - Lorentzian

    Args:
        params (torch.Tensor): Pytorch tensor obtained from lmfit_to_torch_values function containing all parameter thicknesses
        wavelengths (torch.Tensor): tensor containing wavelengths
        n (int): Number of slots in each filter wheel
        Au_mat (torch_tmm.BaseMaterial): Object determining material model of gold
        Ti_mat (torch_tmm.BaseMaterial): Object determining material model of titanium
        aSi_mat (torch_tmm.BaseMaterial): Object determining amorphous silicon material
        env (torch_tmm.BaseLayer): Object determining environment for simulation
        subs (torch_tmm.BaseLayer): Object determining substrate for simulation
        angles (torch.Tensor): tensor containing angles of incidence used in simulation
        dtype (torch.dtype):  type of the tensors
        device (str or torch.device): target device for the tensors
        FWHM (float): FWHM of Lorentzian

    Returns:
        torch.Tensor: Frobenius Norm of difference between estimated Lorentzian and Lorentzian

    """

    # 1. Generation of the P matrix of dimensions N x w containing products of encoder spectra
    P=filter_wheel(params,n,Au_mat,Ti_mat,aSi_mat,env,subs,wavelengths,angles,dtype,device)

    # 2. Construction of Lorentzian
    Lorentzian=lorentzian_sweep(wavelengths,alpha,n,dtype,device)

    # 3. Coefficient matrix
    C=P@Lorentzian

    # 4. Approximating the Lorentzian
    Lorentzian_app=torch.linalg.lstsq(P,C,driver='gelsd')

    # 4.5 Normalization of the Lorentzian approximation
    Lorentzian_app_normalized=(Lorentzian_app[0]-Lorentzian_app[0].min())/(Lorentzian_app[0].max()-Lorentzian_app[0].min())

    # 5. Calculating Frobenius Norm of difference between approximated Lorentzian and original Lorentzian
    result=torch.norm(Lorentzian_app_normalized-Lorentzian,p='fro')

    return result