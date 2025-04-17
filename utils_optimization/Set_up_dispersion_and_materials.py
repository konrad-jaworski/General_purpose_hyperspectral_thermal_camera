from torch_tmm import Model, BaseLayer, BaseMaterial
from torch_tmm.dispersion import Constant_epsilon,Drude,LorentzComplete
import torch

def set_up_dispersion_and_materials(dtype,device):
    """
    Function used to set up dispersion and material models for:
        - environment,
        - substrate,
        - Au,
        - Ti,
        - aSi.
    It's also set up environment and substrate as layers used further for geometry simulations.


    Args:
        dtype: torch tensor depicting dtype
        device: torch tensor depicting device

    Returns:
        Ti_mat (torch_mm.BaseMaterial): Determine material properties of the Ti
        Au_mat (torch_mm.BaseMaterial): Determine material properties of the Au
        aSi_mat (torch_mm.BaseMaterial): Determine material properties of the aSi
        env (torch_mm.BaseLayer): Determine environment as layer type object for simulation properties
        subs (torch_mm.BaseLayer): Determine substrate as layer type object for simulation properties
    """

    # Setting up dispersion models for environment and substrate
    env_disp = [Constant_epsilon(torch.nn.Parameter(torch.tensor(1,dtype=dtype,device=device)), dtype, device)]
    subs_disp = [Constant_epsilon(torch.nn.Parameter(torch.tensor(3.41**2,dtype=dtype,device=device)), dtype, device)]

    # Setting up dispersion models for Au, Ti and aSi
    Au_disp=[Drude(A=torch.nn.Parameter(torch.tensor(1,dtype=dtype,device=device)),E0=torch.nn.Parameter(torch.tensor(80.92,dtype=dtype,device=device)),C=torch.nn.Parameter(torch.tensor(0.25,dtype=dtype,device=device)),dtype=dtype,device=device)]
    Ti_disp=[Drude(A=torch.nn.Parameter(torch.tensor(1,dtype=dtype,device=device)),E0=torch.nn.Parameter(torch.tensor(10.06881,dtype=dtype,device=device)),C=torch.nn.Parameter(torch.tensor(0.15458,dtype=dtype,device=device)),device=device,dtype=dtype)]
    aSi_disp=[LorentzComplete(A=torch.nn.Parameter(torch.tensor(0.002,dtype=dtype,device=device)),E0=torch.nn.Parameter(torch.tensor(0.08,dtype=dtype,device=device)),C=torch.nn.Parameter(torch.tensor(0.014,dtype=dtype,device=device)),Con=torch.nn.Parameter(torch.tensor(12,dtype=dtype,device=device)),dtype=dtype,device=device)]

    # Setting up material models for environments and substrate
    env_mat = BaseMaterial(env_disp,name = 'air',dtype= dtype,device= device)
    subs_mat = BaseMaterial(subs_disp,name = 'c_Si',dtype= dtype, device=device)

    # Setting up material models for the Ti, Au and aSi
    Ti_mat=BaseMaterial(Ti_disp,name = 'Ti',dtype= dtype,device= device)
    Au_mat=BaseMaterial(Au_disp,name = 'Au',dtype= dtype,device= device)
    aSi_mat=BaseMaterial(aSi_disp,name = 'aSi',dtype= dtype,device= device)


    # Setting up environment and substrate as layers used for geometry simulation
    env=BaseLayer(env_mat,thickness=torch.nn.Parameter(torch.tensor(0,dtype=dtype,device=device)),LayerType='env')
    subs=BaseLayer(subs_mat,thickness=torch.nn.Parameter(torch.tensor(0,dtype=dtype,device=device)),LayerType='subs')

    return Ti_mat,Au_mat,aSi_mat,env,subs

