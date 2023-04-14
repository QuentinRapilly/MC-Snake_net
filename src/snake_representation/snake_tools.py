import torch
from .basis_function import compute_cubic_spline_weights, compute_exponential_spline_weights

def sample_contour(control_points : torch.tensor, nb_samples : int, basis_function : str = "exponential_spline", device = "cpu"):

    if nb_samples == 0 :
        return torch.zeros((0,2)).to(device)

    possible_basis_functions = ["exponential_spline", "cubic_spline"]

    assert basis_function in possible_basis_functions
    
    M = control_points.shape[0]
    
    weigth_function = {"exponential_spline" : compute_exponential_spline_weights, "cubic_spline" : compute_cubic_spline_weights}

    with torch.no_grad():
        t = torch.unsqueeze(torch.linspace(0, 1-1/nb_samples, nb_samples),1)
        m = torch.unsqueeze(torch.arange(start=0, end=M),0)
        weigth = weigth_function[basis_function](t,m,M)[:,:,None].to(device)

    #print("weigth : {}, cp : {}".format(weigth.device, control_points.device))

    prod = torch.unsqueeze(control_points,0)*weigth

    samples = torch.sum(prod, dim=1)

    return samples


def polar_to_cartesian_cp(c : torch.Tensor, r : torch.Tensor, theta : torch.Tensor) -> torch.Tensor :

    theta = (theta*2*torch.pi)/torch.sum(theta, dim=-1, keepdim=True)
    print(theta)
    theta = torch.cumsum(theta, dim=-1)
    print(f"r : {r.shape}, theta : {torch.cos(theta).shape}, c : {c[...,0].shape}")
    x = r*torch.cos(theta) + c[...,0][...,None]
    y = r*torch.sin(theta) + c[...,1][...,None]
    cp = torch.stack((x, y)).permute(1,2,0)

    return(cp)


def sample_circle(M : int = 6, r : float =0.3) -> torch.Tensor:

    theta = -2*torch.arange(0, M, step = 1)*torch.pi/M
    x = r*torch.cos(theta) + 0.5
    y = r*torch.sin(theta) + 0.5

    cp = torch.stack((x, y)).T

    return cp
