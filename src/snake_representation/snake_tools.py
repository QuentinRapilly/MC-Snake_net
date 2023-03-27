import torch
from .basis_function import compute_cubic_spline_weights, compute_exponential_spline_weights

def sample_contour(control_points : torch.tensor, nb_samples : int, M : int, basis_function : str = "cubic_spline", device = "cpu"):

    if nb_samples == 0 :
        return torch.zeros((0,2)).to(device)

    possible_basis_functions = ["exponential_spline", "cubic_spline"]

    assert basis_function in possible_basis_functions
    
    weigth_function = {"exponential_spline" : compute_exponential_spline_weights, "cubic_spline" : compute_cubic_spline_weights}

    with torch.no_grad():
        t = torch.unsqueeze(torch.linspace(0, 1-1/nb_samples, nb_samples),1)
        m = torch.unsqueeze(torch.arange(start=0, end=M),0)
        weigth = weigth_function[basis_function](t,m,M)[:,:,None].to(device)

    #print("weigth : {}, cp : {}".format(weigth.device, control_points.device))

    prod = torch.unsqueeze(control_points,0)*weigth

    samples = torch.sum(prod, dim=1)

    return samples


