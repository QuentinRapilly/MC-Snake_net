import torch
from basis_function import create_periodic_exponential_spline

def sample_contour(control_points : torch.tensor, nb_samples : int, M : int, basis_function : str = "exponential_spline"):

    possible_basis_functions = ["exponential_spline"]

    assert basis_function in possible_basis_functions

    if basis_function == "exponential_spline":
        fct = create_periodic_exponential_spline(M)

    with torch.no_grad():
        t = torch.unsqueeze(torch.linspace(0, 1-1/nb_samples, nb_samples),1)
        m = torch.unsqueeze(torch.range(start=0, end=M-1),0)
        weigth = fct(M*t-m)[None,:,:,None]

    samples = torch.sum(torch.unsqueeze(control_points,1)*weigth, dim=2)

    return samples


