import torch
from math import pi,cos, floor, ceil


def create_interval_mask(t, lower_bound, upper_bound):
    """
        Return a binary with 1 where lower_bound <= t < upper_bound and 0 elsewhere.
    """
    return (lower_bound <= t)*(t<upper_bound)*1.


### Tools for the exponential spline ###

def create_exponential_spline(M):

    def basis_function_t(t):
        abs_t = torch.abs(t)

        mask_0 = create_interval_mask(abs_t, 0, 0.5)
        values_0 = mask_0*(torch.cos(2*pi*abs_t/M)*cos(pi/M) - cos(2*pi/M))

        mask_1 = create_interval_mask(abs_t, 0.5, 1.5)
        values_1 = mask_1*(torch.sin(pi*(3/2-abs_t)/M)*torch.sin(pi*(3/2-abs_t)/M))
        
        phi_M_t = 1/(1-cos(2*pi/M))*(values_0 + values_1)

        return phi_M_t

    return basis_function_t


def create_periodic_exponential_spline(M):

    basis_fct_aux = create_exponential_spline(M)

    a = 3/2 
    # Pour comprendre le role de a, relire la these sur l'implem 
    # (globalement c'est la moitie du support de la fonction de base)

    n_min = floor(-a/M - (M-1)/M)
    n_max = ceil(1-a/M)

    # j'ai du aggrandir les bornes. Pourquoi, je ne sais pas ... à creuser

    def basis_periodic_function(t):
        phi_t = basis_fct_aux(t-M*n_min)
        for n in range(n_min+1,n_max+1):
            phi_t = phi_t + basis_fct_aux(t-M*n)
        return phi_t

    return basis_periodic_function


def compute_exponential_spline_weights(t,m,M):
    basis_function = create_periodic_exponential_spline(M)

    weights = basis_function(M*t-m)

    return weights



### Tools for the cubic spline ###

def create_cubic_spline():

    def basis_function(t):
        abs_t = torch.abs(t)

        mask_1 = create_interval_mask(abs_t, 0, 1.)
        values_1 = mask_1*(2/3 - abs_t**2 + 0.5*abs_t**3)

        mask_2 = create_interval_mask(abs_t, 1., 2.)
        values_2 = mask_2*((1/6)*(2-abs_t)**3)

        phi = values_1 + values_2

        return phi
    
    return basis_function

def create_periodic_cubic_spline(M):

    basis_fct_aux = create_cubic_spline()

    n_min = floor(-2/M)
    n_max = ceil(3/M)

    def basis_periodic_function(t):
        phi_t = basis_fct_aux(t-M*n_min)
        for n in range(n_min, n_max+1):
            phi_t = phi_t + basis_fct_aux(t-M*n)
        return phi_t

    return basis_periodic_function

def compute_cubic_spline_weights(t,m,M):
    
    basis_function = create_periodic_cubic_spline(M)

    weights = basis_function(t-m)

    return weights
