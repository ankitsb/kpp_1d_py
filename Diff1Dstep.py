import numpy as np

def diff1Dstep(Fp, kappa, gam, Hz, Hzw, FFlux, BForce, Nz, dt):
    """
    DIFF1Dstep
    This function advances the field Fp (F previous) by one time step.
    This is done using a flux formulation where first the diffusive fluxes are calculated 
    using the diffusivity kappa, then forcing fluxes are added (FFlux). There is also the option 
    to add a body force on the RHS (BForce).

    Parameters:
        Fp (np.ndarray): Profile of the field at the current time step on rho grid.
        kappa (np.ndarray): Vector of diffusivities of length Nz+1 (on w-points).
        gam (np.ndarray): Gamma values, used in the diffusion calculation.
        Hz (np.ndarray): Width of rho blocks.
        Hzw (np.ndarray): Width of w blocks.
        FFlux (np.ndarray): Additional fluxes (such as forcing/T-S fluxes) added.
        BForce (np.ndarray): Body force (rho-points) added to the RHS.
        Nz (int): The number of grid points.
        dt (float): Time step size.

    Returns:
        F (np.ndarray): The updated field after advancing by one time step.
        FRIC (np.ndarray): The friction term.
    """

    # Calculate fluxes using centered 2nd-order finite difference
    Flux = np.zeros(Nz + 1)
    Flux[1:-1] = -kappa[1:-1] * ((Fp[1:] - Fp[:-1]) / Hzw - gam[1:-1])

    # Add forcing fluxes
    Flux += FFlux

    # Step forward field
    FRIC = -(Flux[1:] - Flux[:-1]) / Hz
    F = Fp + dt * FRIC + dt * BForce

    return F#, FRIC
