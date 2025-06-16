# Diff1Dconst.py

import numpy as np

# --- scoord function equivalent to ROMS stretching ---
def scoord(h, zeta, zice, Vtransform, Vstretching, theta_s, theta_b, hc, N, igrid, idry, hmin, type, loglayer):
    """
    Simplified vertical coordinate generator matching ROMS logic.
    Only handles basic options used in original script.
    """
    if igrid == 0:
        s = (np.arange(1, N + 1) - N - 0.5) / N  # s_rho
    elif igrid == 1:
        s = (np.arange(0, N + 1) - N) / N  # s_w
    else:
        raise ValueError("Invalid igrid value.")

    if Vstretching == 1:
        if theta_s > 0:
            csrf = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        else:
            csrf = -s ** 2
        if theta_b > 0:
            Cs = (np.exp(theta_b * csrf) - 1) / (1 - np.exp(-theta_b))
        else:
            Cs = csrf
    else:
        Cs = s  # no stretching

    if Vtransform == 1:
        z = (hc * s + h * Cs) / (hc + h)
        z = z * h
    elif Vtransform == 2:
        z = zeta + (zeta + h) * ((hc * s + h * Cs) / (hc + h))
    else:
        raise ValueError("Invalid Vtransform value")

    return z, s, Cs

# --- Vertical coordinate setup ---
N = 50
h = 400
# Vtransform = 1
# Vstretching = 1
# theta_s = 5
# theta_b = 0
# hc = 75
zbot = -300

# # Generate vertical coordinates
# z_rho_full, s_rho, Cs_rho = scoord(h, 0, 0, Vtransform, Vstretching, theta_s,
#                                    theta_b, hc, N, igrid=0, idry=1, hmin=1, type=0, loglayer=1)
# z_rho = z_rho_full[z_rho_full > zbot][::-1]  # flip to increasing order

# z_w_full, s_w, Cs_w = scoord(h, 0, 0, Vtransform, Vstretching, theta_s,
#                              theta_b, hc, N, igrid=1, idry=1, hmin=1, type=0, 
# loglayer=1)

z_w_full = np.linspace(-h, 0, N + 1)  # z_w from -h to 0
z_rho_full = np.linspace(-(h - h/(2*N)), - h/(2*N), N)  # z_rho from h-h/(2*N) to 0
z_rho = z_rho_full[z_rho_full > zbot]#[::-1]  # flip to increasing order

Nz = len(z_rho)
z_w = z_w_full[-(Nz + 1):]#[::-1]

# Grid spacing
Hz = z_w[1:] - z_w[:-1]          # thickness of each rho cell
Hzw = z_rho[1:] - z_rho[:-1]     # spacing between rho points

# --- Physical constants ---
rho0 = 1025           # kg/m^3
alpha = 2.489e-4      # 1/degC
beta = 7.453e-4       # 1/psu
Cp = 4000             # J/kg/degC
g = 9.81              # m/s^2
lat = 0               # Latitude
f = 2 * 7.29e-5 * np.sin(lat * np.pi / 180)  # Coriolis parameter

# --- Shortwave absorption constants ---
lmd_mu1 = 0.35
lmd_mu2 = 23
lmd_r1 = 0.58
cff1 = 1 / lmd_mu1
cff2 = 1 / lmd_mu2

def swdk(z):
    z = np.atleast_1d(z)
    return np.exp(z[0] * cff1) * lmd_r1 + np.exp(z[0] * cff2) * (1 - lmd_r1)

# --- Boundary layer constants (KPP) ---
Ric = 0.3           # Critical Richardson number for MLD
vonKar = 0.41       # von Kármán constant
epsl = 0.1          # Surface layer fraction (epsilon)
Cstar = 10          # C* parameter
small = 1e-10       # Small numerical parameter

# Dimensionless flux profile constants (Large 1994)
zetam = -0.2
zetas = -1
am = 1.257
cm = 8.36
as_ = -28.86  # renamed from 'as' to avoid keyword conflict
cs = 98.96

# Turbulent velocity scale constant
Vtc = 1.25 * np.sqrt(0.2) / (np.sqrt(cs * epsl) * Ric * vonKar * vonKar)

# Diffusivity shape function coefficients
a0 = 0
a1 = 1

# Peters 1988 constants for alternate mixing schemes
# These are currently unused
K0_P88_L_m = 5.6e-8
K0_P88_L_s = 3.0e-9
EX_P88_L_m = -8.2
EX_P88_L_s = -9.6
K0_P88_U_m = 5e-4
K0_P88_U_s = 5e-4
EX_P88_U_m = -1.5
EX_P88_U_s = -2.5
Ri0_P88_m = 0.2
Ri0_P88_s = 0.2