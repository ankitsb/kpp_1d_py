# diff1d_main.py

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.io import savemat, loadmat
import os
import matplotlib.pyplot as plt

from Diff1Dconst import *  # this contains constants
from Diff1Dmix import *
from Diff1Dstep import diff1Dstep

# Setup
#############  RUN NUMBER #############
run = 2
out_folder = './testing/'

#############  TIME stepping  #############
dt = 120
tfin = 5 * 86400 # seconds
t = np.arange(0, tfin + dt, dt) # time vector
Nt = len(t)
NOUT = 1 # output every NOUT steps

#############  Surface forcing #############
tau_x = -0.08 * np.ones(Nt)
tau_y = np.zeros(Nt)
ssflux = np.zeros(Nt)
shflux = -180 * np.ones(Nt)

DIR = 0
srflux = 275
if DIR:
    hr = (t / 3600) % 24
    srflux = srflux * 4 * (np.cos((2 * (hr / 24) - 1) * np.pi)) ** 2
    srflux[(hr / 24 < 0.25) | (hr / 24 > 0.75)] = 0
else:
    srflux = srflux * np.ones(Nt)

#############  Body forcing #############
# Depth independent nudging:
TS_RST = 1 / (15 * 86400)
u_RST = 1 / (200000000 * 86400)
v_RST = 1 / (200000000 * 86400)

# Pressure-gradient force for EUC:
PGFscale = 120
z_temp = np.arange(-10000, 0.1, 0.1)
PGFamp = -tau_x[0] / rho0 / np.trapezoid(np.exp(-(-z_temp / PGFscale) ** 3), z_temp)
PGF_X = PGFamp * np.exp(-(-z_rho / PGFscale) ** 3)

# Vertical advection:
w = np.zeros((Nz + 1, Nt))

# TIW forcing:
SYM = 0
period = 15 * 86400
amplitude = 2.8e-6
dvdy = amplitude * np.sin(2 * np.pi / period * t)
dvdy_v_expr = '(5.2e-9 / 2.8e-6) * z_rho + 1'
dvdy_v = eval(dvdy_v_expr)
dvdy = np.tile(dvdy, (Nz, 1)) * np.tile(dvdy_v.reshape(-1, 1), (1, Nt))


#############  Vertical mixing configuration #############
# Interior:
INT = 1
kv0, kt0, ks0 = 1e-4, 1e-5, 1e-5

# Background:
# if INT == 1: # KPP
Ri0, K0 = 0.7, 2e-3
# elif INT == 2: # PP
Ri0_PP, K0_PP, PR = 0.2, 0.01, 0
# elif INT == 3:
P88_Kmax = 0.1

# Boundary mixing configuration
KPPBL = 1 #use boundary layer mixing.
EKMO = 1 #use Ekman/MO length restrictions.
nl = 1 #use non-local fluxes?
KPPMLD = -15 #Initial guess mixed layer.

#############  Initial conditions #############
# #  Initial profiles from TPOS20 Sep-Dec:
# mat = loadmat('TPOS20_140W_TS.mat')
# Z = mat['Z'].squeeze()
# U = mat['U']
# V = mat['V']
# Tmat = mat['T']
# Smat = mat['S']

# zI = Z
# uI = 0*U[:, inds] # np.mean(U[:, inds], axis=1)
# vI = 0*V[:, inds] # np.zeros_like(uI)
# TI = Tmat[:, inds] # np.mean(Tmat[:, inds], axis=1)
# SI = Smat[:, inds] # np.mean(Smat[:, inds], axis=1)
# bI = g * alpha * TI - g * beta * SI
# zwI = (zI[1:] + zI[:-1]) / 2


mat = xr.open_dataset('MR_profile_0day.nc')
Z = -mat.z.values
Tmat = mat.t.values
Smat = mat.s.values
U = xr.zeros_like(mat.t).values
V = xr.zeros_like(mat.t).values

zI = Z
uI = 0*U
vI = 0*V
TI = Tmat
SI = Smat
bI = g * alpha * TI - g * beta * SI
zwI = (zI[1:] + zI[:-1]) / 2


# inds = np.where((np.array([d[1] for d in map(lambda x: list(x), map(np.datetime64, time))]) >= 9))[0]

inds = 0

#####################################################
#  Set up grid and constants

# Allocate arrays
u, v, T, S, b = [np.zeros((Nz, Nt)) for _ in range(5)]
kv, kt, ks = [np.full((Nz + 1, Nt), val) for val in (kv0, kt0, ks0)]
gamv, gamt, gams = [np.zeros((Nz + 1, Nt)) for _ in range(3)]
bulkRiN = np.zeros((Nz + 1, Nt))
bulkRiD = np.zeros((Nz + 1, Nt))
Hsbl = np.zeros(Nt)

# Interpolate initial conditions
u[:, 0] = interp1d(zI, uI, kind='cubic', fill_value="extrapolate")(z_rho)
v[:, 0] = interp1d(zI, vI, kind='cubic', fill_value="extrapolate")(z_rho)
T[:, 0] = interp1d(zI, TI, kind='cubic', fill_value="extrapolate")(z_rho)
S[:, 0] = interp1d(zI, SI, kind='cubic', fill_value="extrapolate")(z_rho)
b[:, 0] = g * alpha * T[:, 0] - g * beta * S[:, 0]
Hsbl[0] = KPPMLD

# Main loop
for ti in range(Nt - 1):
    if ti % 50 == 0:
        print(f"Doing step {ti} of {Nt - 1}")

    # Calculate mixing parameters from time dependent forcing:
    Ustar = np.sqrt(np.sqrt(tau_x[ti]**2 + tau_y[ti]**2) / rho0)
    hekman = 0.7 * Ustar / max(abs(f), 1e-10)
    wt0 = shflux[ti] / Cp / rho0
    ws0 = ssflux[ti] / rho0

    # # Compute diffusivity
    # RiKPP_Numer, RiKPP_Denom = compute_RiKPP_terms(b, u, v, z_rho)
    kv, kt, ks, Hsbl, gamv, gamt, gams = diff1Dmix( #
    INT=INT,
    KPPBL=KPPBL, ti=ti, nl=nl,
    u=u, v=v, b=b,
    kv=kv, kt=kt, ks=ks,
    K0=K0, Ri0=Ri0,
    K0_PP=K0_PP, Ri0_PP=Ri0_PP, PR=PR,
    P88_Kmax=P88_Kmax,
    Ustar=Ustar, EKMO=EKMO, hekman=hekman,
    srflux=srflux, wt0=wt0, ws0=ws0,
    KPPMLD=KPPMLD, Hsbl=Hsbl, gamv=gamv, gamt=gamt, gams=gams
    )

    # bulkRiN[:, ti] = RiKPP_Numer
    # bulkRiD[:, ti] = RiKPP_Denom

    # Calculate heat flux down:
    TAU_T = np.concatenate(([0]*Nz, [shflux[ti] / Cp]))
    for zi in range(len(z_w)):
        TAU_T[zi] += srflux[ti] / Cp * swdk(z_w[zi])

    # Calculate momentum and salinity fluxes:
    TAU_X = np.concatenate(([0]*Nz, [tau_x[ti]]))
    TAU_Y = np.concatenate(([0]*Nz, [tau_y[ti]]))
    TAU_S = np.concatenate(([0]*Nz, [ssflux[ti]]))

    # Calculate body forces:
    # depth-independent restoring:
    URST = -u_RST * (u[:, ti] - u[:, 0])
    VRST = -v_RST * (v[:, ti] - v[:, 0])
    TRST = -TS_RST * (T[:, ti] - T[:, 0])
    SRST = -TS_RST * (S[:, ti] - S[:, 0])

    # Vertical advection:
    UVAD = np.zeros(Nz + 1)
    UVAD[1:-1] = -np.diff(u[:, ti]) / np.diff(z_rho) * w[1:-1, ti]
    UVAD = np.mean(UVAD)

    VVAD = np.zeros(Nz + 1)
    VVAD[1:-1] = -np.diff(v[:, ti]) / np.diff(z_rho) * w[1:-1, ti]
    VVAD = np.mean(VVAD)

    TVAD = np.zeros(Nz + 1)
    TVAD[1:-1] = -np.diff(T[:, ti]) / np.diff(z_rho) * w[1:-1, ti]
    TVAD = np.mean(TVAD)

    # Zonal pressure gradient:
    UPGF = PGF_X

    # TIW Stretching:
    UDIV = dvdy[:, ti] * u[:, ti]

    BF_X = URST + UPGF + UDIV
    BF_Y = VRST
    BF_T = TRST
    BF_S = SRST

    # Calculate step ti+1:

    u[:, ti + 1] = diff1Dstep(u[:, ti], kv[:, ti], gamv[:, ti], Hz, Hzw, -TAU_X / rho0, BF_X, Nz, dt)
    v[:, ti + 1] = diff1Dstep(v[:, ti], kv[:, ti], gamv[:, ti], Hz, Hzw, -TAU_Y / rho0, BF_Y, Nz, dt)
    T[:, ti + 1] = diff1Dstep(T[:, ti], kt[:, ti], gamt[:, ti], Hz, Hzw, -TAU_T / rho0, BF_T, Nz, dt)
    S[:, ti + 1] = diff1Dstep(S[:, ti], ks[:, ti], gams[:, ti], Hz, Hzw, -TAU_S / rho0, BF_S, Nz, dt)

    b[:, ti + 1] = g * alpha * T[:, ti + 1] - g * beta * S[:, ti + 1]

ds = xr.Dataset(
    data_vars=dict(
        u=(["depth", "time"], u),
        v=(["depth", "time"], v),
        T=(["depth", "time"], T),
        S=(["depth", "time"], S),
        b=(["depth", "time"], b), ),
    coords=dict(
        depth=("depth", z_rho),
        time=t, ),
    attrs=dict(description="KPP python output"), )

ds.to_netcdf(os.path.join(out_folder, 'KPPpy_out.nc') )

# # Save output
# if os.path.exists(out_folder):
#     print(f"Saving run {run} to folder {out_folder}")
#     savemat(os.path.join(out_folder, f'run_{run:03d}.mat'), {
#         'u': u, 'v': v, 'T': T, 'S': S, 'b': b,
#         'kv': kv, 'kt': kt, 'ks': ks,
#         'run': run
#     })
