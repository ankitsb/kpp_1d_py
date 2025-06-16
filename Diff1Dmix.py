import numpy as np
from scipy.interpolate import interp1d
from Diff1Dconst import *

def diff1Dmix(INT, ti, nl, KPPBL, u, v, b, kv, kt, ks,
              # KPP
              K0=None, Ri0=None,
              # PP
              K0_PP=None, Ri0_PP=None, PR=None,
              # PP88
              P88_Kmax=None,
              # Boundary mixing
              Ustar=None, EKMO=None, hekman=None,
              srflux=None, wt0=None, ws0=None,
              KPPMLD=None, Hsbl=None, gamv=None, gamt=None, gams=None
              ):
    
    N2 = np.zeros_like(z_w)
    N2[1:-1] = (b[1:, ti] - b[:-1, ti]) / Hzw
    N2[0] = N2[1]
    N2[-1] = N2[-2]

    # Shear squared
    Sh2 = np.zeros_like(z_w)
    Sh2[1:-1] = ((u[1:, ti] - u[:-1, ti]) / Hzw)**2 + ((v[1:, ti] - v[:-1, ti]) / Hzw)**2
    Sh2[0] = Sh2[1]
    Sh2[-1] = Sh2[-2]

    Ri = N2 / (Sh2 + small)

    kint = np.zeros_like(kv[:, ti])

    if INT == 1:  # KPP Interior
        kint[Ri < 0] = K0
        mask = (Ri > 0) & (Ri < Ri0)
        kint[mask] = K0 * (1 - (Ri[mask] / Ri0) ** 2) ** 3

    elif INT == 2:  # PP Interior
        kint[Ri < 0] = K0_PP
        kint[Ri >= 0] = K0_PP * (1 + Ri[Ri >= 0] / Ri0_PP) ** -2
        kv[:, ti] += kint
        kint[Ri < 0] = K0_PP
        kint[Ri >= 0] = kv[Ri >= 0, ti] / (1 + Ri[Ri >= 0] / Ri0_PP)
        if PR == 1:
            kint_avg = 0.5 * (kv[:, ti] + kt[:, ti])
            kv[:, ti] = kint_avg
            kt[:, ti] = kint_avg
            ks[:, ti] = kint_avg
        elif PR == 2:
            kv[:, ti] = kt[:, ti]

    elif INT == 3:  # PP88 Interior
        kint[Ri < 0] = K0_P88_U_m
        kint[Ri > 0] = K0_P88_L_m * Ri[Ri > 0] ** EX_P88_L_m + \
                    K0_P88_U_m * (1 + Ri[Ri > 0] / Ri0_P88_m) ** EX_P88_U_m
        kv[:, ti] += kint

        kint[Ri < 0] = K0_P88_U_s
        kint[Ri > 0] = K0_P88_L_s * Ri[Ri > 0] ** EX_P88_L_s + \
                    K0_P88_U_s * (1 + Ri[Ri > 0] / Ri0_P88_s) ** EX_P88_U_s
        kt[:, ti] += kint
        ks[:, ti] += kint

        kv[kv[:, ti] > P88_Kmax, ti] = P88_Kmax
        kt[kt[:, ti] > P88_Kmax, ti] = P88_Kmax
        ks[ks[:, ti] > P88_Kmax, ti] = P88_Kmax

    # Add to total diffusivity
    kv[:, ti] += kint
    kt[:, ti] += kint
    ks[:, ti] += kint

    # KPP boundary layer mixing
    if KPPBL:
        # radiative flux at depth z:
        wr = np.array([srflux[ti] * (1 - swdk(zz)) / Cp / rho0 for zz in z_w])

        # Buoyancy flux:
        Bf = g * alpha * (wr + wt0) - g * beta * ws0

        # Monin-Obukhov length:
        L = Ustar ** 3 / vonKar / Bf
        L[np.abs(L) < small] = small * np.sign(L[np.abs(L) < small])

        # stability parameter:
        zeta = -z_w / L

        # Calculate velocity scales: %%%%%%%%%
        wm = np.zeros_like(z_w)
        ws = np.zeros_like(z_w)

        # stable:
        cond = zeta >= 0
        wm[cond] = vonKar * Ustar / (1 + 5 * zeta[cond])
        ws[cond] = wm[cond]

        # unstable:
        zeta_t = np.maximum(zeta, epsl * (-z_w[-1]) / L)
        cond = (zeta_t < 0) & (zeta_t >= zetam)
        wm[cond] = vonKar * Ustar * (1 - 16 * zeta_t[cond]) ** (-0.25)
        cond = zeta_t < zetam
        wm[cond] = vonKar * Ustar * (am - cm * zeta_t[cond]) ** (-1 / 3)

        cond = (zeta_t < 0) & (zeta_t >= zetas)
        ws[cond] = vonKar * Ustar * (1 - 16 * zeta_t[cond]) ** (-0.5)
        cond = zeta_t < zetas
        ws[cond] = vonKar * Ustar * (as_ - cs * zeta_t[cond]) ** (-1 / 3)

        # Calculate bulk Richardson number:
        bw = interp1d(z_rho, b[:, ti], kind='cubic', fill_value='extrapolate')(z_w)
        uw = interp1d(z_rho, u[:, ti], kind='cubic', fill_value='extrapolate')(z_w)
        vw = interp1d(z_rho, v[:, ti], kind='cubic', fill_value='extrapolate')(z_w)

        RiKPP_num = -(bw[-1] - bw) * z_w
        RiKPP_denom = (uw[-1] - uw) ** 2 + (vw[-1] - vw) ** 2 + Vtc * (-z_w) * ws * np.sqrt(np.abs(N2)) + small

        RiKPP = RiKPP_num / RiKPP_denom
        RiKPP[-1] = 0

        # Interpolate to find MLD:
        KPPMLD = interp1d(RiKPP, z_w, kind='linear', fill_value='extrapolate')(Ric)

        # Interpolate to find MO length with new BLD:
        LB = interp1d(z_w, L, kind='linear', fill_value='extrapolate')(KPPMLD)

        # Restrict to be less than ekman and MO depths:
        if EKMO and LB > 0:
            KPPMLD = -min([-KPPMLD, hekman, LB])
        # Restrict to be greater than the minimum z_rho:
        KPPMLD = min(KPPMLD, max(z_rho))

        # Output:
        Hsbl[ti] = KPPMLD

        # Calculate the diffusivity at the MLD
        kv[:, ti], kt[:, ti], ks[:, ti], gamt[:, ti], gams[:, ti] = compute_kpp_diffusivity(
            kv, kt, ks, wm, ws, ti, KPPMLD, wt0, wr, ws0, nl, gamt, gams)

    return kv, kt, ks, Hsbl, gamv, gamt, gams


def compute_kpp_diffusivity(kv, kt, ks, wm, ws, ti, KPPMLD, wt0, wr, ws0, nl, gamt, gams):

    # Vertical gradients
    wmDZ = (wm[1:]-wm[0:-1])/Hz
    wsDZ = (ws[1:]-ws[0:-1])/Hz

    # Interpolate vertical gradients to z_w
    wmDZ = interp1d(z_rho, wmDZ, kind='cubic', fill_value='extrapolate')(z_w)
    wsDZ = interp1d(z_rho, wsDZ, kind='cubic', fill_value='extrapolate')(z_w)

    # Derivatives of interior diffusivity profiles
    kvDZ = (kv[1:,ti]-kv[0:-1,ti])/Hz
    ktDZ = (kt[1:,ti]-kt[0:-1,ti])/Hz
    ksDZ = (ks[1:,ti]-ks[0:-1,ti])/Hz

    kvDZ = interp1d(z_rho, kvDZ, kind='cubic', fill_value='extrapolate')(z_w)
    ktDZ = interp1d(z_rho, ktDZ, kind='cubic', fill_value='extrapolate')(z_w)
    ksDZ = interp1d(z_rho, ksDZ, kind='cubic', fill_value='extrapolate')(z_w)

    # Interpolate variables to MLD
    kvN = np.interp(KPPMLD, z_w, kv[:, ti])
    ktN = np.interp(KPPMLD, z_w, kt[:, ti])
    ksN = np.interp(KPPMLD, z_w, ks[:, ti])
    wmN = np.interp(KPPMLD, z_w, wm)
    wsN = np.interp(KPPMLD, z_w, ws)
    kvDZN = np.interp(KPPMLD, z_w, kvDZ)
    ktDZN = np.interp(KPPMLD, z_w, ktDZ)
    ksDZN = np.interp(KPPMLD, z_w, ksDZ)
    wmDZN = np.interp(KPPMLD, z_w, wmDZ)
    wsDZN = np.interp(KPPMLD, z_w, wsDZ)

    # Shape function coordinate
    sig = z_w / (KPPMLD + small)

    # Calculate intermediate shape function coefficients:
    G1v = kvN / (-KPPMLD * wmN + small)
    G1t = ktN / (-KPPMLD * wsN + small)
    G1s = ksN / (-KPPMLD * wsN + small)

    G1vDZ = -kvDZN / (wmN + small) - kvN * wmDZN / (-KPPMLD * wmN**2 + small)
    G1tDZ = -ktDZN / (wsN + small) - ktN * wsDZN / (-KPPMLD * wsN**2 + small)
    G1sDZ = -ksDZN / (wsN + small) - ksN * wsDZN / (-KPPMLD * wsN**2 + small)

    # Calculate shape function coefficients:
    a2v = -2 + 3 * G1v - G1vDZ
    a3v = 1 - 2 * G1v + G1vDZ

    a2t = -2 + 3 * G1t - G1tDZ
    a3t = 1 - 2 * G1t + G1tDZ

    a2s = -2 + 3 * G1s - G1sDZ
    a3s = 1 - 2 * G1s + G1sDZ

    # Calculate shape function:
    Gv = a0 + a1 * sig + a2v * sig**2 + a3v * sig**3
    Gt = a0 + a1 * sig + a2t * sig**2 + a3t * sig**3
    Gs = a0 + a1 * sig + a2s * sig**2 + a3s * sig**3

    # Calculate diffusivities: Apply only within the boundary layer (0 < sig < 1)
    mask = (sig > 0) & (sig < 1)
    kv[mask, ti] = (-KPPMLD) * wm[mask] * Gv[mask]
    kt[mask, ti] = (-KPPMLD) * ws[mask] * Gt[mask]
    ks[mask, ti] = (-KPPMLD) * ws[mask] * Gs[mask]

    if nl:
        zeta = np.zeros_like(z_w)  # You can replace this with actual bulk Richardson number profile
        unstable_mask = zeta < 0

        gamt[unstable_mask, ti] = Cstar * vonKar * (cs * vonKar * epsl)**(1/3) * (wt0 + wr[unstable_mask]) / (ws[unstable_mask] * (-KPPMLD) + small)

        gams[unstable_mask, ti] = Cstar * vonKar * (cs * vonKar * epsl)**(1/3) * ws0 / (ws[unstable_mask] * (-KPPMLD) + small)

        gamt[(sig > 1) | (sig < 0)] = 0
        gams[(sig > 1) | (sig < 0)] = 0

    return kv[:, ti], kt[:, ti], ks[:, ti], gamt[:, ti], gams[:, ti]
