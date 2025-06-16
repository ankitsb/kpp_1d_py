import matplotlib.pyplot as plt
import numpy as np

# Assuming the required data is available as NumPy arrays
# Replace these with actual variables in your code (e.g., u, v, T, S, etc.)
# u, v: velocity fields
# T, S: temperature and salinity fields
# b: buoyancy field
# z_rho, z_w: grid points for rho and w
# kv, kt: diffusivities
# Cp, rho0: constants (specific heat capacity and reference density)
# ti: time index
# Pa1, Pa2, Pa3, Pa4, Pa5, Pa6, Pa7, Pa8: axes handles for the plots

def update_plot(u, v, T, S, b, z_rho, z_w, kv, kt, Cp, rho0, ti, Pa1, Pa2, Pa3, Pa4, Pa5, Pa6, Pa7, Pa8):
    # Update for u and v
    Pa11.remove()
    Pa12.remove()
    Pa11 = Pa1.plot(np.mean(u[:, ti], axis=1), z_rho, 'k--', linewidth=2)
    Pa12 = Pa1.plot(np.mean(v[:, ti], axis=1), z_rho, 'r--', linewidth=2)

    # Update for T
    Pa21.remove()
    Pa21 = Pa2.plot(np.mean(T[:, ti], axis=1), z_rho, 'k--', linewidth=2)

    # Update for S
    Pa31.remove()
    Pa31 = Pa3.plot(np.mean(S[:, ti], axis=1), z_rho, 'k--', linewidth=2)

    # Update for Shears
    Pa41.remove()
    Pa42.remove()
    shear_u = np.mean((u[1:, ti] - u[:-1, ti]) / (z_rho[1:] - z_rho[:-1]), axis=1)
    shear_v = np.mean((v[1:, ti] - v[:-1, ti]) / (z_rho[1:] - z_rho[:-1]), axis=1)
    Pa41 = Pa4.plot(shear_u, z_w[1:-1], 'k--', linewidth=2)
    Pa42 = Pa4.plot(shear_v, z_w[1:-1], 'r--', linewidth=2)

    # Update for Stratification
    Pa51.remove()
    stratification = np.mean((b[1:, ti] - b[:-1, ti]) / (z_rho[1:] - z_rho[:-1]), axis=1)
    Pa51 = Pa5.plot(stratification, z_w[1:-1], 'k--', linewidth=2)

    # Update for Reduced shear squared
    Pa61.remove()
    shear_squared = np.mean(((u[1:, ti] - u[:-1, ti]) / (z_rho[1:] - z_rho[:-1]) +
                             (v[1:, ti] - v[:-1, ti]) / (z_rho[1:] - z_rho[:-1]))**2 -
                            4 * (b[1:, ti] - b[:-1, ti]) / (z_rho[1:] - z_rho[:-1]), axis=1)
    Pa61 = Pa6.plot(shear_squared, z_w[1:-1], 'k--', linewidth=2)

    # Update for Diffusivities
    Pa71.remove()
    Pa72.remove()
    Pa71 = Pa7.plot(np.mean(kv[:, ti], axis=1), z_w, 'k--', linewidth=2)
    Pa72 = Pa7.plot(np.mean(kt[:, ti], axis=1), z_w, 'b--', linewidth=2)

    # Uncomment if force balance plots are needed
    # Pa81.remove()
    # Pa82.remove()
    # Pa83.remove()
    # Pa84.remove()
    # Pa85.remove()
    # Pa81 = Pa8.plot(np.mean(UTND[:, ti], axis=1), z_rho, 'k--', linewidth=2)
    # Pa82 = Pa8.plot(np.mean(URST[:, ti], axis=1), z_rho, 'b--', linewidth=2)
    # Pa83 = Pa8.plot(np.mean(UMFX[:, ti], axis=1), z_rho, 'r--', linewidth=2)
    # Pa84 = Pa8.plot(np.mean(UDIV[:, ti], axis=1), z_rho, 'g--', linewidth=2)
    # Pa85 = Pa8.plot(np.mean(UPGF[:, ti], axis=1), z_rho, 'm--', linewidth=2)

    # Update for Jq
    Pa81.remove()
    Jq = np.mean(kt[1:-1, ti] * Cp * rho0 * (T[1:, ti] - T[:-1, ti]) / (z_rho[1:] - z_rho[:-1]), axis=1)
    Pa81 = Pa7.plot(Jq, z_w[1:-1], 'k--', linewidth=2)

    # Redraw the plot
    plt.draw()

# Example of calling this function (replace with your actual data)
# update_plot(u, v, T, S, b, z_rho, z_w, kv, kt, Cp, rho0, ti, Pa1, Pa2, Pa3, Pa4, Pa5, Pa6, Pa7, Pa8)
