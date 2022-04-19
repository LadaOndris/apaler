import matplotlib.pyplot as plt
import numpy as np
from geometry import degs_to_rads


def plot_error_depending_on_surface_altitude_differences():
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    angles = np.linspace(45, 90, 500)
    angles_rads = degs_to_rads(angles)

    error_scale = 1 / np.tan(angles_rads)

    plt.plot(angles, error_scale)
    ax.set_xlabel('Laser angle [degrees]')
    ax.set_ylabel('Error scale')
    fig.tight_layout()
    fig.show()


plot_error_depending_on_surface_altitude_differences()
