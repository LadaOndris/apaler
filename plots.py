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


def plot_error_due_to_invalid_xy_position_of_camera():
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    angles1 = np.linspace(10, 90, 500)
    beta1 = 90 - angles1

    angles2 = np.linspace(90, 170, 500)[1:]
    beta2 = angles2 - 90

    angles = np.concatenate([angles1, angles2], axis=-1)
    betas = np.concatenate([beta1, beta2], axis=-1)
    betas_rads = degs_to_rads(betas)

    error_scale = 1 / np.cos(betas_rads)

    plt.plot(angles, error_scale)
    ax.set_xlabel('Angle between planes [degrees]')
    ax.set_ylabel('Error scale')
    fig.tight_layout()
    fig.show()


def plot_error_due_to_invalid_z_position_of_camera():
    def plot_angles(ax, from_angle, to_angle):
        angles = np.linspace(from_angle, to_angle, 500)
        angles_rads = degs_to_rads(angles)
        error_scale = 1 / np.tan(angles_rads)
        ax.plot(angles, error_scale)

    def set_labels(ax):
        ax.set_xlabel('Plane angle [degrees]')
        ax.set_ylabel('Error scale')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    plot_angles(ax1, 5, 45)
    plot_angles(ax2, 45, 90)

    set_labels(ax1)
    set_labels(ax2)
    fig.tight_layout()
    fig.show()


plot_error_due_to_invalid_z_position_of_camera()
