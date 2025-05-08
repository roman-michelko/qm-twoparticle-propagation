#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: plot_prob_denisty_example.py
LAST EDIT: 2025-05-08
AUTHOR: Roman Michelko

DESCRIPTION:
-------------
Minimal demo — plots the probability–density data that `main.py` saves.
"""

__author__  = "Roman Michelko"
__license__ = "MIT"

import os, sys, re
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Varables and definitions
PD_DIR = r"../prob_density/"
IMG_DIR = r"plots/"

PD_OUTPUT_FILE = r"prob_dens{:d}.csv"
IMG_OUTPUT_FILE = r"prob_dens_cmap{:d}.png"

IMG_DPI = 300
CSV_DELIMITER = ","

PLOT_REGION = {
                "x_p": [-8.0, 8.0],
                "x_b": [-2.0, 6.0],
            }

plt.rcParams['text.usetex'] = True # Enable LaTeX in plots

#%% Functions
def load_bounded_prob(filename: str) -> tuple[NDArray[np.float64], float, float, int]:
    
    # Load parameters and probability slice
    try:
        with open(filename, "r") as fr:
            # Read first line
            line1 = fr.readline().strip().split(CSV_DELIMITER)
            L_LF = float(line1[0])
            N_LF = int(line1[1])
            time = float(line1[2])
            
            # Read second line
            j_min, i_min = map(int, fr.readline().strip().split(CSV_DELIMITER))
            
            # Read rest of the file
            nonzero_prob = np.loadtxt(fr, delimiter=CSV_DELIMITER)
    
    except (IOError, OSError) as e:
        print(f"Error opening or reading the file \"{filename}\": {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"Error parsing numerical values in \"{filename}\": {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error while loading probability density data from \"{filename}\": {e}")
        sys.exit(1)
    
    
    # Allocate all-zeros array
    prob = np.zeros((N_LF, N_LF), dtype=np.float64)
    
    if np.ndim(nonzero_prob) == 0:
        return prob
    
    num_i, num_j = nonzero_prob.shape
    
    # Insert the loaded non-zero probability region at the specified offset
    prob[i_min : i_min + num_i, j_min : j_min + num_j] = nonzero_prob
    
    return prob, time, L_LF, N_LF

def prob_window(x: NDArray[np.float64], y: NDArray[np.float64], x_range: list, y_range: list) -> tuple[NDArray[bool], tuple[int, int]]:
    x_mask = (x >= min(x_range)) & (x <= max(x_range))
    y_mask = (y >= min(y_range)) & (y <= max(y_range))
    
    Nx = np.count_nonzero(x_mask)
    Ny = np.count_nonzero(y_mask)
    
    x_mask_2D, y_mask_2D = np.meshgrid(x_mask, y_mask)
    mask_2D = x_mask_2D & y_mask_2D
    
    return mask_2D, (Ny, Nx)

def mplot(x_p, x_b, P_LAB, n, time):
    # Compute marginal probabilities
    dx_p = x_p[0,1] - x_p[0,0]
    dx_b = x_b[1,0] - x_b[0,0]
    P_p = np.sum(P_LAB, axis=0) * dx_b
    P_b = np.sum(P_LAB, axis=1) * dx_p
    
    # Set figure and main axes
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Set sub-axes
    divider = make_axes_locatable(ax)
    ax_p = divider.append_axes("top", 1.0, pad=0.4, sharex=ax)
    ax_b = divider.append_axes("left", 1.0, pad=0.4, sharey=ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    
    # Plot marginal probability densities
    ax_p.plot(x_p[0, :], P_p, color="blue", linewidth=1.0)
    ax_b.plot(P_b, x_b[:, 0], color="orange", linewidth=1.0)
    ax_b.invert_xaxis()
    
    ax_p.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_b.tick_params(axis='y', which='both', labelleft=False, left=False)
    
    # Plot probabability denisty on 2D color-map
    ax.imshow(P_LAB, extent=[x_p.min(), x_p.max(), x_b.min(), x_b.max()],
              origin='lower', cmap='RdYlBu', aspect='auto')

    fig.colorbar(ax.images[0], cax=cax)
    
    # Set labels
    ax.set_xlabel(r'$x_\mathrm{p}$', labelpad=1.5)
    ax.set_ylabel(r'$x_{\mathrm{b}}$', labelpad=1.5)
    ax.set_title(rf"$|\Psi(x_\mathrm{{p}}, x_\mathrm{{b}}, t={time:.2f})|^2$", pad=10)
    
    ax_p.set_ylabel(r'$P_{\mathrm{p}}(x_\mathrm{p}, t)$', labelpad=3.0)
    ax_b.set_xlabel(r'$P_{\mathrm{b}}(x_\mathrm{b}, t)$', labelpad=1.5)
    ax_p.set_title(r"Projectile", pad=4.0)
    ax_b.set_title(r"Barrier", pad=4.0)
    
    # Save image
    plt.savefig(IMG_DIR+IMG_OUTPUT_FILE.format(n), dpi=IMG_DPI, bbox_inches='tight')
    plt.close(fig)

#%% Main function
def main():
    global P_LAB, x_p, x_b
    
    # List input directory
    try:
        files = os.listdir(PD_DIR)
    
        # Create regex pattern
        pattern = re.compile(r"^" + PD_OUTPUT_FILE.replace(r"{:d}", r"(\d+)") + r"$")
    
        # Match and extract numbers from filenames
        matches = []
        for f in files:
            m = pattern.match(f)
            if m:
                matches.append((int(m.group(1)), f))

        if not matches:
            print(f"No files matching pattern '{PD_OUTPUT_FILE}' found in '{PD_DIR}'")
            sys.exit(1)

        # Sort by the extracted number
        s_files = [f for _, f in sorted(matches)]
    
        print(f"Found {len(s_files)} files.")

    except FileNotFoundError:
        print(f"Directory '{PD_DIR}' does not exist.")
        sys.exit(1)
    
    # Check if output directory exists
    if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
    
    # Main time loop
    for n, file in enumerate(s_files):
        print(f"Plotting from: {file}")
        
        # Load probability denisty array
        P_LAB, time, L_LF, N_LF = load_bounded_prob(PD_DIR + file)
        
        # Define meshgrid for plotting
        Delta = L_LF / N_LF
        xp_range = Delta * np.arange(-N_LF >> 1, N_LF >> 1)
        xb_range = Delta * np.arange(-N_LF >> 1, N_LF >> 1)
        x_p, x_b = np.meshgrid(xp_range.astype(np.float64), xb_range.astype(np.float64))
        
        # Select probabiltiy subregion
        P_mask, P_shape = prob_window(xp_range, xb_range, PLOT_REGION["x_p"], PLOT_REGION["x_b"])
        
        x_p = x_p[P_mask].reshape(P_shape)
        x_b = x_b[P_mask].reshape(P_shape)
        P_LAB = P_LAB[P_mask].reshape(P_shape)
        
        # Create plot
        mplot(x_p, x_b, P_LAB, n, time)

#%% Run program
if __name__ == "__main__":
    main()