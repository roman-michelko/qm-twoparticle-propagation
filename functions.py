# -*- coding: utf-8 -*-
"""
FILE: functions.py
LAST EDIT: 2025-05-08
AUTHOR: Roman Michelko

DESCRIPTION:
-------------
Support module — contains data structures, utility functions, and physical calculations (e.g., wavefunction, reflection/transmission, interpolation)
used by `main.py` for setting up and running the computation.
"""

__author__  = "Roman Michelko"
__license__ = "MIT"
__version__ = "1.0.0"

import json, sys
import numpy as np
import csv

from enum import Enum
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import TextIO

#%% Constants
REQUIRED_TOP_KEYS = {"interaction_regime", "numerical_parameters", "computation_parameters", "physical_parameters"}
REQUIRED_NP_KEYS =  {"power_of_two", "interval_length"}
REQUIRED_CP_KEYS =  {"time_range", "time_step"}
REQUIRED_PP_KEYS =  {"k_0", "lambda", "alpha", "sigma_xpb", "d"}

CSV_DELIMITER = ","
PD_ZERO_MARGIN = 1e-7

#%% Classes definitions
class InteractionRegime(Enum):
    SINGLE_DELTA = 0
    RESONANT = 1
    
    @classmethod
    def set_regime(cls, regime: str):
        if regime in ["sd", "single_delta"]:
            return cls.SINGLE_DELTA
        elif regime in ["res", "resonant"]:
            return cls.RESONANT
        else:
            raise ValueError(f"Unknown regime: \"{regime}\".")
    
    def regime(self) -> str:
        if self   == self.SINGLE_DELTA:
            return r"single delta"
        elif self == self.RESONANT:
            return r"resonant"

@dataclass
class NumParams:
    power_of_two:    int
    interval_length: float
    
    N: int = field(init=False)
    N_LF: int = field(init=False)
    d_Xx:  float = field(init=False)
    d_Kk:  float = field(init=False)
    
    def __post_init__(self):
        if not isinstance(self.power_of_two, int):
            raise TypeError("\"power_of_two\" must be an integer.")
        
        if self.power_of_two <= 0:
            raise ValueError("\"power_of_two\" must be positive.")
        
        if self.interval_length <= 0:
            raise ValueError("\"interval_length\" must be positive.")
        
        self.N = int(2 ** self.power_of_two)
        self.N_LF = self.N >> 1
        self.d_Xx = self.L / self.N
        self.d_Kk = 2 * np.pi / self.L
    
    @property
    def L(self) -> float:
        return self.interval_length

@dataclass
class CompParams:
    time_range: tuple[float, float]
    time_step:  float
    
    t_start: float = field(init=False)
    t_end:   float = field(init=False)
    Nt: int = field(init=False)

    def __post_init__(self):
        if not (isinstance(self.time_range, (list, tuple)) and len(self.time_range) == 2):
            raise ValueError("\"time_range\" must be a list or tuple of two elements.")
        if not isinstance(self.time_step, float):
            raise TypeError("\"time_step\" must be a float or int.")
        
        if any(x < 0 for x in self.time_range) or self.time_step < 0:
             raise ValueError("Time values must be non-negative.")
        
        self.t_start = self.time_range[0]
        self.t_end = self.time_range[1]
        
        if self.t_start >= self.t_end:
            raise ValueError("End time must be greater than start time.")
        
        if (self.t_end - self.t_start) < self.time_step:
            raise ValueError("Time step too large for given time range.")
            
        if self.time_step < 1e-2:
            raise ValueError("Time step too small.")
        
        self.Nt = int((self.t_end - self.t_start) / self.time_step) + 1
    
    @property
    def t_step(self) -> float:
        return self.time_step

@dataclass
class PhysParams:
    k_0:        float
    lmbda:      float
    alpha:      float
    sigma_xpb:  float
    d:          float
    regime:     InteractionRegime
    
    beta: float = field(init=False)
    
    def __post_init__(self):
        if self.k_0 <= 0:
            raise ValueError("\"k_0\" must be positive.")
        if self.lmbda <= 0:
            raise ValueError("\"lambda\" must be positive.")
        if not (0 < self.alpha < 0.5):
            raise ValueError("\"alpha\" must be in the interval (0, 0.5).")
        if self.sigma_xpb <= 0:
            raise ValueError("\"sigma_xpb\" must be positive.")
        if self.regime != InteractionRegime.SINGLE_DELTA:
            if (self.d is None) or (self.d <= 0):
                raise ValueError("\"d\" must be a positive number with resonant potential.")
        
        self.beta = 1 - self.alpha

# %% Function definitions
def load_config(file_name: str) -> tuple[InteractionRegime, NumParams, CompParams, PhysParams]:
    """
    Loads and validates simulation parameters from a JSON config file.

    Parameters:
        file_name: path to the configuration file

    Returns:
        tuple
        (InteractionRegime, NumParams, CompParams, PhysParams)

    Raises:    
        SystemExit
        If the file is missing, or contains invalid parameters.

    Notes:
        Issues warnings for potentially inaccurate simulation settings.
    """
    
    # Load JSON data from file
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
        
        print(f"Config file: \"{file_name}\" loaded successfully.")

        # Validate outer-level keys
        if not REQUIRED_TOP_KEYS.issubset(data):
            raise KeyError("Missing one or more top-level keys in the config file.")

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    
    # Set interaction regime
    try:
        interaction_regime = InteractionRegime.set_regime(data["interaction_regime"])
    
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    
    # Set numerical parameters
    try:
        num_data = data["numerical_parameters"]
        if not REQUIRED_NP_KEYS.issubset(num_data):
            raise KeyError("Missing one or more parameters in the config file.")
        
        num_params = NumParams(
            power_of_two = num_data["power_of_two"],
            interval_length = num_data["interval_length"]
        )
    
    except (TypeError, KeyError, ValueError) as e:
        print(f"Error parsing numerical parameters: {e}")
        sys.exit(1)
    
    # Set computation parameters
    try:
        cmp_data = data["computation_parameters"]
        if not REQUIRED_CP_KEYS.issubset(cmp_data):
            raise KeyError("Missing one or more parameters in the config file.")
        
        comp_params = CompParams(
            time_range = cmp_data["time_range"], 
            time_step = cmp_data["time_step"]
        )
    
    except (TypeError, KeyError, ValueError) as e:
        print(f"Error parsing computation parameters: {e}")
        sys.exit(1)
    
    # Set physical_parameters
    try:
        phy_data = data["physical_parameters"]
        if interaction_regime == InteractionRegime.SINGLE_DELTA:
            REQUIRED_PP_KEYS.discard("d")
        if not REQUIRED_PP_KEYS.issubset(phy_data):
            raise KeyError("Missing one or more parameters in the config file.")
        
        phys_params = PhysParams(
            k_0   = phy_data["k_0"],
            lmbda = phy_data["lambda"],
            alpha = phy_data["alpha"],
            sigma_xpb = phy_data["sigma_xpb"],
            d = phy_data["d"],
            regime=interaction_regime
            )
    
    except (TypeError, KeyError, ValueError) as e:
        print(f"Error parsing physical parameters: {e}")
        sys.exit(1)
        
    # Raise warnings
    is_warning = False
    
    # Maximal time
    t_max = phys_params.k_0 * num_params.d_Xx / num_params.d_Kk
    if t_max < comp_params.t_end:
        print("Warning: End time higher than maximal time; results may be inaccurate.")
        is_warning = True
    
    # Numerical parmeters
    if (interaction_regime == InteractionRegime.RESONANT) and (phys_params.d > 0.2):
        print("Warning: Large value of parameter \"d\"; results may be inaccurate.")
        is_warning = True
        
    # Initial spatial uncertanity
    if phys_params.sigma_xpb > 0.2:
        print("Warning: Large value of parameter \"sigma_xpb\"; results may be inaccurate.")
        is_warning = True
        
    # Initial average relative wavenumber
    sigma_kk = 0.5 * np.sqrt(phys_params.alpha**2 + phys_params.beta**2) / phys_params.sigma_xpb
    if 5 * sigma_kk > phys_params.k_0:
        print("Warning: Small value of parameter \"k_0\"; results may be inaccurate.")
        is_warning = True
        
    # Quantization of Kk and Xx grids
    if 3.54 * num_params.d_Xx > phys_params.sigma_xpb:
        print("Warning: Coarse discretization step \"Delta_Xx\"; results may be inaccurate.")
        is_warning = True
        
    if 7.07 * num_params.d_Kk > 1 / phys_params.sigma_xpb:
        print("Warning: Coarse discretization step \"Delta_Kk\"; results may be inaccurate.")
        is_warning = True
    
    if is_warning:
        res = input("Continue program despite warnings? (y/n): ").strip().lower()
        if res != 'y':
            print("Exiting the program.")
            sys.exit(2)
        
    return interaction_regime, num_params, comp_params, phys_params

def allocate_meshgrid(N: int, Delta: float) -> NDArray[np.float64]:
    """
    Allocates 2D square meshgrid of points given number of points and discretization step.

    Parameters:
        N:     even number of points in each direction of gird
        Delta: discretization step in each direction

    Returns:
        (x, y): A 2D real arrays of shape (N, N), containing the meshgrids of points x and y
    """
    
    x_range = y_range = Delta * np.arange(-N >> 1, N >> 1)
    x, y = np.meshgrid(x_range.astype(np.float64), y_range.astype(np.float64))
    
    return x, y

def calc_c(K: NDArray[np.float64], k: NDArray[np.float64], phys_params:PhysParams) -> NDArray[np.complex128]:
    """
    Evaluates amplitude coefficient function c(K, k) on a meshgrid of points (K, k).

    Parameters:
        K, k: 2D arrays of values K and k corresponding to points on meshgrid
        phys_params: dataclass of all relevant physical parameters

    Returns:
        c: A 2D complex array of shape (N, N), containing values c(K, k) on the meshgrids
    """
    k_0 = phys_params.k_0; alpha = phys_params.alpha; beta = phys_params.beta; sg_xpb = phys_params.sigma_xpb
    
    # Determine k_p0
    k_p0 = k_0 / beta
    
    # Compute exponential part
    exp_part   = np.exp(-((k_p0 - alpha * K - k)*sg_xpb)**2 - ((-beta*K + k)*sg_xpb)**2, dtype=np.complex128)
    
    # Compute phase part
    phase_part = np.exp(-1j* (k_p0 - alpha * K - k), dtype=np.complex128)
    
    # Compute error function part
    erf_part   = 1.0
    
    # Merge parts
    c_fun = exp_part * phase_part * erf_part
    
    # c(K, k<0) = 0
    c_fun[k < 0] = 0
    
    return 2 * sg_xpb * (2 * np.pi)**(-0.5) * c_fun

def coeff_r(k: NDArray[np.float64], interaction_regime: InteractionRegime, phys_params: PhysParams) -> NDArray[np.complex128]:
    """
    Evaluates amplitude coefficient of reflection for an array of values k on meshgrid.

    Parameters:
        k: 2D array of values k corresponding to points on meshgrid
        interaction_regime: enumeration of possible interaction regimes
        phys_params: instance of PhysParams class containg of all relevant physical parameters, including:
            - lmbda: interaction strenght
            - d: resonant parameter

    Returns:
        r: A 2D complex arrays of shape (N, N), containing values r(k) on the meshgrids
    """
    lmbda = phys_params.lmbda; d = phys_params.d
    
    # Single delta potential regime
    if interaction_regime == InteractionRegime.SINGLE_DELTA:
        return lmbda / (1j * k - lmbda)
    
    # Resonant potential regime
    else:
        phi = k * d
        r = np.empty_like(k, dtype=np.complex128)
        
        numer = 2 * lmbda * (k * np.cos(phi) + lmbda * np.sin(phi))
        denom = k * (1j * k - 2 * lmbda) * np.exp(-1j * phi) - 2 * lmbda**2 * np.sin(phi)
        
        mask = (denom != 0)
        r[np.logical_not(mask)]  = 0
        r[mask]   = numer[mask] / denom[mask] 
        
        return r

def coeff_t(k: NDArray[np.float64], interaction_regime: InteractionRegime, phys_params: PhysParams) -> NDArray[np.complex128]:
    """
    Evaluates amplitude coefficient of transmission for an array of values k on meshgrid.

    Parameters:
        k: 2D array of values k corresponding to points on meshgrid
        interaction_regime: enumeration of possible interaction regimes
        phys_params: instance of PhysParams class containg of all relevant physical parameters, including:
            - lmbda: interaction strenght
            - d: resonant parameter

    Returns:
        t: A 2D complex arrays of shape (N, N), containing values t(k) on the meshgrids
    """
    lmbda = phys_params.lmbda; d = phys_params.d
    
    # Single delta potential regime
    if interaction_regime == InteractionRegime.SINGLE_DELTA:
        return 1j * k / (1j * k - lmbda)
    
    # Resonant potential regime
    else:
        phi = k * g.d
        t = np.empty_like(k, dtype=np.complex128)
        
        numer = k**2
        denom = k * (k + 2j * lmbda) * np.exp(-1j * phi) + 2j * lmbda**2 * np.sin(phi)
        
        mask = (denom != 0)
        t[np.logical_not(mask)]  = 0
        t[mask] = numer[mask] / denom[mask] 
        
        return t

def eigen_energy(K: NDArray[np.float64], k: NDArray[np.float64], phys_params:PhysParams):
    """
    Evaluates eigen energy E(K, k) on a meshgrid of points (K, k) .

    Parameters:
        K, k: 2D arrays of values K and k corresponding to points on meshgrid
        phys_params: instance of PhysParams class containg of all relevant physical parameters

    Returns:
        E: A 2D real arrays of shape (N, N), containing values of eigen energy
    """
    
    return 0.5 * (phys_params.alpha * phys_params.beta * K**2 + k**2)

def calc_Psi(
                time: float, num_params: NumParams, phys_params: PhysParams,
                K:  NDArray[np.float64], k:  NDArray[np.float64], E_Kk:  NDArray[np.float64],
                c: NDArray[np.complex128], r_times_c:  NDArray[np.complex128], t_times_c:  NDArray[np.complex128]
            ) -> NDArray[np.complex128]:
    """
    Computes the wave function on the discrete grid of points (X_p, x_q)
    
    Parameters:
        time: time


    Returns:
        A 2D complex array of shape (N, N), containging the values of the wave function calculated on the grid of points (X_p, x_q) in time;  [Psi(X_p, x_q)]
    """
    N = num_params.N; d_Xx = num_params.d_Xx; k_0 = phys_params.k_0

    # Calculate arrays chi_inc, chi_R and chi_T
    chi_inc = c * np.exp( -1j * time * E_Kk / k_0)
    chi_R   = r_times_c * np.exp( -1j * time * E_Kk / k_0)
    chi_T   = t_times_c * np.exp( -1j * time * E_Kk / k_0)

    # Flip axis k in chi_R; chi_R(K, k) -> chi_R(K, -k)
    chi_R = chi_R[::-1 , :]

    # Reorder arrays for inverse FFT
    chi_minus = np.fft.ifftshift(chi_inc + chi_R)
    chi_plus  = np.fft.ifftshift(chi_T)

    # Perform FFT
    Psi_minus = np.fft.ifft2(chi_minus)
    Psi_plus  = np.fft.ifft2(chi_plus)
    
    # Reorder arrays to be zero centered
    Psi_minus = np.fft.fftshift(Psi_minus)
    Psi_plus  = np.fft.fftshift(Psi_plus)

    # Join arrays on axis x
    Psi_minus = Psi_minus[0 : N >> 1, :]
    Psi_plus  = Psi_plus[N >> 1 : N, :]
    Psi = np.concatenate((Psi_minus, Psi_plus), axis=0, dtype=np.complex128)
    
    return 2 * np.pi * d_Xx**(-2.0) * Psi

def bilin_interpol(P_com: NDArray[np.float64], x_p: NDArray[np.float64], x_b: NDArray[np.float64], num_params: NumParams, phys_params: PhysParams) -> NDArray[np.float64]:
    """
    Performs bilinear interpolation on a 2D grid.
    
    Parameters:
        P_com: 2D array of shape (N, N) containing probability density in points of CoM grid
        x_p, x_b: 2D arrays of shape (N/2, N/2) containing meshgrids of points x_p and x_b
        num_params: instance of NumParams class containing numerical parameters, including:
            - N: number of COM-frame grid points in each dimension
            - d_Xx: grid spacing of spatial coordinates
            - L: length of the computation region
        phys_params: instance of PhysParams class containg of all relevant physical parameters, including:
            - alpha: projectile mass fraction parameter
            - beta: barrier mass fraction parameter
    
    Returns:
        Interpolated values of probability density in the points of laboratory grid (x_p, x_b) in two dimensional array of shape (N/2, N/2)
    """
    N = num_params.N; L = num_params.L; alpha = phys_params.alpha; beta = phys_params.beta
    
    # Map Lab grid to CoM
    X_ij = alpha * x_p + beta * x_b
    x_ij = x_p - x_b
    
    # Estimate indecies
    rr = N * (0.5 + X_ij / L)
    ss = N * (0.5 + x_ij / L)
    
    # Get closest lower indices
    r_l = np.floor(rr).astype(int)
    s_l = np.floor(ss).astype(int)
    
    # Get closest upper indices
    r_u = r_l + 1
    s_u = s_l + 1
    
    # Compute interpolation weights
    h = ss - s_l
    w = rr - r_l
    
    mask_h  = h != 0
    mask_w  = w != 0
    mask_hw = mask_h & mask_w
    
    # Retrieve values from the four surrounding grid points
    p1 = P_com[s_l, r_l]
    
    p2 = np.zeros_like(p1)
    p2[mask_h] = P_com[s_u[mask_h], r_l[mask_h]]

    p3 = np.zeros_like(p1)
    p3[mask_hw] = P_com[s_u[mask_hw], r_u[mask_hw]]

    p4 = np.zeros_like(p1)
    p4[mask_w] = P_com[s_l[mask_w], r_u[mask_w]]

    # Bilinear interpolation formula
    P_lab = (1 - h) * (1 - w) * p1 + h * (1 - w) * p2 + h * w * p3 + (1 - h) * w * p4
    
    return P_lab

def calc_avg_pos(P_lab: NDArray[np.float64], x_p: NDArray[np.float64], x_b: NDArray[np.float64], num_params: NumParams, time:float, fw: TextIO) -> None:
    """
    Calculates average positions of projectile and barrier based on a 2D probability density in laboratory frame.

    Parameters:
        P_lab: 2D array of shape (N, N) containing the probability density in laboratory-frame coordinates
        x_p, x_b: 2D arrays of shape (N, N) containing meshgrids of projectile and target positions
        num_params: instance of NumParams class containing numerical parameters, including:
            - N_LF: number of LAB-frame grid points in each dimension
            - d_Xx: grid spacing in laboratory coordinates
        time: float value of the current time step
        fw: open file object for writing the output CSV row

    Returns:
        None. Results are written as a row to the provided output file:
            [time, total probability, transmission, reflection,
             avg_x_p (reflected), avg_x_b (reflected),
             avg_x_p (transmitted), avg_x_b (transmitted)]
    """
    N_LF = num_params.N_LF
    
    ## x_p < x_b
    # Mask lower diagonal where i < j
    j, i = np.tril_indices(N_LF)
    
    P_ref   = P_lab[j, i]
    x_p_ref = x_p[j, i]
    x_b_ref = x_b[j, i]
    
    # Calculate conditional probability
    cp_ref = np.sum(P_ref)
    R = cp_ref * (num_params.d_Xx)**2
    
    # Calculate averages
    x_p_ref  = np.sum(x_p_ref * P_ref) / cp_ref
    x_b_ref  = np.sum(x_b_ref * P_ref) / cp_ref
    
    ## x_p >= x_b
    # Mask upper diagonal where i >= j
    j, i = np.triu_indices(N_LF)
    
    P_trans = P_lab[j, i]
    x_p_trans = x_p[j, i]
    x_b_trans = x_b[j, i]
    
    # Calculate conditional probability
    cp_trans = np.sum(P_trans)
    T = cp_trans * (num_params.d_Xx)**2
    
    # Calculate averages
    x_p_trans  = np.sum(x_p_trans * P_trans) / cp_trans
    x_b_trans  = np.sum(x_b_trans * P_trans) / cp_trans
    
    # Append new row of results to output file
    try:
        writer = csv.writer(fw, delimiter=CSV_DELIMITER)  # use your chosen delimiter here
        writer.writerow([
            f"{time:.2f}", f"{R + T: .6e}", f"{T:.6e}", f"{R:.6e}", f"{x_p_ref:.6e}", f"{x_b_ref:.6e}",
            f"{x_p_trans:.6e}", f"{x_b_trans:.6e}"
        ])
    
    except Exception as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

def print_progress(step: int, N_steps: int, bar_len=50) -> None:
    """Prints and inline terminal progress-bar.

    Parameters:
        step: current step of main computation loop
        N_steps: number of total steps
        bar_len: bar width in characters (default 50).
    """
    progress = step / (N_steps - 1)
    pbar_len = int(bar_len * progress)
    pbar = "#" * pbar_len + "-" * (bar_len - pbar_len)
    sys.stdout.write(f"\rProgress: |{pbar}| {progress * 100: 6.2f}% ")
    
    if progress == 1.0: print("\nFinished!")
    
    sys.stdout.flush()

def save_bounded_prob(P_lab: NDArray[np.float64], num_params: NumParams, filename: str, time: float) -> None:
    """
    Creates a datafile containg a minimal rectangular region, where 2D probability density can be regared as non-zero (>=PD_ZERO_MARGIN).
    Probability density slice is stored in plain-text file.
    
    Parameters:
        P_lab: 2D array of shape (N, N) containing the probability density in laboratory-frame coordinates
        num_params: instance of NumParams class containing numerical parameters, including:
            - N_LF: number of LAB-frame grid points in each dimension
            - d_Xx: grid spacing in laboratory coordinates
        filename: name of column-separated output file
        time: float value of the current time step

    Returns:
        None. Results are written as a row to the provided output file.

        Output file layout:
        -------------------
        <L_LF> <N_LF> <time>
        <col_offset> <row_offset>                   
        <data block>
        EOF
    """
    # Find subset where probability density can be regared as nonzero
    nonzero_mask = np.argwhere(P_lab >= PD_ZERO_MARGIN)
    
    FIRST_ROW = [f"{0.5 * num_params.L}", f"{num_params.N_LF}", f"{time:.2f}"]
    
    # Case where probability is zero everywhere
    if nonzero_mask.size == 0:
        with open(filename, "w") as fw:
            fw.write(CSV_DELIMITER.join(FIRST_ROW) + "\n")
            # Zero offset
            fw.write(f"0{CSV_DELIMITER}0\n")
            # Zero everywhere
            fw.write("0.0\n")
        return
    
    # Define boudning box
    i_min, j_min = nonzero_mask.min(axis=0)
    i_max, j_max = nonzero_mask.max(axis=0)
    
    # Extract subregion within bounding box
    if (i_min, j_min) == (0, 0) and (i_max, j_max) == (num_params.N_LF - 1, num_params.N_LF - 1): #Non-zero everywhere
        nonzero_prob = P_lab
    else:
        nonzero_prob = P_lab[i_min : i_max, j_min : j_max]
    
    # Save subregion of nonzero probability
    try:
        with open(filename, "w") as fw:
            # Save computation region parameters
            fw.write(CSV_DELIMITER.join(FIRST_ROW) + "\n")
            # Save data offset
            fw.write(f"{j_min:d}{CSV_DELIMITER}{i_min:d}\n")
            # Save rest of the subregion normally
            np.savetxt(fw, nonzero_prob, delimiter=CSV_DELIMITER)
    
    except (IOError, OSError) as e:
        print(f"Error writing to probability density output file: {e}")
        sys.exit(1)
    
    except Exception as e:
         print(f"Failed while saving sub-region: {e}")