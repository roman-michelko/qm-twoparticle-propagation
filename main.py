#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: main.py
LAST EDIT: 2025-05-08
AUTHOR: Roman Michelko

DESCRIPTION:
-------------
Main computation script — loads configuration, computes the quantum wavefunction in the center-of-mass frame, transforms 
the corresponding probability denisty to the laboratory frame, evaluates average particle positions and saves probability desnity
evaluated on grid of laboratory frame coordinates.
"""

__author__  = "Roman Michelko"
__license__ = "MIT"
__version__ = "1.0.0"

import os, sys, re, argparse
import numpy as np
from functions import *

#%% Constants
DEFAULT_CONFIG_FILE = r"config_file.json"
DEFAULT_OUTPUT_FILE = r"output_file.csv"

PD_OUTPUT_DIR  = r"prob_density/"
PD_OUTPUT_FILE = r"prob_dens{:d}.csv"

OUTPUT_FILE_HEADER = [
    "time", 
    "total_norm", 
    "trans_prob", 
    "ref_prob", 
    "avg_x_p_ref", 
    "avg_x_b_ref", 
    "avg_x_p_trans", 
    "avg_x_b_trans"
]
    
#%% Main function
def main(argv: list):
    
    # Check Python version
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or higher is required to run this script.")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute and save average particle positions and the probability density in the laboratory frame.")
    parser.add_argument("-i", "--input", default=DEFAULT_CONFIG_FILE, help="Path to input config file")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, help="Path where the output data file will be saved")
    args = parser.parse_args(argv[1:])
    
    # Load config file
    interaction_regime, num_params, comp_params, phys_params = load_config(args.input)
    
    # Allocate meshgrids
    K, k   = allocate_meshgrid(num_params.N, num_params.d_Kk)
    x_p, x_b = allocate_meshgrid(num_params.N_LF, num_params.d_Xx)
    
    # Compute 2D array of amplitude coeffiecient functiom "c(K, k)" values on K, k meshgrid
    c = calc_c(K, k, phys_params)
    
    # Compute 2D arrays of relfection and transmission coefficietns
    r = coeff_r(k, interaction_regime, phys_params)
    t = coeff_t(k, interaction_regime, phys_params)
    
    # Pre-compute time-independent parts of function \chi_R(K, k) and \chi_T(K, k)
    r_times_c = r * c
    t_times_c = t * c
    
    # Pre-compute eigen energies
    E_Kk = eigen_energy(K, k, phys_params)
    
    # Check probability density output dir
    if not os.path.exists(PD_OUTPUT_DIR):
        try:
            os.makedirs(PD_OUTPUT_DIR)
            
        except Exception as e:
            print(f"Failed to create directory \"{PD_OUTPUT_DIR}\": {e}")
            sys.exit(1)
    
    # Check probability density output files
    exists_out_file = None
    
    if os.path.exists(args.output): exists_out_file = args.output
    else:
        pattern_str = PD_OUTPUT_FILE.replace("{:d}", r"(\d+)")
        regex = re.compile(f"^{pattern_str}$")
        for filename in os.listdir(PD_OUTPUT_DIR):
            if regex.match(filename): exists_out_file = filename
    
    if exists_out_file is not None:
        res = input(f"File \"{exists_out_file}\" already exists and will be overwritten. Do you want to continue (y/n): ").strip().lower()
        if res != 'y':
            print("Exiting the program.")
            sys.exit(2)
    
    # Open data output file
    try:
        fw = open(args.output, mode="w", newline="")
        fw.write(CSV_DELIMITER.join(OUTPUT_FILE_HEADER) + "\n")
    
    except (IOError, OSError) as e:
        print(f"Error opening file \"{args.output}\" for writing: {e}")
        sys.exit(1)
    
    # Main time loop
    time = comp_params.t_start
    for n in range(comp_params.Nt):
        
        # Compute wavefunction
        Psi = calc_Psi(time, num_params, phys_params, K, k, E_Kk, c, r_times_c, t_times_c)
        P_COM = np.real(np.conj(Psi) * Psi)
        
        # Billinear transform to Laboratory frame
        P_LAB = bilin_interpol(P_COM, x_p, x_b, num_params, phys_params)
        
        # Calcualte average particle positions
        calc_avg_pos(P_LAB, x_p, x_b, num_params, time, fw)
        
        # Save probability density
        save_bounded_prob(P_LAB, num_params, PD_OUTPUT_DIR+PD_OUTPUT_FILE.format(n), time)
        
        # Print progress
        print_progress(n, comp_params.Nt)
        
        # Append time
        time += comp_params.time_step
    
    #Close data output file
    try:
        fw.close()
    except (IOError, OSError) as e:
        print(f"Error opening file \"{args.output}\" for writing: {e}")
    
    # End program
    sys.exit(0)
    
#%% Run main
if __name__ == "__main__":
    main(sys.argv)