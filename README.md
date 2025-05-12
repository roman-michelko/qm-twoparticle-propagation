# qm-twoparticle-propagation
Numerical computation code for finding the time-dependent solution of a one-dimensional two-particle system. Includes FFT-based wave function propagation, conditional probability analysis, and post-collision observables.
***
## Overview
This code calculates one-dimensional scattering dynamics between two distinguishable quantum particles. The initial state of the system can be configured by specifying physical parameters, expressed in system-specific natural units, in a configuration file. The system's wave function is propagated in the center-of-mass reference frame using a Fast Fourier Transform to obtain time-resolved probability densities. These densities are then converted to a grid in the laboratory frame, representing individual particle positions, using bilinear interpolation. Blocks of computed probability density data are saved as plain-text files. Observables such as reflection and transmission probabilities, as well as particle position expectation values, are determined from the laboratory-frame probability density and saved accordingly. For a complete description of the physical model, theoretical background, and interpretation of results, refer to our accompanying preprint on [arXiv:2504.16575](https://arxiv.org/abs/2504.16575). This repository provides the source code used to perform the computations presented in the manuscript.
### Method Summary
1. **Initialization**: User-defined parameters are read-from configuration file in JSON format.
2. **Grid Setup**: Meshgrids in momentum space ($K, k$) and laboratory coordinates ($x_\mathrm{p}, x_\mathrm{b}$) are allocated.
3. **Preprocessing**: Amplitude coefficients, reflection/transmission amplitudes, and energy eigenvalues are precomputed.
4. **Time Evolution**:
   - The wavefunction is propagated using FFT.
   - The CoM-frame probability density is transformed into the lab frame via bilinear interpolation.
   - Observables are evaluated and stored.
1. **Output**:
   - Expectation values of particle positions and conditional probabilities at each time step are saved to a CSV file.
   - Spatial probability densities are saved per timestep in the `prob_density/` directory.
### File Structure
- `main.py` - Main script
- `functions.py` - Support module containing function definitions and other utilities
- `config_file.json` - A configuration file example
- `output_file.csv` - Output file example
- `prob_density/` - Output directory for time-resolved probability density snapshots
- `examples/plot_prob_density_example` - Illustrative script for visualizing computed probability densities from generated data files
### Natural units and initial physical setup of the system
The computation is carried in a system of natural units, in which all physical quantities are expressed as dimensionless ratios relative to characteristic reference scales. This choice simplifies the mathematical formulation and brings out the universal structure of the underlying collision dynamics. Three base units of length, time and mass are defined in this natural unit system as:
- **Unit of length ($\ell$)** - defined by initial average relative distance between two particles\
  $$1\ \ell =x_0=x_\mathrm{p0}-x_\mathrm{b0}$$
- **Unit of time ($\Large \tau$)** - The time it would take for two particles, initially separated by a distance of exactly $1\ \ell$, to collide if their relative velocity was exactly equal to their initial average relative velocity $v_0=v_\mathrm{p0}-v_\mathrm{b0}$.\
  $$1 \ {\Large \tau} = |x_0/v_0|$$
- **Unit of mass ($\mu$)** - The reduced mass of the two-particle system, defined as:\
  $$1 \ \mu =\frac{m_\mathrm{p}m_\mathrm{b}}{m_\mathrm{p}+m_\mathrm{b}}.$$\
With the choice of natural units, the dimensionless initial average relative wave number $k_0$ corresponds to a quantity given by the following expression:\
$$k_0=\frac{\mu\ell^2}{\hbar {\Large \tau}}.$$
#### Interaction potential
The Schrödinger equation governing the two-particle system, expressed in the natural unit system, takes the form:\
$$ik_0\frac{\partial\Psi}{\partial t}=-\frac{\alpha\beta}{2}\frac{\partial^2 \Psi}{\partial X^2}-\frac{1}{2}\frac{\partial^2\Psi}{\partial x^2}+\tilde{V}(x)\Psi(X, x, t),$$\
where $\tilde{V}(x)$ denotes dimensionless interaction potential. The computation supports two possible distinct interaction regimes mediated by following two types of dimensionless interaction potential:
- **Single delta function (Contact) interaction**  
  A localized point interaction modeled by a single delta function at $x=0$:\
  $$\tilde{V}(x)=\lambda\delta(x).$$\
  This regime represents an instantaneous, short-range contact interaction between the particles.
- **Double delta function (Resonant) interaction**
  A symmetric potential formed by two delta functions located at positions $x=\pm d/2$:\
  $$\tilde{V}(x) = \lambda \left(\delta(x+d/2) + \delta(x-d/2)\right).$$\
  This regime supports quantum interference and resonance effects tuned by the resonance parameter $d$.

The interaction type is selected in the configuration file via the `"interaction_regime"` parameter (where `"sd"` stands for single delta and `"res"` stands for resonant double delta).



***
## Usage Instructions
See this section for instructions on how to properly setup the configuration file and execute the main program.
### Configuration File Setup
Numerical, physical, and computational parameters are specified in a JSON configuration file. A valid configuration must contain the following fields:

- **`interaction_regime`**: Specifies the type of interaction. Valid options are:
  - `"sd"` - Single delta function interaction
  - `"rd"` - Resonant double delta function interaction

- **`numerical_parameters`**: Defines the resolution and spatial domain.
  - `power_of_two` *(int)*: Defines the grid size as `N = 2**power_of_two`.
  - `interval_length` *(float)*: Length of the spatial interval in each direction of center of mass coordinates (in natural units); denoted as `L`.

- **`computation_parameters`**: Time domain for the computation.
  - `time_range` *(list of two floats)*: Start and end time of the computation `[t_start, t_end]`
  - `time_step` *(float)*: Temporal step size

- **`physical_parameters`**: Describes physical properties of the initial state and interaction potential.
  - `k_0` *(float)*: Mean relative wavenumber of initial incident wave packet
  - `lambda` *(float)*: Strength of interaction constant
  - `alpha` *(float)*: Ratio of the projectile particle mass to the total mass. Must lie in the interval $\alpha \in (0, 0.5)$, corresponding to a lighter projectile and a heavier barrier.
  - `sigma_xpb` *(float)*: Initial spatial uncertainty of both particles 
  -  `d` *(float or null)*: Resonance parameter - the distance between two delta barriers. Required only for the double-barrier regime.

#### Configuration file example

```json
{
  "interaction_regime": "sd",
  "numerical_parameters": {
    "power_of_two": 12,
    "interval_length": 100.0
  },
  "computation_parameters": {
    "time_range": [2, 8],
    "time_step": 0.5
  },
  "physical_parameters": {
    "k_0": 50.0,
    "lambda": 50.0,
    "alpha": 0.01,
    "sigma_xpb": 0.1,
    "d": null
  }
}
```

### Executing program
Once a valid configuration file has been created, the program can be executed by running the `main.py` script from the terminal as follows:

```bash
python main.py -i config.json -o results.csv
```
#### Available command line arguments
- `-i` or `--input`: Path to the input configuration file (default: `config_file.json`)
- `-o` or `--output`: Path to the output CSV file containing computed observables at each time step (default: `output.csv`)

If the command line arguments are omitted, the script will use the default filenames located in the current working directory.
### Parameters validation
After executing the main script, the program verifies all parameters specified by the user in the configuration file. If any parameters are missing or invalid, the program notifies the user and terminates. Additionally, if the parameters are valid but there is a potential risk of numerical inaccuracy in the results, a warning is issued to inform the user of this possibility.
### Requirements
- Python ≥ 3.9
- NumPy, Matplotlib
***
## Output files
Upon successful execution of the program, the following output files are generated:
#### Main output file (default: `output_file.csv`)
This file contains physical observables computed at each time step. Each line corresponds to one time point and contains comma-separated values (in natural units) in the following order:

- `time` - Physical time of current time step
- `total_norm` - Total probability norm (should be close to 1)
- `trans_prob` - Transmission probability (projectile passes the barrier)
- `ref_prob` - Reflection probability (projectile is reflected)
- `avg_x_p_ref` - Mean position of the projectile in the reflected part of the wave packet
- `avg_x_b_ref` - Mean position of the barrier in the reflected part of the wave packet
- `avg_x_p_trans` - Mean position of the projectile in the transmitted part of the wave packet
- `avg_x_b_trans` - Mean position of the barrier in the transmitted part of the wave packet

**Example line form `output_file.csv`:**
```csv
2.00, 1.000002e+00,5.000610e-01,4.999411e-01,-9.509179e-01,1.931608e-02,1.009759e+00,9.417300e-05
```

#### Probability density output file (default: `prob_dens%d.csv`)
For each time step of the computation, a file named `prob_dens%d.csv` is saved in the `prob_density/` directory. Each file contains a 2-dimensional array representing the computed probability density on a laboratory-frame grid. For efficiency, only the minimal rectangular region where the probability density is non-negligible is stored. These files have the following layout:

```csv
<L_LF>,<N_LF>,<time>
<col_offset>,<row_offset>
<data block>
```
##### Line 1 - contains global information about grid and time
- `<L_LF>` - Length of the lab-frame spatial domain
- `<N_LF>` - Number of grid points per axis in the lab frame
- `<time>` - Time at which this probability density snapshot was computed

**Example:**
```csv
50.0,2048,3.00
```
##### Line 2 - contains probability density slice offset
This line indicates the top-left coordinate (in grid indices) of the sliced nonzero subregion within the full array:
- `<col_offset>` - Starting column index (along $x_\mathrm{p}$)
- `<row_offset>` - Starting row index (along $x_\mathrm{b}$)

This allows the data block to be positioned correctly when reconstructing the full grid, with the rest of the array being zero.

**Example:**
```csv
858,1001
```
##### Rest of the file
The remainder of the file contains a 2-dimensional array of floating-point values representing the computed probability density on the laboratory-frame grid. Each column corresponds to a fixed value of $x_\mathrm{p}$ (projectile position), and each row corresponds to a fixed value of $x_\mathrm{b}$ (barrier position).
***
## Examples
Example output files – `output_file.csv` and the probability density snapshots `prob_density/prob_dens%d.csv` – are included in the repository. These results were generated using the configuration specified in `config_file.json`, corresponding to the parameter set listed in the table below:

| Parameter                      | Value        | Description                                           |
| ------------------------------ | ------------ | ----------------------------------------------------- |
| Interaction regime             | sd           | Single delta (contact) interaction                    |
| $k_0$                          | 50           | Mean relative wavenumber                              |
| $\lambda$                      | 49.75        | Dimensionless interaction strength                    |
| $\alpha$                       | 1/101        | Mass fraction of the projectile                       |
| $\sigma_{x\mathrm{pb}}$        | 0.1          | Initial spatial uncertainty                           |
| power of two                   | 12           | Grid size in each direction: $N=2^{12} = 4096$ points |
| $L$                            | 100          | Length of square COM spatial domain                   |
| time range                     | $t\in[2, 8]$ | Time evolution interval                               |
| time step                      | 0.5          | Time step                                             |
| $d$                            | None         | Not used in single delta regime                       |
| Total probability of tunneling | ~50%         |                                                       |
### Visualization of Results
A minimalist plotting script is provided in:
```
examples/plot_prob_density_example.py
```
To run the script, navigate to the `examples/` directory and execute:
```
python plot_prob_density_example.py
```

The script will:
- Load all `prob_dens%d.csv` files from the `../prob_density/` directory  
- Reconstruct the full lab-frame probability density grid for each time step  
- Extract a predefined subregion for visualization  of probability densities
- Compute marginal probability densities of both particles
- Generate and save colormap plots with marginal distributions as `.png` images in the `examples/plots/` directory

Output images are named sequentially as:

```
prob_dens_cmap0.png, prob_dens_cmap1.png, ...
```

You can adjust the plotted region by modifying the `PLOT_REGION` dictionary inside the script.

***
## Authors
**Roman Michelko**  
**Peter Bokes** 

Department of Physics, Institute of Nuclear and Physical Engineering  
Faculty of Electrical Engineering and Information Technology  
Slovak University of Technology in Bratislava  
841 04 Bratislava, Slovak Republic

*Contact:*  
- Roman Michelko: [xmichelko@stuba.sk](mailto:xmichelko@stuba.sk)  
- Peter Bokes: [peter.bokes@stuba.sk](mailto:peter.bokes@stuba.sk)
