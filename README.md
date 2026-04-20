# gkma_tool

Green-Kubo Modal Analysis (GKMA) for thermal transport in amorphous and disordered solids.

Decomposes lattice thermal conductivity into frequency-resolved mode contributions (propagons, diffusons, locons) from a LAMMPS NVE trajectory and a dynamical matrix.

## Prerequisites

**Python 3.8+** with the following packages:

```
numpy
scipy
numba
matplotlib
```

Install with:

```
pip install numpy scipy numba matplotlib
```

## Required input files

Different subcommands need different files. The table below shows what each subcommand reads.

| File | Description | Needed by |
|------|-------------|-----------|
| `*.data` | LAMMPS data file (atom types, masses, coordinates, box) | all |
| `dynmat.dat` | Dynamical matrix in ESKM format from `dynamical_matrix all eskm` | all |
| `dump_gkma.lammpstrj` | NVE trajectory dump | `kappa` |
| `heatflux_total.dat` | Total heat flux time series (optional, for GK validation) | `kappa` |

### Data file format

Standard LAMMPS data file with `atom_style atomic`. Must contain Masses and Atoms sections. The Atoms section uses the format `id type x y z`.

### Dynamical matrix

Produced by the LAMMPS `dynamical_matrix` command with the `eskm` keyword. This gives a mass-weighted dynamical matrix whose eigenvalues are (2*pi*freq)^2 in THz^2. See `lammps/in.dynmat` for a template.

### NVE dump

The trajectory dump must contain per-atom positions, velocities, potential energy and the 6-component stress tensor. The expected column names are:

```
id x y z vx vy vz c_pe c_s[1] c_s[2] c_s[3] c_s[4] c_s[5] c_s[6]
```

Variants `xu yu zu`, `c_PE`, `c_stress[1..6]` are also detected automatically. The dump must be sorted by atom ID (`dump_modify ... sort id`). See `lammps/in.gkma` for a production template.

### Heat flux file (optional)

A text file with columns `step Jx Jy Jz` at the same dump frequency as the trajectory. Used only for cross-checking the GKMA sum against standard Green-Kubo. If absent the validation step is skipped.

## Usage

All subcommands share a common set of flags. Run from the directory containing your input files.

### Compute eigensystem only

```
python gkma_tool.py eigensystem --datafile 300K.data --dynmat-file dynmat.dat --plot
```

Produces `eigenfreqs.npy`, `eigenvecs.npy`, `masses.npy`, `positions.npy`, `vdos.dat`, `participation_ratio.dat`, `mse.dat`. These are cached and reused by subsequent commands.

### VDOS

```
python gkma_tool.py vdos --datafile 300K.data --dynmat-file dynmat.dat --plot
```

Computes the eigensystem (or loads it if cached) and prints the VDOS summary. With `--plot` saves `plot_vdos.png`.

### Participation ratio

```
python gkma_tool.py pr --datafile 300K.data --dynmat-file dynmat.dat --plot
```

Prints PR statistics and with `--plot` saves `plot_pr_mse.png` (PR vs frequency and MSE vs PR scatter).

### Ioffe-Regel crossover

```
python gkma_tool.py ioffe-regel --datafile 300K.data --dynmat-file dynmat.dat --plot
```

Computes the branch-resolved (longitudinal/transverse) Ioffe-Regel crossover frequencies from the eigenvector structure factor F(q,nu). With `--plot` saves `plot_ioffe_regel.png`.

Additional flags: `--ir-nq-radial` (default 50), `--ir-nq-angular` (default 40).

### Full thermal conductivity

```
python gkma_tool.py kappa \
    --datafile 300K.data \
    --dynmat-file dynmat.dat \
    --dump-file dump_gkma.lammpstrj \
    --T 300 \
    --timestep-ps 0.0001 \
    --dump-freq 100 \
    --plot
```

Runs the full pipeline: eigensystem, Ioffe-Regel, trajectory processing, thermal properties and (with `--plot`) all visualisations.

Additional flags:

- `--hf-file`: heat flux file for GK validation
- `--nbins`: frequency bins (default 120)
- `--freq-max`: max frequency in THz (default 60)
- `--max-frames`: limit number of trajectory frames
- `--corr-fraction`: fraction of frames for correlation window (default 0.25)
- `--effective-thickness`: for 2D materials, effective thickness in Angstrom
- `--skip-ir`: skip Ioffe-Regel analysis
- `--nworkers`: parallel workers (default: all CPUs)

### JSON config file

Instead of passing all flags on the command line, you can write a JSON file:

```json
{
    "datafile": "300K.data",
    "dynmat_file": "dynmat.dat",
    "dump_file": "dump_gkma.lammpstrj",
    "T": 300.0,
    "timestep_ps": 0.0001,
    "dump_freq": 100,
    "freq_max": 60.0,
    "nbins": 120
}
```

Then:

```
python gkma_tool.py kappa --config params.json --plot
```

CLI flags override the config file, and the config file overrides built-in defaults.

## Output files

### Eigensystem stage

| File | Contents |
|------|----------|
| `eigenfreqs.npy` | Eigenfrequencies in THz (3N array) |
| `eigenvecs.npy` | Eigenvectors (3N x 3N matrix) |
| `masses.npy` | Atomic masses in amu |
| `positions.npy` | Equilibrium positions in Angstrom |
| `vdos.dat` | VDOS: freq_THz, DOS_modes_per_THz |
| `participation_ratio.dat` | freq_THz, PR |
| `mse.dat` | freq_THz, PR, MSE_Angstrom |

### Ioffe-Regel stage

| File | Contents |
|------|----------|
| `Fqnu_L.npy` / `Fqnu_T.npy` / `Fqnu_total.npy` | Eigenvector structure factor F(q,nu) |
| `Fqnu_axes.npz` | q_vals and freq_bins arrays |
| `ioffe_regel.dat` | IR crossover frequencies and sharpness vs frequency |

### Thermal conductivity stage

| File | Contents |
|------|----------|
| `kappa_vs_freq.dat` | kappa(nu) classical and quantum-corrected, modes per bin |
| `kappa_accumulation.dat` | Cumulative kappa vs frequency |
| `kappa_mode_map.npy` | Mode-mode kappa correlation matrix |
| `spectral_diffusivity.dat` | Thermal diffusivity D(nu), Debye and random-walk references |
| `dispersion_CL.npy` / `dispersion_CT.npy` | Velocity-current L/T power spectra |
| `dispersion_axes.npz` | q_vals and freq arrays for dispersion |
| `dispersion_fits.dat` | DHO fits: peak frequency, linewidth, group velocity, MFP |
| `relaxation_times.dat` | Mode relaxation times: freq_THz, PR, tau_ps |
| `mode_contributions.dat` | Propagon/diffuson/locon kappa breakdown |

### Trajectory intermediates

| File | Contents |
|------|----------|
| `Q_binned_all.npy` | Binned modal heat flux (frames x bins x 3) |
| `Q_total_all.npy` | Total modal heat flux (frames x 3) |
| `modal_amp_store.npy` | Modal amplitudes for relaxation time calculation |
| `j_velocity_current.npy` | Velocity-current data for dispersion |
| `trajectory_meta.npz` | Metadata (bin centres, edges, volume, dt, etc.) |

## LAMMPS templates

Two LAMMPS input templates are provided in `lammps/`:

- `in.dynmat`: minimisation followed by `dynamical_matrix all eskm` to produce `dynmat.dat`
- `in.gkma`: NVT thermalisation followed by NVE production run with per-atom energy and stress dumps

Both require editing the potential block and adjustable parameters at the top of the file.

## Notes

- The eigensystem is the most memory-intensive step. For N atoms the eigenvector matrix is 3N x 3N floats. A 5000-atom system needs about 6 GB of RAM for the eigenvectors alone.
- Trajectory processing is I/O bound. Performance scales roughly linearly with the number of frames.
- All intermediate `.npy` files are cached. Subsequent runs of any subcommand will detect and reuse them. Delete them to force recomputation.
- Units throughout are LAMMPS metal: eV, Angstrom, ps, amu, bar. The reported thermal conductivity is in W/(m*K).
- For 2D materials, use `--effective-thickness` to specify the out-of-plane dimension used for volume normalisation.
