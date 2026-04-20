#!/usr/bin/env python3
"""
gkma_tool  –  Green-Kubo Modal Analysis for amorphous solids

Subcommands
  eigensystem    Diagonalise the dynamical matrix (VDOS, PR, MSE)
  vdos           Vibrational density of states
  pr             Participation ratio and mode spatial extent
  ioffe-regel    Branch-resolved Ioffe-Regel crossover analysis
  kappa          Full GKMA thermal conductivity pipeline

Run  gkma_tool.py <command> --help  for per-command options.
"""

import argparse
import json
import numpy as np
import time
import sys
import os
import concurrent.futures
from multiprocessing import cpu_count
from numba import njit, prange
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Physical constants (metal units: eV, Angstrom, ps, amu)
# ---------------------------------------------------------------------------
KB_EV         = 8.617333262e-5
AMU_ANGPS2_EV = 1.0364269e-4
BAR_ANG3_EV   = 6.24150974e-7
KAPPA_CONV    = 1602.176634
H_EV_PS       = 4.135667696e-3
TWO_PI        = 2.0 * np.pi

# ---------------------------------------------------------------------------
#  Defaults (overridden by --config JSON or CLI flags)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    datafile            = "300K.data",
    dynmat_file         = "dynmat.dat",
    dump_file           = "dump_gkma.lammpstrj",
    hf_file             = "heatflux_total.dat",
    T                   = 300.0,
    timestep_ps         = 0.0001,
    dump_freq           = 100,
    nbins               = 120,
    freq_max            = 60.0,
    max_frames          = None,
    corr_fraction       = 0.25,
    effective_thickness = 0,
    ir_nq_radial        = 50,
    ir_nq_angular       = 40,
    disp_nq             = 20,
    disp_nq_angular     = 20,
    disp_subsample      = 5,
    relax_freq_max      = 40.0,
    mse_nslices         = 40,
    nworkers            = None,
)

# ---------------------------------------------------------------------------
#  Utility
# ---------------------------------------------------------------------------
def timestamp():
    return time.strftime("%H:%M:%S")

def section(title):
    w = 60
    print(f"\n{'='*w}\n  {title}  [{timestamp()}]\n{'='*w}")

def build_config(args):
    """Merge DEFAULTS <- JSON config file <- CLI overrides."""
    cfg = dict(DEFAULTS)
    if hasattr(args, 'config') and args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    # CLI flags override
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    cfg['nworkers'] = cfg.get('nworkers') or cpu_count()
    return cfg

# ---------------------------------------------------------------------------
#  Quantum correction
# ---------------------------------------------------------------------------
def quantum_correction(freq_THz, T):
    x = H_EV_PS * freq_THz / (KB_EV * T)
    out = np.zeros_like(x)
    valid = (freq_THz > 0.01) & (x < 500)
    ex = np.exp(x[valid])
    out[valid] = x[valid]**2 * ex / (ex - 1.0)**2
    return out

# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------
def read_datafile(path):
    natoms = 0
    type_mass = {}
    atom_type = {}
    positions = {}
    box = np.zeros((3, 2))

    with open(path) as f:
        sec = None
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "atoms" in s and "Atoms" not in s:
                natoms = int(s.split()[0])
            elif "atom types" in s:
                pass
            elif "xlo xhi" in s:
                box[0] = [float(s.split()[0]), float(s.split()[1])]
            elif "ylo yhi" in s:
                box[1] = [float(s.split()[0]), float(s.split()[1])]
            elif "zlo zhi" in s:
                box[2] = [float(s.split()[0]), float(s.split()[1])]
            elif s == "Masses":
                sec = "masses"
                continue
            elif s.startswith("Atoms"):
                sec = "atoms"
                continue
            elif s in ("Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"):
                sec = None
                continue

            if sec == "masses":
                p = s.split()
                if len(p) >= 2:
                    try:
                        type_mass[int(p[0])] = float(p[1])
                    except ValueError:
                        pass
            elif sec == "atoms":
                p = s.split()
                if len(p) >= 5:
                    try:
                        aid = int(p[0])
                        atom_type[aid] = int(p[1])
                        positions[aid] = [float(p[2]), float(p[3]), float(p[4])]
                    except ValueError:
                        pass

    masses = np.array([type_mass[atom_type[i+1]] for i in range(natoms)])
    pos = np.array([positions[i+1] for i in range(natoms)])
    volume = np.prod(box[:, 1] - box[:, 0])
    return natoms, masses, pos, box, volume


def parse_dump_header(f):
    line = f.readline()
    if not line:
        return None
    ts = int(f.readline().strip())
    f.readline()
    na = int(f.readline().strip())
    f.readline()
    box = np.zeros((3, 2))
    for d in range(3):
        p = f.readline().split()
        box[d] = [float(p[0]), float(p[1])]
    hdr = f.readline().strip()
    cols = hdr.replace("ITEM: ATOMS", "").split()
    return ts, na, box, cols


def read_dump_frame_fast(f, natoms, ncols):
    lines = []
    for _ in range(natoms):
        lines.append(f.readline())
    data = np.empty((natoms, ncols))
    for i, ln in enumerate(lines):
        data[i] = ln.split()
    data = data[data[:, 0].argsort()]
    return data


def detect_dump_columns(col_names):
    return {c: i for i, c in enumerate(col_names)}

# ---------------------------------------------------------------------------
#  FFT-based correlation
# ---------------------------------------------------------------------------
def correlate_fft(A, B, max_corr):
    nf = A.shape[0]
    nfft = 2 * nf
    out = np.zeros(max_corr)
    for a in range(min(A.shape[1], B.shape[1])):
        Af = np.fft.rfft(A[:, a], n=nfft)
        Bf = np.fft.rfft(B[:, a], n=nfft)
        out += np.fft.irfft(np.conj(Bf) * Af, n=nfft)[:max_corr]
    norm = np.arange(nf, nf - max_corr, -1, dtype=float)
    out /= norm
    return out


def kappa_from_corr(corr, dt_ps, volume, T):
    integral = np.trapz(corr, dx=dt_ps)
    return volume / (3.0 * KB_EV * T**2) * integral * KAPPA_CONV

# ---------------------------------------------------------------------------
#  Numba kernels
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _compute_ir_qpoints(q_vals, q_dirs, positions, eigvecs_3d, freqs,
                        mode_bin, valid, n3, nqr, nqa, natoms, nfbin):
    FL = np.zeros((nqr, nfbin), dtype=np.float64)
    FT = np.zeros((nqr, nfbin), dtype=np.float64)
    Ftot = np.zeros((nqr, nfbin), dtype=np.float64)
    counts = np.zeros((nqr, nfbin), dtype=np.float64)

    for iq in prange(nqr):
        q_mag = q_vals[iq]
        fl_local = np.zeros(nfbin, dtype=np.float64)
        ft_local = np.zeros(nfbin, dtype=np.float64)
        cnt_local = np.zeros(nfbin, dtype=np.float64)

        for idir in range(nqa):
            qhat = q_dirs[idir]
            qvec = q_mag * qhat
            phase = np.zeros(natoms, dtype=np.complex128)
            for i in range(natoms):
                phase_arg = (positions[i, 0]*qvec[0]
                           + positions[i, 1]*qvec[1]
                           + positions[i, 2]*qvec[2])
                phase[i] = np.exp(1j * phase_arg)

            proj_L_mag2 = np.zeros(n3, dtype=np.float64)
            proj_T_mag2 = np.zeros(n3, dtype=np.float64)
            for s in range(n3):
                if not valid[s]:
                    continue
                proj_x = 0.0 + 0.0j
                proj_y = 0.0 + 0.0j
                proj_z = 0.0 + 0.0j
                for i in range(natoms):
                    p = phase[i]
                    proj_x += p * eigvecs_3d[i, 0, s]
                    proj_y += p * eigvecs_3d[i, 1, s]
                    proj_z += p * eigvecs_3d[i, 2, s]
                dot_q = qhat[0]*proj_x + qhat[1]*proj_y + qhat[2]*proj_z
                pL_x = dot_q * qhat[0]
                pL_y = dot_q * qhat[1]
                pL_z = dot_q * qhat[2]
                pT_x = proj_x - pL_x
                pT_y = proj_y - pL_y
                pT_z = proj_z - pL_z
                proj_L_mag2[s] = ((pL_x.real**2 + pL_x.imag**2)
                                + (pL_y.real**2 + pL_y.imag**2)
                                + (pL_z.real**2 + pL_z.imag**2))
                proj_T_mag2[s] = ((pT_x.real**2 + pT_x.imag**2)
                                + (pT_y.real**2 + pT_y.imag**2)
                                + (pT_z.real**2 + pT_z.imag**2))

            for s in range(n3):
                if valid[s]:
                    b = mode_bin[s]
                    fl_local[b] += proj_L_mag2[s]
                    ft_local[b] += proj_T_mag2[s]
                    cnt_local[b] += 1.0

        for b in range(nfbin):
            FL[iq, b] = fl_local[b]
            FT[iq, b] = ft_local[b]
            Ftot[iq, b] = fl_local[b] + ft_local[b]
            counts[iq, b] = cnt_local[b]

    return FL, FT, Ftot, counts


@njit(parallel=True)
def _compute_modal_flux_spatial(E_i, sigma, eigvecs_3d, modal_amp,
                                volume, n3, natoms):
    Q_modal = np.zeros((3, n3))
    for s in prange(n3):
        amp = modal_amp[s]
        qx = 0.0; qy = 0.0; qz = 0.0
        for i in range(natoms):
            ex = eigvecs_3d[i, 0, s]
            ey = eigvecs_3d[i, 1, s]
            ez = eigvecs_3d[i, 2, s]
            qx += E_i[i] * ex
            qy += E_i[i] * ey
            qz += E_i[i] * ez
            qx += sigma[i,0,0]*ex + sigma[i,0,1]*ey + sigma[i,0,2]*ez
            qy += sigma[i,1,0]*ex + sigma[i,1,1]*ey + sigma[i,1,2]*ez
            qz += sigma[i,2,0]*ex + sigma[i,2,1]*ey + sigma[i,2,2]*ez
        Q_modal[0, s] = (qx * amp) / volume
        Q_modal[1, s] = (qy * amp) / volume
        Q_modal[2, s] = (qz * amp) / volume
    return Q_modal


@njit(parallel=True)
def _compute_dispersion(pos_t, vel, q_vals_d, q_dirs_d,
                        nq_d, nqa_d, natoms):
    j_frame = np.zeros((nq_d, nqa_d, 3), dtype=np.complex128)
    for iq in prange(nq_d):
        for idir in range(nqa_d):
            qvec = q_vals_d[iq] * q_dirs_d[idir]
            for i in range(natoms):
                dot_prod = (pos_t[i, 0]*qvec[0]
                          + pos_t[i, 1]*qvec[1]
                          + pos_t[i, 2]*qvec[2])
                phase = np.exp(1j * dot_prod)
                j_frame[iq, idir, 0] += phase * vel[i, 0]
                j_frame[iq, idir, 1] += phase * vel[i, 1]
                j_frame[iq, idir, 2] += phase * vel[i, 2]
    return j_frame

# ---------------------------------------------------------------------------
#  MSE helper (for multiprocessing pool)
# ---------------------------------------------------------------------------
def _compute_mse_chunk(mode_indices, eigvecs, positions, natoms, nslices):
    def gauss2(x, a1, b1, c1, a2, b2, c2):
        return (a1 * np.exp(-((x - b1) / max(c1, 0.01))**2)
              + a2 * np.exp(-((x - b2) / max(c2, 0.01))**2))

    mse_out = np.zeros(len(mode_indices))
    for idx, s in enumerate(mode_indices):
        e = eigvecs[:, s].reshape(natoms, 3)
        amp2 = np.sum(e**2, axis=1)
        if amp2.max() < 1e-30:
            continue
        widths = []
        for dim in range(3):
            coords = positions[:, dim]
            lo, hi = coords.min(), coords.max()
            span = hi - lo
            if span < 1e-6:
                widths.append(0.0)
                continue
            edges = np.linspace(lo, hi, nslices + 1)
            centres = 0.5 * (edges[:-1] + edges[1:])
            profile = np.zeros(nslices)
            bins = np.clip(np.digitize(coords, edges) - 1, 0, nslices - 1)
            np.add.at(profile, bins, amp2)
            if profile.max() < 1e-30:
                widths.append(0.0)
                continue
            profile /= profile.max()
            try:
                peak = centres[np.argmax(profile)]
                p0 = [1.0, peak, span/4, 0.5, peak + span/4, span/4]
                popt, _ = curve_fit(gauss2, centres, profile, p0=p0,
                                   maxfev=2000,
                                   bounds=([0, lo, 0.1, 0, lo, 0.1],
                                           [2, hi, span, 2, hi, span]))
                widths.append(0.5 * (abs(popt[2]) + abs(popt[5])))
            except Exception:
                widths.append(span / 4)
        mse_out[idx] = np.mean(widths) if widths else 0.0
    return mse_out

# ---------------------------------------------------------------------------
#  Ioffe-Regel helpers
# ---------------------------------------------------------------------------
def _sharpness_array(q_vals, freq_bins, Fqnu):
    nf = len(freq_bins)
    sharpness = np.zeros(nf)
    for j in range(nf):
        col = Fqnu[:, j]
        if col.max() < 1e-10:
            continue
        mean_val = col.mean()
        sharpness[j] = col.max() / mean_val if mean_val > 0 else 0.0
    return sharpness


def _estimate_ir(q_vals, freq_bins, Fqnu, threshold=2.0, min_freq=0.5):
    sharpness = _sharpness_array(q_vals, freq_bins, Fqnu)
    for j in range(len(freq_bins)):
        if freq_bins[j] < min_freq:
            continue
        if 0 < sharpness[j] < threshold:
            return freq_bins[j]
    return freq_bins[-1]

# ---------------------------------------------------------------------------
#  DHO fit helper
# ---------------------------------------------------------------------------
def _fit_dho(freq, spectrum):
    if spectrum.max() < 1e-20:
        return 0.0, 0.0
    peak_idx = np.argmax(spectrum)
    omega0 = freq[peak_idx]
    if omega0 < 0.01:
        return 0.0, 0.0
    half_max = spectrum[peak_idx] / 2.0
    above = spectrum > half_max
    if np.sum(above) < 2:
        return omega0, omega0 * 0.5
    indices = np.where(above)[0]
    f_lo = freq[indices[0]]
    f_hi = freq[indices[-1]]
    gamma = (f_hi - f_lo) / 2.0
    return omega0, max(gamma, 0.01)

# ===================================================================
#  STAGE 1: Eigensystem
# ===================================================================
def compute_eigensystem(cfg):
    section("EIGENSYSTEM")

    natoms, masses, positions, box, volume = read_datafile(cfg['datafile'])
    n3 = 3 * natoms
    print(f"  {natoms} atoms")

    print(f"  Reading dynamical matrix from {cfg['dynmat_file']}...")
    D = np.loadtxt(cfg['dynmat_file']).reshape(n3, n3)
    D = 0.5 * (D + D.T)

    print(f"  Diagonalising {n3}x{n3} matrix...")
    t0 = time.time()
    eigenvalues, eigvecs = np.linalg.eigh(D)
    print(f"  Done in {time.time()-t0:.1f} s")

    freqs = np.where(eigenvalues >= 0,
                     np.sqrt(eigenvalues) / TWO_PI,
                     -np.sqrt(np.abs(eigenvalues)) / TWO_PI)

    n_neg = np.sum(freqs < 0)
    pos_f = freqs[freqs > 0.1]
    print(f"  {n_neg} imaginary modes")
    print(f"  Freq range {pos_f.min():.3f} to {pos_f.max():.3f} THz")

    # VDOS
    nbin_v = int(cfg['freq_max'] * 10)
    edges_v = np.linspace(0, cfg['freq_max'], nbin_v + 1)
    centres_v = 0.5 * (edges_v[:-1] + edges_v[1:])
    dos, _ = np.histogram(pos_f, bins=edges_v)
    dfreq = centres_v[1] - centres_v[0]
    dos = dos.astype(float) / dfreq

    # Participation ratio
    print("  Computing participation ratio...")
    e_all = eigvecs.reshape(natoms, 3, n3)
    m2 = np.sum(e_all**2, axis=1)
    num = np.sum(m2, axis=0)**2
    den = np.sum(m2**2, axis=0)
    pr = np.where(den > 0, num / (natoms * den), 0.0)

    # Mode spatial extent
    print("  Computing MSE...")
    nw = cfg['nworkers']
    chunks = np.array_split(np.arange(n3), min(nw * 4, n3))

    from functools import partial
    from multiprocessing import Pool

    _mse_func = partial(_compute_mse_chunk,
                        eigvecs=eigvecs, positions=positions,
                        natoms=natoms, nslices=cfg['mse_nslices'])
    with Pool(nw) as pool:
        results = pool.map(_mse_func, chunks)
    mse = np.concatenate(results)
    print(f"  MSE range {mse[mse>0].min():.2f} to {mse.max():.2f} Angstrom")

    # Save
    np.save("eigenfreqs.npy", freqs)
    np.save("eigenvecs.npy", eigvecs)
    np.save("masses.npy", masses)
    np.save("positions.npy", positions)

    np.savetxt("vdos.dat",
               np.column_stack([centres_v, dos]),
               header="freq_THz  DOS_modes_per_THz",
               fmt="%12.6f  %12.6f")

    np.savetxt("participation_ratio.dat",
               np.column_stack([freqs, pr]),
               header="freq_THz  PR",
               fmt="%12.6f  %12.8f")

    np.savetxt("mse.dat",
               np.column_stack([freqs, pr, mse]),
               header="freq_THz  PR  MSE_Angstrom",
               fmt="%12.6f  %12.8f  %12.4f")

    gb = eigvecs.nbytes / 1e9
    print(f"  Saved eigenfreqs.npy, eigenvecs.npy ({gb:.2f} GB), masses.npy")
    print(f"  Saved vdos.dat, participation_ratio.dat, mse.dat")

    return dict(freqs=freqs, eigvecs=eigvecs, masses=masses,
                positions=positions, box=box, volume=volume,
                natoms=natoms, pr=pr, mse=mse,
                vdos_centres=centres_v, vdos_dos=dos)


def load_eigensystem(cfg):
    """Load pre-computed eigensystem from .npy/.dat files."""
    required = ["eigenfreqs.npy", "eigenvecs.npy", "masses.npy",
                "positions.npy", "vdos.dat", "participation_ratio.dat",
                "mse.dat"]
    missing = [f for f in required if not os.path.isfile(f)]
    if missing:
        return None
    print("  Loading cached eigensystem...")
    freqs = np.load("eigenfreqs.npy")
    eigvecs = np.load("eigenvecs.npy")
    masses = np.load("masses.npy")
    positions = np.load("positions.npy")
    pr = np.loadtxt("participation_ratio.dat")[:, 1]
    mse_d = np.loadtxt("mse.dat")
    mse = mse_d[:, 2]
    vd = np.loadtxt("vdos.dat")
    na, _, pos, box, vol = read_datafile(cfg['datafile'])
    return dict(freqs=freqs, eigvecs=eigvecs, masses=masses,
                positions=positions, box=box, volume=vol,
                natoms=na, pr=pr, mse=mse,
                vdos_centres=vd[:, 0], vdos_dos=vd[:, 1])


def ensure_eigensystem(cfg):
    """Load or compute eigensystem."""
    eig = load_eigensystem(cfg)
    if eig is None:
        eig = compute_eigensystem(cfg)
    return eig

# ===================================================================
#  STAGE 2: Ioffe-Regel
# ===================================================================
def compute_ioffe_regel(cfg, eig):
    section("IOFFE-REGEL ANALYSIS")

    freqs = eig['freqs']
    eigvecs = eig['eigvecs']
    positions = eig['positions']
    natoms = eig['natoms']
    n3 = 3 * natoms

    tree = cKDTree(positions)
    dd, _ = tree.query(positions, k=2)
    d_nn = np.mean(dd[:, 1])
    q_max = np.pi / d_nn
    print(f"  d_nn = {d_nn:.3f} A,  q_max = {q_max:.3f} 1/A")

    nqr = cfg['ir_nq_radial']
    nqa = cfg['ir_nq_angular']
    q_vals = np.linspace(0.05, q_max, nqr)

    pos_f = freqs[freqs > 0.1]
    nfbin = 200
    freq_edges = np.linspace(0, pos_f.max() * 1.05, nfbin + 1)
    freq_bins = 0.5 * (freq_edges[:-1] + freq_edges[1:])
    mode_bin = np.clip(np.digitize(freqs, freq_edges) - 1, 0, nfbin - 1)

    rng = np.random.default_rng(42)
    theta = np.arccos(2.0 * rng.random(nqa) - 1.0)
    phi = 2.0 * np.pi * rng.random(nqa)
    q_dirs = np.column_stack([np.sin(theta)*np.cos(phi),
                              np.sin(theta)*np.sin(phi),
                              np.cos(theta)])

    eigvecs_3d = np.ascontiguousarray(eigvecs.reshape(natoms, 3, n3))
    valid = freqs > 0.1

    print(f"  Computing FL/FT  {nqr} q x {nqa} dirs...")
    t0 = time.time()
    FL, FT, Ftot, counts = _compute_ir_qpoints(
        q_vals, q_dirs, positions, eigvecs_3d, freqs,
        mode_bin, valid, n3, nqr, nqa, natoms, nfbin)
    print(f"  Done in {time.time()-t0:.1f} s")

    mask = counts > 0
    FL[mask] /= counts[mask]
    FT[mask] /= counts[mask]
    Ftot[mask] /= counts[mask]

    for j in range(nfbin):
        for arr in (FL, FT, Ftot):
            mx = arr[:, j].max()
            if mx > 0:
                arr[:, j] /= mx

    ir_L = _estimate_ir(q_vals, freq_bins, FL)
    ir_T = _estimate_ir(q_vals, freq_bins, FT)
    ir_tot = _estimate_ir(q_vals, freq_bins, Ftot)
    print(f"  IR frequencies  L={ir_L:.1f}  T={ir_T:.1f}  total={ir_tot:.1f} THz")

    np.save("Fqnu_L.npy", FL)
    np.save("Fqnu_T.npy", FT)
    np.save("Fqnu_total.npy", Ftot)
    np.savez("Fqnu_axes.npz", q_vals=q_vals, freq_bins=freq_bins)

    sharpness_L = _sharpness_array(q_vals, freq_bins, FL)
    sharpness_T = _sharpness_array(q_vals, freq_bins, FT)
    np.savetxt("ioffe_regel.dat",
               np.column_stack([freq_bins, sharpness_L, sharpness_T]),
               header=(f"IR_L={ir_L:.2f} IR_T={ir_T:.2f} IR_tot={ir_tot:.2f} THz\n"
                       f"freq_THz  sharpness_L  sharpness_T"),
               fmt="%12.6f  %12.6f  %12.6f")
    print("  Saved Fqnu_L/T/total.npy, Fqnu_axes.npz, ioffe_regel.dat")

    return dict(q_vals=q_vals, freq_bins=freq_bins,
                FL=FL, FT=FT, Ftot=Ftot,
                ir_L=ir_L, ir_T=ir_T, ir_tot=ir_tot)

# ===================================================================
#  STAGE 3: Trajectory processing
# ===================================================================
def process_trajectory(cfg, eig):
    section("TRAJECTORY PROCESSING")

    freqs = eig['freqs']
    eigvecs = eig['eigvecs']
    masses = eig['masses']
    positions = eig['positions']
    box = eig['box']
    natoms = eig['natoms']
    n3 = 3 * natoms

    dt_ps = cfg['timestep_ps'] * cfg['dump_freq']
    nbins = cfg['nbins']
    freq_max = cfg['freq_max']

    freq_edges = np.linspace(0, freq_max, nbins + 1)
    bin_centres = 0.5 * (freq_edges[:-1] + freq_edges[1:])
    bin_idx = np.clip(np.digitize(freqs, freq_edges) - 1, 0, nbins - 1)

    if cfg['effective_thickness'] and cfg['effective_thickness'] > 0:
        Lx = box[0, 1] - box[0, 0]
        Ly = box[1, 1] - box[1, 0]
        volume = Lx * Ly * cfg['effective_thickness']
        print(f"  2D effective volume = {volume:.1f} A^3")
    else:
        volume = np.prod(box[:, 1] - box[:, 0])
        print(f"  3D volume = {volume:.1f} A^3")

    print(f"  Scanning {cfg['dump_file']}...")
    with open(cfg['dump_file']) as f:
        hdr = parse_dump_header(f)
        if hdr is None:
            sys.exit("ERROR: empty dump file")
        ts0, na, _, col_names = hdr
        assert na == natoms, f"Atom count mismatch: data={natoms} dump={na}"
        ncols = len(col_names)
        cm = detect_dump_columns(col_names)
        print(f"  Columns ({ncols}): {' '.join(col_names)}")

        nframes = 1
        for _ in range(natoms):
            f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                nframes += 1
                for _ in range(3 + natoms + 4):
                    f.readline()

    nframes_file = nframes
    if cfg['max_frames']:
        nframes = min(nframes, cfg['max_frames'])
    print(f"  {nframes_file} frames in file, using {nframes}")

    ix = cm.get('x', cm.get('xu', 2))
    iy = cm.get('y', cm.get('yu', 3))
    iz = cm.get('z', cm.get('zu', 4))
    ivx, ivy, ivz = cm['vx'], cm['vy'], cm['vz']

    ipe = None
    for key in ('c_pe', 'c_PE', 'pe', 'PE'):
        if key in cm:
            ipe = cm[key]
            break
    if ipe is None:
        sys.exit("ERROR: no per-atom PE column found in dump")

    istress = []
    for pattern in [['c_s[1]','c_s[2]','c_s[3]','c_s[4]','c_s[5]','c_s[6]'],
                    ['c_stress[1]','c_stress[2]','c_stress[3]',
                     'c_stress[4]','c_stress[5]','c_stress[6]']]:
        if all(p in cm for p in pattern):
            istress = [cm[p] for p in pattern]
            break
    if not istress:
        sys.exit("ERROR: no per-atom stress columns found in dump")

    sub = cfg['disp_subsample']
    nq_d = cfg['disp_nq']
    nqa_d = cfg['disp_nq_angular']

    tree = cKDTree(positions)
    dd, _ = tree.query(positions, k=2)
    d_nn = np.mean(dd[:, 1])
    q_max_d = np.pi / d_nn
    q_vals_d = np.linspace(0.1, q_max_d, nq_d)

    rng = np.random.default_rng(123)
    th = np.arccos(2.0 * rng.random(nqa_d) - 1.0)
    ph = 2.0 * np.pi * rng.random(nqa_d)
    q_dirs_d = np.column_stack([np.sin(th)*np.cos(ph),
                                np.sin(th)*np.sin(ph),
                                np.cos(th)])

    n_disp_frames = (nframes + sub - 1) // sub
    j_store = np.zeros((nq_d, nqa_d, n_disp_frames, 3), dtype=np.complex128)

    relax_mask = (freqs > 0.1) & (freqs < cfg['relax_freq_max'])
    relax_indices = np.where(relax_mask)[0]
    n_relax = len(relax_indices)
    print(f"  Storing modal amplitudes for {n_relax} modes")
    modal_amp_store = np.zeros((nframes, n_relax))

    eigvecs_3d = np.ascontiguousarray(eigvecs.reshape(natoms, 3, n3))
    eigvecs_transposed = np.ascontiguousarray(eigvecs.T)

    max_corr = int(nframes * cfg['corr_fraction'])
    Q_binned_all = np.zeros((nframes, nbins, 3))
    Q_total_all = np.zeros((nframes, 3))

    print(f"  Processing {nframes} frames...")
    t0 = time.time()
    disp_frame = 0

    with open(cfg['dump_file']) as f:
        for frame in range(nframes):
            hdr = parse_dump_header(f)
            if hdr is None:
                nframes = frame
                break
            data = read_dump_frame_fast(f, natoms, ncols)

            vel = data[:, [ivx, ivy, ivz]]
            pe = data[:, ipe]
            stress_raw = data[:, istress]
            vel_flat = vel.reshape(n3)

            ke = 0.5 * masses * np.sum(vel**2, axis=1) * AMU_ANGPS2_EV
            E_i = pe + ke

            s_eV = stress_raw * BAR_ANG3_EV
            sigma = np.zeros((natoms, 3, 3))
            sigma[:, 0, 0] = s_eV[:, 0]
            sigma[:, 1, 1] = s_eV[:, 1]
            sigma[:, 2, 2] = s_eV[:, 2]
            sigma[:, 0, 1] = sigma[:, 1, 0] = s_eV[:, 3]
            sigma[:, 0, 2] = sigma[:, 2, 0] = s_eV[:, 4]
            sigma[:, 1, 2] = sigma[:, 2, 1] = s_eV[:, 5]

            modal_amp = eigvecs_transposed @ vel_flat

            Q_modal = _compute_modal_flux_spatial(
                E_i, sigma, eigvecs_3d, modal_amp, volume, n3, natoms)

            for b in range(nbins):
                m = (bin_idx == b)
                if np.any(m):
                    Q_binned_all[frame, b, :] = np.sum(Q_modal[:, m], axis=1)
            Q_total_all[frame] = np.sum(Q_modal, axis=1)

            modal_amp_store[frame] = modal_amp[relax_indices]

            if frame % sub == 0 and disp_frame < n_disp_frames:
                pos_t = data[:, [ix, iy, iz]]
                j_store[:, :, disp_frame, :] = _compute_dispersion(
                    pos_t, vel, q_vals_d, q_dirs_d, nq_d, nqa_d, natoms)
                disp_frame += 1

            if (frame + 1) % 2000 == 0:
                el = time.time() - t0
                rate = (frame + 1) / el
                print(f"    frame {frame+1}/{nframes}  ({rate:.1f} fr/s)")

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f} s")

    Q_binned_all = Q_binned_all[:nframes]
    Q_total_all = Q_total_all[:nframes]
    modal_amp_store = modal_amp_store[:nframes]
    j_store = j_store[:, :, :disp_frame, :]
    max_corr = min(max_corr, nframes // 2)

    np.save("Q_binned_all.npy", Q_binned_all)
    np.save("Q_total_all.npy", Q_total_all)
    np.save("modal_amp_store.npy", modal_amp_store)
    np.save("j_velocity_current.npy", j_store)
    np.savez("trajectory_meta.npz",
             bin_centres=bin_centres, freq_edges=freq_edges,
             bin_idx=bin_idx, volume=volume, dt_ps=dt_ps,
             nframes=nframes, max_corr=max_corr,
             q_vals_d=q_vals_d, q_dirs_d=q_dirs_d,
             relax_indices=relax_indices, disp_subsample=sub)
    print("  Saved intermediate files")

    return dict(Q_binned=Q_binned_all, Q_total=Q_total_all,
                modal_amp=modal_amp_store, j_store=j_store,
                bin_centres=bin_centres, freq_edges=freq_edges,
                bin_idx=bin_idx, volume=volume, dt_ps=dt_ps,
                nframes=nframes, max_corr=max_corr,
                q_vals_d=q_vals_d, q_dirs_d=q_dirs_d,
                relax_indices=relax_indices, disp_sub=sub)


def load_trajectory(cfg):
    """Load pre-computed trajectory data."""
    required = ["Q_binned_all.npy", "Q_total_all.npy",
                "modal_amp_store.npy", "j_velocity_current.npy",
                "trajectory_meta.npz"]
    if any(not os.path.isfile(f) for f in required):
        return None
    print("  Loading cached trajectory data...")
    meta = np.load("trajectory_meta.npz", allow_pickle=True)
    return dict(
        Q_binned=np.load("Q_binned_all.npy"),
        Q_total=np.load("Q_total_all.npy"),
        modal_amp=np.load("modal_amp_store.npy"),
        j_store=np.load("j_velocity_current.npy"),
        bin_centres=meta['bin_centres'],
        freq_edges=meta['freq_edges'],
        bin_idx=meta['bin_idx'],
        volume=float(meta['volume']),
        dt_ps=float(meta['dt_ps']),
        nframes=int(meta['nframes']),
        max_corr=int(meta['max_corr']),
        q_vals_d=meta['q_vals_d'],
        q_dirs_d=meta['q_dirs_d'],
        relax_indices=meta['relax_indices'],
        disp_sub=int(meta['disp_subsample']),
    )

# ===================================================================
#  STAGE 4: Thermal properties
# ===================================================================
def calculate_thermal_properties(cfg, eig, traj):
    section("THERMAL PROPERTIES")

    T = cfg['T']
    freqs = eig['freqs']
    nbins = cfg['nbins']
    bc = traj['bin_centres']
    volume = traj['volume']
    dt_ps = traj['dt_ps']
    nframes = traj['nframes']
    max_corr = traj['max_corr']
    Q_b = traj['Q_binned']
    Q_t = traj['Q_total']
    nw = cfg['nworkers']

    # -- kappa(nu) --
    print("  Computing thermal conductivity...")
    modes_per_bin = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        modes_per_bin[b] = np.sum(traj['bin_idx'] == b)

    kappa_bin = np.zeros(nbins)
    for b in range(nbins):
        if modes_per_bin[b] == 0:
            continue
        corr = correlate_fft(Q_b[:, b, :], Q_t, max_corr)
        kappa_bin[b] = kappa_from_corr(corr, dt_ps, volume, T)

    kappa_cl = np.sum(kappa_bin)
    qc = quantum_correction(bc, T)
    kappa_qc = kappa_bin * qc
    kappa_qt = np.sum(kappa_qc)
    cum_cl = np.cumsum(kappa_bin)
    cum_qc = np.cumsum(kappa_qc)
    norm_cl = cum_cl / kappa_cl if kappa_cl != 0 else cum_cl
    norm_qc = cum_qc / kappa_qt if kappa_qt != 0 else cum_qc

    print(f"  kappa classical {kappa_cl:.4f} W/(m*K)")
    print(f"  kappa quantum   {kappa_qt:.4f} W/(m*K)")

    np.savetxt("kappa_vs_freq.dat",
               np.column_stack([bc, kappa_bin, kappa_qc, modes_per_bin]),
               header=(f"T={T} V={volume:.2f} frames={nframes} dt={dt_ps}\n"
                       f"kappa_cl={kappa_cl:.6f} kappa_qc={kappa_qt:.6f}\n"
                       f"freq_THz  kappa_cl  kappa_qc  modes"),
               fmt="%12.6f  %14.8e  %14.8e  %6d")

    np.savetxt("kappa_accumulation.dat",
               np.column_stack([bc, cum_cl, cum_qc, norm_cl, norm_qc]),
               header="freq_THz  cum_cl  cum_qc  norm_cl  norm_qc",
               fmt="%12.6f  %14.8e  %14.8e  %10.6f  %10.6f")

    # -- GK validation --
    try:
        Jl = np.loadtxt(cfg['hf_file'], comments="#")
        if Jl.shape[1] >= 4:
            Jl = Jl[:, 1:4]
        Jl /= volume
        nf_v = min(len(Jl), nframes)
        cv = correlate_fft(Jl[:nf_v], Jl[:nf_v], min(max_corr, nf_v // 2))
        kgk = kappa_from_corr(cv, dt_ps, volume, T)
        ratio = kappa_cl / kgk if kgk != 0 else 0
        print(f"  GK validation  LAMMPS={kgk:.2f}  GKMA_sum={kappa_cl:.2f}  ratio={ratio:.4f}")
    except Exception:
        print("  GK validation skipped")

    # -- Mode-mode map --
    print("  Computing mode-mode correlation map...")
    pairs_b1 = np.array([b1 for b1 in range(nbins) for b2 in range(b1, nbins)
                         if modes_per_bin[b1] > 0 and modes_per_bin[b2] > 0],
                        dtype=np.int32)
    pairs_b2 = np.array([b2 for b1 in range(nbins) for b2 in range(b1, nbins)
                         if modes_per_bin[b1] > 0 and modes_per_bin[b2] > 0],
                        dtype=np.int32)
    npairs = len(pairs_b1)
    print(f"    {npairs} bin pairs")

    coeff = (volume / (3.0 * KB_EV * T**2)) * KAPPA_CONV
    results_mm = np.zeros(npairs)

    def mode_map_worker(idx):
        b1 = pairs_b1[idx]
        b2 = pairs_b2[idx]
        corr = correlate_fft(Q_b[:, b1, :], Q_b[:, b2, :], max_corr)
        integral = np.trapz(corr, dx=dt_ps)
        return idx, coeff * integral

    with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as executor:
        for idx, val in executor.map(mode_map_worker, range(npairs)):
            results_mm[idx] = val

    kappa_map = np.zeros((nbins, nbins))
    for idx in range(npairs):
        b1, b2 = pairs_b1[idx], pairs_b2[idx]
        kappa_map[b1, b2] = results_mm[idx]
        kappa_map[b2, b1] = results_mm[idx]
    np.save("kappa_mode_map.npy", kappa_map)

    # -- Spectral diffusivity --
    print("  Computing spectral diffusivity...")
    vdos = np.loadtxt("vdos.dat")
    vc, vd = vdos[:, 0], vdos[:, 1]
    dos_interp = np.interp(bc, vc, vd)

    fit_mask = (bc > 0.3) & (bc < 5.0) & (dos_interp > 0)
    if np.sum(fit_mask) > 3:
        def debye_dos(nu, vD):
            return 12 * np.pi * volume * nu**2 / vD**3
        popt, _ = curve_fit(debye_dos, bc[fit_mask], dos_interp[fit_mask], p0=[15.0])
        v_D = popt[0]
    else:
        v_D = 15.0
    print(f"  Debye velocity {v_D:.2f} km/s")

    dnu = bc[1] - bc[0]
    D_cl = np.zeros(nbins)
    D_qc = np.zeros(nbins)
    safe = (dos_interp > 1e-6) & (bc > 0.1)
    D_cl[safe] = kappa_bin[safe] * volume / (KB_EV * dos_interp[safe] * dnu)
    qc_safe = qc.copy()
    qc_safe[qc_safe < 1e-6] = 1e-6
    D_qc[safe] = kappa_qc[safe] * volume / (KB_EV * qc_safe[safe] * dos_interp[safe] * dnu)

    n_density = eig['natoms'] / volume
    a_nn = (1.0 / n_density)**(1.0/3.0)
    D_a = v_D * a_nn * np.ones(nbins)
    D_RW = (1.0/3.0) * TWO_PI * bc / (np.pi * n_density**(2.0/3.0))

    np.savetxt("spectral_diffusivity.dat",
               np.column_stack([bc, D_cl, D_qc, D_a, D_RW]),
               header=(f"v_Debye={v_D:.4f} km/s  a_nn={a_nn:.4f} Angstrom\n"
                       f"freq_THz  D_classical  D_quantum  D_a  D_RW"),
               fmt="%12.6f  %14.6e  %14.6e  %14.6e  %14.6e")

    # -- Velocity-current dispersion --
    print("  Computing velocity-current dispersion...")
    j_store = traj['j_store']
    q_vals_d = traj['q_vals_d']
    q_dirs_d = traj['q_dirs_d']
    sub = traj['disp_sub']
    dt_disp = dt_ps * sub
    n_disp = j_store.shape[2]
    nq_d = len(q_vals_d)
    nqa_d = q_dirs_d.shape[0]

    freq_fft = np.fft.rfftfreq(2 * n_disp, d=dt_disp)
    nf_fft = len(freq_fft)

    CL = np.zeros((nq_d, nf_fft))
    CT = np.zeros((nq_d, nf_fft))

    for iq in range(nq_d):
        for idir in range(nqa_d):
            qhat = q_dirs_d[idir]
            j_t = j_store[iq, idir, :, :]
            j_L_scalar = j_t @ qhat
            j_T_vec = j_t - np.outer(j_L_scalar, qhat)
            fft_L = np.fft.rfft(j_L_scalar, n=2*n_disp)
            CL[iq] += np.abs(fft_L)**2
            for a in range(3):
                fft_T = np.fft.rfft(j_T_vec[:, a], n=2*n_disp)
                CT[iq] += np.abs(fft_T)**2
        CL[iq] /= nqa_d
        CT[iq] /= nqa_d

    CL_norm = CL.copy()
    CT_norm = CT.copy()
    for j in range(nf_fft):
        mx = CL_norm[:, j].max()
        if mx > 0:
            CL_norm[:, j] /= mx
        mx = CT_norm[:, j].max()
        if mx > 0:
            CT_norm[:, j] /= mx

    np.save("dispersion_CL.npy", CL_norm)
    np.save("dispersion_CT.npy", CT_norm)
    np.savez("dispersion_axes.npz", q_vals=q_vals_d, freq=freq_fft)

    # -- Dispersion fits / MFP --
    print("  Fitting dispersions for mean free path...")
    omega0_L = np.zeros(nq_d)
    gamma_L = np.zeros(nq_d)
    omega0_T = np.zeros(nq_d)
    gamma_T = np.zeros(nq_d)

    for iq in range(nq_d):
        omega0_L[iq], gamma_L[iq] = _fit_dho(freq_fft, CL[iq])
        omega0_T[iq], gamma_T[iq] = _fit_dho(freq_fft, CT[iq])

    vg_L = np.abs(np.gradient(omega0_L, q_vals_d)) * TWO_PI
    vg_T = np.abs(np.gradient(omega0_T, q_vals_d)) * TWO_PI

    mfp_L = np.zeros(nq_d)
    mfp_T = np.zeros(nq_d)
    safe_L = gamma_L > 0.01
    safe_T = gamma_T > 0.01
    mfp_L[safe_L] = vg_L[safe_L] / (2.0 * gamma_L[safe_L] * TWO_PI)
    mfp_T[safe_T] = vg_T[safe_T] / (2.0 * gamma_T[safe_T] * TWO_PI)

    np.savetxt("dispersion_fits.dat",
               np.column_stack([q_vals_d, omega0_L, gamma_L, vg_L, mfp_L,
                                omega0_T, gamma_T, vg_T, mfp_T]),
               header=("q(1/A) w0_L(THz) G_L(THz) vg_L(A/ps) mfp_L(A) "
                       "w0_T(THz) G_T(THz) vg_T(A/ps) mfp_T(A)"),
               fmt="%10.5f  " * 9)

    # -- Relaxation times --
    print("  Computing relaxation times...")
    relax_idx = traj['relax_indices']
    modal_amp = traj['modal_amp']
    max_corr_r = min(nframes // 2, 10000)
    n_modes = modal_amp.shape[1]
    tau = np.zeros(n_modes)

    def relax_worker(m):
        sig = modal_amp[:, m]
        if np.std(sig) < 1e-20:
            return m, max_corr_r * dt_ps
        nf = len(sig)
        nfft = 2 * nf
        sf = np.fft.rfft(sig, n=nfft)
        acf = np.fft.irfft(np.abs(sf)**2, n=nfft)[:max_corr_r]
        norm = np.arange(nf, nf - max_corr_r, -1, dtype=float)
        acf /= norm
        if acf[0] > 0:
            acf /= acf[0]
        else:
            acf[:] = 1.0
        env = np.abs(acf)
        below = np.where(env < 0.36787944117)[0]
        if len(below) > 0:
            return m, below[0] * dt_ps
        return m, max_corr_r * dt_ps

    with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as executor:
        for m, val in executor.map(relax_worker, range(n_modes)):
            tau[m] = val

    np.savetxt("relaxation_times.dat",
               np.column_stack([freqs[relax_idx], eig['pr'][relax_idx], tau]),
               header="freq_THz  PR  tau_ps",
               fmt="%12.6f  %12.8f  %14.6e")
    print("  Saved relaxation_times.dat")

    # -- Mode contributions --
    ir_file = "ioffe_regel.dat"
    ir_tot = cfg['freq_max']
    try:
        with open(ir_file) as fir:
            hdr = fir.readline()
            for part in hdr.split():
                if part.startswith("IR_tot="):
                    ir_tot = float(part.split("=")[1])
    except Exception:
        pass

    pr_binned = np.zeros(nbins)
    fe = traj['freq_edges']
    for b in range(nbins):
        in_bin = (freqs >= fe[b]) & (freqs < fe[b+1]) & (freqs > 0.1)
        if np.any(in_bin):
            pr_binned[b] = np.mean(eig['pr'][in_bin])
    mob_edge = cfg['freq_max']
    for b in range(nbins):
        if bc[b] > ir_tot and pr_binned[b] > 0 and pr_binned[b] < 0.05:
            mob_edge = bc[b]
            break

    k_prop = np.sum(kappa_qc[bc < ir_tot])
    k_diff = np.sum(kappa_qc[(bc >= ir_tot) & (bc < mob_edge)])
    k_loc  = np.sum(kappa_qc[bc >= mob_edge])

    np.savetxt("mode_contributions.dat",
               np.array([[ir_tot, mob_edge, k_prop, k_diff, k_loc, kappa_qt]]),
               header=(f"IR_freq={ir_tot:.2f} mob_edge={mob_edge:.2f}\n"
                       f"IR_freq  mob_edge  k_propagon  k_diffuson  k_locon  k_total  (W/mK)"),
               fmt="%10.4f  %10.4f  %12.6f  %12.6f  %12.6f  %12.6f")
    print(f"  Propagons  {k_prop:.4f} W/(m*K)")
    print(f"  Diffusons  {k_diff:.4f} W/(m*K)")
    print(f"  Locons     {k_loc:.4f} W/(m*K)")

# ===================================================================
#  STAGE 5: Plotting
# ===================================================================
def generate_plots(cfg):
    section("PLOTS")

    T = cfg['T']
    fmax = cfg['freq_max']

    # VDOS + PR
    print("  Plotting VDOS and PR...")
    vdos = np.loadtxt("vdos.dat")
    prd = np.loadtxt("participation_ratio.dat")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]
    ax.fill_between(vdos[:, 0], vdos[:, 1], alpha=0.3, color='C0')
    ax.plot(vdos[:, 0], vdos[:, 1], 'C0', lw=1)
    ax.set_ylabel('DOS (modes/THz)')
    ax.set_title('Vibrational Density of States')
    ax.set_xlim(0, fmax)

    ax = axes[1]
    pos = prd[:, 0] > 0.1
    ax.scatter(prd[pos, 0], prd[pos, 1], s=0.5, c='k', alpha=0.3, rasterized=True)
    ax.axhline(0.2, color='gray', ls='--', lw=0.6, label='PR=0.2')
    ax.axhline(0.05, color='gray', ls=':', lw=0.6, label='PR=0.05')
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Participation Ratio')
    ax.set_xlim(0, fmax)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('plot_vdos_pr.png', dpi=300, bbox_inches='tight')
    plt.close()

    # MSE scatter
    print("  Plotting MSE scatter...")
    mse_d = np.loadtxt("mse.dat")
    fig, ax = plt.subplots(figsize=(7, 5))
    pos = mse_d[:, 0] > 0.1
    sc = ax.scatter(mse_d[pos, 1], mse_d[pos, 2], s=1.5, c=mse_d[pos, 0],
                    cmap='viridis', alpha=0.4, rasterized=True)
    ax.set_xlabel('Participation Ratio')
    ax.set_ylabel('Mode Spatial Extent (A)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.colorbar(sc, label='Frequency (THz)')
    ax.set_title('Mode Spatial Extent vs Participation Ratio')
    plt.tight_layout()
    plt.savefig('plot_mse_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # F(q,nu) dispersion
    if os.path.isfile("Fqnu_axes.npz"):
        print("  Plotting F(q,nu) dispersion...")
        axes_d = np.load("Fqnu_axes.npz")
        q_v = axes_d['q_vals']
        f_v = axes_d['freq_bins']
        for label, fname in [('Longitudinal', 'Fqnu_L.npy'),
                             ('Transverse', 'Fqnu_T.npy')]:
            if not os.path.isfile(fname):
                continue
            F = np.load(fname)
            fig, ax = plt.subplots(figsize=(7, 5))
            ext = [q_v[0], q_v[-1], f_v[0], f_v[-1]]
            ax.imshow(F.T, origin='lower', aspect='auto', extent=ext,
                      cmap='hot', interpolation='bilinear')
            try:
                with open("ioffe_regel.dat") as fir:
                    h = fir.readline()
                    for p in h.split():
                        if p.startswith("IR_L=") and 'Long' in label:
                            ax.axhline(float(p.split('=')[1]),
                                       color='cyan', ls='--', lw=1.2)
                        elif p.startswith("IR_T=") and 'Trans' in label:
                            ax.axhline(float(p.split('=')[1]),
                                       color='cyan', ls='--', lw=1.2)
            except Exception:
                pass
            ax.set_xlabel(r'$|q|$ (1/A)')
            ax.set_ylabel('Frequency (THz)')
            ax.set_title(f'{label} Dispersion')
            plt.tight_layout()
            tag = 'L' if 'Long' in label else 'T'
            plt.savefig(f'plot_dispersion_eigvec_{tag}.png', dpi=300, bbox_inches='tight')
            plt.close()

    # PR + Ioffe-Regel coloured
    if os.path.isfile("mode_contributions.dat"):
        print("  Plotting PR with Ioffe-Regel boundaries...")
        mc = np.loadtxt("mode_contributions.dat")
        ir_freq = mc[0]
        mob = mc[1]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pos = prd[:, 0] > 0.1
        fp, pp = prd[pos, 0], prd[pos, 1]
        prop_m = (fp < ir_freq) & (pp > 0.05)
        diff_m = (fp >= ir_freq) & (fp < mob) & (pp >= 0.05)
        loc_m = (pp < 0.05) | (fp >= mob)
        for m, c, l in [(prop_m, 'C0', 'Propagon'),
                        (diff_m, 'C1', 'Diffuson'),
                        (loc_m, 'C3', 'Locon')]:
            ax.scatter(fp[m], pp[m], s=0.8, c=c, alpha=0.4,
                       label=f'{l} ({np.sum(m)})', rasterized=True)
        ax.axvline(ir_freq, color='cyan', ls='--', lw=1)
        ax.axvline(mob, color='magenta', ls='--', lw=1)
        ax.axhline(0.05, color='gray', ls=':', lw=0.5)
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Participation Ratio')
        ax.legend(fontsize=7, markerscale=8)
        ax.set_xlim(0, fmax)
        plt.tight_layout()
        plt.savefig('plot_pr_ioffe_regel.png', dpi=300, bbox_inches='tight')
        plt.close()

    # kappa vs freq
    if os.path.isfile("kappa_vs_freq.dat"):
        print("  Plotting spectral thermal conductivity...")
        kd = np.loadtxt("kappa_vs_freq.dat")
        ad = np.loadtxt("kappa_accumulation.dat")
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax = axes[0]
        w = kd[1, 0] - kd[0, 0]
        ax.bar(kd[:, 0], kd[:, 1], width=w, alpha=0.35, color='C0', label='Classical')
        ax.bar(kd[:, 0], kd[:, 2], width=w, alpha=0.35, color='C1', label='Quantum')
        ax.set_ylabel(r'$\kappa$ per bin (W/m K)')
        ax.legend()
        ax.axhline(0, color='gray', lw=0.5)

        ax = axes[1]
        ax.plot(ad[:, 0], ad[:, 3], 'C0-', lw=1.5, label='Classical')
        ax.plot(ad[:, 0], ad[:, 4], 'C1-', lw=1.5, label='Quantum')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel(r'Normalised cumulative $\kappa$')
        ax.axhline(1, color='gray', ls='--', lw=0.5)
        ax.set_ylim(-0.05, 1.15)
        ax.legend()
        plt.tight_layout()
        plt.savefig('plot_kappa_vs_freq.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Accumulation with mode regions
        if os.path.isfile("mode_contributions.dat"):
            mc = np.loadtxt("mode_contributions.dat")
            ir_freq = mc[0]; mob = mc[1]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.fill_betweenx([0, 1.15], 0, ir_freq, alpha=0.08, color='C0')
            ax.fill_betweenx([0, 1.15], ir_freq, mob, alpha=0.08, color='C1')
            ax.fill_betweenx([0, 1.15], mob, fmax, alpha=0.08, color='C3')
            ax.plot(ad[:, 0], ad[:, 3], 'k-', lw=1.5, label='Classical')
            ax.plot(ad[:, 0], ad[:, 4], 'C1--', lw=1.5, label='Quantum')
            ax.axvline(ir_freq, color='gray', ls=':', lw=0.7)
            ax.axvline(mob, color='gray', ls=':', lw=0.7)
            ax.set_xlabel('Frequency (THz)')
            ax.set_ylabel(r'Normalised cumulative $\kappa$')
            ax.set_xlim(0, fmax)
            ax.set_ylim(-0.05, 1.15)
            ax.axhline(1, color='gray', ls='--', lw=0.5)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig('plot_kappa_accumulation.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Mode contributions bar
            vals = [mc[2], mc[3], mc[4]]
            labels = ['Propagons', 'Diffusons', 'Locons']
            colors = ['C0', 'C1', 'C3']
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, vals, color=colors, alpha=0.7)
            for b, v in zip(bars, vals):
                pct = 100*v/mc[5] if mc[5] != 0 else 0
                ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                        f'{v:.3f}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
            ax.set_ylabel(r'$\kappa$ (W/m K)')
            ax.set_title(f'Mode Contributions (quantum, {T:.0f} K)')
            plt.tight_layout()
            plt.savefig('plot_mode_contributions.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Spectral diffusivity
    if os.path.isfile("spectral_diffusivity.dat"):
        print("  Plotting spectral diffusivity...")
        sd = np.loadtxt("spectral_diffusivity.dat")
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = (sd[:, 0] > 0.3) & (sd[:, 2] > 0)
        ax.scatter(sd[pos, 0], sd[pos, 2], s=3, c='k', alpha=0.4,
                   label=r'$D(\omega)$ quantum', rasterized=True)
        ax.plot(sd[:, 0], sd[:, 3], 'b--', lw=1.2, label=r'$D_a = v_D \cdot a$')
        ax.plot(sd[:, 0], sd[:, 4], 'r--', lw=1.2, label=r'$D_{RW}$')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel(r'Diffusivity (A$^2$/ps)')
        ax.set_xlim(0, fmax)
        ax.legend(fontsize=8)
        ax.set_title('Spectral Thermal Diffusivity')
        plt.tight_layout()
        plt.savefig('plot_spectral_diffusivity.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Mode-mode map
    if os.path.isfile("kappa_mode_map.npy"):
        print("  Plotting mode-mode correlation map...")
        kmap = np.load("kappa_mode_map.npy")
        bc_k = np.loadtxt("kappa_vs_freq.dat")[:, 0]
        fig, ax = plt.subplots(figsize=(7, 6))
        vmax = np.percentile(np.abs(kmap[kmap != 0]), 99) if np.any(kmap != 0) else 1
        ax.imshow(kmap.T, origin='lower', aspect='auto',
                  extent=[bc_k[0], bc_k[-1], bc_k[0], bc_k[-1]],
                  cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Frequency (THz)')
        ax.set_title(r'Mode-mode $\kappa$ correlation map')
        plt.colorbar(ax.images[0], label=r'$\kappa_{nm}$ (W/m K)')
        plt.tight_layout()
        plt.savefig('plot_kappa_mode_map.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Velocity-current dispersion
    if os.path.isfile("dispersion_axes.npz"):
        print("  Plotting velocity-current dispersion...")
        da = np.load("dispersion_axes.npz")
        for label, fname in [('Longitudinal', 'dispersion_CL.npy'),
                             ('Transverse', 'dispersion_CT.npy')]:
            if not os.path.isfile(fname):
                continue
            C = np.load(fname)
            fig, ax = plt.subplots(figsize=(7, 5))
            ext = [da['q_vals'][0], da['q_vals'][-1],
                   da['freq'][0], da['freq'][-1]]
            fcut = min(fmax, da['freq'][-1])
            ax.imshow(C.T, origin='lower', aspect='auto', extent=ext,
                      cmap='inferno', interpolation='bilinear')
            ax.set_ylim(0, fcut)
            ax.set_xlabel(r'$|q|$ (1/A)')
            ax.set_ylabel('Frequency (THz)')
            ax.set_title(f'{label} Velocity-Current Dispersion')
            plt.tight_layout()
            tag = 'L' if 'Long' in label else 'T'
            plt.savefig(f'plot_dispersion_velcurr_{tag}.png', dpi=300, bbox_inches='tight')
            plt.close()

    # MFP
    if os.path.isfile("dispersion_fits.dat"):
        print("  Plotting mean free path...")
        df = np.loadtxt("dispersion_fits.dat")
        fig, ax = plt.subplots(figsize=(7, 5))
        mL = df[:, 3] > 0
        mT = df[:, 8] > 0
        if np.any(mL):
            ax.scatter(df[mL, 1], df[mL, 4], c='C0', s=30, label='Longitudinal')
        if np.any(mT):
            ax.scatter(df[mT, 5], df[mT, 8], c='C1', s=30, label='Transverse')
        try:
            with open("spectral_diffusivity.dat") as fsd:
                for line in fsd:
                    if 'a_nn' in line:
                        for p in line.split():
                            if p.startswith('a_nn='):
                                a_nn = float(p.split('=')[1])
                                ax.axhline(a_nn, color='gray', ls='--', lw=0.8,
                                           label=f'a = {a_nn:.2f} A')
                        break
        except Exception:
            pass
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Mean Free Path (A)')
        ax.set_title('Vibrational Mean Free Path')
        ax.legend(fontsize=8)
        ax.set_xlim(0, fmax)
        plt.tight_layout()
        plt.savefig('plot_mfp.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Relaxation times
    if os.path.isfile("relaxation_times.dat"):
        print("  Plotting relaxation times...")
        rt = np.loadtxt("relaxation_times.dat")
        pos = (rt[:, 0] > 0.1) & (rt[:, 2] > 0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(rt[pos, 0], rt[pos, 2], s=1, c=rt[pos, 1], cmap='viridis',
                   alpha=0.4, rasterized=True)
        plt.colorbar(ax.collections[0], label='Participation Ratio')
        nu_ref = np.linspace(0.5, cfg['relax_freq_max'], 200)
        tau_ref = rt[pos, 2].max() * (0.5 / nu_ref)**2
        ax.plot(nu_ref, tau_ref, 'r--', lw=0.8, alpha=0.5, label=r'$\propto \omega^{-2}$')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel(r'$\tau$ (ps)')
        ax.set_title(f'Mode Relaxation Times at {T:.0f} K')
        ax.set_xlim(0, cfg['relax_freq_max'])
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('plot_relaxation_times.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("  All plots saved.")

# ===================================================================
#  Subcommand handlers
# ===================================================================
def cmd_eigensystem(args):
    cfg = build_config(args)
    compute_eigensystem(cfg)
    if args.plot:
        generate_plots(cfg)


def cmd_vdos(args):
    cfg = build_config(args)
    eig = ensure_eigensystem(cfg)

    section("VDOS")
    print(f"  Frequency range: {eig['vdos_centres'][0]:.2f} "
          f"to {eig['vdos_centres'][-1]:.2f} THz")
    print(f"  {len(eig['vdos_centres'])} bins")
    total_modes = eig['vdos_dos'].sum() * (eig['vdos_centres'][1] - eig['vdos_centres'][0])
    print(f"  Integrated modes: {total_modes:.0f}  (3N = {3*eig['natoms']})")

    if args.plot:
        fmax = cfg['freq_max']
        vdos = np.loadtxt("vdos.dat")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(vdos[:, 0], vdos[:, 1], alpha=0.3, color='C0')
        ax.plot(vdos[:, 0], vdos[:, 1], 'C0', lw=1)
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('DOS (modes/THz)')
        ax.set_title('Vibrational Density of States')
        ax.set_xlim(0, fmax)
        plt.tight_layout()
        plt.savefig('plot_vdos.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved plot_vdos.png")


def cmd_pr(args):
    cfg = build_config(args)
    eig = ensure_eigensystem(cfg)

    section("PARTICIPATION RATIO")
    freqs = eig['freqs']
    pr = eig['pr']
    pos = freqs > 0.1
    print(f"  PR range: {pr[pos].min():.6f} to {pr[pos].max():.6f}")
    print(f"  Mean PR: {pr[pos].mean():.4f}")
    print(f"  Modes with PR < 0.05: {np.sum(pr[pos] < 0.05)}")
    print(f"  Modes with PR > 0.20: {np.sum(pr[pos] > 0.20)}")

    if args.plot:
        fmax = cfg['freq_max']
        prd = np.loadtxt("participation_ratio.dat")
        mse_d = np.loadtxt("mse.dat")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        m = prd[:, 0] > 0.1
        ax.scatter(prd[m, 0], prd[m, 1], s=0.5, c='k', alpha=0.3, rasterized=True)
        ax.axhline(0.2, color='gray', ls='--', lw=0.6, label='PR=0.2')
        ax.axhline(0.05, color='gray', ls=':', lw=0.6, label='PR=0.05')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Participation Ratio')
        ax.set_xlim(0, fmax)
        ax.legend(fontsize=7)

        ax = axes[1]
        m = mse_d[:, 0] > 0.1
        sc = ax.scatter(mse_d[m, 1], mse_d[m, 2], s=1.5, c=mse_d[m, 0],
                        cmap='viridis', alpha=0.4, rasterized=True)
        ax.set_xlabel('Participation Ratio')
        ax.set_ylabel('Mode Spatial Extent (A)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.colorbar(sc, ax=ax, label='Frequency (THz)')

        plt.tight_layout()
        plt.savefig('plot_pr_mse.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved plot_pr_mse.png")


def cmd_ioffe_regel(args):
    cfg = build_config(args)
    eig = ensure_eigensystem(cfg)
    ir = compute_ioffe_regel(cfg, eig)

    if args.plot:
        fmax = cfg['freq_max']
        prd = np.loadtxt("participation_ratio.dat")

        # F(q,nu) maps
        axes_d = np.load("Fqnu_axes.npz")
        q_v = axes_d['q_vals']
        f_v = axes_d['freq_bins']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (label, fname, ir_key) in zip(axes,
            [('Longitudinal', 'Fqnu_L.npy', ir['ir_L']),
             ('Transverse', 'Fqnu_T.npy', ir['ir_T'])]):
            F = np.load(fname)
            ext = [q_v[0], q_v[-1], f_v[0], f_v[-1]]
            ax.imshow(F.T, origin='lower', aspect='auto', extent=ext,
                      cmap='hot', interpolation='bilinear')
            ax.axhline(ir_key, color='cyan', ls='--', lw=1.2,
                       label=f'IR = {ir_key:.1f} THz')
            ax.set_xlabel(r'$|q|$ (1/A)')
            ax.set_ylabel('Frequency (THz)')
            ax.set_title(f'{label}')
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('plot_ioffe_regel.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved plot_ioffe_regel.png")


def cmd_kappa(args):
    cfg = build_config(args)
    eig = ensure_eigensystem(cfg)

    ir = None
    if not args.skip_ir:
        ir = compute_ioffe_regel(cfg, eig)

    traj = load_trajectory(cfg)
    if traj is None:
        traj = process_trajectory(cfg, eig)

    calculate_thermal_properties(cfg, eig, traj)

    if args.plot:
        generate_plots(cfg)

    section("COMPLETE")
    print(f"  Finished at {timestamp()}")

# ===================================================================
#  CLI definition
# ===================================================================
def add_common_args(parser):
    """Arguments shared across all subcommands."""
    parser.add_argument('--config', type=str, default=None,
                        help='JSON config file (overrides defaults)')
    parser.add_argument('--datafile', type=str, default=None,
                        help='LAMMPS data file')
    parser.add_argument('--dynmat-file', dest='dynmat_file', type=str, default=None,
                        help='Dynamical matrix file (eskm format)')
    parser.add_argument('--T', type=float, default=None,
                        help='Temperature (K)')
    parser.add_argument('--freq-max', dest='freq_max', type=float, default=None,
                        help='Max frequency for binning (THz)')
    parser.add_argument('--nworkers', type=int, default=None,
                        help='Parallel workers (default: all CPUs)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after computation')


def main():
    parser = argparse.ArgumentParser(
        prog='gkma_tool',
        description='Green-Kubo Modal Analysis for amorphous solids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    sub = parser.add_subparsers(dest='command', required=True)

    # -- eigensystem --
    p_eig = sub.add_parser('eigensystem',
        help='Diagonalise dynamical matrix (VDOS, PR, MSE)')
    add_common_args(p_eig)
    p_eig.set_defaults(func=cmd_eigensystem)

    # -- vdos --
    p_vdos = sub.add_parser('vdos',
        help='Vibrational density of states')
    add_common_args(p_vdos)
    p_vdos.set_defaults(func=cmd_vdos)

    # -- pr --
    p_pr = sub.add_parser('pr',
        help='Participation ratio and mode spatial extent')
    add_common_args(p_pr)
    p_pr.set_defaults(func=cmd_pr)

    # -- ioffe-regel --
    p_ir = sub.add_parser('ioffe-regel',
        help='Branch-resolved Ioffe-Regel crossover')
    add_common_args(p_ir)
    p_ir.add_argument('--ir-nq-radial', dest='ir_nq_radial', type=int, default=None)
    p_ir.add_argument('--ir-nq-angular', dest='ir_nq_angular', type=int, default=None)
    p_ir.set_defaults(func=cmd_ioffe_regel)

    # -- kappa --
    p_k = sub.add_parser('kappa',
        help='Full GKMA thermal conductivity pipeline')
    add_common_args(p_k)
    p_k.add_argument('--dump-file', dest='dump_file', type=str, default=None,
                     help='NVE trajectory dump file')
    p_k.add_argument('--hf-file', dest='hf_file', type=str, default=None,
                     help='LAMMPS total heat flux file (for GK validation)')
    p_k.add_argument('--timestep-ps', dest='timestep_ps', type=float, default=None)
    p_k.add_argument('--dump-freq', dest='dump_freq', type=int, default=None)
    p_k.add_argument('--nbins', type=int, default=None)
    p_k.add_argument('--max-frames', dest='max_frames', type=int, default=None)
    p_k.add_argument('--corr-fraction', dest='corr_fraction', type=float, default=None)
    p_k.add_argument('--effective-thickness', dest='effective_thickness',
                     type=float, default=None,
                     help='For 2D materials: effective thickness in Angstrom')
    p_k.add_argument('--skip-ir', action='store_true',
                     help='Skip Ioffe-Regel analysis')
    p_k.set_defaults(func=cmd_kappa)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
