#!/usr/bin/env python3
"""
Check helium_spectra_calc.py against P.J. Nacher's Fortran.

Two independent halves:

  Line positions and strengths, against the spectre*.dat files in this
  directory. These are what HeSpectra's Fortran wrote and what
  spectreVoigt_w0w12 reads back: one row per transition, holding
  B, frequency (GHz), strength, lower level and upper level, both 1-based,
  then the plot range and the polarisation.

  The Voigt line shape, against spectreVoigt's own funcV/qsimp, transcribed
  below, and its wG Doppler formula.

Needs only numpy. Run it from anywhere:  python test/test_against_fortran.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from helium_spectra_calc import HeliumSpectraCalculator, voigt_K

# Tolerances: the .dat files carry 8 significant digits, so agreement is
# limited by their formatting rather than by either calculation.
FREQ_TOL = 2e-4      # GHz
STRENGTH_TOL = 2e-4  # relative

CASES = [
    ('spectre3plusB.dat', 'he3', 'plus', 'He3', 3),
    ('spectre3minusB.dat', 'he3', 'minus', 'He3', 3),
    ('spectre3piB.dat', 'he3', 'pi', 'He3', 3),
    ('spectre4plusB.dat', 'he4', 'plus', 'He4', 1),
    ('spectre4minusB.dat', 'he4', 'minus', 'He4', 1),
    ('spectre4piB.dat', 'he4', 'pi', 'He4', 1),
]

failures = []


def report(label, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + label + ('' if ok else f'  <- {detail}'))
    if not ok:
        failures.append(f'{label}: {detail}')


def read_spectre(path):
    """One row per transition, then xp1, xp2 and the polarisation label."""
    rows, tail = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append(tuple(float(p) for p in parts[:3])
                        + (int(parts[3]), int(parts[4])))
        elif parts:
            tail.append(line.strip())
    return rows, tail


# ----------------------------------------------------------------- Fortran
# spectreVoigt's integrand and its adaptive Simpson, kept close to the
# original so this stays a check on our Voigt rather than a restatement of it.

def _funcV(z, pos, wG, wL, xc=0.0):
    ln2 = np.log(2.0)
    mor1 = ln2 * (wL / wG) ** 2
    mor2 = (2.0 * np.sqrt(ln2) / wG * (pos - xc) - z) ** 2
    return np.exp(-z ** 2) / (mor1 + mor2)


def _trapzd(a, b, previous, n, args):
    if n == 1:
        return 0.5 * (b - a) * (_funcV(a, *args) + _funcV(b, *args))
    it = 2 ** (n - 2)
    delta = (b - a) / it
    x = a + 0.5 * delta + delta * np.arange(it)
    return 0.5 * (previous + (b - a) * np.sum(_funcV(x, *args)) / it)


def nacher_voigt(pos, wG, wL, eps=1e-6, jmax=20):
    """const * qsimp(funcV, -7, 7), exactly as the Fortran forms it."""
    ln2 = np.log(2.0)
    const = np.sqrt(ln2) * wL / wG / np.pi
    args = (pos, wG, wL)
    ost = os_ = -1e30
    s = 0.0
    for j in range(1, jmax + 1):
        st = _trapzd(-7.0, 7.0, ost, j, args)
        s = (4.0 * st - ost) / 3.0
        if j > 1 and abs(s - os_) < eps * abs(os_):
            break
        os_, ost = s, st
    return const * s


# --------------------------------------------------------------------- run
calc = HeliumSpectraCalculator()

print('1. Line positions, strengths and level indices vs spectre*.dat')
for name, iso, pol, isotope, n_wL0 in CASES:
    path = HERE / name
    if not path.exists():
        report(name, False, 'file missing')
        continue
    rows, tail = read_spectre(path)
    B = rows[0][0]
    full = calc.calculate_full_results(B, 300.0)
    tr = full['transitions'][iso][pol]
    ours = {(int(lo) + 1, int(up) + 1): (float(e), float(s))
            for e, s, lo, up in zip(tr['energies'], tr['forces'],
                                    tr['ind_lower'], tr['ind_upper'])}

    label = f'{name} ({tail[-1] if tail else "?"}, B = {B} T)'
    if len(ours) != len(rows):
        report(label, False, f'{len(rows)} lines in file, {len(ours)} from us')
        continue

    worst_f = worst_s = 0.0
    missing = []
    for _, freq, strength, lo, up in rows:
        if (lo, up) not in ours:
            missing.append((lo, up))
            continue
        e, s = ours[(lo, up)]
        worst_f = max(worst_f, abs(e - freq))
        worst_s = max(worst_s, abs(s - strength))
    if missing:
        report(label, False, f'transitions absent from ours: {missing}')
        continue
    report(f'{label}: {len(rows)} lines, worst df {worst_f:.2e} GHz, '
           f'dS {worst_s:.2e}',
           worst_f < FREQ_TOL and worst_s < STRENGTH_TOL,
           f'df {worst_f:.2e}, dS {worst_s:.2e}')

print()
print('2. Doppler width vs the Fortran wG formula')
for isotope, mass in (('He3', 3.0160293e-3), ('He4', 4.00260e-3)):
    for T in (77.0, 300.0, 1000.0):
        want = (np.sqrt(2 * 8.314472 * T / mass) / 1.0829e-6
                * 2 * np.sqrt(np.log(2.0)) / 1e9)
        got = calc.doppler_fwhm(T, isotope)
        report(f'wG {isotope} at {T:.0f} K = {got:.6f} GHz',
               abs(got - want) < 1e-12, f'expected {want:.9f}')

print()
print('3. Voigt line shape vs the Fortran funcV/qsimp integral')
wG = calc.doppler_fwhm(300.0, 'He3')
worst = 0.0
for wL in (0.012, 0.12, 0.6, 1.2, 3.2):
    for pos in (0.0, 0.3, 1.0, 2.5, 6.0, 15.0, 40.0):
        ref = nacher_voigt(pos, wG, wL)
        got = float(voigt_K(pos, wG, wL))
        worst = max(worst, abs(got - ref) / max(abs(ref), 1e-300))
report(f'K(x, y) matches to {worst:.2e} relative, over wL 0.012-3.2 GHz',
       worst < 1e-5, f'{worst:.2e}')
report('K(x, 0) is the Doppler Gaussian',
       np.allclose(voigt_K(np.linspace(-8, 8, 400), wG, 0.0),
                   np.exp(-(np.linspace(-8, 8, 400)
                            * 2 * np.sqrt(np.log(2.0)) / wG) ** 2), atol=0),
       'differs')

print()
print('4. Which lines carry wL0: our J=0 admixture vs the Fortran row split')
print('   (the Fortran hardcodes rows 1-3 for He3 and row 1 for He4)')
for name, iso, pol, isotope, n_wL0 in CASES:
    path = HERE / name
    if not path.exists():
        continue
    rows, tail = read_spectre(path)
    B = rows[0][0]
    if isotope == 'He3':
        H = calc.H3PB0 + calc.mu * B * calc.Hzee3P
        n_spin = 2
    else:
        H = calc.Hf4P + calc.mu * B * calc.Hzee4P
        n_spin = 1
    w, V = np.linalg.eigh(H)
    weights = calc.j0_weights(V[:, w.argsort()], n_spin=n_spin)

    ours = [i + 1 for i, r in enumerate(rows) if weights[r[4] - 1] > 0.5]
    theirs = list(range(1, n_wL0 + 1))
    if ours == theirs:
        print(f'  ok   {name:22s} rows {ours}')
    else:
        # Not counted as a failure: see the note printed below.
        print(f'  note {name:22s} we find rows {ours}, the Fortran uses {theirs}')
        for i in [r for r in ours if r not in theirs]:
            _, freq, strength, lo, up = rows[i - 1]
            print(f'         row {i}: {freq:9.4f} GHz, {lo} -> {up}, '
                  f'J=0 weight {weights[up - 1]:.3f}, strength {strength:.4g}')

print()
print('=' * 72)
if failures:
    print(f'{len(failures)} FAILURE(S):')
    for f in failures:
        print('  - ' + f)
    sys.exit(1)
print('helium_spectra_calc.py reproduces the Fortran line lists and line shape.')
print()
print('Note on section 4: spectreVoigt gives wL0 to a hardcoded first three')
print('rows, which is right for the 22-line sigma lists but one short for the')
print('26-line pi list, where four transitions reach the 2^3P_0 doublet. It')
print('makes no difference while wL0 and wL12 are set equal, as the rates')
print('quoted in the Fortran prompts make them.')
