#!/usr/bin/env python3
"""
Check HeliumSpectraCalculator.pumping_rates and the page's pumping readout.

Needs only numpy:  python test/test_pumping.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'web'))

import bridge
from helium_spectra_calc import HeliumSpectraCalculator, voigt_K

calc = HeliumSpectraCalculator()
failures = []


def report(label, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + label + ('' if ok else f'  <- {detail}'))
    if not ok:
        failures.append(f'{label}: {detail}')


print('1. A Gaussian laser widens the Gaussian part in quadrature')
# Brute force: convolve the atomic Voigt with the laser's Gaussian spectrum
# and compare with voigt_K at sqrt(wG^2 + wLaser^2).
wG = calc.doppler_fwhm(300.0, 'He3')
dx = 0.002
grid = np.arange(-60.0, 60.0, dx)
worst = 0.0
for wL in (0.0, 0.12, 1.2):
    for wLaser in (0.5, 2.0):
        sigma = wLaser / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        laser = np.exp(-0.5 * (grid / sigma) ** 2)
        laser /= laser.sum()
        for detuning in (0.0, 1.0, 2.5, 13.8):
            brute = float(np.sum(laser * voigt_K(detuning - grid, wG, wL)))
            closed = float(voigt_K(detuning, np.hypot(wG, wLaser), wL)
                           * wG / np.hypot(wG, wLaser))
            worst = max(worst, abs(brute - closed) / closed)
report(f'agrees with a brute-force convolution to {worst:.1e}', worst < 1e-4, f'{worst:.1e}')
# The factor wG/wG_eff is K's peak-height normalisation. It is common to every
# line, so it cancels from the relative rates pumping_rates is used for.

print()
print('2. The 5 T, 100 mbar case, sigma- peak on A1-A4, 2 GHz laser')
def strong_sigma_minus_readout(pressure):
    """compute() then pumping() for the sigma- peak on A1-A4, as the page does"""
    table = bridge.compute(5.0, 300.0, 'He3', 'Frequency Offset', pressure)['table']
    index = next(i for i, r in enumerate(table)
                 if r['polarization'] == 'σ-' and len(r['members']) == 4)
    return bridge.pumping(index)


levels = strong_sigma_minus_readout(100.0)['levels']
for i, lv in enumerate(levels):
    print(f'     A{i + 1}: {lv["text"]:>7}{"  pumped" if lv["targeted"] else ""}')
report('A1-A4 marked as pumped', [lv['targeted'] for lv in levels] == [True] * 4 + [False] * 2)
report('A1-A4 at 85-100%: full-strength lines, a little off the centroid',
       all(0.85 < lv['rel'] < 1.0 for lv in levels[:4]),
       str([round(lv['rel'], 3) for lv in levels[:4]]))
report('A5 and A6 leak at 0.4-0.6%', all(0.004 < lv['rel'] < 0.006 for lv in levels[4:]),
       str([round(lv['rel'], 5) for lv in levels[4:]]))
report('A5 is emptied mostly via its probe line A5 -> B13',
       levels[4]['via'] and levels[4]['via'][0]['name'] == 'A₅ → B₁₃'
       and int(levels[4]['via'][0]['share']) > 90, str(levels[4]['via']))

print()
print('3. Pressure is what makes the leak: 1 mbar vs 100 mbar')
ratio = levels[4]['rel'] / strong_sigma_minus_readout(1.0)['levels'][4]['rel']
report(f'A5 leak grows {ratio:.0f}x from 1 to 100 mbar', ratio > 20, f'{ratio:.1f}')

print()
print('4. 100% means a full-strength line exactly on resonance')
# With the laser on a line, that line's own part of the readout must be exactly
# its strength S. Other lines from the same level can only add to it: at
# 100 mbar a weak line 6 GHz from a strong one of the same level reads ~30x its
# own strength, nearly all through the strong line's wing.
full = calc.calculate_full_results(5.0, 300.0)
wG = calc.doppler_fwhm(300.0, 'He3')
wL = calc.collision_per_mbar['He3'] * 100.0
reference = float(voigt_K(0.0, np.hypot(wG, bridge.PUMP_LASER_FWHM), wL))
worst_own, never_below = 0.0, True
for pol in ('plus', 'minus', 'pi'):
    tr = full['transitions']['he3'][pol]
    for j, (nu, S) in enumerate(zip(tr['energies'], tr['forces'])):
        rates, per_line = calc.pumping_rates(
            tr['energies'], tr['forces'], tr['ind_lower'], nu, wG, wL,
            bridge.PUMP_LASER_FWHM, 6)
        worst_own = max(worst_own, abs(per_line[j] / reference - S))
        never_below &= rates[tr['ind_lower'][j]] / reference >= S - 1e-12
report(f"a line's own part is exactly S, to {worst_own:.1e}", worst_own < 1e-12,
       f'{worst_own:.1e}')
report('other lines only ever add to it', never_below)

print()
print('5. The readout is made on request, for the table compute() last returned')
table = bridge.compute(5.0, 300.0, 'He3', 'Frequency Offset', 100.0)['table']
report('rows carry no readout of their own', all('pumping' not in r for r in table))
report('an out-of-range row gives None', bridge.pumping(len(table)) is None)
report('every row has a readout', all(bridge.pumping(i) for i in range(len(table))))

print()
print('=' * 72)
if failures:
    print(f'{len(failures)} FAILURE(S):')
    for f in failures:
        print('  - ' + f)
    sys.exit(1)
print('Pumping rates behave as intended.')
