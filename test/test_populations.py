#!/usr/bin/env python3
"""
Check the spin-temperature weighting of the spectra by the He3 polarization M.

Needs only numpy:  python test/test_populations.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'web'))

import bridge
from helium_spectra_calc import HeliumSpectraCalculator

calc = HeliumSpectraCalculator()
failures = []

SERIES = [f'{iso}_{pol}' for iso in ('he3', 'he4') for pol in ('plus', 'minus', 'pi')]


def report(label, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + label + ('' if ok else f'  <- {detail}'))
    if not ok:
        failures.append(f'{label}: {detail}')


def nacher_popa(B, M):
    """popa from spectreVoigt_w0w12, energy-ordered levels 1-6, summing to 1"""
    eb = (1 + M) / (1 - M)
    if B > 0.1619:
        powers = [-1.5, -0.5, 0.5, -0.5, 1.5, 0.5]
    else:
        powers = [-1.5, -0.5, 0.5, 1.5, -0.5, 0.5]
    w = np.array([eb ** p for p in powers])
    return w / (eb ** 1.5 + 2 * eb ** 0.5 + 2 * eb ** -0.5 + eb ** -1.5)


def nacher_popy(M):
    """popy from spectreVoigt_He4w0w12, energy-ordered levels 1-3"""
    eb = (1 + M) / (1 - M)
    w = np.array([1 / eb, 1.0, eb])
    return w / w.sum()


print('1. M = 0 leaves every spectrum unchanged')
for B in (0.0001, 0.13, 1.0, 5.0):
    full = calc.calculate_full_results(B, 300.0)
    pops = full['populations']
    same = (np.all(pops['he3'] == 1.0) and np.all(pops['he4'] == 1.0))
    report(f'B = {B} T: all populations exactly 1', same,
           f"{pops['he3']}, {pops['he4']}")

print()
print("2. Populations vs Nacher's popa / popy tables")
# The Fortran switches tables at 0.1619 T; the true crossing is a hair lower,
# so fields right at it are skipped.
for B in (0.0001, 0.05, 0.13, 0.17, 1.0, 4.0):
    for M in (-0.7, 0.3, 0.9):
        pops = calc.calculate_full_results(B, 300.0, M=M)['populations']
        got3 = pops['he3'] / len(pops['he3'])
        got4 = pops['he4'] / len(pops['he4'])
        err = max(np.max(np.abs(got3 - nacher_popa(B, M))),
                  np.max(np.abs(got4 - nacher_popy(M))))
        report(f'B = {B} T, M = {M:+.1f}: worst difference {err:.1e}',
               err < 1e-12, f'{got3} vs {nacher_popa(B, M)}')

print()
print('3. Each line is weighted by its lower level')
B, M = 1.0, 0.6
full = calc.calculate_full_results(B, 300.0, M=M)
base = calc.calculate_full_results(B, 300.0)
pops = full['populations']
# Strength is linear in the forces, so rebuilding the spectra from forces
# weighted by hand must reproduce what calculate_full_results gives
args = []
for iso, D in (('he3', 'D3'), ('he4', 'D4')):
    for pol in ('plus', 'minus', 'pi'):
        tr = base['transitions'][iso][pol]
        args += [tr['energies'], tr['forces'] * pops[iso][tr['ind_lower']]]
    args.append(base['doppler_widths'][D])
rebuilt = calc.generate_spectra_data(*args)
worst = max(np.max(np.abs(rebuilt[k] - full['spectra_data'][k])) for k in SERIES)
report(f'rebuilt spectra agree to {worst:.1e}', worst < 1e-12, f'{worst:.1e}')
same_forces = all(np.array_equal(full['transitions'][i][p]['forces'],
                                 base['transitions'][i][p]['forces'])
                  for i in ('he3', 'he4') for p in ('plus', 'minus', 'pi'))
report('returned transition forces stay per atom', same_forces, 'changed with M')

print()
print('4. Near zero field, sigma+ at M is sigma- at -M')
# The Zeeman shifts at 0.1 mT are ~MHz against a ~2 GHz Doppler width
for M in (0.4, 0.95):
    up = calc.calculate_full_results(0.0001, 300.0, M=M)['spectra_data']
    down = calc.calculate_full_results(0.0001, 300.0, M=-M)['spectra_data']
    for iso in ('he3', 'he4'):
        a, b = up[f'{iso}_plus'], down[f'{iso}_minus']
        rel = np.max(np.abs(a - b)) / np.max(a)
        report(f'{iso}, M = {M}: differ by {rel:.1e} of the peak', rel < 5e-3,
               f'{rel:.1e}')
    rel = np.max(np.abs(up['he3_plus'] - up['he3_minus'])) / np.max(up['he3_plus'])
    report(f'and at M = {M} the two differ ({rel:.2f} of the peak)', rel > 0.05,
           f'{rel:.2e}')

print()
print('5. M out of range is refused')
for M in (-1.0, 1.0, 1.5):
    try:
        calc.calculate_full_results(1.0, 300.0, M=M)
        report(f'M = {M} raises', False, 'no error')
    except ValueError:
        report(f'M = {M} raises', True)

print()
print('6. The page: populations in the diagram, table unchanged')
for isotope in ('He3', 'He4'):
    at0 = bridge.compute(1.0, 300.0, isotope, 'Frequency Offset', 0.0, 0.0)
    atM = bridge.compute(1.0, 300.0, isotope, 'Frequency Offset', 0.0, 0.5)
    shares = [lv['pop'] for lv in atM['levels']['S']]
    report(f'{isotope}: level populations sum to 1', abs(sum(shares) - 1) < 1e-12,
           f'{sum(shares)}')
    top = max(atM['levels']['S'], key=lambda lv: lv['pop'])
    report(f'{isotope}: M > 0 favours the highest m_F', top['mf'] ==
           max(lv['mf'] for lv in atM['levels']['S']), f"m_F {top['mf']}")
    report(f'{isotope}: table intensities do not depend on M',
           [r['intensity'] for r in at0['table']] == [r['intensity'] for r in atM['table']],
           'differ')
    report(f'{isotope}: spectra do', at0['spectra']['plus'] != atM['spectra']['plus'],
           'identical')
    report(f'{isotope}: absorption equals intensity at M = 0',
           all(r['absorption'] == r['intensity'] for r in at0['table'] + at0['lines']),
           'differ')
    # Absorption by hand: each line's strength times its level's population.
    # The intensities read back are rounded to 4 decimals, so each member can
    # be off by 5e-5 times its population, plus 5e-5 for the absorption's own.
    pops = calc.calculate_full_results(1.0, 300.0, M=0.5)['populations'][isotope.lower()]
    worst = 0.0
    for row in atM['table'] + atM['lines']:
        want = sum(float(m['intensity']) * pops[m['lower']] for m in row['members'])
        bound = 5e-5 * (1 + sum(pops[m['lower']] for m in row['members']))
        worst = max(worst, abs(float(row['absorption']) - want) / bound)
    report(f'{isotope}: absorption is intensity x population, within rounding',
           worst <= 1.0, f'{worst:.2f} of the rounding bound')
    bridge.compute(1.0, 300.0, isotope, 'Frequency Offset', 0.0, 0.5)
    readout = bridge.pumping_line(0)
    member = readout['members'][0]
    want = float(member['intensity']) * pops[member['lower']]
    report(f'{isotope}: pumping_line members carry absorption',
           abs(float(member['absorption']) - want) < 2e-4, member['absorption'])
    report(f'{isotope}: title gives M', 'M = +0.50' in atM['title']
           and 'M =' not in at0['title'], atM['title'])

print()
print('=' * 72)
if failures:
    print(f'{len(failures)} FAILURE(S):')
    for f in failures:
        print('  ' + f)
    sys.exit(1)
print('Spin-temperature populations behave as intended.')
