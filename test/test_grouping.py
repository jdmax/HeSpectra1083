#!/usr/bin/env python3
"""
Check the centroid grouping in web/bridge.py.

A line joins a peak when it lies within GROUP_THRESHOLD of the peak's
intensity-weighted centroid. Needs only numpy:  python test/test_grouping.py
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


def report(label, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + label + ('' if ok else f'  <- {detail}'))
    if not ok:
        failures.append(f'{label}: {detail}')


def centroid(group):
    e = np.asarray(group['energies'], float)
    s = np.asarray(group['forces'], float)
    return float(np.sum(e * s) / np.sum(s))


print('1. Every line lands in exactly one peak, over a field and temperature scan')
worst_offset = (0.0, None)
lost = []
for iso in ('he3', 'he4'):
    for B in np.arange(0.05, 7.01, 0.25):
        for T in (77.0, 300.0, 1000.0):
            full = calc.calculate_full_results(float(B), T)
            for pol in ('plus', 'minus', 'pi'):
                tr = full['transitions'][iso][pol]
                groups = bridge.group_transitions(
                    tr['energies'], tr['forces'], tr['ind_lower'], tr['ind_upper'])
                got = sorted((int(l), int(u)) for g in groups
                             for l, u in zip(g['ind_lower'], g['ind_upper']))
                want = sorted((int(l), int(u))
                              for l, u in zip(tr['ind_lower'], tr['ind_upper']))
                if got != want:
                    lost.append((iso, round(float(B), 2), T, pol))
                for g in groups:
                    c = centroid(g)
                    d = float(np.max(np.abs(np.asarray(g['energies']) - c)))
                    if d > worst_offset[0]:
                        worst_offset = (d, (iso, round(float(B), 2), T, pol))
report('no line lost or duplicated', not lost, str(lost[:3]))

# Members are within the threshold of the centroid when they join, but the
# centroid can move as later, weaker lines join; how far does that go?
print(f'  largest distance of any member from its final centroid: '
      f'{worst_offset[0]:.3f} GHz  at {worst_offset[1]}')
report('no member ends more than 1.1x the threshold from its centroid',
       worst_offset[0] <= 1.1 * bridge.GROUP_THRESHOLD, f'{worst_offset[0]:.3f}')

print()
print('2. The case that motivated the change: He3 sigma- at 5 T')
full = calc.calculate_full_results(5.0, 300.0)
tr = full['transitions']['he3']['minus']
groups = bridge.group_transitions(
    tr['energies'], tr['forces'], tr['ind_lower'], tr['ind_upper'])
strong = max(groups, key=lambda g: float(np.sum(g['forces'])))
lowers = sorted(int(l) + 1 for l in strong['ind_lower'])
print(f'  strongest peak: lower levels {lowers}, centroid {centroid(strong):.3f} GHz,'
      f' span {max(strong["energies"]) - min(strong["energies"]):.3f} GHz')
report('strong peak is exactly A1-A4', lowers == [1, 2, 3, 4], str(lowers))
a5b12 = [g for g in groups
         if any(int(l) == 4 and int(u) == 11
                for l, u in zip(g['ind_lower'], g['ind_upper']))]
report('A5 -> B12 is a peak of its own',
       len(a5b12) == 1 and len(a5b12[0]['energies']) == 1,
       str([len(g['energies']) for g in a5b12]))

print()
print('3. Table rows report the intensity-weighted centroid')
row = next(r for r in bridge.compute(5.0, 300.0, 'He3', 'Frequency Offset')['table']
           if r['polarization'] == 'σ-' and len(r['members']) == 4)
report(f'row frequency {row["frequency"]} is the centroid',
       row['frequency'] == f'{centroid(strong):.3f}',
       f'expected {centroid(strong):.3f}')

print()
print('=' * 72)
if failures:
    print(f'{len(failures)} FAILURE(S):')
    for f in failures:
        print('  - ' + f)
    sys.exit(1)
print('Centroid grouping behaves as intended.')
