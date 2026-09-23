#!/usr/bin/env python3
"""
Browser bridge for the static build of the helium spectra calculator.

Runs inside Pyodide. Imports helium_spectra_calc and returns plain
JSON-able structures; Plotly.js does all the drawing on the JavaScript side.
The presentation logic began as a port of the Streamlit app
(helium_spectra_ui.py), and has since departed from it: peaks are grouped
around intensity-weighted centroids rather than by single linkage.
"""

import numpy as np
from helium_spectra_calc import HeliumSpectraCalculator, voigt_K

# c expressed so that (nm) = C_NM_GHZ / (GHz)
C_NM_GHZ = 299792458.0

# A line joins a peak when it lies within this many GHz of the peak's
# intensity-weighted centroid (see group_transitions)
GROUP_THRESHOLD = 2.0

# Spectral width of the pumping laser assumed by the pumping readout: a
# Gaussian of this FWHM in GHz, centred on the selected peak's centroid.
PUMP_LASER_FWHM = 2.0

# A lower level counts as one the peak pumps, and is always labelled, when one
# of its lines in the peak is at least this fraction of the peak's strongest.
PUMP_TARGET_FRACTION = 0.1

# Collisional broadening lives in helium_spectra_calc, which follows
# P.J. Nacher's spectreVoigt_w0w12: a Voigt line shape built from a Doppler
# FWHM and two Lorentz FWHMs, wL0 for the 2^3P_0 lines and wL12 for the rest.
#
# It broadens the LINE SHAPE only. The line positions and strengths come from
# a model valid to a few mbar; collisional shifts (~1.4 MHz/mbar) and any line
# mixing among the 2^3P sublevels are not included, so above a few mbar the
# widths are right but the positions are still the low-pressure ones.

_calculator = None

# What pumping() needs to produce the readout for one row of the table that
# compute() last returned. The readout is only ever shown for the selected
# row, so it is made on request rather than for all ~50 rows on every
# recompute, which would roughly double the cost of dragging a slider.
_last_pumping = None


def _get_calculator():
    """Cached calculator instance, the Pyodide equivalent of @st.cache_data"""
    global _calculator
    if _calculator is None:
        _calculator = HeliumSpectraCalculator()
    return _calculator


def group_transitions(energies, forces, ind_lower, ind_upper, threshold=GROUP_THRESHOLD):
    """Group transitions into peaks around intensity-weighted centroids.

    Lines are taken strongest first. Each joins the nearest existing peak
    whose centroid lies within `threshold` GHz, and that centroid is then
    recomputed; a line with no peak in reach starts its own. Strong lines
    therefore define the peaks, and weak ones attach to whichever is nearest
    without dragging its centre far.

    This replaces single linkage, which admitted any line within the
    threshold of the previous member, so a group could chain out to any
    width: at 5 T an A5 line of 0.01% relative strength joined the strong
    sigma- peak from 2.5 GHz off its centre.
    """
    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if energies.size == 0:
        return []

    # Strongest first, ties broken by frequency. Many strengths are exactly
    # equal by symmetry and LAPACK's last-bit noise differs between numpy
    # builds, so both keys are rounded; otherwise tied lines could be taken in
    # a different order, and grouped differently, in the browser than here.
    order = np.lexsort((np.round(energies, 6), -np.round(forces, 9)))
    peaks = []           # [member indices, sum of S, sum of S*nu]
    for i in order:
        best, best_distance = None, None
        for peak in peaks:
            distance = abs(energies[i] - peak[2] / peak[1])
            if distance <= threshold and (best is None or distance < best_distance):
                best, best_distance = peak, distance
        if best is None:
            peaks.append([[i], forces[i], forces[i] * energies[i]])
        else:
            best[0].append(i)
            best[1] += forces[i]
            best[2] += forces[i] * energies[i]

    groups = []
    for members, _, _ in sorted(peaks, key=lambda pk: pk[2] / pk[1]):
        idx = np.array(sorted(members, key=lambda k: energies[k]))
        groups.append({
            'energies': list(energies[idx]),
            'forces': list(forces[idx]),
            'ind_lower': list(np.asarray(ind_lower)[idx]),
            'ind_upper': list(np.asarray(ind_upper)[idx]),
        })
    return groups


def to_subscript(s: str) -> str:
    """Converts a string of digits to Unicode subscript characters."""
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    return "".join(subscript_map.get(char, char) for char in s)


def format_transition_name(ind_lower, ind_upper, isotope):
    """Format transition name based on isotope and indices"""
    # Fix subscripts to index 1 and subscript characters
    lower_sub = to_subscript(str(int(ind_lower) + 1))
    upper_sub = to_subscript(str(int(ind_upper) + 1))

    if isotope == 'He3':
        return f"A{lower_sub} → B{upper_sub}"
    else:  # He4
        return f"Y{lower_sub} → Z{upper_sub}"


def build_transitions_table(transitions, isotope, c1_ghz, pump=None):
    """Grouped transitions as a list of row dicts, sorted by intensity.

    With `pump`, also records in _last_pumping what pumping() needs to give
    the readout for any row.
    """
    global _last_pumping
    rows = []
    peaks = {}

    for pol_index, (pol_name, pol_data) in enumerate([('σ+', transitions['plus']),
                                                      ('σ-', transitions['minus']),
                                                      ('π', transitions['pi'])]):
        groups = group_transitions(
            pol_data['energies'],
            pol_data['forces'],
            pol_data['ind_lower'],
            pol_data['ind_upper']
        )

        # The intensity-weighted centroid of each peak, which is what the
        # grouping measures distance from and so what the row reports. A plain
        # mean would let a line of negligible strength pull the position.
        centroids = [float(np.sum(np.asarray(g['energies']) * np.asarray(g['forces']))
                           / np.sum(g['forces'])) for g in groups]
        peaks[pol_index] = (pol_data, groups, centroids)

        for k, group in enumerate(groups):
            total_intensity = float(np.sum(group['forces']))
            centroid = centroids[k]

            centroid_abs = c1_ghz + centroid
            centroid_wavelength = C_NM_GHZ / centroid_abs if centroid_abs else 0.0

            # Format transition names
            transition_names = []
            for i in range(len(group['ind_lower'])):
                transition_names.append(format_transition_name(
                    group['ind_lower'][i], group['ind_upper'][i], isotope))

            # Per-transition detail, in frequency order, for the level diagram
            # hover. 'offset' is each line's distance from the centroid, the
            # quantity the grouping tests, and 'share' its part of the peak.
            order = np.argsort(group['energies'])
            members = []
            for i in order:
                energy = float(group['energies'][i])
                force = float(group['forces'][i])
                abs_freq = c1_ghz + energy
                members.append({
                    'name': format_transition_name(
                        group['ind_lower'][i], group['ind_upper'][i], isotope),
                    'lower': int(group['ind_lower'][i]),
                    'upper': int(group['ind_upper'][i]),
                    'frequency': f"{energy:.3f}",
                    'wavelength': f"{C_NM_GHZ / abs_freq:.6f}" if abs_freq else "0",
                    'intensity': f"{force:.4f}",
                    'share': f"{100.0 * force / total_intensity:.1f}"
                             if total_intensity else "0.0",
                    # Rounded first, and +0.0 turns -0.0 into 0.0: a lone
                    # line sits ~1e-14 from its own centroid, which would
                    # otherwise print '-0.000' or '+0.000' depending on the
                    # numpy build's last-bit rounding.
                    'offset': f"{round(energy - centroid, 3) + 0.0:+.3f}",
                })

            energies = np.asarray(group['energies'], dtype=float)
            span_min, span_max = float(energies.min()), float(energies.max())

            rows.append({
                # Sort key only, dropped before the rows are returned.
                # Symmetry makes many groups exactly equal in intensity, and
                # LAPACK's last-bit noise differs between numpy builds, so the
                # key is rounded and carries explicit tiebreakers: without them
                # tied rows come out in a different order on different machines.
                '_sort': (-round(total_intensity, 12), pol_index,
                          round(centroid, 9)),
                # app.js maps this symbol to the series colour, so the palette
                # is defined in one place.
                'polarization': pol_name,
                # The peak's centroid, formatted here so the table and the
                # selection marker show identical values
                'frequency': f"{centroid:.3f}",
                'wavelength': f"{centroid_wavelength:.6f}",
                'transitions': ', '.join(transition_names),
                'intensity': f"{total_intensity:.4f}",
                'lower': [int(v) for v in group['ind_lower']],
                'upper': [int(v) for v in group['ind_upper']],
                # How wide the group actually is, which the centroid above
                # does not reveal
                'span': f"{span_max - span_min:.3f}",
                'span_min': span_min,
                'span_max': span_max,
                'members': members,
                '_peak': (pol_index, k),
            })

    # Strongest peaks first
    rows.sort(key=lambda r: r['_sort'])
    for row in rows:
        del row['_sort']
    _last_pumping = {
        'pump': pump, 'isotope': isotope, 'peaks': peaks,
        'rows': [row.pop('_peak') for row in rows],
    } if pump else None
    for row in rows:
        row.pop('_peak', None)
    return rows


def _format_rate(rel):
    """A relative rate as a short percentage, with sensible precision."""
    pct = 100.0 * rel
    if pct >= 10:
        return f"{pct:.0f}%"
    if pct >= 1:
        return f"{pct:.1f}%"
    if pct >= 0.01:
        return f"{pct:.2f}%"
    return "<0.01%" if pct > 0 else "0%"


def _pumping_context(pump, pol_data, centroids, isotope):
    """Rates for a laser on every peak of one polarisation, in one Voigt call.

    Each row's readout is then only bookkeeping, and the line names are
    formatted once per polarisation rather than once per row and level.
    """
    rates, per_line = pump['calc'].pumping_rates(
        pol_data['energies'], pol_data['forces'], pol_data['ind_lower'],
        np.asarray(centroids), pump['wG'], pump['wL'], PUMP_LASER_FWHM,
        pump['n_lower'])
    lower = np.asarray(pol_data['ind_lower'], dtype=int)
    upper = np.asarray(pol_data['ind_upper'], dtype=int)
    return {
        'rates': rates,
        'per_line': per_line,
        'energies': np.asarray(pol_data['energies'], dtype=float),
        'names': [format_transition_name(lo, up, isotope) for lo, up in zip(lower, upper)],
        'by_level': [np.where(lower == i)[0] for i in range(pump['n_lower'])],
        # The rate a full-strength line (S = 1) would give exactly on
        # resonance with the same laser: the readout's 100%. Normalising to the
        # levels the peak pumps instead made those read ~100% by construction.
        'reference': float(voigt_K(0.0, np.hypot(pump['wG'], PUMP_LASER_FWHM),
                                   pump['wL'])),
    }


def _pumping_readout(context, k, group, centroid):
    """How fast a laser on peak k empties each lower level.

    The laser is a Gaussian of PUMP_LASER_FWHM on the peak's centroid, driving
    every line of the peak's polarisation. Rates are per atom, as a fraction of
    what a full-strength line exactly on resonance would give, so pumped levels
    show how well the laser covers their lines and leaks read on the same scale.
    """
    rates = context['rates'][k]
    per_line = context['per_line'][k]
    reference = context['reference']

    strongest = max(group['forces'])
    targeted = sorted({int(lo) for lo, s in zip(group['ind_lower'], group['forces'])
                       if s >= PUMP_TARGET_FRACTION * strongest})

    levels = []
    for i, mine in enumerate(context['by_level']):
        rel = float(rates[i] / reference) if reference > 0 else 0.0
        via = []
        if rates[i] > 0:
            # The two lines doing most of it, if they do at least 1%. The
            # sort key is rounded, as elsewhere, so near-ties order the same
            # way on every numpy build.
            share = per_line[mine] / rates[i]
            for j in mine[np.lexsort((mine, -np.round(share, 9)))][:2]:
                if per_line[j] < 0.01 * rates[i]:
                    break
                via.append({
                    'name': context['names'][j],
                    'share': f"{100.0 * per_line[j] / rates[i]:.0f}",
                    'offset': f"{round(context['energies'][j] - centroid, 1) + 0.0:+.1f}",
                })
        levels.append({
            'rel': rel,
            'text': _format_rate(rel),
            'targeted': i in targeted,
            'has_lines': bool(mine.size),
            'via': via,
        })
    return {'laser_fwhm': f"{PUMP_LASER_FWHM:.1f}", 'levels': levels}



def pumping(row_index):
    """The pumping readout for one row of the table compute() last returned.

    None if that compute had no readout or the row is out of range.
    """
    last = _last_pumping
    if last is None or not 0 <= row_index < len(last['rows']):
        return None
    pol_index, k = last['rows'][row_index]
    pol_data, groups, centroids = last['peaks'][pol_index]
    context = _pumping_context(last['pump'], pol_data, [centroids[k]], last['isotope'])
    return _pumping_readout(context, 0, groups[k], centroids[k])


def pumping_js(row_index):
    """pumping() converted to plain JS objects, or None"""
    import js
    from pyodide.ffi import to_js
    readout = pumping(int(row_index))
    return None if readout is None else to_js(readout, dict_converter=js.Object.fromEntries)

def _spectra_series(spectra_data, isotope, x_axis_type):
    """x/y series for the spectra plot, per isotope and x-axis choice"""
    if isotope == 'He3':
        freq_range = spectra_data['freq_range'] - 40  # He3 offset
        abs_freq = spectra_data['abs_freq_he3']
        plus_data = spectra_data['he3_plus']
        minus_data = spectra_data['he3_minus']
        pi_data = spectra_data['he3_pi']
    else:  # He4
        freq_range = spectra_data['freq_range']
        abs_freq = spectra_data['abs_freq_he4']
        plus_data = spectra_data['he4_plus']
        minus_data = spectra_data['he4_minus']
        pi_data = spectra_data['he4_pi']

    if x_axis_type == 'Frequency Offset':
        x_data = freq_range
        x_label = 'Frequency Offset (GHz)'
    else:  # Wavelength
        # Avoid division by zero
        non_zero_freq = abs_freq != 0
        x_data = np.full_like(abs_freq, fill_value=np.nan, dtype=float)
        x_data[non_zero_freq] = (C_NM_GHZ / abs_freq[non_zero_freq])
        x_label = 'Wavelength (nm)'

    return {
        'x': x_data.tolist(),
        'x_label': x_label,
        'plus': plus_data.tolist(),
        'minus': minus_data.tolist(),
        'pi': pi_data.tolist(),
    }


def _level_diagram(energy_levels, isotope):
    """Level positions, ticks and ranges for the energy level diagram.

    The P states are drawn shifted up by P_OFFSET so both manifolds share one
    axis; the y tick labels are relabelled back to each manifold's own scale.
    Energies returned for the P states already include the offset.
    """
    # 1. Select data and labels based on isotope
    if isotope == 'He3':
        W_S, mF_S = energy_levels['W3S'], energy_levels['mf3S']
        W_P, mF_P = energy_levels['W3P'], energy_levels['mf3P']
        label_S, label_P = 'A', 'B'
        mF_values = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        mF_labels = ['-5/2', '-3/2', '-1/2', '1/2', '3/2', '5/2']
    else:  # He4
        W_S, mF_S = energy_levels['W4S'], energy_levels['mf4S']
        W_P, mF_P = energy_levels['W4P'], energy_levels['mf4P']
        label_S, label_P = 'Y', 'Z'
        mF_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        mF_labels = ['-2', '-1', '0', '1', '2']

    # 2. Dynamically calculate the offset and round it to the nearest 10
    if len(W_S) > 0 and len(W_P) > 0:
        total_span = (np.max(W_P) - np.min(W_P)) + (np.max(W_S) - np.min(W_S))
        VISUAL_GAP = total_span * 0.15
        P_OFFSET = np.max(W_S) - np.min(W_P) + VISUAL_GAP
        P_OFFSET = round(P_OFFSET / 10) * 10  # Round to nearest 10
    else:
        P_OFFSET = 50.0

    W_P_offset = W_P + P_OFFSET

    # 3. Axis range
    y_min_S, y_max_S = (np.min(W_S), np.max(W_S)) if len(W_S) > 0 else (0, 1)
    y_max_P_offset = np.max(W_P_offset) if len(W_P) > 0 else P_OFFSET + 1
    y_range_buffer = (y_max_P_offset - y_min_S) * 0.1
    y_range = [y_min_S - y_range_buffer, y_max_P_offset + y_range_buffer]

    # 4. Custom tick generation: each manifold labelled on its own scale
    tickvals, ticktext = [], []
    if len(W_S) > 0:
        s_tickvals = np.linspace(y_min_S, y_max_S, 5)
        tickvals.extend(s_tickvals.tolist())
        ticktext.extend([f'{int(round(v / 10) * 10)}' for v in s_tickvals])
    if len(W_P) > 0:
        p_original_tickvals = np.linspace(np.min(W_P), np.max(W_P), 5)
        tickvals.extend((p_original_tickvals + P_OFFSET).tolist())
        ticktext.extend([f'{int(round(v / 10) * 10)}' for v in p_original_tickvals])

    line_width = 0.4

    return {
        'S': [{'mf': float(mF_S[i]),
               'e': float(W_S[i]),
               'label': f" {label_S}{to_subscript(str(i + 1))}"}
              for i in range(len(W_S))],
        'P': [{'mf': float(mF_P[i]),
               'e': float(W_P_offset[i]),
               'label': f" {label_P}{to_subscript(str(i + 1))}"}
              for i in range(len(W_P))],
        # The P energies above include this; the hover subtracts it so each
        # level is reported on its own manifold's scale.
        'p_offset': float(P_OFFSET),
        'mF_values': [float(v) for v in mF_values],
        'mF_labels': mF_labels,
        'tickvals': tickvals,
        'ticktext': ticktext,
        'y_range': [float(y_range[0]), float(y_range[1])],
        'line_width': line_width,
        'label_x': float(mF_values[0] - line_width),
        'label_S_y': float(np.mean(W_S)) if len(W_S) > 0 else 0.0,
        'label_P_y': float(np.mean(W_P_offset)) if len(W_P) > 0 else 0.0,
    }


def compute(B, Temp, isotope, x_axis_type, pressure=0.0):
    """Everything the page needs for one parameter combination"""
    calculator = _get_calculator()

    # Collisional width for the isotope on display. The Fortran takes wL0 and
    # wL12 separately; the rate it quotes is the same for both, so they are
    # set equal here and the J=0 weighting inside calculate_full_results has
    # no effect until they differ. calculate_full_results broadens both
    # isotopes with these, and only the selected one is read back below.
    rate = calculator.collision_per_mbar[isotope]
    wL = rate * float(pressure)
    full_results = calculator.calculate_full_results(B, Temp, wL0=wL, wL12=wL)

    transitions = (full_results['transitions']['he3'] if isotope == 'He3'
                   else full_results['transitions']['he4'])

    # The grouping threshold is a fixed 2 GHz; the line width is the physical
    # scale that decides whether lines actually blend into one peak, so the
    # page shows it for comparison. Widths are quoted as FWHM, which is what
    # the threshold is comparable to; calculate_full_results returns the
    # Doppler width as a 1/e half-width.
    # Doppler FWHM as the Fortran computes it, from the molar mass; this is
    # the same width the line shapes are built with.
    doppler_fwhm = calculator.doppler_fwhm(Temp, isotope)
    lorentz_fwhm = wL
    # Olivero & Longbothum's approximation, good to ~0.02%
    voigt_fwhm = (0.5346 * lorentz_fwhm
                  + np.sqrt(0.2166 * lorentz_fwhm ** 2 + doppler_fwhm ** 2))

    title = f'{isotope} Spectra at B = {B:.4f} T, T = {Temp:.0f} K'
    if wL > 0:
        title += f', P = {float(pressure):.0f} mbar'

    return {
        'title': title,
        'spectra': _spectra_series(full_results['spectra_data'], isotope, x_axis_type),
        'table': build_transitions_table(
            transitions, isotope, calculator.c1_ghz,
            pump={'calc': calculator, 'wG': doppler_fwhm, 'wL': wL,
                  'n_lower': len(full_results['energy_levels'][
                      'W3S' if isotope == 'He3' else 'W4S'])}),
        'levels': _level_diagram(full_results['energy_levels'], isotope),
        'doppler': f"{doppler_fwhm:.3f}",
        'lorentz': f"{lorentz_fwhm:.3f}",
        'voigt': f"{voigt_fwhm:.3f}",
        'group_threshold': f"{GROUP_THRESHOLD:.1f}",
    }


def compute_js(B, Temp, isotope, x_axis_type, pressure=0.0):
    """compute() converted to plain JS objects/arrays (no PyProxy to free)"""
    import js
    from pyodide.ffi import to_js
    return to_js(compute(B, Temp, isotope, x_axis_type, pressure),
                 dict_converter=js.Object.fromEntries)

