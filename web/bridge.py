#!/usr/bin/env python3
"""
Browser bridge for the static build of the helium spectra calculator.

Runs inside Pyodide. Imports helium_spectra_calc unchanged and returns plain
JSON-able structures; Plotly.js does all the drawing on the JavaScript side.
The presentation logic below is a direct port of helium_spectra_ui.py with
Streamlit, pandas and plotly stripped out, so the numbers are unchanged.
"""

import numpy as np
from helium_spectra_calc import HeliumSpectraCalculator

# c expressed so that (nm) = C_NM_GHZ / (GHz), matching helium_spectra_ui.py
C_NM_GHZ = 299792458.0

POL_COLORS = {'σ+': 'blue', 'σ-': 'red', 'π': 'green'}

_calculator = None


def _get_calculator():
    """Cached calculator instance, the Pyodide equivalent of @st.cache_data"""
    global _calculator
    if _calculator is None:
        _calculator = HeliumSpectraCalculator()
    return _calculator


def group_transitions(energies, forces, ind_lower, ind_upper, threshold=2.0):
    """Group transitions that are within threshold GHz of each other"""
    if len(energies) == 0:
        return []

    # Sort by energy
    sorted_indices = np.argsort(energies)
    sorted_energies = energies[sorted_indices]
    sorted_forces = forces[sorted_indices]
    sorted_ind_lower = ind_lower[sorted_indices]
    sorted_ind_upper = ind_upper[sorted_indices]

    groups = []
    current_group = {
        'energies': [sorted_energies[0]],
        'forces': [sorted_forces[0]],
        'ind_lower': [sorted_ind_lower[0]],
        'ind_upper': [sorted_ind_upper[0]]
    }

    for i in range(1, len(sorted_energies)):
        if sorted_energies[i] - current_group['energies'][-1] <= threshold:
            # Add to current group
            current_group['energies'].append(sorted_energies[i])
            current_group['forces'].append(sorted_forces[i])
            current_group['ind_lower'].append(sorted_ind_lower[i])
            current_group['ind_upper'].append(sorted_ind_upper[i])
        else:
            # Start new group
            groups.append(current_group)
            current_group = {
                'energies': [sorted_energies[i]],
                'forces': [sorted_forces[i]],
                'ind_lower': [sorted_ind_lower[i]],
                'ind_upper': [sorted_ind_upper[i]]
            }

    # Don't forget the last group
    groups.append(current_group)

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


def build_transitions_table(transitions, isotope, c1_ghz):
    """Grouped transitions as a list of row dicts, sorted by intensity"""
    rows = []

    for pol_index, (pol_name, pol_data) in enumerate([('σ+', transitions['plus']),
                                                      ('σ-', transitions['minus']),
                                                      ('π', transitions['pi'])]):
        groups = group_transitions(
            pol_data['energies'],
            pol_data['forces'],
            pol_data['ind_lower'],
            pol_data['ind_upper']
        )

        for group in groups:
            # Calculate average values
            avg_energy = float(np.mean(group['energies']))
            total_intensity = float(np.sum(group['forces']))

            # Calculate average absolute frequency from the average relative frequency
            avg_abs_freq = c1_ghz + avg_energy

            # Calculate average wavelength from the average absolute frequency
            if avg_abs_freq != 0:
                avg_wavelength = C_NM_GHZ / avg_abs_freq
            else:
                avg_wavelength = 0.0

            # Format transition names
            transition_names = []
            for i in range(len(group['ind_lower'])):
                transition_names.append(format_transition_name(
                    group['ind_lower'][i], group['ind_upper'][i], isotope))

            rows.append({
                # Sort key only, dropped before the rows are returned.
                # Symmetry makes many groups exactly equal in intensity, and
                # LAPACK's last-bit noise differs between numpy builds, so the
                # key is rounded and carries explicit tiebreakers: without them
                # tied rows come out in a different order on different machines.
                '_sort': (-round(total_intensity, 12), pol_index,
                          round(avg_energy, 9)),
                'polarization': pol_name,
                'color': POL_COLORS.get(pol_name, 'grey'),
                # Formatted here so the table, the selection marker and the
                # original Streamlit app all show identical values.
                'frequency': f"{avg_energy:.3f}",
                'wavelength': f"{avg_wavelength:.6f}",
                'transitions': ', '.join(transition_names),
                'intensity': f"{total_intensity:.4f}",
                'lower': [int(v) for v in group['ind_lower']],
                'upper': [int(v) for v in group['ind_upper']],
            })

    # Sort by intensity (descending), as the Streamlit DataFrame did
    rows.sort(key=lambda r: r['_sort'])
    for row in rows:
        del row['_sort']
    return rows


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


def compute(B, Temp, isotope, x_axis_type):
    """Everything the page needs for one (B, T, isotope, x-axis) combination"""
    calculator = _get_calculator()
    full_results = calculator.calculate_full_results(B, Temp)

    transitions = (full_results['transitions']['he3'] if isotope == 'He3'
                   else full_results['transitions']['he4'])

    return {
        'title': f'{isotope} Spectra at B = {B:.4f} T, T = {Temp:.0f} K',
        'spectra': _spectra_series(full_results['spectra_data'], isotope, x_axis_type),
        'table': build_transitions_table(transitions, isotope, calculator.c1_ghz),
        'levels': _level_diagram(full_results['energy_levels'], isotope),
    }


def compute_js(B, Temp, isotope, x_axis_type):
    """compute() converted to plain JS objects/arrays (no PyProxy to free)"""
    import js
    from pyodide.ffi import to_js
    return to_js(compute(B, Temp, isotope, x_axis_type),
                 dict_converter=js.Object.fromEntries)
