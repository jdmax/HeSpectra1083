#!/usr/bin/env python3
"""
Helium Spectrum Calculator - Streamlit UI
Interactive interface for helium spectra calculations

Need to install: "pip install streamlit plotly numpy pandas"
To run: "streamlit run helium_spectra_ui.py"
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from helium_spectra_calc import HeliumSpectraCalculator


@st.cache_data
def get_calculator():
    """Cached calculator instance for better performance"""
    return HeliumSpectraCalculator()


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
    lower_sub = to_subscript(str(ind_lower + 1))
    upper_sub = to_subscript(str(ind_upper + 1))

    if isotope == 'He3':
        return f"A{lower_sub} → B{upper_sub}"
    else:  # He4
        return f"Y{lower_sub} → Z{upper_sub}"


def create_transitions_table(transitions, isotope, energy_offset, c1_ghz):
    """Create a DataFrame with grouped transitions"""
    all_groups_data = []

    for pol_name, pol_data in [('σ+', transitions['plus']),
                               ('σ-', transitions['minus']),
                               ('π', transitions['pi'])]:
        groups = group_transitions(
            pol_data['energies'],
            pol_data['forces'],
            pol_data['ind_lower'],
            pol_data['ind_upper']
        )

        for group in groups:
            # Calculate average values
            avg_energy = np.mean(group['energies'])
            total_intensity = np.sum(group['forces'])

            # Calculate absolute frequency and wavelength
            if isotope == 'He3':
                abs_freq = c1_ghz + avg_energy - 40
            else:  # He4
                abs_freq = c1_ghz + avg_energy

            # Avoid division by zero for frequency
            avg_wavelength = (299792458.0 / (abs_freq * 1e9)) * 1e9 if abs_freq != 0 else 0

            # Format transition names
            transition_names = []
            for i in range(len(group['ind_lower'])):
                name = format_transition_name(
                    group['ind_lower'][i],
                    group['ind_upper'][i],
                    isotope
                )
                transition_names.append(name)

            all_groups_data.append({
                'Polarization': pol_name,
                'Average Relative Frequency (GHz)': f"{avg_energy:.3f}",
                'Average Wavelength (nm)': f"{avg_wavelength:.6f}",
                'Transitions in Group': ', '.join(transition_names),
                'Group Intensity': f"{total_intensity:.4f}",
                '_intensity_value': total_intensity,  # Keep numeric value for sorting
                'ind_lower_list': group['ind_lower'],  # Keep raw indices for plotting
                'ind_upper_list': group['ind_upper']  # Keep raw indices for plotting
            })

    # Create DataFrame and sort by intensity (descending)
    if not all_groups_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_groups_data)
    df = df.sort_values('_intensity_value', ascending=False)
    df = df.drop('_intensity_value', axis=1)  # Remove the helper column

    return df


def create_plotly_figure(spectra_data, isotope, x_axis_type, B_field, temperature, selected_frequency=None,
                         selected_wavelength=None):
    """Create a plotly figure for the spectra"""

    # Choose data based on isotope
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

    # Choose x-axis data
    if x_axis_type == 'Frequency Offset':
        x_data = freq_range
        x_label = 'Frequency Offset (GHz)'
    else:  # Wavelength
        # Avoid division by zero
        non_zero_freq = abs_freq != 0
        x_data = np.full_like(abs_freq, fill_value=np.nan, dtype=float)
        x_data[non_zero_freq] = (299792458.0 / (abs_freq[non_zero_freq] * 1e9)) * 1e9
        x_label = 'Wavelength (nm)'

    # Create plotly figure
    fig = go.Figure()

    # Add traces for each polarization
    fig.add_trace(go.Scatter(
        x=x_data, y=plus_data,
        mode='lines',
        name='σ+',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x_data, y=minus_data,
        mode='lines',
        name='σ-',
        line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x_data, y=pi_data,
        mode='lines',
        name='π',
        line=dict(color='green', width=2)
    ))

    # Add vertical line for selected transition
    if selected_frequency is not None and selected_wavelength is not None:
        if x_axis_type == 'Frequency Offset':
            # Use the selected frequency
            x_position = selected_frequency
        else:  # Wavelength
            # Use the selected wavelength
            x_position = selected_wavelength

        # Add vertical line
        fig.add_vline(
            x=x_position,
            line_width=2,
            line_dash="dash",
            line_color="orange",
            annotation_text="Selected",
            annotation_position="top right"
        )

    # Update layout
    fig.update_layout(
        title=f'{isotope} Spectra at B = {B_field:.2f} T, T = {temperature:.0f} K',
        xaxis_title=x_label,
        yaxis_title='Intensity',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


# /// START OF MODIFIED CODE ///
def create_energy_level_diagram(energy_levels, isotope, selected_transition_group=None, pol_color='blue'):
    """Create a Plotly figure for the energy level diagram against mF, split into two subplots."""
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

    # 2. Create subplot figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # 3. Add P-states to top subplot (row=1)
    line_width = 0.4
    for i in range(len(W_P)):
        mf, energy = mF_P[i], W_P[i]
        fig.add_trace(go.Scatter(x=[mf - line_width / 2, mf + line_width / 2], y=[energy, energy],
                                 mode='lines', line_color='black', showlegend=False), row=1, col=1)
        fig.add_annotation(x=mf + line_width / 2, y=energy, text=f" {label_P}{to_subscript(str(i + 1))}",
                           showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10), row=1, col=1)

    # 4. Add S-states to bottom subplot (row=2)
    for i in range(len(W_S)):
        mf, energy = mF_S[i], W_S[i]
        fig.add_trace(go.Scatter(x=[mf - line_width / 2, mf + line_width / 2], y=[energy, energy],
                                 mode='lines', line_color='black', showlegend=False), row=2, col=1)
        fig.add_annotation(x=mf + line_width / 2, y=energy, text=f" {label_S}{to_subscript(str(i + 1))}",
                           showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10), row=2, col=1)

    # 5. Plot selected transitions using annotations as arrows between subplots
    if selected_transition_group:
        for i in range(len(selected_transition_group['lower'])):
            idx_lower, idx_upper = selected_transition_group['lower'][i], selected_transition_group['upper'][i]
            x_S, y_S = mF_S[idx_lower], W_S[idx_lower]
            x_P, y_P = mF_P[idx_upper], W_P[idx_upper]

            fig.add_annotation(
                x=x_S, y=y_S, ax=x_P, ay=y_P,
                xref='x1', yref='y2', axref='x1', ayref='y1',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=pol_color
            )

    # 6. Configure layout and axes
    y_min_P, y_max_P = (np.min(W_P), np.max(W_P)) if len(W_P) > 0 else (0, 1)
    y_buffer_P = (y_max_P - y_min_P) * 0.15 if (y_max_P - y_min_P) > 0 else 1
    y_range_P = [y_min_P - y_buffer_P, y_max_P + y_buffer_P]

    y_min_S, y_max_S = (np.min(W_S), np.max(W_S)) if len(W_S) > 0 else (0, 1)
    y_buffer_S = (y_max_S - y_min_S) * 0.15 if (y_max_S - y_min_S) > 0 else 1
    y_range_S = [y_min_S - y_buffer_S, y_max_S + y_buffer_S]

    fig.update_layout(
        title="Energy Level Diagram", height=700, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(title_text="Magnetic Quantum Number m_F", tickmode='array', tickvals=mF_values, ticktext=mF_labels,
                     row=2, col=1)
    fig.update_yaxes(title_text="2³P Energy (GHz)", range=y_range_P, showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="2³S Energy (GHz)", range=y_range_S, showgrid=True, row=2, col=1)

    return fig


# /// END OF MODIFIED CODE ///


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Helium Spectra Calculator", page_icon="🔬",
        layout="wide", initial_sidebar_state="expanded"
    )

    if 'b_field_value' not in st.session_state: st.session_state.b_field_value = 1.0
    if 'temp_value' not in st.session_state: st.session_state.temp_value = 300
    if 'isotope' not in st.session_state: st.session_state.isotope = 'He3'
    if 'x_axis_type' not in st.session_state: st.session_state.x_axis_type = 'Frequency Offset'
    if 'show_transitions' not in st.session_state: st.session_state.show_transitions = True

    st.title("Helium 1083 nm Line Calculator")
    st.markdown("""
    Calculate and visualize helium spectra near 1083 nm with Zeeman splitting for ³He and ⁴He isotopes.
    Adjust magnetic field and temperature to see real-time changes in the spectra. Accurate only at gas pressure up to a few mbar, above which additional corrections are needed.
    """)

    # Sidebar controls
    st.sidebar.header("Parameters")
    st.sidebar.subheader("Magnetic Field")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        B_field_slider = st.slider(
            "B Field (T)", 0.01, 7.0, st.session_state.b_field_value, 0.01, key="b_slider")
    with col2:
        B_field_input = st.number_input(
            "Exact B (T)", 0.01, 10.0, st.session_state.b_field_value, 0.01, "%.3f", key="b_input")

    if B_field_slider != st.session_state.b_field_value:
        st.session_state.b_field_value = B_field_slider;
        st.rerun()
    elif B_field_input != st.session_state.b_field_value:
        st.session_state.b_field_value = B_field_input;
        st.rerun()
    B_field = st.session_state.b_field_value

    st.sidebar.subheader("Temperature")
    col3, col4 = st.sidebar.columns(2)
    with col3:
        temp_slider = st.slider(
            "Temperature (K)", 77, 1000, st.session_state.temp_value, 1, key="temp_slider")
    with col4:
        temp_input = st.number_input(
            "Exact T (K)", 77, 1000, st.session_state.temp_value, 1, "%d", key="temp_input")

    if temp_slider != st.session_state.temp_value:
        st.session_state.temp_value = temp_slider;
        st.rerun()
    elif temp_input != st.session_state.temp_value:
        st.session_state.temp_value = temp_input;
        st.rerun()
    temperature = st.session_state.temp_value

    st.sidebar.subheader("Display Options")
    isotope = st.sidebar.radio(
        "Isotope", ["He3", "He4"], index=0 if st.session_state.isotope == 'He3' else 1, key="isotope_radio")

    if 'isotope_memory' not in st.session_state: st.session_state.isotope_memory = isotope
    if isotope != st.session_state.isotope_memory:
        st.session_state.isotope_memory = isotope
        if 'transitions_df_select' in st.session_state: del st.session_state['transitions_df_select']
        st.rerun()
    st.session_state.isotope = isotope

    x_axis_type = st.sidebar.radio(
        "X-axis", ["Frequency Offset", "Wavelength"],
        index=0 if st.session_state.x_axis_type == 'Frequency Offset' else 1, key="x_axis_radio")
    if x_axis_type != st.session_state.x_axis_type: st.session_state.x_axis_type = x_axis_type

    show_transitions = st.sidebar.checkbox(
        "Show Transitions Table & Diagram", value=st.session_state.show_transitions, key="show_transitions_checkbox")
    if show_transitions != st.session_state.show_transitions: st.session_state.show_transitions = show_transitions

    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Information"):
        st.markdown("""...""")  # Omitted for brevity

    col_main, col_info = st.columns([3, 1])
    with col_main:
        st.markdown(f"""
        **Current Parameters:** B = {B_field:.3f} T, T = {temperature:.0f} K, Isotope = {st.session_state.isotope}
        """)
        with st.spinner('Calculating spectra...'):
            calculator = get_calculator()
            full_results = calculator.calculate_full_results(B_field, temperature)
            spectra_data = full_results['spectra_data']

            selected_frequency, selected_wavelength = None, None
            selected_transition_group, pol_color = None, 'grey'

            if st.session_state.isotope == 'He3':
                transitions = full_results['transitions']['he3']
            else:
                transitions = full_results['transitions']['he4']

            df_transitions = create_transitions_table(
                transitions, st.session_state.isotope,
                full_results['energy_offsets']['eC1'] if st.session_state.isotope == 'He3' else
                full_results['energy_offsets']['he4_offset'], calculator.c1_ghz)

            if st.session_state.show_transitions and 'transitions_df_select' in st.session_state:
                if st.session_state.transitions_df_select['selection']['rows']:
                    selected_row_index = st.session_state.transitions_df_select['selection']['rows'][0]
                    if selected_row_index < len(df_transitions):
                        selected_row = df_transitions.iloc[selected_row_index]
                        selected_frequency = float(selected_row['Average Relative Frequency (GHz)'])
                        selected_wavelength = float(selected_row['Average Wavelength (nm)'])
                        selected_transition_group = {
                            'lower': selected_row['ind_lower_list'], 'upper': selected_row['ind_upper_list']
                        }
                        pol_color = {'σ+': 'blue', 'σ-': 'red', 'π': 'green'}.get(selected_row['Polarization'], 'grey')

            fig = create_plotly_figure(
                spectra_data, st.session_state.isotope, st.session_state.x_axis_type,
                B_field, temperature, selected_frequency, selected_wavelength)
            st.plotly_chart(fig, use_container_width=True)

            if st.session_state.show_transitions:
                st.markdown("---")
                col_table, col_diagram = st.columns(2)
                with col_table:
                    st.subheader("Transitions Table")
                    st.markdown("*Select a row to highlight the transition group.*")
                    st.dataframe(
                        df_transitions, key="transitions_df_select", on_select="rerun",
                        selection_mode="single-row", use_container_width=True, hide_index=True,
                        column_config={
                            "Polarization": st.column_config.TextColumn("Polarization", width="small"),
                            "Average Relative Frequency (GHz)": st.column_config.TextColumn("Avg Rel Freq (GHz)"),
                            "Average Wavelength (nm)": st.column_config.TextColumn("Avg λ (nm)"),
                            "Transitions in Group": st.column_config.TextColumn("Transitions"),
                            "Group Intensity": st.column_config.TextColumn("Intensity"),
                            "ind_lower_list": None, "ind_upper_list": None})
                with col_diagram:
                    st.subheader("Energy Level Diagram")
                    st.markdown("*Shows sublevels vs. m_F and selected transitions.*")
                    fig_levels = create_energy_level_diagram(
                        full_results['energy_levels'], st.session_state.isotope,
                        selected_transition_group, pol_color)
                    st.plotly_chart(fig_levels, use_container_width=True)

    with col_info:
        st.subheader("Quick Settings")
        if st.button("🧊 Liquid Nitrogen (77K)"): st.session_state.temp_value = 77; st.rerun()
        if st.button("🏠 Room Temperature (300K)"): st.session_state.temp_value = 300; st.rerun()
        if st.button("🔥 High Temperature (800K)"): st.session_state.temp_value = 800; st.rerun()
        st.markdown("---")
        if st.button("⚡ Low Field (0.01T)"): st.session_state.b_field_value = 0.01; st.rerun()
        if st.button("🧲 High Field (1T)"): st.session_state.b_field_value = 1.0; st.rerun()
        if st.button("🚀 Higher Field (5T)"): st.session_state.b_field_value = 5.0; st.rerun()
        st.markdown("---")

        st.subheader("Export Data")
        if st.session_state.isotope == 'He3':
            freq_data, abs_freq = spectra_data['freq_range'] - 40, spectra_data['abs_freq_he3']
            plus_data, minus_data, pi_data = spectra_data['he3_plus'], spectra_data['he3_minus'], spectra_data['he3_pi']
        else:
            freq_data, abs_freq = spectra_data['freq_range'], spectra_data['abs_freq_he4']
            plus_data, minus_data, pi_data = spectra_data['he4_plus'], spectra_data['he4_minus'], spectra_data['he4_pi']
        df_download = pd.DataFrame({
            'frequency_offset_ghz': freq_data, 'absolute_freq_ghz': abs_freq,
            'wavelength_nm': (299792458.0 / (abs_freq * 1e9)) * 1e9 if abs_freq.any() != 0 else 0,
            'sigma_plus': plus_data, 'sigma_minus': minus_data, 'pi': pi_data
        })
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Spectra CSV", data=csv,
            file_name=f"{st.session_state.isotope}_B{B_field:.2f}T_T{temperature}K.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("""...""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()