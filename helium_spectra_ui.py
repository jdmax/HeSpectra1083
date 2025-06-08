#!/usr/bin/env python3
"""
Helium Spectrum Calculator - Streamlit UI
Interactive interface for helium spectra calculations

Need to install: "pip install streamlit plotly numpy pandas"
To run: "streamlit run helium_spectra_ui.py"
"""

import streamlit as st
import plotly.graph_objects as go
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


def format_transition_name(ind_lower, ind_upper, isotope):
    """Format transition name based on isotope and indices"""
    if isotope == 'He3':
        return f"A_{ind_lower} → B_{ind_upper}"
    else:  # He4
        return f"Y_{ind_lower} → Z_{ind_upper}"


def create_transitions_table(transitions, isotope, energy_offset, c1_ghz):
    """Create a DataFrame with grouped transitions"""
    all_groups = []

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
                abs_freq = c1_ghz + avg_energy
            else:  # He4
                abs_freq = c1_ghz + avg_energy

            avg_wavelength = 299792458.0 / abs_freq  # nm

            # Format transition names
            transition_names = []
            for i in range(len(group['ind_lower'])):
                name = format_transition_name(
                    group['ind_lower'][i],
                    group['ind_upper'][i],
                    isotope
                )
                transition_names.append(name)

            all_groups.append({
                'Polarization': pol_name,
                'Average Relative Frequency (GHz)': f"{avg_energy:.3f}",
                'Average Wavelength (nm)': f"{avg_wavelength:.6f}",
                'Transitions in Group': ', '.join(transition_names),
                'Group Intensity': f"{total_intensity:.4f}",
                '_intensity_value': total_intensity  # Keep numeric value for sorting
            })

    # Create DataFrame and sort by intensity (descending)
    df = pd.DataFrame(all_groups)
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
        x_data = 299792458.0 / abs_freq  # Convert to wavelength in nm
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


def main():
    """Main Streamlit application"""

    # Set page configuration
    st.set_page_config(
        page_title="Helium Spectra Calculator",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state variables
    if 'b_field_value' not in st.session_state:
        st.session_state.b_field_value = 1.0
    if 'temp_value' not in st.session_state:
        st.session_state.temp_value = 300
    if 'isotope' not in st.session_state:
        st.session_state.isotope = 'He3'
    if 'x_axis_type' not in st.session_state:
        st.session_state.x_axis_type = 'Frequency Offset'
    if 'show_transitions' not in st.session_state:
        st.session_state.show_transitions = False

    # Title and description
    st.title("Helium 1083 nm Line Calculator")
    st.markdown("""
    Calculate and visualize helium spectra near 1083 nm with Zeeman splitting for ³He and ⁴He isotopes.
    Adjust magnetic field and temperature to see real-time changes in the spectra.
    """)

    # Sidebar controls
    st.sidebar.header("Parameters")

    # Magnetic field controls with synchronization
    st.sidebar.subheader("Magnetic Field")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        B_field_slider = st.slider(
            "B Field (T)",
            min_value=0.01,
            max_value=7.0,
            value=st.session_state.b_field_value,
            step=0.01,
            key="b_slider"
        )

    with col2:
        B_field_input = st.number_input(
            "Exact B (T)",
            min_value=0.01,
            max_value=10.0,
            value=st.session_state.b_field_value,
            step=0.01,
            format="%.3f",
            key="b_input"
        )

    # Synchronization logic for B field
    if B_field_slider != st.session_state.b_field_value:
        st.session_state.b_field_value = B_field_slider
        st.rerun()
    elif B_field_input != st.session_state.b_field_value:
        st.session_state.b_field_value = B_field_input
        st.rerun()

    B_field = st.session_state.b_field_value

    # Temperature controls with synchronization
    st.sidebar.subheader("Temperature")
    col3, col4 = st.sidebar.columns(2)

    with col3:
        temp_slider = st.slider(
            "Temperature (K)",
            min_value=77,
            max_value=1000,
            value=st.session_state.temp_value,
            step=1,
            key="temp_slider"
        )

    with col4:
        temp_input = st.number_input(
            "Exact T (K)",
            min_value=77,
            max_value=1000,
            value=st.session_state.temp_value,
            step=1,
            format="%d",
            key="temp_input"
        )

    # Synchronization logic for temperature
    if temp_slider != st.session_state.temp_value:
        st.session_state.temp_value = temp_slider
        st.rerun()
    elif temp_input != st.session_state.temp_value:
        st.session_state.temp_value = temp_input
        st.rerun()

    temperature = st.session_state.temp_value

    # Other controls
    st.sidebar.subheader("Display Options")
    isotope = st.sidebar.radio(
        "Isotope",
        ["He3", "He4"],
        index=0 if st.session_state.isotope == 'He3' else 1,
        key="isotope_radio"
    )

    # Update session state if isotope changed
    if isotope != st.session_state.isotope:
        st.session_state.isotope = isotope

    x_axis_type = st.sidebar.radio(
        "X-axis",
        ["Frequency Offset", "Wavelength"],
        index=0 if st.session_state.x_axis_type == 'Frequency Offset' else 1,
        key="x_axis_radio"
    )

    # Update session state if x_axis_type changed
    if x_axis_type != st.session_state.x_axis_type:
        st.session_state.x_axis_type = x_axis_type

    # Add checkbox for showing transitions table
    show_transitions = st.sidebar.checkbox(
        "Show Transitions Table",
        value=st.session_state.show_transitions,
        key="show_transitions_checkbox"
    )

    if show_transitions != st.session_state.show_transitions:
        st.session_state.show_transitions = show_transitions

    # Add some spacing
    st.sidebar.markdown("---")

    # Information panel
    with st.sidebar.expander("ℹ️ Information"):
        st.markdown("""
        **Parameters:**
        - **B Field**: 0.01 - 10.0 Tesla
        - **Temperature**: 77 - 1000 Kelvin
        - **Doppler Width**: ∝ √(T/300)

        **Polarizations:**
        - **σ+**: Right circular polarization
        - **σ-**: Left circular polarization  
        - **π**: Linear polarization

        **Physics:**
        - Zeeman splitting in magnetic field
        - Hyperfine structure included
        - Doppler broadening from temperature

        **Transitions Table:**
        - Groups transitions within 2 GHz
        - Shows average frequencies and wavelengths
        - Lists all transitions in each group
        """)

    # Main content area
    col_main, col_info = st.columns([3, 1])

    with col_main:
        # Show current parameters
        st.markdown(f"""
        **Current Parameters:** B = {B_field:.3f} T, T = {temperature:.0f} K, Isotope = {st.session_state.isotope}
        """)

        # Calculate spectra
        with st.spinner('Calculating spectra...'):
            calculator = get_calculator()
            full_results = calculator.calculate_full_results(B_field, temperature)
            spectra_data = full_results['spectra_data']

            # Initialize variables for selected transition
            selected_frequency = None
            selected_wavelength = None

            # Show transitions table if requested
            if st.session_state.show_transitions:
                st.subheader("Transitions Table")
                st.markdown("*Select the first column of a row to highlight the transition in the spectrum*")

                # Get the appropriate transitions based on isotope
                if st.session_state.isotope == 'He3':
                    transitions = full_results['transitions']['he3']
                else:
                    transitions = full_results['transitions']['he4']

                # Create transitions table
                df_transitions = create_transitions_table(
                    transitions,
                    st.session_state.isotope,
                    full_results['energy_offsets']['eC1'] if st.session_state.isotope == 'He3' else
                    full_results['energy_offsets']['he4_offset'],
                    calculator.c1_ghz
                )

                # Display the table with selection enabled
                selected = st.dataframe(
                    df_transitions,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    column_config={
                        "Polarization": st.column_config.TextColumn(
                            "Polarization",
                            width="small"
                        ),
                        "Average Relative Frequency (GHz)": st.column_config.TextColumn(
                            "Avg Rel Freq (GHz)",
                            width="medium"
                        ),
                        "Average Wavelength (nm)": st.column_config.TextColumn(
                            "Avg λ (nm)",
                            width="medium"
                        ),
                        "Transitions in Group": st.column_config.TextColumn(
                            "Transitions",
                            width="large"
                        ),
                        "Group Intensity": st.column_config.TextColumn(
                            "Group Intensity",
                            width="medium"
                        )
                    }
                )

                # Check if a row is selected
                if selected and selected.selection.rows:
                    selected_row_idx = selected.selection.rows[0]
                    selected_row = df_transitions.iloc[selected_row_idx]

                    # Parse the frequency and wavelength from the selected row
                    selected_frequency = float(selected_row['Average Relative Frequency (GHz)'])
                    selected_wavelength = float(selected_row['Average Wavelength (nm)'])
                #
                # # Download button for transitions table
                # csv_transitions = df_transitions.to_csv(index=False)
                # st.download_button(
                #     label="📊 Download Transitions Table",
                #     data=csv_transitions,
                #     file_name=f"{st.session_state.isotope}_transitions_B{B_field:.2f}T.csv",
                #     mime="text/csv"
                # )

            # Create and display plot with selected transition highlighted
            fig = create_plotly_figure(
                spectra_data,
                st.session_state.isotope,
                st.session_state.x_axis_type,
                B_field,
                temperature,
                selected_frequency,
                selected_wavelength
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("Quick Settings")

        # Preset buttons
        if st.button("🧊 Liquid Nitrogen (77K)"):
            st.session_state.temp_value = 77
            st.rerun()

        if st.button("🏠 Room Temperature (300K)"):
            st.session_state.temp_value = 300
            st.rerun()

        if st.button("🔥 High Temperature (800K)"):
            st.session_state.temp_value = 800
            st.rerun()

        st.markdown("---")

        if st.button("⚡ Low Field (0.01T)"):
            st.session_state.b_field_value = 0.01
            st.rerun()

        if st.button("🧲 High Field (1T)"):
            st.session_state.b_field_value = 1.0
            st.rerun()

        if st.button("🚀 Higher Field (5T)"):
            st.session_state.b_field_value = 5.0
            st.rerun()

        st.markdown("---")

        # Download options
        st.subheader("Export Data")

        if st.button("📊 Download Spectra CSV"):
            # Generate CSV data
            if st.session_state.isotope == 'He3':
                freq_data = spectra_data['freq_range'] - 40
                abs_freq = spectra_data['abs_freq_he3']
                plus_data = spectra_data['he3_plus']
                minus_data = spectra_data['he3_minus']
                pi_data = spectra_data['he3_pi']
            else:
                freq_data = spectra_data['freq_range']
                abs_freq = spectra_data['abs_freq_he4']
                plus_data = spectra_data['he4_plus']
                minus_data = spectra_data['he4_minus']
                pi_data = spectra_data['he4_pi']

            df = pd.DataFrame({
                'frequency_offset_ghz': freq_data,
                'absolute_freq_ghz': abs_freq,
                'wavelength_nm': 299792458.0 / abs_freq,
                'sigma_plus': plus_data,
                'sigma_minus': minus_data,
                'pi': pi_data
            })

            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{st.session_state.isotope}_B{B_field:.2f}T_T{temperature}K.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Based on original Fortran code by P.J. Nacher, <a href="https://www.lkb.fr/polarisedhelium/">Laboratoire Kastler Brossel</a> <a href="https://doi.org/10.1140/epjd/e2002-00176-1">(Courtade et al. 2002)</a> | Python translation by J. Maxwell <a href="https://www.jlab.org">Jefferson Laboratory</a>, 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()