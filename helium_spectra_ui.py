#!/usr/bin/env python3
"""
Helium Spectrum Calculator - Streamlit UI
Interactive interface for helium spectra calculations

Need to install: "pip install streamlit plotly numpy"
To run: "streamlit run helium_spectra_ui.py"
"""

import streamlit as st
import plotly.graph_objects as go
from helium_spectra_calc import HeliumSpectraCalculator


@st.cache_data
def get_calculator():
    """Cached calculator instance for better performance"""
    return HeliumSpectraCalculator()


def create_plotly_figure(spectra_data, isotope, x_axis_type, B_field, temperature):
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
        """)

    # Main content area
    col_main, col_info = st.columns([3, 1])

    with col_main:
        # Show current parameters
        st.markdown(f"""
        **Current Parameters:** B = {B_field:.3f} T, T = {temperature:.0f} K, Isotope = {st.session_state.isotope}
        """)

        # Calculate and display spectra
        with st.spinner('Calculating spectra...'):
            calculator = get_calculator()
            spectra_data = calculator.calculate_spectra(B_field, temperature)

            # Create and display plot
            fig = create_plotly_figure(spectra_data, st.session_state.isotope, st.session_state.x_axis_type, B_field,
                                       temperature)
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

        if st.button("📊 Download CSV Data"):
            # Generate CSV data
            import pandas as pd

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