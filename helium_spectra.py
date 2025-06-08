#!/usr/bin/env python3
"""
Helium Spectrum Command-line Interface
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from helium_spectra_calc import HeliumSpectraCalculator


def get_user_input():
    """Get magnetic field and temperature values from user"""
    print("Field (Tesla), Temperature (K)?  (Doppler width He3: 1.1875 GHz @300 K)")
    B = float(input("Field (Tesla): "))
    Temp = float(input("Temperature (K): "))
    return B, Temp


def write_energy_levels(out_folder, energy_levels):
    """Write energy level files"""
    # Write He3 energy levels
    with open(out_folder + "Ai.dat", "w") as f:
        for i in range(6):
            f.write(f"{energy_levels['W3S'][i]:.8f} ")

    with open(out_folder + "Bj.dat", "w") as f:
        for i in range(18):
            f.write(f"{energy_levels['W3P'][i]:.8f} ")

    # Write He4 energy levels
    with open(out_folder + "Yi.dat", "w") as f:
        for i in range(3):
            f.write(f"{energy_levels['W4S'][i]:.8f} ")

    with open(out_folder + "Zi.dat", "w") as f:
        for i in range(9):
            f.write(f"{energy_levels['W4P'][i]:.8f} ")


def write_transitions(filename, B, transition_data):
    """Write transition data to file"""
    energies = transition_data['energies']
    forces = transition_data['forces']
    ind_lower = transition_data['ind_lower']
    ind_upper = transition_data['ind_upper']

    with open(filename, "w") as f:
        for i in range(len(energies)):
            f.write(f"{B:10.5f} {energies[i]:15.8e} {forces[i]:15.8e} {ind_lower[i]:3d} {ind_upper[i]:3d}\n")


def write_all_transitions(out_folder, B, transitions):
    """Write all transition files"""
    # He3 transitions
    write_transitions(out_folder + "He3plus.dat", B, transitions['he3']['plus'])
    write_transitions(out_folder + "He3minus.dat", B, transitions['he3']['minus'])
    write_transitions(out_folder + "He3pi.dat", B, transitions['he3']['pi'])

    # He4 transitions
    write_transitions(out_folder + "He4plus.dat", B, transitions['he4']['plus'])
    write_transitions(out_folder + "He4minus.dat", B, transitions['he4']['minus'])
    write_transitions(out_folder + "He4pi.dat", B, transitions['he4']['pi'])


def write_spectra_files(out_folder, spectra_data):
    """Write spectra data to files"""
    freq_range = spectra_data['freq_range']
    abs_freq_he3 = spectra_data['abs_freq_he3']
    abs_freq_he4 = spectra_data['abs_freq_he4']

    # Write He3 spectra to file
    with open(out_folder + "spHe3.dat", "w") as f:
        f.write(" AbsFreq(GHz) RelFreq(GHz) sigplus sigminus pi wavelength(nm)\n")
        for i in range(len(freq_range)):
            # Calculate wavelength in nm
            wavelength_nm = 299792458 / abs_freq_he3[i]
            f.write(f"{abs_freq_he3[i]:15.8f} {freq_range[i] - 40:15.8f} {spectra_data['he3_plus'][i]:15.8f} "
                    f"{spectra_data['he3_minus'][i]:15.8f} {spectra_data['he3_pi'][i]:15.8f} {wavelength_nm:15.8f}\n")

    # Write He4 spectra to file
    with open(out_folder + "spHe4.dat", "w") as f:
        f.write(" AbsFreq(GHz) RelFreq(GHz) sigplus sigminus pi wavelength(nm)\n")
        for i in range(len(freq_range)):
            # Calculate wavelength in nm
            wavelength_nm = 299792458 / abs_freq_he4[i]
            f.write(f"{abs_freq_he4[i]:15.8f} {freq_range[i]:15.8f} {spectra_data['he4_plus'][i]:15.8f} "
                    f"{spectra_data['he4_minus'][i]:15.8f} {spectra_data['he4_pi'][i]:15.8f} {wavelength_nm:15.8f}\n")


def create_plots(out_folder, spectra_data):
    """Create plots of the spectra"""
    freq_range = spectra_data['freq_range']
    abs_freq_he3 = spectra_data['abs_freq_he3']
    abs_freq_he4 = spectra_data['abs_freq_he4']

    # Create plots of the spectra with both relative and absolute scales
    plt.figure(figsize=(12, 12))

    # He3 plot with relative frequencies
    plt.subplot(2, 2, 1)
    plt.plot(freq_range - 40, spectra_data['he3_plus'], label='σ+')
    plt.plot(freq_range - 40, spectra_data['he3_minus'], label='σ-')
    plt.plot(freq_range - 40, spectra_data['he3_pi'], label='π')
    plt.title('He3 Spectra (Relative)')
    plt.xlabel('Frequency offset (GHz)')
    plt.ylabel('Intensity')
    plt.legend()

    # He3 plot with absolute frequencies
    plt.subplot(2, 2, 2)
    plt.plot(abs_freq_he3, spectra_data['he3_plus'], label='σ+')
    plt.plot(abs_freq_he3, spectra_data['he3_minus'], label='σ-')
    plt.plot(abs_freq_he3, spectra_data['he3_pi'], label='π')
    plt.title('He3 Spectra (Absolute)')
    plt.xlabel('Absolute Frequency (GHz)')
    plt.ylabel('Intensity')
    plt.legend()

    # He4 plot with relative frequencies
    plt.subplot(2, 2, 3)
    plt.plot(freq_range, spectra_data['he4_plus'], label='σ+')
    plt.plot(freq_range, spectra_data['he4_minus'], label='σ-')
    plt.plot(freq_range, spectra_data['he4_pi'], label='π')
    plt.title('He4 Spectra (Relative)')
    plt.xlabel('Frequency offset (GHz)')
    plt.ylabel('Intensity')
    plt.legend()

    # He4 plot with absolute frequencies
    plt.subplot(2, 2, 4)
    plt.plot(abs_freq_he4, spectra_data['he4_plus'], label='σ+')
    plt.plot(abs_freq_he4, spectra_data['he4_minus'], label='σ-')
    plt.plot(abs_freq_he4, spectra_data['he4_pi'], label='π')
    plt.title('He4 Spectra (Absolute)')
    plt.xlabel('Absolute Frequency (GHz)')
    plt.ylabel('Intensity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_folder + 'helium_spectra.png')

    # Create another plot with wavelength on x-axis
    plt.figure(figsize=(12, 6))

    # Convert frequency to wavelength (nm)
    wavelength_he3 = 299792458. / abs_freq_he3
    wavelength_he4 = 299792458. / abs_freq_he4

    plt.subplot(1, 2, 1)
    plt.plot(wavelength_he3, spectra_data['he3_plus'], label='σ+')
    plt.plot(wavelength_he3, spectra_data['he3_minus'], label='σ-')
    plt.plot(wavelength_he3, spectra_data['he3_pi'], label='π')
    plt.title('He3 Spectra (Wavelength)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(wavelength_he4, spectra_data['he4_plus'], label='σ+')
    plt.plot(wavelength_he4, spectra_data['he4_minus'], label='σ-')
    plt.plot(wavelength_he4, spectra_data['he4_pi'], label='π')
    plt.title('He4 Spectra (Wavelength)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_folder + 'helium_spectra_wavelength.png')

    plt.show()


def main():
    """Main function that calculates the structure and spectra of 1083 nm transitions in He3 and He4."""
    # Output folder
    out_folder = "output/"
    os.makedirs(out_folder, exist_ok=True)

    # Create calculator instance
    calculator = HeliumSpectraCalculator()

    # Get user input
    B, Temp = get_user_input()

    # Calculate all results
    results = calculator.calculate_full_results(B, Temp)

    # Extract components
    spectra_data = results['spectra_data']
    energy_levels = results['energy_levels']
    transitions = results['transitions']

    # Write all output files
    write_energy_levels(out_folder, energy_levels)
    write_all_transitions(out_folder, B, transitions)
    write_spectra_files(out_folder, spectra_data)

    # Create plots
    create_plots(out_folder, spectra_data)


if __name__ == "__main__":
    main()