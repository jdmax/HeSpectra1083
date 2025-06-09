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
    """Write energy level files with index, frequency, and magnetic quantum numbers"""

    # Write He3 S state energy levels (6 states)
    # States are ordered by energy, corresponding to F,mF quantum numbers
    # For S=1, I=1/2: F=3/2 (mF = +3/2, +1/2, -1/2, -3/2) and F=1/2 (mF = +1/2, -1/2)
    he3_s_quantum_numbers = [
        (0, -3 / 2),  # F=3/2, mF=-3/2
        (1, -1 / 2),  # F=3/2, mF=-1/2
        (2, -1 / 2),  # F=1/2, mF=-1/2
        (3, +1 / 2),  # F=1/2, mF=+1/2
        (4, +1 / 2),  # F=3/2, mF=+1/2
        (5, +3 / 2)  # F=3/2, mF=+3/2
    ]

    with open(out_folder + "Ai.dat", "w") as f:
        f.write("# He3 23S1 state energy levels\n")
        f.write("# Index  Energy(GHz)      mF\n")
        for i, (idx, mF) in enumerate(he3_s_quantum_numbers):
            f.write(f"{i:6d} {energy_levels['W3S'][i]:15.8f} {mF:6.1f}\n")

    # Write He3 P state energy levels (18 states)
    # States include J=0,1,2 coupled with I=1/2
    # The ordering follows the basis construction in the code
    he3_p_quantum_numbers = []

    # First 9 states are mI = +1/2 block
    # J=2: mJ = +2, +1, 0, -1, -2
    # J=1: mJ = +1, 0, -1
    # J=0: mJ = 0
    mI = +0.5
    for mJ in [+2, +1, 0, +1, 0, -1, 0, -1, -2]:
        he3_p_quantum_numbers.append((mJ, mI, mJ + mI))

    # Next 9 states are mI = -1/2 block
    mI = -0.5
    for mJ in [+2, +1, 0, +1, 0, -1, 0, -1, -2]:
        he3_p_quantum_numbers.append((mJ, mI, mJ + mI))

    with open(out_folder + "Bj.dat", "w") as f:
        f.write("# He3 23P state energy levels\n")
        f.write("# Index  Energy(GHz)      mJ    mI    mF\n")
        for i, (mJ, mI, mF) in enumerate(he3_p_quantum_numbers):
            f.write(f"{i:6d} {energy_levels['W3P'][i]:15.8f} {mJ:6.1f} {mI:5.1f} {mF:6.1f}\n")

    # Write He4 S state energy levels (3 states)
    # Simple triplet state with mS = +1, 0, -1
    he4_s_quantum_numbers = [
        (0, -1),  # mS = -1
        (1, 0),  # mS =  0
        (2, +1)  # mS = +1
    ]

    with open(out_folder + "Yi.dat", "w") as f:
        f.write("# He4 23S1 state energy levels\n")
        f.write("# Index  Energy(GHz)      mS\n")
        for i, (idx, mS) in enumerate(he4_s_quantum_numbers):
            f.write(f"{i:6d} {energy_levels['W4S'][i]:15.8f} {mS:6d}\n")

    # Write He4 P state energy levels (9 states)
    # States are J=2,1,0 with corresponding mJ values
    # Order follows the P4 transformation matrix
    he4_p_quantum_numbers = [
        (0, 2, +2),  # J=2, mJ=+2
        (1, 2, +1),  # J=2, mJ=+1
        (2, 2, 0),  # J=2, mJ=0
        (3, 1, +1),  # J=1, mJ=+1
        (4, 1, 0),  # J=1, mJ=0
        (5, 1, -1),  # J=1, mJ=-1
        (6, 2, -1),  # J=2, mJ=-1
        (7, 2, -2),  # J=2, mJ=-2
        (8, 0, 0)  # J=0, mJ=0
    ]

    with open(out_folder + "Zi.dat", "w") as f:
        f.write("# He4 23P state energy levels\n")
        f.write("# Index  Energy(GHz)      J    mJ\n")
        for i, (idx, J, mJ) in enumerate(he4_p_quantum_numbers):
            f.write(f"{i:6d} {energy_levels['W4P'][i]:15.8f} {J:6d} {mJ:6d}\n")


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