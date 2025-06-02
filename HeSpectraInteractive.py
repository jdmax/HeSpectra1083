# Helium Spectrum Calculation with Zeeman Splitting based on Fortran from P.J. Nacher
# Translation to Python, NumPy, J. Maxwell 2025
# Interactive version with real-time control

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.gridspec as gridspec


class HeliumSpectraCalculator:
    """Class to handle helium spectra calculations"""

    def __init__(self):
        # Initialize all constants and matrices
        self.setup_constants()
        self.setup_matrices()

    def setup_constants(self):
        """Initialize physical constants"""
        # Constants
        self.zero = 0.0
        self.epsilon = 1e-12
        self.r2 = np.sqrt(2)
        self.r3 = np.sqrt(3)
        self.r6 = np.sqrt(6)

        # Fine structure, He4
        self.e14 = 2291.175e-3  # J=1 level in GHz (e2=reference)
        self.e04 = self.e14 + 29616.950e-3  # J=0 level in GHz (e2=reference)
        self.delta = 6.1431e4  # singlet-triplet gap
        self.eM = -17.037  # singlet-triplet mixing
        self.ep14 = self.e14 + self.eM ** 2 / (self.delta - self.e14)  # diagonal parameter

        # Fine structure, He3
        self.f01 = -0.841e-3  # m-dependent contribution to Hfs splittings
        self.f12 = 2.294e-3  # from Hinds85
        self.mratio = 1.81921204 / 1.37074562  # µ/M ratios, DrakeHB
        self.e013 = self.e04 - self.e14 + (self.mratio - 1.0) * self.f01  # Hinds85
        self.e13 = self.e14 + (self.mratio - 1.0) * self.f12  # Hinds85
        self.e03 = self.e13 + self.e013
        self.ep13 = self.e13 + self.eM ** 2 / (self.delta - self.e13)  # diagonal parameter

        # Hyperfine structure, 23S
        self.aS = -6739.701177e-3 * (2.0 / 3.0)  # from Rosner70 (*2/3)

        # Hyperfine structure, 23P - Nacher Optimal adjustment values at 1T
        self.c = -4283.026e-3  # GHz (0.019% lower than CPres)
        self.d = -14.507e-3  # GHz (3.4% higher than DPres/2)
        self.e = 1.4861e-3  # GHz (4.6% higher than EPres/5)

        # Miscellaneous constants
        self.mu = 13996.24189e-3  # Bohr magneton in GHz/T i.e., µB/hbar from DrakeHB
        self.gsS = 2.002237319
        self.gsP = 2.002238838
        self.gl3 = 0.999827935
        self.gl4 = 0.999873626
        self.gi = 0.0023174823

        self.D2C9 = 810.599e-3  # gap C9-D2 in B=0 Shiner95

        # C1 line absolute position in GHz for wavelength calculations
        self.c1_ghz = 2.766933041e+5

    def setup_matrices(self):
        """Initialize transformation and Hamiltonian matrices"""
        # Matrix P4 for |J,mJ> coupling
        self.P4 = np.zeros((9, 9))
        self.P4[0, 0] = 1.0
        self.P4[1, 1] = 1.0 / self.r2
        self.P4[1, 3] = 1.0 / self.r2
        self.P4[2, 2] = 1.0 / self.r6
        self.P4[2, 4] = 2.0 / self.r6
        self.P4[2, 6] = 1.0 / self.r6
        self.P4[3, 5] = 1.0 / self.r2
        self.P4[3, 7] = 1.0 / self.r2
        self.P4[4, 8] = 1.0
        self.P4[5, 1] = -1.0 / self.r2
        self.P4[5, 3] = 1.0 / self.r2
        self.P4[6, 2] = -1.0 / self.r2
        self.P4[6, 6] = 1.0 / self.r2
        self.P4[7, 5] = -1.0 / self.r2
        self.P4[7, 7] = 1.0 / self.r2
        self.P4[8, 2] = 1.0 / self.r3
        self.P4[8, 4] = -1.0 / self.r3
        self.P4[8, 6] = 1.0 / self.r3
        self.invP4 = np.linalg.inv(self.P4)

        # Setup He4 matrices
        self.setup_he4_matrices()

        # Setup He3 matrices
        self.setup_he3_matrices()

    def setup_he4_matrices(self):
        """Setup He4 Hamiltonian matrices"""
        # Hfs for He4
        self.Hf4P = np.zeros((9, 9))
        self.Hf4P[5, 5] = self.e14
        self.Hf4P[6, 6] = self.e14  # fine structure term in coupled basis |J,mj>
        self.Hf4P[7, 7] = self.e14
        self.Hf4P[8, 8] = self.e04
        # Compute invP.Hf.P, @ is Numpy matrix multiplication
        self.Hf4P = self.invP4 @ self.Hf4P @ self.P4

        # Zeeman term for He4
        self.Hzee4P = np.zeros((9, 9))
        self.Hzee4P[0, 0] = self.gl4 + self.gsP
        self.Hzee4P[1, 1] = self.gsP
        self.Hzee4P[2, 2] = -self.gl4 + self.gsP
        self.Hzee4P[3, 3] = self.gl4
        self.Hzee4P[5, 5] = -self.gl4
        self.Hzee4P[6, 6] = self.gl4 - self.gsP
        self.Hzee4P[7, 7] = -self.gsP
        self.Hzee4P[8, 8] = -self.gl4 - self.gsP

    def setup_he3_matrices(self):
        """Setup He3 Hamiltonian matrices"""
        # Reduced matrix F6/As for He3
        self.F6red = np.zeros((6, 6))
        self.F6red[0, 0] = 1.0 / 2
        self.F6red[1, 3] = 1.0 / self.r2
        self.F6red[2, 2] = -1.0 / 2
        self.F6red[2, 4] = 1.0 / self.r2
        self.F6red[3, 1] = 1.0 / self.r2
        self.F6red[3, 3] = -1.0 / 2
        self.F6red[4, 2] = 1.0 / self.r2
        self.F6red[5, 5] = 1.0 / 2

        # 23S state for He3
        self.Hhf3S = self.aS * self.F6red

        # Zeeman term for 23S He3
        self.Hzee3S = np.zeros((6, 6))
        self.Hzee3S[0, 0] = self.gsS + self.gi / 2.0
        self.Hzee3S[1, 1] = self.gi / 2.0
        self.Hzee3S[2, 2] = -self.gsS + self.gi / 2.0
        self.Hzee3S[3, 3] = self.gsS - self.gi / 2.0
        self.Hzee3S[4, 4] = -self.gi / 2.0
        self.Hzee3S[5, 5] = -self.gsS - self.gi / 2.0

        # Setup He3 P state matrices
        self.setup_he3_p_matrices()

    def setup_he3_p_matrices(self):
        """Setup He3 P state matrices"""
        # 23P state for He3
        e0 = self.e03
        e1 = self.e13  # true energy; e2=0

        # Hfs for He3
        self.Hf3P = np.zeros((9, 9))
        self.Hf3P[5, 5] = e1
        self.Hf3P[6, 6] = e1  # fine structure in coupled basis |J,mj>
        self.Hf3P[7, 7] = e1
        self.Hf3P[8, 8] = e0
        # Compute invP.Hf.P
        self.Hf3P = self.invP4 @ self.Hf3P @ self.P4

        # Hyperfine structure for 23P He3
        self.Hcor = np.zeros((18, 18))
        self.Hcor[0, 0] = self.d + 2.0 * self.e
        self.Hcor[1, 1] = -4.0 * self.e
        self.Hcor[2, 2] = -self.d + 2.0 * self.e
        self.Hcor[3, 1] = 3.0 * self.e
        self.Hcor[3, 3] = self.d
        self.Hcor[4, 2] = -3.0 * self.e
        self.Hcor[5, 5] = -self.d
        self.Hcor[6, 4] = 3.0 * self.e
        self.Hcor[6, 6] = self.d - 2.0 * self.e
        self.Hcor[7, 5] = -3.0 * self.e
        self.Hcor[7, 7] = 4.0 * self.e
        self.Hcor[8, 8] = -self.d - 2.0 * self.e
        self.Hcor[9, 1] = self.r2 * (self.d + 3.0 * self.e)
        self.Hcor[9, 3] = -self.r2 * self.e
        self.Hcor[10, 2] = self.r2 * (self.d - 3.0 * self.e)
        self.Hcor[10, 4] = self.r2 * 2.0 * self.e
        self.Hcor[11, 5] = -self.r2 * self.e
        self.Hcor[12, 2] = self.r2 * 6.0 * self.e
        self.Hcor[12, 4] = self.r2 * self.d
        self.Hcor[12, 6] = -self.r2 * self.e
        self.Hcor[13, 5] = self.r2 * self.d
        self.Hcor[13, 7] = self.r2 * 2.0 * self.e
        self.Hcor[14, 8] = -self.r2 * self.e
        self.Hcor[15, 5] = self.r2 * 6.0 * self.e
        self.Hcor[15, 7] = self.r2 * (self.d - 3.0 * self.e)
        self.Hcor[16, 8] = self.r2 * (self.d + 3.0 * self.e)

        # Apply diagonal symmetry
        for i in range(1, 18):
            for j in range(i + 1, 18):
                self.Hcor[i, j] = self.Hcor[j, i]

        # Apply second diagonal symmetry
        for i in range(9):
            for j in range(9):
                self.Hcor[18 - j - 1, 18 - i - 1] = self.Hcor[j, i]

        # Calculate H3PB0
        self.H3PB0 = np.copy(self.Hcor)
        for i in range(9):
            for j in range(9):
                self.H3PB0[i, j] += self.Hf3P[i, j]
                self.H3PB0[i + 9, j + 9] += self.Hf3P[i, j]

        for i in range(6):
            for j in range(6):
                self.H3PB0[3 * i, 3 * j] += self.c * self.F6red[i, j]
                self.H3PB0[3 * i + 1, 3 * j + 1] += self.c * self.F6red[i, j]
                self.H3PB0[3 * i + 2, 3 * j + 2] += self.c * self.F6red[i, j]

        # Zeeman term for 23P He3
        self.Hzee3P = np.zeros((18, 18))
        self.Hzee3P[0, 0] = self.gl3 + self.gsP
        self.Hzee3P[1, 1] = self.gsP
        self.Hzee3P[2, 2] = -self.gl3 + self.gsP
        self.Hzee3P[3, 3] = self.gl3
        self.Hzee3P[5, 5] = -self.gl3
        self.Hzee3P[6, 6] = self.gl3 - self.gsP
        self.Hzee3P[7, 7] = -self.gsP
        self.Hzee3P[8, 8] = -self.gl3 - self.gsP

        for i in range(9):
            self.Hzee3P[i + 9, i + 9] = self.Hzee3P[i, i] - self.gi / 2.0  # assign second block
            self.Hzee3P[i, i] = self.Hzee3P[i, i] + self.gi / 2.0  # modify first block

    def calculate_spectra(self, B, Temp=300):
        """Calculate spectra for given magnetic field and temperature"""
        D3 = 1.1875 * np.sqrt(Temp / 300)  # Doppler width for He3
        D4 = D3 * np.sqrt(3.0 / 4.0)  # Doppler width for He4

        # Zero-field computation first for energy references
        H3S = self.Hhf3S
        W3S, V3S = np.linalg.eigh(H3S)
        idx = W3S.argsort()
        W3S = W3S[idx]
        V3S = V3S[:, idx]

        H3P = self.H3PB0
        W3P, V3P = np.linalg.eigh(H3P)
        idx = W3P.argsort()
        W3P = W3P[idx]
        V3P = V3P[:, idx]

        # He4 zero field
        W4S = np.zeros(3)
        W4S[2] = self.mu * B * self.gsS
        W4S[1] = self.zero
        W4S[0] = -self.mu * B * self.gsS
        V4S = np.eye(3)

        H4P = self.Hf4P
        W4P, V4P = np.linalg.eigh(H4P)
        idx = W4P.argsort()
        W4P = W4P[idx]
        V4P = V4P[:, idx]

        eC1 = W3P[8] - W3S[5]  # Reference for energy offsets
        eC9 = W3P[16] - W3S[2]

        # He4 offset
        eD2 = W4P[2] - W4S[1]
        C1C9 = (W3P[16] - W3S[2]) - eC1
        he4_offset = eD2 + self.D2C9 - C1C9

        # Now compute for chosen field
        # Calculate for He3
        H3S = self.Hhf3S + self.mu * B * self.Hzee3S
        W3S, V3S = np.linalg.eigh(H3S)
        idx = W3S.argsort()
        W3S = W3S[idx]
        V3S = V3S[:, idx]

        H3P = self.H3PB0 + self.mu * B * self.Hzee3P
        W3P, V3P = np.linalg.eigh(H3P)
        idx = W3P.argsort()
        W3P = W3P[idx]
        V3P = V3P[:, idx]

        # Calculate transition matrix elements for He3
        T3p = self.calculate_sigma_plus_he3(V3S, V3P)
        T3m = self.calculate_sigma_minus_he3(V3S, V3P)
        T3pi = self.calculate_pi_he3(V3S, V3P)

        # Sort and store transitions for He3
        r3pe, r3pf, _, _ = self.sort_transitions(T3p, W3P, W3S, self.epsilon, eC1)
        r3me, r3mf, _, _ = self.sort_transitions(T3m, W3P, W3S, self.epsilon, eC1)
        r3pie, r3pif, _, _ = self.sort_transitions(T3pi, W3P, W3S, self.epsilon, eC1)

        # Calculate for He4
        W4S = np.zeros(3)
        W4S[2] = self.mu * B * self.gsS
        W4S[1] = self.zero
        W4S[0] = -self.mu * B * self.gsS
        V4S = np.eye(3)

        H4P = self.Hf4P + self.mu * B * self.Hzee4P
        W4P, V4P = np.linalg.eigh(H4P)
        idx = W4P.argsort()
        W4P = W4P[idx]
        V4P = V4P[:, idx]

        # Calculate transition matrix elements for He4
        T4p = self.calculate_sigma_plus_he4(V4S, V4P)
        T4m = self.calculate_sigma_minus_he4(V4S, V4P)
        T4pi = self.calculate_pi_he4(V4S, V4P)

        # Sort and store transitions for He4
        r4pe, r4pf, _, _ = self.sort_transitions(T4p, W4P, W4S, self.epsilon, he4_offset)
        r4me, r4mf, _, _ = self.sort_transitions(T4m, W4P, W4S, self.epsilon, he4_offset)
        r4pie, r4pif, _, _ = self.sort_transitions(T4pi, W4P, W4S, self.epsilon, he4_offset)

        # Generate Doppler-broadened spectra
        spectra_data = self.generate_spectra_data(
            r3pe, r3pf, r3me, r3mf, r3pie, r3pif, D3,
            r4pe, r4pf, r4me, r4mf, r4pie, r4pif, D4
        )

        return spectra_data

    def calculate_sigma_plus_he3(self, V3S, V3P):
        """Calculate sigma+ transition matrix elements for He3"""
        T3p = np.zeros((18, 6))
        for j in range(18):
            for i in range(6):
                T3p[j, i] = (V3S[0, i] * V3P[0, j] +
                             V3S[1, i] * V3P[3, j] +
                             V3S[2, i] * V3P[6, j] +
                             V3S[3, i] * V3P[9, j] +
                             V3S[4, i] * V3P[12, j] +
                             V3S[5, i] * V3P[15, j]) ** 2
        return T3p

    def calculate_sigma_minus_he3(self, V3S, V3P):
        """Calculate sigma- transition matrix elements for He3"""
        T3m = np.zeros((18, 6))
        for j in range(18):
            for i in range(6):
                T3m[j, i] = (V3S[0, i] * V3P[2, j] +
                             V3S[1, i] * V3P[5, j] +
                             V3S[2, i] * V3P[8, j] +
                             V3S[3, i] * V3P[11, j] +
                             V3S[4, i] * V3P[14, j] +
                             V3S[5, i] * V3P[17, j]) ** 2
        return T3m

    def calculate_pi_he3(self, V3S, V3P):
        """Calculate pi transition matrix elements for He3"""
        T3pi = np.zeros((18, 6))
        for j in range(18):
            for i in range(6):
                T3pi[j, i] = (V3S[0, i] * V3P[1, j] +
                              V3S[1, i] * V3P[4, j] +
                              V3S[2, i] * V3P[7, j] +
                              V3S[3, i] * V3P[10, j] +
                              V3S[4, i] * V3P[13, j] +
                              V3S[5, i] * V3P[16, j]) ** 2
        return T3pi

    def calculate_sigma_plus_he4(self, V4S, V4P):
        """Calculate sigma+ transition matrix elements for He4"""
        T4p = np.zeros((9, 3))
        for j in range(9):
            for i in range(3):
                T4p[j, i] = (V4S[0, 3 - i - 1] * V4P[0, j] +
                             V4S[1, 3 - i - 1] * V4P[3, j] +
                             V4S[2, 3 - i - 1] * V4P[6, j]) ** 2
        return T4p

    def calculate_sigma_minus_he4(self, V4S, V4P):
        """Calculate sigma- transition matrix elements for He4"""
        T4m = np.zeros((9, 3))
        for j in range(9):
            for i in range(3):
                T4m[j, i] = (V4S[0, 3 - i - 1] * V4P[2, j] +
                             V4S[1, 3 - i - 1] * V4P[5, j] +
                             V4S[2, 3 - i - 1] * V4P[8, j]) ** 2
        return T4m

    def calculate_pi_he4(self, V4S, V4P):
        """Calculate pi transition matrix elements for He4"""
        T4pi = np.zeros((9, 3))
        for j in range(9):
            for i in range(3):
                T4pi[j, i] = (V4S[0, 3 - i - 1] * V4P[1, j] +
                              V4S[1, 3 - i - 1] * V4P[4, j] +
                              V4S[2, 3 - i - 1] * V4P[7, j]) ** 2
        return T4pi

    def sort_transitions(self, T, WP, WS, epsilon, energy_offset=0.0):
        """Sort transitions by energy (with offset applied) and filter by magnitude"""
        count = 0
        for j in range(T.shape[0]):
            for i in range(T.shape[1]):
                if T[j, i] >= epsilon:
                    count += 1

        re = np.zeros(count)
        rf = np.zeros(count)
        ind_lower = np.zeros(count, dtype=int)
        ind_upper = np.zeros(count, dtype=int)

        idx = 0
        for j in range(T.shape[0]):
            for i in range(T.shape[1]):
                if T[j, i] >= epsilon:
                    re[idx] = WP[j] - WS[i] - energy_offset
                    rf[idx] = T[j, i]
                    ind_lower[idx] = i
                    ind_upper[idx] = j
                    idx += 1

        sort_idx = np.argsort(re)
        return re[sort_idx], rf[sort_idx], ind_lower[sort_idx], ind_upper[sort_idx]

    def generate_spectra_data(self, r3pe, r3pf, r3me, r3mf, r3pie, r3pif, D3,
                              r4pe, r4pf, r4me, r4mf, r4pie, r4pif, D4):
        """Generate Doppler-broadened spectra data"""
        # Generate frequency axis
        freq_range = np.arange(-200, 300.1, 0.1)

        # Create absolute frequency axes
        abs_freq_he3 = self.c1_ghz + freq_range - 40
        abs_freq_he4 = self.c1_ghz + freq_range

        # Initialize arrays for spectra
        he3_plus = np.zeros_like(freq_range)
        he3_minus = np.zeros_like(freq_range)
        he3_pi = np.zeros_like(freq_range)
        he4_plus = np.zeros_like(freq_range)
        he4_minus = np.zeros_like(freq_range)
        he4_pi = np.zeros_like(freq_range)

        # Calculate spectra for He3
        for i in range(len(r3pe)):
            he3_plus += r3pf[i] * np.exp(-((freq_range - r3pe[i] - 40) / D3) ** 2)

        for i in range(len(r3me)):
            he3_minus += r3mf[i] * np.exp(-((freq_range - r3me[i] - 40) / D3) ** 2)

        for i in range(len(r3pie)):
            he3_pi += r3pif[i] * np.exp(-((freq_range - r3pie[i] - 40) / D3) ** 2)

        # Calculate spectra for He4
        for i in range(len(r4pe)):
            he4_plus += r4pf[i] * np.exp(-((freq_range - r4pe[i]) / D4) ** 2)

        for i in range(len(r4me)):
            he4_minus += r4mf[i] * np.exp(-((freq_range - r4me[i]) / D4) ** 2)

        for i in range(len(r4pie)):
            he4_pi += r4pif[i] * np.exp(-((freq_range - r4pie[i]) / D4) ** 2)

        return {
            'freq_range': freq_range,
            'abs_freq_he3': abs_freq_he3,
            'abs_freq_he4': abs_freq_he4,
            'he3_plus': he3_plus,
            'he3_minus': he3_minus,
            'he3_pi': he3_pi,
            'he4_plus': he4_plus,
            'he4_minus': he4_minus,
            'he4_pi': he4_pi
        }


class InteractiveSpectraPlotter:
    """Interactive plotter for helium spectra with controls"""

    def __init__(self):
        self.calculator = HeliumSpectraCalculator()
        self.current_B = 1.0  # Default B field
        self.current_isotope = 'He3'  # Default isotope
        self.current_x_axis = 'frequency'  # Default x-axis type
        self.current_temp = 300  # Default temperature

        # Initialize plot
        self.setup_plot()

    def setup_plot(self):
        """Setup the interactive plot with controls"""
        self.fig = plt.figure(figsize=(14, 10))

        # Create grid layout
        gs = gridspec.GridSpec(4, 4, figure=self.fig,
                               left=0.1, right=0.75, bottom=0.25, top=0.9,
                               hspace=0.3, wspace=0.3)

        # Main plot area
        self.ax = self.fig.add_subplot(gs[:, :3])

        # Control area
        controls_gs = gridspec.GridSpecFromSubplotSpec(4, 1, gs[:, 3], hspace=0.4)

        # Slider for B field
        slider_ax = self.fig.add_subplot(controls_gs[0, 0])
        self.B_slider = Slider(
            slider_ax, 'B (T)', 0.01, 10.0,
            valinit=self.current_B, valstep=0.01
        )

        # Slider for Temperature
        temp_ax = self.fig.add_subplot(controls_gs[1, 0])
        self.temp_slider = Slider(
            temp_ax, 'T (K)', 77, 1000,
            valinit=self.current_temp, valstep=1
        )

        # Radio buttons for isotope selection
        isotope_ax = self.fig.add_subplot(controls_gs[2, 0])
        self.isotope_radio = RadioButtons(
            isotope_ax, ('He3', 'He4'), active=0
        )
        isotope_ax.set_title('Isotope', fontweight='bold')

        # Radio buttons for x-axis type
        xaxis_ax = self.fig.add_subplot(controls_gs[3, 0])
        self.xaxis_radio = RadioButtons(
            xaxis_ax, ('Frequency', 'Wavelength'), active=0
        )
        xaxis_ax.set_title('X-axis', fontweight='bold')

        # Connect event handlers
        self.B_slider.on_changed(self.update_B_field)
        self.temp_slider.on_changed(self.update_temperature)
        self.isotope_radio.on_clicked(self.update_isotope)
        self.xaxis_radio.on_clicked(self.update_xaxis)

        # Initial plot
        self.update_plot()

    def update_B_field(self, val):
        """Update B field value and replot"""
        self.current_B = val
        self.update_plot()

    def update_temperature(self, val):
        """Update temperature value and replot"""
        self.current_temp = val
        self.update_plot()

    def update_isotope(self, label):
        """Update isotope selection and replot"""
        self.current_isotope = label
        self.update_plot()

    def update_xaxis(self, label):
        """Update x-axis type and replot"""
        self.current_x_axis = label
        self.update_plot()

    def update_plot(self):
        """Update the main plot with current parameters"""
        # Calculate spectra
        spectra_data = self.calculator.calculate_spectra(self.current_B, self.current_temp)

        # Clear previous plot
        self.ax.clear()

        # Choose data based on isotope
        if self.current_isotope == 'He3':
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
        if self.current_x_axis == 'frequency':
            x_data = freq_range
            x_label = 'Frequency Offset (GHz)'
        else:  # wavelength
            x_data = 299792458.0 / abs_freq  # Convert to wavelength in nm
            x_label = 'Wavelength (nm)'

        # Plot spectra
        self.ax.plot(x_data, plus_data, 'b-', linewidth=2, label='σ+')
        self.ax.plot(x_data, minus_data, 'r-', linewidth=2, label='σ-')
        self.ax.plot(x_data, pi_data, 'g-', linewidth=2, label='π')

        # Formatting
        self.ax.set_xlabel(x_label, fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=12)
        self.ax.set_title(
            f'{self.current_isotope} Spectra at B = {self.current_B:.2f} T, T = {self.current_temp:.0f} K',
            fontsize=14, fontweight='bold')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # Auto-scale for better viewing
        self.ax.relim()
        self.ax.autoscale_view()

        # Add some margin to y-axis
        y_min, y_max = self.ax.get_ylim()
        self.ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        # Redraw
        self.fig.canvas.draw()

    def show(self):
        """Display the interactive plot"""
        plt.show()


def main():
    """Main function - now launches interactive interface"""
    print("Launching Interactive Helium Spectra Calculator...")

    # Create and show interactive plotter
    plotter = InteractiveSpectraPlotter()
    plotter.show()


if __name__ == "__main__":
    main()