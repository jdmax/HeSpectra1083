#!/usr/bin/env python3
"""
Helium Spectrum Calculation Module
Based on Fortran code by P.J. Nacher
Python translation by J. Maxwell 2025

This module contains the calculation logic for helium spectra with Zeeman splitting.
"""

import numpy as np

# --------------------------------------------------------------------------
# Voigt line shape, following P.J. Nacher's spectreVoigt_w0w12 Fortran.
#
# He tabulates
#     K(x, y) = (y/pi) * Integral exp(-z^2) / (y^2 + (x - z)^2) dz
# with
#     x = 2*sqrt(ln2) * (nu - nu0) / wG        (wG = Doppler FWHM)
#     y = sqrt(ln2) * wL / wG                  (wL = Lorentz FWHM)
# evaluating the integral by Simpson quadrature over z in [-7, 7].
#
# That integral is by definition the real part of the Faddeeva function,
# Re[w(x + iy)], so it is evaluated directly here rather than by quadrature:
# same quantity, no truncation at |z| = 7, and fast enough to stay
# interactive. K(x, 0) = exp(-x^2) exactly, so at zero pressure this reduces
# to the Gaussian the module used before.
# --------------------------------------------------------------------------

_WEIDEMAN_N = 32
_weideman_coefficients = None


def _weideman_a(n=_WEIDEMAN_N):
    """Cached coefficients for Weideman's rational approximation to w(z)."""
    global _weideman_coefficients
    if _weideman_coefficients is None:
        m = 2 * n
        k = np.arange(-m + 1, m)
        L = np.sqrt(n / np.sqrt(2.0))
        t = L * np.tan(k * np.pi / (2 * m))
        f = np.concatenate(([0.0], np.exp(-t ** 2) * (L ** 2 + t ** 2)))
        a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * m)
        _weideman_coefficients = (np.flipud(a[1:n + 1]), L)
    return _weideman_coefficients


def faddeeva(z):
    """w(z) = exp(-z^2) erfc(-i z), for Im(z) >= 0.

    Weideman's method (SIAM J. Numer. Anal. 31, 1497 (1994)). Pure numpy, so
    it runs under Pyodide without pulling in scipy; y is a half width and is
    never negative, so the upper half plane is all that is needed.
    """
    a, L = _weideman_a()
    denominator = L - 1j * np.asarray(z)
    Z = (L + 1j * np.asarray(z)) / denominator
    p = np.polyval(a, Z)
    return 2.0 * p / denominator ** 2 + (1.0 / np.sqrt(np.pi)) / denominator


def voigt_K(detuning, wG, wL):
    """Nacher's K(x, y): unit peak height in the Doppler-only limit.

    detuning, wG (Doppler FWHM) and wL (Lorentz FWHM) are all in GHz.
    """
    root_ln2 = np.sqrt(np.log(2.0))
    x = 2.0 * root_ln2 * np.asarray(detuning, dtype=float) / wG
    if wL <= 0.0:
        # Exactly what the module computed before any pressure support
        return np.exp(-x ** 2)
    y = root_ln2 * wL / wG
    return np.real(faddeeva(x + 1j * y))


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

        # Doppler width, as spectreVoigt_w0w12 computes it:
        #   wG = sqrt(2 R T / M) / lambda * 2 sqrt(ln2)      [FWHM]
        self.R_gas = 8.314472           # J/(mol K)
        self.lambda_1083 = 1.0829e-6    # m
        self.molar_mass = {'He3': 3.0160293e-3, 'He4': 4.00260e-3}   # kg/mol

        # Collisional broadening, Lorentz FWHM in GHz per mbar. The Fortran
        # takes wL0 (2^3P_0 lines: C8, C9 for He3; D0 for He4) and wL12 (all
        # others) as separate inputs; these are the typical values its prompts
        # quote, from Nikiel et al., Eur. Phys. J. D 67, 200 (2013).
        self.collision_per_mbar = {'He3': 0.012, 'He4': 0.0104}

        # /// START OF MODIFIED CODE ///
        # mF values for the basis states, used to identify the mF of the eigenstates
        self.mF3S_basis = np.array([1.5, 0.5, -0.5, 0.5, -0.5, -1.5])
        self.mF3P_basis = np.array([2.5, 1.5, 0.5, 1.5, 0.5, -0.5, 0.5, -0.5, -1.5,
                                    1.5, 0.5, -0.5, 0.5, -0.5, -1.5, -0.5, -1.5, -2.5])
        # For He4, I=0, so mF = mJ. The basis is the uncoupled |mL, mS> basis.
        # mJ = mL + mS. For the 9 P-states, basis order is (mL,mS) = (1,1),(0,1),(-1,1),(1,0), etc.
        self.mF4P_basis = np.array([2.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, -2.0])
        self.mF4S_basis = np.array([-1.0, 0.0, 1.0])  # mS for S=1 state, ordered by energy
        # /// END OF MODIFIED CODE ///

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

    def doppler_fwhm(self, Temp, isotope):
        """Doppler FWHM in GHz, from the molar mass as the Fortran does.

        The module's own D3/D4 are 1/e half-widths and relate to this by
        wG = 2 sqrt(ln2) D. Note D4 was taken as D3*sqrt(3/4); the exact mass
        ratio is sqrt(M3/M4) = 0.86805, so this runs 0.25% wider for He4.
        """
        speed = np.sqrt(2.0 * self.R_gas * Temp / self.molar_mass[isotope])
        return speed / self.lambda_1083 * 2.0 * np.sqrt(np.log(2.0)) / 1e9

    def j0_weights(self, V_P, n_spin=1):
        """Fraction of each 2^3P eigenstate that is J = 0.

        P4 carries the 9 uncoupled |mL,mS> states into the coupled |J,mJ>
        basis, where index 8 is |J=0, mJ=0>. For He3 the 18 states are that
        9-dim space times the two nuclear spin projections, stored as two
        blocks of 9, so the J=0 population is summed over both.

        The Fortran assigns each line to wL0 or wL12 by name, which presumes
        J is a good quantum number. At several tesla it is not, so the width
        is interpolated by this admixture instead; below ~0.2 T the weights
        are 0 or 1 and the two agree.
        """
        weights = np.zeros(V_P.shape[1])
        for block in range(n_spin):
            rows = slice(block * 9, block * 9 + 9)
            amplitude = self.P4 @ V_P[rows, :]
            weights += np.abs(amplitude[8, :]) ** 2
        return weights

    def line_widths(self, ind_upper, j0_weight, wL0, wL12):
        """Per-transition Lorentz FWHM, blended by the upper state's J=0 part"""
        p0 = j0_weight[np.asarray(ind_upper, dtype=int)]
        return p0 * wL0 + (1.0 - p0) * wL12

    def calculate_full_results(self, B, Temp=300, wL0=0.0, wL12=0.0):
        """
        Calculate complete results including all intermediate values needed for file output.

        Returns a dictionary with:
        - spectra_data: Doppler-broadened spectra
        - energy_levels: All energy eigenvalues (W3S, W3P, W4S, W4P)
        - transitions: All transition data (energies, forces, indices)
        - doppler_widths: D3 and D4
        """
        # Doppler 1/e half-widths, from the molar masses as the Fortran does.
        # These were 1.1875*sqrt(T/300) and D3*sqrt(3/4). The He3 value is the
        # same formula rounded to five figures, so it shifts by 0.009%; the He4
        # mass ratio is sqrt(M3/M4) = 0.86805 rather than sqrt(3/4) = 0.86603,
        # so He4 widths come out 0.25% larger than before.
        root_ln2 = np.sqrt(np.log(2.0))
        D3 = self.doppler_fwhm(Temp, 'He3') / (2.0 * root_ln2)
        D4 = self.doppler_fwhm(Temp, 'He4') / (2.0 * root_ln2)

        # Zero-field computation first for energy references
        H3S_zero = self.Hhf3S
        W3S_zero, V3S_zero = np.linalg.eigh(H3S_zero)
        idx = W3S_zero.argsort()
        W3S_zero = W3S_zero[idx]

        H3P_zero = self.H3PB0
        W3P_zero, V3P_zero = np.linalg.eigh(H3P_zero)
        idx = W3P_zero.argsort()
        W3P_zero = W3P_zero[idx]

        # He4 zero field
        H4P_zero = self.Hf4P
        W4P_zero, V4P_zero = np.linalg.eigh(H4P_zero)
        idx = W4P_zero.argsort()
        W4P_zero = W4P_zero[idx]

        eC1 = W3P_zero[8] - W3S_zero[5]  # Reference for energy offsets
        eC9 = W3P_zero[16] - W3S_zero[2]

        # He4 offset
        # At B=0, W4S is just [0,0,0] for the triplet state. We take mS=0 as reference.
        eD2 = W4P_zero[2] - 0.0
        C1C9 = (W3P_zero[16] - W3S_zero[2]) - eC1
        he4_offset = eD2 + self.D2C9 - C1C9

        # Now compute for chosen field
        # Calculate for He3
        H3S = self.Hhf3S + self.mu * B * self.Hzee3S
        W3S, V3S = np.linalg.eigh(H3S)
        dominant_basis_indices_3S = np.argmax(np.abs(V3S), axis=0)
        mf3S = self.mF3S_basis[dominant_basis_indices_3S]
        idx = W3S.argsort()
        W3S = W3S[idx]
        V3S = V3S[:, idx]
        mf3S = mf3S[idx]

        H3P = self.H3PB0 + self.mu * B * self.Hzee3P
        W3P, V3P = np.linalg.eigh(H3P)
        dominant_basis_indices_3P = np.argmax(np.abs(V3P), axis=0)
        mf3P = self.mF3P_basis[dominant_basis_indices_3P]
        idx = W3P.argsort()
        W3P = W3P[idx]
        V3P = V3P[:, idx]
        mf3P = mf3P[idx]

        # Calculate transition matrix elements for He3
        T3p = self.calculate_sigma_plus_he3(V3S, V3P)
        T3m = self.calculate_sigma_minus_he3(V3S, V3P)
        T3pi = self.calculate_pi_he3(V3S, V3P)

        # Sort and store transitions for He3
        r3pe, r3pf, indap, indbp = self.sort_transitions(T3p, W3P, W3S, self.epsilon, eC1)
        r3me, r3mf, indam, indbm = self.sort_transitions(T3m, W3P, W3S, self.epsilon, eC1)
        r3pie, r3pif, indapi, indbpi = self.sort_transitions(T3pi, W3P, W3S, self.epsilon, eC1)

        # Calculate for He4
        W4S = np.zeros(3)
        W4S[2] = self.mu * B * self.gsS
        W4S[1] = self.zero
        W4S[0] = -self.mu * B * self.gsS
        V4S = np.eye(3)
        mf4S = self.mF4S_basis  # Already sorted by energy: -1, 0, 1

        H4P = self.Hf4P + self.mu * B * self.Hzee4P
        W4P, V4P = np.linalg.eigh(H4P)
        dominant_basis_indices_4P = np.argmax(np.abs(V4P), axis=0)
        mf4P = self.mF4P_basis[dominant_basis_indices_4P]
        idx = W4P.argsort()
        W4P = W4P[idx]
        V4P = V4P[:, idx]
        mf4P = mf4P[idx]

        # Calculate transition matrix elements for He4
        T4p = self.calculate_sigma_plus_he4(V4S, V4P)
        T4m = self.calculate_sigma_minus_he4(V4S, V4P)
        T4pi = self.calculate_pi_he4(V4S, V4P)

        # Sort and store transitions for He4
        r4pe, r4pf, indyp, indzp = self.sort_transitions(T4p, W4P, W4S, self.epsilon, he4_offset)
        r4me, r4mf, indym, indzm = self.sort_transitions(T4m, W4P, W4S, self.epsilon, he4_offset)
        r4pie, r4pif, indypi, indzpi = self.sort_transitions(T4pi, W4P, W4S, self.epsilon, he4_offset)

        # Lorentz width per transition, set by how much J=0 character the
        # upper state carries (see j0_weights)
        p0_3 = self.j0_weights(V3P, n_spin=2)
        p0_4 = self.j0_weights(V4P, n_spin=1)
        width = self.line_widths
        wl3 = (width(indbp, p0_3, wL0, wL12),
               width(indbm, p0_3, wL0, wL12),
               width(indbpi, p0_3, wL0, wL12))
        wl4 = (width(indzp, p0_4, wL0, wL12),
               width(indzm, p0_4, wL0, wL12),
               width(indzpi, p0_4, wL0, wL12))

        # Generate Doppler- (and, with wL > 0, collision-) broadened spectra
        spectra_data = self.generate_spectra_data(
            r3pe, r3pf, r3me, r3mf, r3pie, r3pif, D3,
            r4pe, r4pf, r4me, r4mf, r4pie, r4pif, D4,
            wl3=wl3, wl4=wl4
        )

        # Return comprehensive results
        return {
            'spectra_data': spectra_data,
            'energy_levels': {
                'W3S': W3S, 'mf3S': mf3S,
                'W3P': W3P, 'mf3P': mf3P,
                'W4S': W4S, 'mf4S': mf4S,
                'W4P': W4P, 'mf4P': mf4P
            },
            'transitions': {
                'he3': {
                    'plus': {'energies': r3pe, 'forces': r3pf, 'ind_lower': indap, 'ind_upper': indbp},
                    'minus': {'energies': r3me, 'forces': r3mf, 'ind_lower': indam, 'ind_upper': indbm},
                    'pi': {'energies': r3pie, 'forces': r3pif, 'ind_lower': indapi, 'ind_upper': indbpi}
                },
                'he4': {
                    'plus': {'energies': r4pe, 'forces': r4pf, 'ind_lower': indyp, 'ind_upper': indzp},
                    'minus': {'energies': r4me, 'forces': r4mf, 'ind_lower': indym, 'ind_upper': indzm},
                    'pi': {'energies': r4pie, 'forces': r4pif, 'ind_lower': indypi, 'ind_upper': indzpi}
                }
            },
            'doppler_widths': {'D3': D3, 'D4': D4},
            'energy_offsets': {'eC1': eC1, 'he4_offset': he4_offset}
        }

    def calculate_spectra(self, B, Temp=300):
        """
        Calculate only the Doppler-broadened spectra (for backward compatibility).
        For full results including energy levels and transitions, use calculate_full_results().
        """
        full_results = self.calculate_full_results(B, Temp)
        return full_results['spectra_data']

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
                              r4pe, r4pf, r4me, r4mf, r4pie, r4pif, D4,
                              wl3=None, wl4=None):
        """Generate broadened spectra data.

        wl3 / wl4 are (plus, minus, pi) tuples of per-transition Lorentz FWHM
        in GHz. Left as None, or all zero, every line is the plain Doppler
        Gaussian this method produced before pressure was supported.
        """
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

        # A line's shape is K(x, y) with wG = 2 sqrt(ln2) D, which is exactly
        # exp(-((nu - nu0)/D)^2) when its Lorentz width is zero.
        root_ln2 = np.sqrt(np.log(2.0))

        # K depends only on |detuning| once wG and wL are fixed, so it is
        # tabulated per distinct wL and interpolated, as the Fortran does with
        # its tabVoigt0/tabVoigt12 arrays. The grid has to reach several
        # hundred GHz to cover every line pair, which a uniform fine step would
        # make more expensive than evaluating K outright, so it is fine across
        # the core and coarse out in the wings, where K falls off as 1/x^2 and
        # is smooth. Interpolation error stays near 1e-5 of the peak.
        CORE_REACH, CORE_STEP, TAIL_STEP = 30.0, 0.01, 0.1

        def voigt_table(wG, wL, reach):
            core = np.arange(0.0, min(reach, CORE_REACH) + CORE_STEP, CORE_STEP)
            if reach > CORE_REACH:
                tail = np.arange(core[-1] + TAIL_STEP, reach + TAIL_STEP, TAIL_STEP)
                grid = np.concatenate((core, tail))
            else:
                grid = core
            return grid, voigt_K(grid, wG, wL)

        def accumulate(spectrum, energies, forces, widths, D, offset):
            wG = 2.0 * root_ln2 * D
            tables = {}
            if len(energies):
                # Cover every line, so np.interp never runs off the end of the
                # table and clamps to a non-zero tail value
                centres = np.asarray(energies, dtype=float) + offset
                reach = float(np.max(np.abs(
                    np.subtract.outer(freq_range[[0, -1]], centres)))) + TAIL_STEP
            for i in range(len(energies)):
                wL = 0.0 if widths is None else float(widths[i])
                detuning = freq_range - energies[i] - offset
                if wL <= 0.0:
                    # Analytically this is voigt_K(detuning, wG, 0), but
                    # evaluated in the original order so that zero-pressure
                    # output stays bit-identical to the pre-Voigt module.
                    spectrum += forces[i] * np.exp(-(detuning / D) ** 2)
                    continue
                key = round(wL, 12)
                if key not in tables:
                    tables[key] = voigt_table(wG, wL, reach)
                grid, values = tables[key]
                spectrum += forces[i] * np.interp(np.abs(detuning), grid, values)

        wl3 = wl3 if wl3 is not None else (None, None, None)
        wl4 = wl4 if wl4 is not None else (None, None, None)

        # Calculate spectra for He3
        accumulate(he3_plus, r3pe, r3pf, wl3[0], D3, 40)
        accumulate(he3_minus, r3me, r3mf, wl3[1], D3, 40)
        accumulate(he3_pi, r3pie, r3pif, wl3[2], D3, 40)

        # Calculate spectra for He4
        accumulate(he4_plus, r4pe, r4pf, wl4[0], D4, 0)
        accumulate(he4_minus, r4me, r4mf, wl4[1], D4, 0)
        accumulate(he4_pi, r4pie, r4pif, wl4[2], D4, 0)

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