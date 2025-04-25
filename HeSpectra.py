import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Main function that calculates the structure and spectra of 1083 nm transitions in He3 and He4.
    It prompts for field value and temperature (for Doppler widths) and outputs results.
    """
    # Constants
    zero = 0.0
    epsilon = 1e-12
    un = 1.0
    dx = 2.0
    tx = 3.0
    sx = 6.0
    r2 = np.sqrt(dx)
    r3 = np.sqrt(tx)
    r6 = np.sqrt(sx)

    # Fine structure, He4
    e14 = 2291.175e-3  # J=1 level in GHz (e2=reference)
    e04 = e14 + 29616.950e-3  # J=0 level in GHz (e2=reference)
    delta = 6.1431e4  # singlet-triplet gap
    eM = -17.037  # singlet-triplet mixing
    ep14 = e14 + eM ** 2 / (delta - e14)  # diagonal parameter

    # Fine structure, He3
    f01 = -0.841e-3  # m-dependent contribution to Hfs splittings
    f12 = 2.294e-3  # from Hinds85
    mratio = 1.81921204 / 1.37074562  # µ/M ratios, DrakeHB
    e013 = e04 - e14 + (mratio - 1.0) * f01  # Hinds85
    e13 = e14 + (mratio - 1.0) * f12  # Hinds85
    e03 = e13 + e013
    ep13 = e13 + eM ** 2 / (delta - e13)  # diagonal parameter

    # Hyperfine structure, 23S
    aS = -6739.701177e-3 * (2.0 / 3.0)  # from Rosner70 (*2/3)

    # Hyperfine structure, 23P
    # Optimal adjustment values at 1T to the full Hamiltonian (including singlet)
    c = -4283.026e-3  # GHz (0.019% lower than CPres)
    d = -14.507e-3  # GHz (3.4% higher than DPres/2)
    e = 1.4861e-3  # GHz (4.6% higher than EPres/5)

    # Miscellaneous constants
    mu = 13996.24189e-3  # Bohr magneton in GHz/T i.e., µB/hbar from DrakeHB
    gsS = 2.002237319
    gsP = 2.002238838
    gl3 = 0.999827935
    gl4 = 0.999873626
    gi = 0.0023174823

    D2C9 = 810.599e-3  # gap C9-D2 in B=0 Shiner95

    # Matrix P4 for |J,mJ> coupling
    P4 = np.zeros((9, 9))
    P4[0, 0] = un
    P4[1, 1] = un / r2
    P4[1, 3] = un / r2
    P4[2, 2] = un / r6
    P4[2, 4] = dx / r6
    P4[2, 6] = un / r6
    P4[3, 5] = un / r2
    P4[3, 7] = un / r2
    P4[4, 8] = un
    P4[5, 1] = -un / r2
    P4[5, 3] = un / r2
    P4[6, 2] = -un / r2
    P4[6, 6] = un / r2
    P4[7, 5] = -un / r2
    P4[7, 7] = un / r2
    P4[8, 2] = un / r3
    P4[8, 4] = -un / r3
    P4[8, 6] = un / r3
    invP4 = np.linalg.inv(P4)

    # Hfs for He4
    Hf4P = np.zeros((9, 9))
    Hf4P[5, 5] = e14
    Hf4P[6, 6] = e14  # fine structure term in coupled basis |J,mj>
    Hf4P[7, 7] = e14
    Hf4P[8, 8] = e04
    # Compute invP.Hf.P
    Hf4P = invP4 @ Hf4P @ P4

    # Zeeman term for He4
    Hzee4P = np.zeros((9, 9))
    Hzee4P[0, 0] = gl4 + gsP
    Hzee4P[1, 1] = gsP
    Hzee4P[2, 2] = -gl4 + gsP
    Hzee4P[3, 3] = gl4
    Hzee4P[5, 5] = -gl4
    Hzee4P[6, 6] = gl4 - gsP
    Hzee4P[7, 7] = -gsP
    Hzee4P[8, 8] = -gl4 - gsP

    # Reduced matrix F6/As for He3
    F6red = np.zeros((6, 6))
    F6red[0, 0] = un / dx
    F6red[1, 3] = un / r2
    F6red[2, 2] = -un / dx
    F6red[2, 4] = un / r2
    F6red[3, 1] = un / r2
    F6red[3, 3] = -un / dx
    F6red[4, 2] = un / r2
    F6red[5, 5] = un / dx

    # 23S state for He3
    Hhf3S = aS * F6red

    # Zeeman term for 23S He3
    Hzee3S = np.zeros((6, 6))
    Hzee3S[0, 0] = gsS + gi / dx
    Hzee3S[1, 1] = gi / dx
    Hzee3S[2, 2] = -gsS + gi / dx
    Hzee3S[3, 3] = gsS - gi / dx
    Hzee3S[4, 4] = -gi / dx
    Hzee3S[5, 5] = -gsS - gi / dx

    # 23P state for He3
    e0 = e03
    e1 = e13  # true energy; e2=0

    # Hfs for He3
    Hf3P = np.zeros((9, 9))
    Hf3P[5, 5] = e1
    Hf3P[6, 6] = e1  # fine structure in coupled basis |J,mj>
    Hf3P[7, 7] = e1
    Hf3P[8, 8] = e0
    # Compute invP.Hf.P
    Hf3P = invP4 @ Hf3P @ P4

    # Hyperfine structure for 23P He3
    Hcor = np.zeros((18, 18))
    Hcor[0, 0] = d + dx * e
    Hcor[1, 1] = -4.0 * e
    Hcor[2, 2] = -d + dx * e
    Hcor[3, 1] = tx * e
    Hcor[3, 3] = d
    Hcor[4, 2] = -tx * e
    Hcor[5, 5] = -d
    Hcor[6, 4] = tx * e
    Hcor[6, 6] = d - dx * e
    Hcor[7, 5] = -tx * e
    Hcor[7, 7] = 4.0 * e
    Hcor[8, 8] = -d - dx * e
    Hcor[9, 1] = r2 * (d + tx * e)
    Hcor[9, 3] = -r2 * e
    Hcor[10, 2] = r2 * (d - tx * e)
    Hcor[10, 4] = r2 * dx * e
    Hcor[11, 5] = -r2 * e
    Hcor[12, 2] = r2 * sx * e
    Hcor[12, 4] = r2 * d
    Hcor[12, 6] = -r2 * e
    Hcor[13, 5] = r2 * d
    Hcor[13, 7] = r2 * dx * e
    Hcor[14, 8] = -r2 * e
    Hcor[15, 5] = r2 * sx * e
    Hcor[15, 7] = r2 * (d - tx * e)
    Hcor[16, 8] = r2 * (d + tx * e)

    # Apply diagonal symmetry
    for i in range(1, 18):
        for j in range(i + 1, 18):
            Hcor[i, j] = Hcor[j, i]

    # Apply second diagonal symmetry
    for i in range(9):
        for j in range(9):
            Hcor[18 - j - 1, 18 - i - 1] = Hcor[j, i]

    # Calculate H3PB0
    H3PB0 = np.copy(Hcor)
    for i in range(9):
        for j in range(9):
            H3PB0[i, j] += Hf3P[i, j]
            H3PB0[i + 9, j + 9] += Hf3P[i, j]

    for i in range(6):
        for j in range(6):
            H3PB0[3 * i, 3 * j] += c * F6red[i, j]
            H3PB0[3 * i + 1, 3 * j + 1] += c * F6red[i, j]
            H3PB0[3 * i + 2, 3 * j + 2] += c * F6red[i, j]

    # Zeeman term for 23P He3
    Hzee3P = np.zeros((18, 18))
    Hzee3P[0, 0] = gl3 + gsP
    Hzee3P[1, 1] = gsP
    Hzee3P[2, 2] = -gl3 + gsP
    Hzee3P[3, 3] = gl3
    Hzee3P[5, 5] = -gl3
    Hzee3P[6, 6] = gl3 - gsP
    Hzee3P[7, 7] = -gsP
    Hzee3P[8, 8] = -gl3 - gsP

    for i in range(9):
        Hzee3P[i + 9, i + 9] = Hzee3P[i, i] - gi / dx  # assign second block
        Hzee3P[i, i] = Hzee3P[i, i] + gi / dx  # modify first block

    # Get user input
    B, Temp = get_user_input()
    D3 = 1.1875 * np.sqrt(Temp / 300)  # Doppler width for He3
    D4 = D3 * np.sqrt(3.0 / 4.0)  # Doppler width for He4

    # Calculate for He3
    H3S = Hhf3S + mu * B * Hzee3S
    W3S, V3S = np.linalg.eigh(H3S)
    # Sort eigenvalues and eigenvectors
    idx = W3S.argsort()
    W3S = W3S[idx]
    V3S = V3S[:, idx]

    H3P = H3PB0 + mu * B * Hzee3P
    W3P, V3P = np.linalg.eigh(H3P)
    # Sort eigenvalues and eigenvectors
    idx = W3P.argsort()
    W3P = W3P[idx]
    V3P = V3P[:, idx]

    # Write energy levels for He3
    with open("Ai.dat", "w") as f:
        for i in range(6):
            f.write(f"{W3S[i]:.8f} ")
    with open("Bj.dat", "w") as f:
        for i in range(18):
            f.write(f"{W3P[i]:.8f} ")

    # Calculate transition matrix elements for He3
    T3p = calculate_sigma_plus_he3(V3S, V3P)
    T3m = calculate_sigma_minus_he3(V3S, V3P)
    T3pi = calculate_pi_he3(V3S, V3P)

    # Sort and store transitions for He3
    r3pe, r3pf, indap, indbp = sort_transitions(T3p, W3P, W3S, epsilon)
    r3me, r3mf, indam, indbm = sort_transitions(T3m, W3P, W3S, epsilon)
    r3pie, r3pif, indapi, indbpi = sort_transitions(T3pi, W3P, W3S, epsilon)

    eC1 = W3P[8] - W3S[5]  # Reference for energy offsets

    # Write transitions for He3
    write_transitions("He3plus.dat", B, r3pe, r3pf, indap, indbp, eC1)
    write_transitions("He3moins.dat", B, r3me, r3mf, indam, indbm, eC1)
    write_transitions("He3pi.dat", B, r3pie, r3pif, indapi, indbpi, eC1)

    # Calculate for He4
    W4S = np.zeros(3)
    W4S[2] = mu * B * gsS  # Fixed bug from original code
    W4S[1] = zero
    W4S[0] = -mu * B * gsS  # Fixed bug from original code
    V4S = np.zeros((3, 3))
    V4S[0, 0] = un
    V4S[1, 1] = un
    V4S[2, 2] = un

    H4P = Hf4P + mu * B * Hzee4P
    W4P, V4P = np.linalg.eigh(H4P)
    # Sort eigenvalues and eigenvectors
    idx = W4P.argsort()
    W4P = W4P[idx]
    V4P = V4P[:, idx]

    # Write energy levels for He4
    with open("Yi.dat", "w") as f:
        for i in range(3):
            f.write(f"{W4S[i]:.8f} ")
    with open("Zi.dat", "w") as f:
        for i in range(9):
            f.write(f"{W4P[i]:.8f} ")

    # Calculate transition matrix elements for He4
    T4p = calculate_sigma_plus_he4(V4S, V4P)
    T4m = calculate_sigma_minus_he4(V4S, V4P)
    T4pi = calculate_pi_he4(V4S, V4P)

    # Sort and store transitions for He4
    r4pe, r4pf, indyp, indzp = sort_transitions(T4p, W4P, W4S, epsilon)
    r4me, r4mf, indym, indzm = sort_transitions(T4m, W4P, W4S, epsilon)
    r4pie, r4pif, indypi, indzpi = sort_transitions(T4pi, W4P, W4S, epsilon)

    eD2 = W4P[2] - W4S[1]
    C1C9 = (W3P[16] - W3S[2]) - eC1  # C9-C1 transition energy difference

    # Write transitions for He4
    write_transitions("He4plus.dat", B, r4pe, r4pf, indyp, indzp, eD2 + D2C9 - C1C9)
    write_transitions("He4moins.dat", B, r4me, r4mf, indym, indzm, eD2 + D2C9 - C1C9)
    write_transitions("He4pi.dat", B, r4pie, r4pif, indypi, indzpi, eD2 + D2C9 - C1C9)

    # Generate Doppler-broadened spectra
    generate_spectra(r3pe, r3pf, r3me, r3mf, r3pie, r3pif, D3,
                     r4pe, r4pf, r4me, r4mf, r4pie, r4pif, D4)


def get_user_input():
    """Get magnetic field and temperature values from user"""
    print("Version of April 25, 2025, bugs fixed")
    print("(unsorted indices for Yis and Zjs in He4pi.dat, and wrong order of Yis)")
    print("")
    print("Field (Tesla), Temperature (K)?  (Doppler width He3: 1.1875 GHz @300 K)")
    B = float(input("Field (Tesla): "))
    Temp = float(input("Temperature (K): "))
    return B, Temp


def calculate_sigma_plus_he3(V3S, V3P):
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


def calculate_sigma_minus_he3(V3S, V3P):
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


def calculate_pi_he3(V3S, V3P):
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


def calculate_sigma_plus_he4(V4S, V4P):
    """Calculate sigma+ transition matrix elements for He4"""
    T4p = np.zeros((9, 3))
    for j in range(9):
        for i in range(3):
            T4p[j, i] = (V4S[0, 3 - i - 1] * V4P[0, j] +
                         V4S[1, 3 - i - 1] * V4P[3, j] +
                         V4S[2, 3 - i - 1] * V4P[6, j]) ** 2
    return T4p


def calculate_sigma_minus_he4(V4S, V4P):
    """Calculate sigma- transition matrix elements for He4"""
    T4m = np.zeros((9, 3))
    for j in range(9):
        for i in range(3):
            T4m[j, i] = (V4S[0, 3 - i - 1] * V4P[2, j] +
                         V4S[1, 3 - i - 1] * V4P[5, j] +
                         V4S[2, 3 - i - 1] * V4P[8, j]) ** 2
    return T4m


def calculate_pi_he4(V4S, V4P):
    """Calculate pi transition matrix elements for He4"""
    T4pi = np.zeros((9, 3))
    for j in range(9):
        for i in range(3):
            T4pi[j, i] = (V4S[0, 3 - i - 1] * V4P[1, j] +
                          V4S[1, 3 - i - 1] * V4P[4, j] +
                          V4S[2, 3 - i - 1] * V4P[7, j]) ** 2
    return T4pi


def sort_transitions(T, WP, WS, epsilon):
    """Sort transitions by energy and filter by magnitude"""
    # Count transitions above epsilon threshold
    count = 0
    for j in range(T.shape[0]):
        for i in range(T.shape[1]):
            if T[j, i] >= epsilon:
                count += 1

    # Create arrays to store filtered transitions
    re = np.zeros(count)
    rf = np.zeros(count)
    ind_lower = np.zeros(count, dtype=int)
    ind_upper = np.zeros(count, dtype=int)

    # Extract transitions above threshold
    idx = 0
    for j in range(T.shape[0]):
        for i in range(T.shape[1]):
            if T[j, i] >= epsilon:
                re[idx] = WP[j] - WS[i]
                rf[idx] = T[j, i]
                ind_lower[idx] = i
                ind_upper[idx] = j
                idx += 1

    # Sort transitions by energy
    sort_idx = np.argsort(re)
    return re[sort_idx], rf[sort_idx], ind_lower[sort_idx], ind_upper[sort_idx]


def write_transitions(filename, B, re, rf, ind_lower, ind_upper, energy_offset=0.0):
    """Write transition data to file"""
    with open(filename, "w") as f:
        for i in range(len(re)):
            f.write(f"{B:10.5f} {re[i] - energy_offset:15.8e} {rf[i]:15.8e} {ind_lower[i]:3d} {ind_upper[i]:3d}\n")


def generate_spectra(r3pe, r3pf, r3me, r3mf, r3pie, r3pif, D3,
                     r4pe, r4pf, r4me, r4mf, r4pie, r4pif, D4):
    """Generate Doppler-broadened spectra for both isotopes"""
    # Generate frequency axis
    freq_range = np.arange(-200, 300.1, 0.1)

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

    # Write He3 spectra to file
    with open("spHe3.dat", "w") as f:
        f.write(" GHz sigplus sigminus pi\n")
        for i in range(len(freq_range)):
            f.write(f"{freq_range[i] - 40:15.8f} {he3_plus[i]:15.8f} {he3_minus[i]:15.8f} {he3_pi[i]:15.8f}\n")

    # Write He4 spectra to file
    with open("spHe4.dat", "w") as f:
        f.write(" GHz sigplus sigminus pi\n")
        for i in range(len(freq_range)):
            f.write(f"{freq_range[i]:15.8f} {he4_plus[i]:15.8f} {he4_minus[i]:15.8f} {he4_pi[i]:15.8f}\n")

    # Create plots of the spectra
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(freq_range - 40, he3_plus, label='σ+')
    plt.plot(freq_range - 40, he3_minus, label='σ-')
    plt.plot(freq_range - 40, he3_pi, label='π')
    plt.title('He3 Spectra')
    plt.xlabel('Frequency offset (GHz)')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freq_range, he4_plus, label='σ+')
    plt.plot(freq_range, he4_minus, label='σ-')
    plt.plot(freq_range, he4_pi, label='π')
    plt.title('He4 Spectra')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Intensity')
    plt.legend()

    plt.tight_layout()
    plt.savefig('helium_spectra.png')
    plt.show()


if __name__ == "__main__":
    main()