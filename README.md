# 1083 nm Helium Spectra 

Spectrum Calculation for 3He and 4He Near 1083 nm with Zeeman Splitting, Doppler Broadening based on Fortran code by P.J. Nacher[[1]](#1) , [LKB, ENS Paris.](https://www.lkb.fr/polarisedhelium/)

## Interfaces

All three share the calculation in `helium_spectra_calc.py`.

| | |
| --- | --- |
| `helium_spectra.py` | Command line; writes spectra and level files |
| `helium_spectra_ui.py` | Streamlit app: `streamlit run helium_spectra_ui.py` |
| [`web/`](web/) | The same interface as a static page, needing no server |

The static build runs `helium_spectra_calc.py` in the browser under Pyodide,
so it can be served from any plain file host — see [web/README.md](web/README.md).
It is published at **<https://jdmax.github.io/HeSpectra1083/>**.

## Line shape

`helium_spectra_calc.py` builds each line as a Voigt profile, following
P.J. Nacher's `spectreVoigt_w0w12` Fortran: a Doppler width from the molar
mass, and collisional broadening from two Lorentz widths (wL0 for the 2³P₀
lines, wL12 for the rest). `calculate_full_results(B, Temp, wL0, wL12)` takes
the Lorentz FWHMs in GHz, as that program's prompts do; leaving them at zero
gives the Doppler-only Gaussian. Typical rates are 12.0 MHz/mbar for ³He and
10.4 for ⁴He[[2]](#2).

## Author
Written in 2025 by [J. Maxwell](https://orcid.org/0000-0003-2710-4646), based on Fortran by P.J. Nacher.

<a id="1">[1]</a> 
E. Courtade et. al.  "Magnetic field effects on the 1 083 nm atomic line of helium"
EPJD Volume 21, pages 25–55, (2002). [https://doi.org/10.1140/epjd/e2002-00176-1](https://doi.org/10.1140/epjd/e2002-00176-1)

<a id="2">[2]</a>
A. Nikiel-Osuchowska et al. "Metastability exchange optical pumping of ³He gas up to hundreds of millibars at 4.7 Tesla"
EPJD Volume 67, 200 (2013). [https://doi.org/10.1140/epjd/e2013-40153-y](https://doi.org/10.1140/epjd/e2013-40153-y)
