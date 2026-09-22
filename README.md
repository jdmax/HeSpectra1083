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

## Author
Written in 2025 by [J. Maxwell](https://orcid.org/0000-0003-2710-4646), based on Fortran by P.J. Nacher.

<a id="1">[1]</a> 
E. Courtade et. al.  "Magnetic field effects on the 1 083 nm atomic line of helium"
EPJD Volume 21, pages 25–55, (2002). [https://doi.org/10.1140/epjd/e2002-00176-1](https://doi.org/10.1140/epjd/e2002-00176-1)
