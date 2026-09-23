# Static (browser) build

The calculator's main interface, with no server behind it. It began as a
port of the Streamlit app (`helium_spectra_ui.py`), which is being
deprecated: the Streamlit version has neither pressure broadening nor the
centroid grouping described below. `helium_spectra_calc.py` runs in the
browser under
[Pyodide](https://pyodide.org) (CPython compiled to WebAssembly); Plotly.js
draws the figures. Because everything runs on the visitor's machine, this can
be served from any plain static file host — a `public_html` directory, for
example.

## Files

| File | Role |
| --- | --- |
| `index.html` | Page structure and controls |
| `style.css` | Layout |
| `app.js` | Widgets, Plotly.js figures, transitions table |
| `bridge.py` | Runs in Pyodide; wraps `helium_spectra_calc.py` and returns plain data |
| `deploy.sh` | Copies the above plus `helium_spectra_calc.py` into a target directory |

`bridge.py` holds the presentation logic: grouping lines into peaks and
formatting the table and diagram. The physics module is imported as-is and is
never duplicated.

## Running locally

Serve the directory over HTTP — opening `index.html` as a `file://` URL will
not work, because the page fetches its Python sources:

```sh
cd web
python -m http.server 8000    # or: py -m http.server 8000
```

Then open <http://localhost:8000/>. In this layout `helium_spectra_calc.py` is
fetched from the repository root one level up, so nothing needs copying during
development.

## Deploying

### GitHub Pages

`.github/workflows/pages.yml` publishes the site on every push to `master`
that touches `web/` or `helium_spectra_calc.py`, and can also be run by hand
from the Actions tab. It calls the same `deploy.sh` used below, so there is
one definition of what gets deployed.

This needs to be enabled once, in **Settings → Pages → Build and deployment →
Source: GitHub Actions**. The site is then served at
<https://jdmax.github.io/HeSpectra1083/>.

### Any other static host

```sh
./deploy.sh ~/public_html/hespectra
```

That copies `index.html`, `style.css`, `app.js`, `bridge.py` and
`helium_spectra_calc.py` into one self-contained directory. No build step, no
Python process on the server, nothing to keep running. The page uses only
relative paths, so it works at a domain root or under a subdirectory.

## External dependencies

The page loads two things from jsDelivr:

- Pyodide `v0.29.5` plus the NumPy wheel — roughly 10–15 MB on a first visit
- Plotly.js `3.0.1` — about 4.5 MB

Both are cached by the browser afterwards, and both versions are pinned:
Pyodide in `PYODIDE_VERSION` at the top of `app.js`, Plotly in the `<script>`
tag in `index.html`.

### If jsDelivr is unreachable

Vendor both locally and the page will have no external dependencies at all:

```sh
cd ~/public_html/hespectra
curl -O https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.0.1/plotly.min.js
# Pyodide: take pyodide.mjs, pyodide.asm.js, pyodide.asm.wasm,
# python_stdlib.zip, pyodide-lock.json and the numpy wheel from
# https://github.com/pyodide/pyodide/releases into ./pyodide/
```

Then point the `<script>` tag at `plotly.min.js` and set `PYODIDE_URL` in
`app.js` to `'pyodide/'`.

## Verification

Two tests in [`test/`](../test/) need only numpy:

- `test_against_fortran.py` checks the physics against P.J. Nacher's Fortran
  (see [Checking it](#checking-it) below).
- `test_grouping.py` checks the centroid grouping over a field and
  temperature scan (see [Reading a peak's membership](#reading-a-peaks-membership)).

The page's output was also run under Pyodide and compared with CPython across
both isotopes, both x-axes and a range of fields and temperatures: every
displayed string is identical and floats agree to about 1e-13, the difference
being LAPACK rounding between the WebAssembly and native NumPy builds.

## Pressure broadening

The pressure control adds **collisional** broadening, which is Lorentzian, not
Doppler. Combined with the Gaussian Doppler profile it gives a Voigt. This
follows P.J. Nacher's `spectreVoigt_w0w12` Fortran and lives in
`helium_spectra_calc.py`, so the command-line and web interfaces share it.

That matters because Lorentzian wings fall off as 1/Δν² rather than
exp(−Δν²). A line that is negligible at a given separation under pure Doppler
broadening can contribute far more once the gas is at pressure — for the
σ⁻ spectrum at 6 T, the signal midway between the strong peak and the A₅/A₆
probe peak rises by a factor of ~130 between 0 and 100 mbar.

### What came from the Fortran

- **The line shape.** Nacher tabulates
  K(x, y) = (y/π) ∫ exp(−z²)/(y² + (x−z)²) dz with
  x = 2√(ln2)(ν−ν₀)/wG and y = √(ln2)·wL/wG, integrating over z ∈ [−7, 7] by
  Simpson quadrature. That integral is Re[w(x+iy)], so `voigt_K()` evaluates
  the Faddeeva function directly — same quantity, no truncation at |z| = 7.
  Checked against a literal transcription of his `funcV`/`qsimp`: agreement to
  5×10⁻¹⁰ over wL from 0.012 to 3.2 GHz. K(x, 0) = exp(−x²), so zero pressure
  reproduces the Doppler-only Gaussian exactly.
- **The Doppler width**, wG = √(2RT/M)/λ · 2√(ln2), from the molar mass. This
  replaces `1.1875·√(T/300)` and `D3·√(3/4)`: **He3 widths shift by +0.012%
  and He4 by +0.247%**, the He4 change because the exact mass ratio is
  √(M₃/M₄) = 0.86805, not √(3/4) = 0.86603.
- **Two Lorentz widths**, wL0 for the 2³P₀ lines and wL12 for the rest, and
  the rates his prompts quote: 12.0 MHz/mbar for ³He, 10.4 for ⁴He.
- **Tabulate-and-interpolate**, as his `tabVoigt0`/`tabVoigt12` arrays do.
  The grid is fine across the core and coarse in the wings, where K falls off
  as 1/x²; interpolation error stays near 1×10⁻⁵ of the peak and the Voigt
  path costs about the same as the old Gaussian one.

### Where it departs

The Fortran assigns each line to wL0 or wL12 by its **row in the input file**,
hardcoded as the first three for ³He and the first for ⁴He. That presumes J is
a good quantum number. Here the spectra are computed at arbitrary field, and
at several tesla J is thoroughly mixed: at 6 T the J=0 character of the ³He
2³P manifold is spread over six states at weights of 0.23 to 0.47. So
`line_widths()` interpolates each transition's Lorentz width by its upper
state's J=0 admixture, from `j0_weights()`.

At 0.13 T, where the `spectre*.dat` fixtures were produced, the two agree for
five of the six files. They differ on the π list: four transitions reach the
2³P₀ doublet there, not three, so the Fortran's hardcoded row range leaves one
out (A₆ → B₁₇ at 29.737 GHz). It makes no difference while wL0 and wL12 are
equal, as the quoted rates make them — the plumbing is there for when they
differ.

### Checking it

[`test/test_against_fortran.py`](../test/test_against_fortran.py) needs only
numpy and compares every line's position, strength and level indices against
the `spectre*.dat` files, plus the Voigt shape against a transcription of
`funcV`/`qsimp` and the Doppler width against the `wG` formula. Current
agreement: **6×10⁻⁷ GHz** on positions, **2×10⁻⁸** on strengths — both limited
by the files' 8-digit formatting — and **2×10⁻⁸** relative on K(x, y).

### What this does not do

It broadens the line *shape* only. The line positions and strengths come from
a model valid to a few mbar: collisional shifts (~1.4 MHz/mbar) and any line
mixing among the 2³P sublevels are not included. Above a few mbar the widths
are right, but the positions are still the low-pressure ones.

## Reading a peak's membership

A row in the transitions table is a *peak*: a group of lines, not a single
one. `group_transitions()` builds peaks around **intensity-weighted
centroids**. Lines are taken strongest first; each joins the nearest peak
whose centroid lies within 2 GHz, and the centroid is recomputed, while a
line with no peak in reach starts its own. Strong lines therefore define the
peaks, and weak ones cannot drag a peak's centre or stretch it outwards. The
frequency and wavelength in the row are that centroid.

This replaced single linkage, where a line joined if it was within 2 GHz of
the *previous* member, so groups could chain out to any width. At 5 T an A₅→B₁₂
line with 0.01% of the peak's intensity was chained onto the strong σ⁻ peak
from 2.5 GHz off its centre, reporting a 3.2 GHz span for four lines that
really span 1.27 GHz. Under the centroid rule it is a peak of its own. Across
a scan of 0.05–7 T and 77–1000 K, no member ends further than 2.0 GHz from its
peak's final centroid.

The threshold stays fixed rather than growing with pressure. Pressure
broadening does blend lines, but it acts through the Lorentzian wings of
*strong* lines, often far off: pumping the strong σ⁻ peak at 5 T and
100 mbar empties A₅ at ~0.5% of the pumped rate, and 98% of that comes from
the wing of the A₅→B₁₃ probe line 13.8 GHz away. No peak definition would
capture that, so the hover's per-line intensities are the tool for it.

Three things in the page show what a peak contains:

- **Span (GHz)** in the table is the peak's full extent. It is highlighted
  when it exceeds the line width, Doppler or Voigt, i.e. when the members have
  not actually merged. That width is printed above the table.
- **The shaded band** on the spectra plot covers the selected peak's extent.
- **Hovering** a transition arrow on the level diagram gives that single line's
  frequency, wavelength, intensity, its share of the peak, and its offset from
  the centroid. Hovering a level bar gives its label, m_F and energy on its own
  manifold's scale.

## Notes

- Row order and peak membership are deterministic across machines. Many
  strengths are exactly equal by symmetry and LAPACK's last-bit noise differs
  between numpy builds, so the sort keys are rounded and ties broken
  explicitly; otherwise the browser and a desktop Python could order or group
  tied lines differently. (The Streamlit version sorted with pandas' unstable
  quicksort and had the same problem.)
- The B slider steps in clean 0.01 T increments. The Streamlit slider had
  `min=0.0001, step=0.01`, which put every stop on the grid 0.0001, 0.0101,
  …, so round values like 4 T were unreachable by dragging. The number box
  still accepts any value from 0.0001 to 10 T.
- The energy level diagram no longer uses the original's
  `margin=dict(t=0,l=0,r=0,b=0)`, which ran the y-axis title into the
  "2³P States" label and clipped the m_F tick labels off the bottom.
- The page follows the operating system's light/dark setting, and the toggle
  at the top of the sidebar overrides it. That choice is remembered per
  browser; until it is used, the OS setting continues to govern, including
  when it changes while the page is open.
- Both modes are separately stepped and validated — the dark colours are not
  the light ones flipped. `PALETTE` at the top of `app.js` holds the chart
  colours for each mode and `style.css` holds the matching page tokens; the
  CSS values are set once per mode in `:root`, under a
  `prefers-color-scheme` block, and under `:root[data-theme="..."]`.
  Every palette was checked against its own background for lightness band,
  chroma, colour-vision separation and contrast.
- The spectra plot's selection marker is achromatic on purpose: it is an
  annotation, not a fourth series. Selected table rows are marked in their own
  polarization colour, matching the level-diagram arrows.
- The energy level bars are a gold pair, chosen for appearance, and this is a
  known trade. Selecting a transition draws arrows in the polarization colour
  inside that same diagram, so the bars are in effect a fourth and fifth
  categorical colour alongside all three series, and no gold pair clears that.
  The values used are the best available in each mode and are ~3.7:1 apart, so
  the two manifolds read clearly. What remains: in light mode the B bars sit at
  normal-vision ΔE 13.0 from σ- (the floor is 15); in dark mode the A bars are
  ΔE 1.2 from the π arrows under deuteranopia and protanopia, meaning those
  viewers see them as one colour. That one is unavoidable on a dark background
  — a bronze dark enough to escape green falls below 3:1 contrast. The A/B
  (Y/Z) labels and the vertical split carry the distinction instead. Setting
  `levelLower` to a neutral (`#848c99` dark, `#6b7280` light) in `app.js`
  removes it if that ever matters.
- Recalculation costs about 30 ms, with or without pressure broadening, so
  the plots follow the sliders directly rather than through a server round
  trip.
