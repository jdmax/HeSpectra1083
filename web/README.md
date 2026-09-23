# Static (browser) build

The same calculator as the Streamlit app, with no server behind it.
`helium_spectra_calc.py` runs unchanged in the browser under
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

`bridge.py` is a direct port of the presentation half of
`helium_spectra_ui.py` with Streamlit, pandas and plotly removed. The physics
module is imported as-is and is never duplicated.

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

## Verifying against the Streamlit app

`bridge.py` was checked against `helium_spectra_ui.py` over
B = 0.0001 / 0.5 / 1 / 4 / 6.5 T and T = 77 / 300 / 450 / 1000 K, for both
isotopes and both x-axes: spectra, transition table and level-diagram
geometry all match exactly. The same cases were then run under Pyodide and
compared with CPython — every displayed string is identical and floats agree
to about 1e-13, the difference being LAPACK rounding between the WebAssembly
and native NumPy builds.

## Reading a peak's membership

A row in the transitions table is a *group* of lines, not a single one, and
`group_transitions()` builds those groups by single linkage: a line joins if it
is within 2 GHz of the **previous** member, not of the group's centre. Groups
can therefore chain out to any width, and the averaged frequency the row shows
does not reveal that. Three things in the page exist to make it visible:

- **Span (GHz)** in the table is the group's full extent. It is highlighted
  when it exceeds the Doppler width, i.e. when the members have not actually
  merged into one peak. The Doppler width for the current temperature is
  printed above the table next to the grouping threshold.
- **The shaded band** on the spectra plot covers the selected group's extent,
  against the peak it is named for.
- **Hovering** a transition arrow on the level diagram gives that single line's
  frequency, wavelength, intensity, its share of the group, and its gap from
  the previous line — the quantity the grouping actually tested. Hovering a
  level bar gives its label, m_F and energy on its own manifold's scale.

A worked example: above **B = 4.039 T** an A₅ line joins the strong σ⁻ group.
It carries about 0.01% of the group's intensity and sits ~1.9 GHz from the
nearest strong line. The threshold is fixed in frequency while the Doppler
width scales with temperature, so that 4.039 T figure is identical at 77 K and
at 600 K — the membership change is an artefact of the grouping, not physics.

## Notes

- Row order in the transitions table is now deterministic. Many groups are
  exactly equal in intensity by symmetry, and the Streamlit version sorted
  them with pandas' (unstable) quicksort, so tied rows could come out in a
  different order on a different machine. `bridge.py` breaks ties explicitly
  by polarization then frequency.
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
- Recalculation costs about 25 ms, so the plots follow the sliders directly
  rather than through a server round trip.
