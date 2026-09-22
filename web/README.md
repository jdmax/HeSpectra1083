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
  chroma, colour-vision separation and contrast. The selection marker is
  achromatic in both modes so it reads as an annotation rather than a fourth
  series; selected table rows are marked in their own polarization colour,
  matching the level-diagram arrows.
- Recalculation costs about 25 ms, so the plots follow the sliders directly
  rather than through a server round trip.
