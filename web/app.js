/*
 * Static browser front-end for the helium 1083 nm line calculator.
 *
 * helium_spectra_calc.py runs unchanged inside Pyodide; bridge.py turns its
 * output into plain objects and everything below draws them with Plotly.js.
 * No server, no build step: these files can be copied straight into a
 * public_html directory.
 */

const PYODIDE_VERSION = 'v0.29.5';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`;

// Module search path inside Pyodide's virtual filesystem
const PY_DIR = '/app';

// helium_spectra_calc.py is fetched from the deployment directory if present,
// otherwise from the repository root one level up (so `web/` works in-place).
const CALC_PATHS = ['helium_spectra_calc.py', '../helium_spectra_calc.py'];
const BRIDGE_PATHS = ['bridge.py'];

// Series colours, validated against the --bg surface in style.css for
// lightness band, chroma, CVD separation and contrast.
const SERIES = {
  plus: { label: 'σ+', color: '#3987e5' },
  minus: { label: 'σ-', color: '#e66767' },
  pi: { label: 'π', color: '#008300' },
};

// bridge.py labels each table row with one of the same symbols
const COLOR_BY_POL = Object.fromEntries(
  Object.values(SERIES).map(s => [s.label, s.color]));

// Chart chrome. Plotly.js ships no named templates, so these are literal
// values mirroring the tokens in style.css.
const THEME = {
  text: '#e8e8e5',
  muted: '#9ea3ad',
  grid: '#2f333c',
  levelUpper: '#d95926',   // 2³P states
  levelLower: '#9085e9',   // 2³S states
  // Achromatic on purpose: the marker is an annotation, not a fourth series,
  // and must not be mistaken for one of the curves it sits among.
  marker: '#c3c2b7',
};

const PLOT_CONFIG = { responsive: true, displaylogo: false };

const state = {
  B: 1.0,
  T: 300,
  isotope: 'He3',
  xAxis: 'Frequency Offset',
  selected: null,   // index into the current table, or null
  data: null,       // last result from bridge.compute_js
};

let bridge = null;
let renderQueued = false;

/* ------------------------------------------------------------------ boot */

const bootEl = document.getElementById('boot');
const bootMsg = document.getElementById('boot-msg');

function bootFail(err) {
  bootEl.classList.add('error');
  bootMsg.textContent = String(err && err.message ? err.message : err);
  console.error(err);
}

async function fetchText(paths) {
  for (const path of paths) {
    try {
      const resp = await fetch(path);
      if (resp.ok) return await resp.text();
    } catch (err) {
      /* try the next candidate */
    }
  }
  throw new Error(`Could not load ${paths[0]} (tried: ${paths.join(', ')})`);
}

async function boot() {
  if (location.protocol === 'file:') {
    bootFail(new Error(
      'This page must be served over HTTP, not opened as a file.\n\n' +
      'From the web/ directory run:\n    python -m http.server 8000\n\n' +
      'then open http://localhost:8000/'));
    return;
  }

  try {
    bootMsg.textContent = 'Loading Python runtime…';
    const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
    const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

    bootMsg.textContent = 'Loading NumPy…';
    await pyodide.loadPackage('numpy');

    bootMsg.textContent = 'Loading calculator…';
    const [calcSrc, bridgeSrc] = await Promise.all([
      fetchText(CALC_PATHS),
      fetchText(BRIDGE_PATHS),
    ]);

    pyodide.FS.mkdirTree(PY_DIR);
    pyodide.FS.writeFile(`${PY_DIR}/helium_spectra_calc.py`, calcSrc);
    pyodide.FS.writeFile(`${PY_DIR}/bridge.py`, bridgeSrc);
    pyodide.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(PY_DIR)})`);

    bridge = pyodide.pyimport('bridge');

    bindControls();
    bootEl.hidden = true;
    document.getElementById('app').hidden = false;
    recompute();
  } catch (err) {
    bootFail(err);
  }
}

/* -------------------------------------------------------------- controls */

function bindControls() {
  // Slider and number box edit the same value, as the Streamlit pair did,
  // but here they stay in sync without a round trip.
  linkNumeric('b-slider', 'b-input', 'B', v => v.toFixed(4));
  linkNumeric('t-slider', 't-input', 'T', v => String(Math.round(v)));

  for (const radio of document.querySelectorAll('input[name="isotope"]')) {
    radio.addEventListener('change', () => {
      state.isotope = radio.value;
      state.selected = null;   // indices mean different levels per isotope
      recompute();
    });
  }

  for (const radio of document.querySelectorAll('input[name="xaxis"]')) {
    radio.addEventListener('change', () => {
      state.xAxis = radio.value;
      recompute();
    });
  }
}

function linkNumeric(sliderId, inputId, key, format) {
  const slider = document.getElementById(sliderId);
  const input = document.getElementById(inputId);

  const apply = (raw, echoTo) => {
    const value = Number(raw);
    if (!Number.isFinite(value)) return;
    const min = Number(input.min);
    const max = Number(input.max);
    state[key] = Math.min(Math.max(value, min), max);
    // Echo to the other widget only, so typing is not fought mid-keystroke
    if (echoTo === 'input') input.value = format(state[key]);
    else slider.value = String(state[key]);
    recompute();
  };

  slider.addEventListener('input', () => apply(slider.value, 'input'));
  input.addEventListener('change', () => apply(input.value, 'slider'));

  slider.value = String(state[key]);
  input.value = format(state[key]);
}

/* ------------------------------------------------------------- rendering */

/** Recalculate in Python, then redraw. Coalesced to one call per frame. */
function recompute() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    state.data = bridge.compute_js(state.B, state.T, state.isotope, state.xAxis);
    if (state.selected !== null && state.selected >= state.data.table.length) {
      state.selected = null;
    }
    drawSpectra();
    drawTable();
    drawLevels();
  });
}

/** Redraw only; selection changes need no recalculation. */
function rerender() {
  drawSpectra();
  drawLevels();
  markSelectedRow();
}

function selectedRow() {
  return state.selected === null ? null : state.data.table[state.selected];
}

function drawSpectra() {
  const s = state.data.spectra;
  const line = color => ({ color, width: 2 });

  const traces = ['plus', 'minus', 'pi'].map(key => ({
    x: s.x, y: s[key], mode: 'lines',
    name: SERIES[key].label, line: line(SERIES[key].color),
  }));

  const shapes = [];
  const row = selectedRow();
  if (row) {
    // Both axes are derived from the same absolute frequency, so the table
    // value can be used directly whichever axis is showing.
    const x = Number(state.xAxis === 'Frequency Offset' ? row.frequency : row.wavelength);
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: x, x1: x, y0: 0, y1: 1,
      line: { color: THEME.marker, width: 2, dash: 'dash' },
    });
  }

  const axis = title => ({
    title: { text: title },
    showgrid: true, gridwidth: 1, gridcolor: THEME.grid,
    zerolinecolor: THEME.grid, linecolor: THEME.grid, tickcolor: THEME.grid,
  });

  const layout = {
    title: { text: state.data.title },
    xaxis: axis(s.x_label),
    yaxis: axis('Intensity'),
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: THEME.text },
    modebar: { color: THEME.muted, activecolor: THEME.text, bgcolor: 'rgba(0,0,0,0)' },
    height: 500,
    showlegend: true,
    legend: {
      yanchor: 'top', y: 0.99, xanchor: 'left', x: 0.01,
      bgcolor: 'rgba(0,0,0,0)', font: { color: THEME.text },
    },
    shapes,
    // Keep pan/zoom while sweeping B and T; reset when the axes change meaning
    uirevision: `${state.isotope}|${state.xAxis}`,
  };

  Plotly.react('spectra-plot', traces, layout, PLOT_CONFIG);
}

function drawLevels() {
  const lv = state.data.levels;
  const half = lv.line_width / 2;

  // One trace per manifold, with nulls separating the individual level bars
  const manifold = (levels, color) => {
    const x = [], y = [];
    for (const level of levels) {
      x.push(level.mf - half, level.mf + half, null);
      y.push(level.e, level.e, null);
    }
    return { x, y, mode: 'lines', line: { color }, hoverinfo: 'skip', showlegend: false };
  };

  const annotations = [];
  for (const levels of [lv.P, lv.S]) {
    for (const level of levels) {
      annotations.push({
        x: level.mf + half, y: level.e, text: level.label,
        showarrow: false, xanchor: 'left', yanchor: 'middle',
        font: { size: 10, color: THEME.muted },
      });
    }
  }

  // Arrows for the selected group of transitions
  const row = selectedRow();
  if (row) {
    for (let i = 0; i < row.lower.length; i++) {
      const from = lv.S[row.lower[i]];
      const to = lv.P[row.upper[i]];
      if (!from || !to) continue;
      annotations.push({
        x: to.mf, y: to.e, ax: from.mf, ay: from.e,
        xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
        showarrow: true, arrowhead: 2, arrowsize: 1,
        arrowwidth: 1.5, arrowcolor: COLOR_BY_POL[row.polarization],
      });
    }
  }

  annotations.push({
    x: lv.label_x, y: lv.label_P_y, text: '2³P States',
    showarrow: false, xanchor: 'right', textangle: -90,
    font: { color: THEME.muted },
  });
  annotations.push({
    x: lv.label_x, y: lv.label_S_y, text: '2³S States',
    showarrow: false, xanchor: 'right', textangle: -90,
    font: { color: THEME.muted },
  });

  const layout = {
    height: 500,
    showlegend: false,
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: THEME.text },
    modebar: { color: THEME.muted, activecolor: THEME.text, bgcolor: 'rgba(0,0,0,0)' },
    // The Streamlit original used zero margins here, which ran the y-axis
    // title into the "2³P States" label and crowded the m_F ticks.
    margin: { t: 8, l: 62, r: 8, b: 44 },
    xaxis: {
      title: { text: 'Magnetic Quantum Number m_F' },
      tickmode: 'array', tickvals: lv.mF_values, ticktext: lv.mF_labels,
      gridcolor: THEME.grid, zerolinecolor: THEME.grid,
      linecolor: THEME.grid, tickcolor: THEME.grid,
    },
    yaxis: {
      title: { text: 'Relative Energy (GHz)' },
      range: lv.y_range, showgrid: true,
      tickmode: 'array', tickvals: lv.tickvals, ticktext: lv.ticktext,
      gridcolor: THEME.grid, zerolinecolor: THEME.grid,
      linecolor: THEME.grid, tickcolor: THEME.grid,
    },
    annotations,
    uirevision: state.isotope,
  };

  Plotly.react('levels-plot',
    [manifold(lv.P, THEME.levelUpper), manifold(lv.S, THEME.levelLower)],
    layout, PLOT_CONFIG);
}

function drawTable() {
  const tbody = document.querySelector('#transitions tbody');
  tbody.replaceChildren();

  state.data.table.forEach((row, index) => {
    const tr = document.createElement('tr');
    tr.dataset.index = String(index);

    const cells = [
      [row.polarization, 'pol'],
      [row.frequency, ''],
      [row.wavelength, ''],
      [row.transitions, ''],
      [row.intensity, ''],
    ];
    for (const [text, cls] of cells) {
      const td = document.createElement('td');
      td.textContent = text;
      if (cls) td.className = cls;
      if (cls === 'pol') td.style.color = COLOR_BY_POL[row.polarization];
      tr.appendChild(td);
    }

    tr.addEventListener('click', () => {
      state.selected = state.selected === index ? null : index;  // click again to clear
      rerender();
    });

    tbody.appendChild(tr);
  });

  markSelectedRow();
}

function markSelectedRow() {
  for (const tr of document.querySelectorAll('#transitions tbody tr')) {
    const on = Number(tr.dataset.index) === state.selected;
    tr.classList.toggle('selected', on);
    // Marked in the row's own polarization colour, matching the arrows the
    // selection draws on the level diagram
    tr.style.boxShadow = on
      ? `inset 3px 0 0 ${COLOR_BY_POL[state.data.table[state.selected].polarization]}`
      : '';
  }
}

boot();
