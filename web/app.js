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

const SERIES_KEYS = ['plus', 'minus', 'pi'];
const SERIES_LABELS = { plus: 'σ+', minus: 'σ-', pi: 'π' };

// Each mode is stepped for its own surface rather than reused from the other,
// and validated there for lightness band, chroma, colour-vision separation and
// contrast. Plotly.js ships no named templates, so the chart chrome is literal
// too; these values mirror the tokens in style.css.
//
// The marker is achromatic in both modes on purpose: it is an annotation, not
// a fourth series, and every chromatic candidate collided with red or green.
//
// The level bars are a gold pair, chosen for appearance. Note what that costs:
// selecting a transition draws arrows in the polarization colour *inside* the
// level diagram, so the bars are effectively a fourth and fifth categorical
// colour alongside all three series, and no gold pair clears that. These are
// the best available in each mode, ~3.7:1 apart so the manifolds read clearly.
//
//   light  A #4d3500 vs pi: CVD dE 11.4, clear - a deep enough bronze escapes
//          B #c98500 vs sigma-: normal-vision dE 13.0, under the 15 floor
//   dark   B #ffd166 vs sigma-: dE 26.3, comfortable
//          A #8f6200 vs pi: CVD dE 1.2 - deuteranopes and protanopes see the
//          A bars and the pi arrows as one colour. Unavoidable on a dark
//          surface: a bronze dark enough to escape green drops below 3:1.
//
// The A/B labels and the vertical split carry the distinction for those
// viewers. Swapping levelLower to a neutral (#848c99 dark, #6b7280 light)
// removes the collapse entirely if it ever matters.
const PALETTE = {
  light: {
    series: { plus: '#2a78d6', minus: '#e66767', pi: '#006d00' },
    text: '#1a1a19',
    muted: '#52514e',
    grid: '#d6d9de',
    levelUpper: '#c98500',   // 2³P states
    levelLower: '#4d3500',   // 2³S states
    marker: '#52514e',
  },
  dark: {
    series: { plus: '#3987e5', minus: '#e66767', pi: '#008300' },
    text: '#e8e8e5',
    muted: '#9ea3ad',
    grid: '#2f333c',
    levelUpper: '#ffd166',
    levelLower: '#8f6200',
    marker: '#c3c2b7',
  },
};

const THEME_KEY = 'hespectra-theme';

/** The viewer's explicit choice, or null while the OS setting governs. */
function storedTheme() {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === 'light' || v === 'dark' ? v : null;
  } catch (err) {
    return null;   // private windows and blocked site data
  }
}

function activeTheme() {
  return storedTheme()
    || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}

/** Colours for the mode currently showing. */
function palette() {
  return PALETTE[activeTheme()];
}

// bridge.py labels each table row with one of the same symbols
function colorForPol(symbol) {
  const pal = palette();
  const key = SERIES_KEYS.find(k => SERIES_LABELS[k] === symbol);
  return key ? pal.series[key] : pal.muted;
}

const PLOT_CONFIG = { responsive: true, displaylogo: false };

const state = {
  B: 1.0,
  T: 300,
  isotope: 'He3',
  xAxis: 'Frequency Offset',
  P: 0,
  selected: null,   // index into the current table, or null
  pump: null,       // pumping readout for the selected row, fetched on demand
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
  linkNumeric('p-slider', 'p-input', 'P', v => String(Math.round(v)));

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

  bindTheme();
}

/* ----------------------------------------------------------------- theme */

function bindTheme() {
  const button = document.getElementById('theme-toggle');

  button.addEventListener('click', () => {
    const next = activeTheme() === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch (err) {
      /* Not persisted, but the attribute below still applies it */
    }
    document.documentElement.dataset.theme = next;
    applyTheme();
  });

  // Follow the OS while the viewer has expressed no preference of their own
  window.matchMedia('(prefers-color-scheme: dark)')
    .addEventListener('change', () => {
      if (!storedTheme()) applyTheme();
    });

  const stored = storedTheme();
  if (stored) document.documentElement.dataset.theme = stored;
  applyTheme();
}

/** Sync the button to the mode showing, and repaint the charts in it. */
function applyTheme() {
  const dark = activeTheme() === 'dark';
  document.querySelector('.theme-icon').textContent = dark ? '☾' : '☀';
  document.getElementById('theme-label').textContent =
    dark ? 'Dark theme' : 'Light theme';
  document.getElementById('theme-toggle').setAttribute(
    'aria-label', `Theme: ${dark ? 'dark' : 'light'}. Switch to ${dark ? 'light' : 'dark'}.`);

  // Colours are baked into the Plotly specs, so the charts need redrawing.
  // No recalculation: the numbers are unchanged.
  if (state.data) {
    drawSpectra();
    drawTable();
    drawLevels();
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
    state.data = bridge.compute_js(
      state.B, state.T, state.isotope, state.xAxis, state.P);
    if (state.selected !== null && state.selected >= state.data.table.length) {
      state.selected = null;
    }
    fetchPumping();
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

/**
 * The pumping readout for the selected row. bridge.py makes it on request
 * rather than for every row of every recompute, as only this one is shown.
 */
function fetchPumping() {
  state.pump = state.selected === null ? null : bridge.pumping_js(state.selected) || null;
}

function selectedRow() {
  return state.selected === null ? null : state.data.table[state.selected];
}

/** m_F as a signed integer or half-integer, matching the axis tick labels. */
function formatMf(value) {
  const sign = value < 0 ? '-' : (value > 0 ? '+' : '');
  const magnitude = Math.abs(value);
  return Number.isInteger(magnitude)
    ? `${sign}${magnitude}`
    : `${sign}${Math.round(magnitude * 2)}/2`;
}

function drawSpectra() {
  const s = state.data.spectra;
  const THEME = palette();

  const traces = SERIES_KEYS.map(key => ({
    x: s.x, y: s[key], mode: 'lines',
    name: SERIES_LABELS[key], line: { color: THEME.series[key], width: 2 },
  }));

  const shapes = [];
  const row = selectedRow();
  if (row) {
    // Both axes are derived from the same absolute frequency, so the table
    // value can be used directly whichever axis is showing.
    const x = Number(state.xAxis === 'Frequency Offset' ? row.frequency : row.wavelength);

    // A band across the group's full extent, not just its mean. The grouping
    // chains at a fixed threshold, so a group can reach well beyond the peak
    // it is named for; this shows how much of the axis it actually claims.
    // members is ordered by frequency, so its ends are the extremes on either
    // axis, and both coordinates are already converted.
    const key = state.xAxis === 'Frequency Offset' ? 'frequency' : 'wavelength';
    const edges = [row.members[0][key], row.members[row.members.length - 1][key]]
      .map(Number);
    if (Math.abs(edges[1] - edges[0]) > 0) {
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: Math.min(...edges), x1: Math.max(...edges), y0: 0, y1: 1,
        fillcolor: THEME.marker, opacity: 0.12, line: { width: 0 }, layer: 'below',
      });
    }

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

// Leaks below this rate (0.1% of a full-strength line) are left off the diagram,
// though they still appear in the level's hover
const PUMP_LABEL_FLOOR = 0.001;

/** Hover lines for each lower level, describing the selected peak's pumping. */
function pumpHoverText(pump, polarization) {
  return pump.levels.map(rate => {
    let text = '<br><br>';
    if (rate.targeted) {
      text += `<b>Pumped by this peak</b>, at ${rate.text}`;
    } else if (!rate.has_lines) {
      text += `No ${polarization} lines from this level`;
    } else {
      text += `<b>Emptied at ${rate.text}</b>`;
    }
    for (const v of rate.via) {
      text += `<br>${v.share}% via ${v.name} (${v.offset} GHz from laser)`;
    }
    return text + `<br><i>100% = a full-strength line on resonance with a`
      + ` ${pump.laser_fwhm} GHz laser on the peak centroid; rates per atom</i>`;
  });
}

function drawLevels() {
  const THEME = palette();
  const lv = state.data.levels;
  const half = lv.line_width / 2;

  // One trace per manifold, with nulls separating the individual level bars.
  // The bars carry their own hover text; `e` is the plotted energy, which for
  // the upper manifold already includes the display offset, so the label uses
  // the level's own scale instead.
  const manifold = (levels, color, offset, extra = []) => {
    const x = [], y = [], text = [];
    levels.forEach((level, i) => {
      const label = `${level.label.trim()}<br>m<sub>F</sub> = ${formatMf(level.mf)}`
        + `<br>${(level.e - offset).toFixed(3)} GHz` + (extra[i] || '');
      x.push(level.mf - half, level.mf + half, null);
      y.push(level.e, level.e, null);
      text.push(label, label, '');
    });
    return {
      x, y, text, mode: 'lines', line: { color, width: 3 },
      hoverinfo: 'text', hoverlabel: { bgcolor: THEME.grid, font: { color: THEME.text } },
      showlegend: false,
    };
  };

  // Invisible hover targets along each selected transition: the arrows are
  // annotations, which cannot be hovered. Plotly's 'closest' hovermode snaps
  // to data points rather than to a position along a line, so the shaft is
  // sampled. The samples stop short of both ends, leaving the level bars their
  // own hover instead of being shadowed by a transition endpoint.
  const hoverTargets = { x: [], y: [], text: [], mode: 'markers',
    marker: { size: 12, color: 'rgba(0,0,0,0)' },
    hoverinfo: 'text', hoverlabel: { bgcolor: THEME.grid, font: { color: THEME.text } },
    showlegend: false };
  const SHAFT_SAMPLES = 14;

  // Pumping readout for the selected peak: how fast a laser on it empties
  // each lower level, where 100% is a full-strength line on resonance.
  // Labelled only where it says something - the pumped levels, and leaks of at
  // least 0.1% - with the full breakdown in the bar's hover.
  const pump = state.pump;
  const pumpExtra = pump ? pumpHoverText(pump, selectedRow().polarization) : [];

  const annotations = [];
  const levelLabel = (level, text) => annotations.push({
    x: level.mf + half, y: level.e, text,
    showarrow: false, xanchor: 'left', yanchor: 'middle',
    font: { size: 10, color: THEME.muted },
  });
  lv.P.forEach(level => levelLabel(level, level.label));
  lv.S.forEach((level, i) => {
    levelLabel(level, level.label);
    const rate = pump && pump.levels[i];
    if (rate && (rate.targeted || rate.rel >= PUMP_LABEL_FLOOR)) {
      // Centred under the bar rather than after its label: at high field the
      // lower levels pair up at nearly the same energy in neighbouring m_F
      // columns, and a label running to the right collides with the next one.
      annotations.push({
        x: level.mf, y: level.e, text: rate.text,
        showarrow: false, xanchor: 'center', yanchor: 'top', yshift: -3,
        font: { size: 9, color: THEME.muted },
      });
    }
  });

  // Arrows for the selected group of transitions. Driven from `members`, which
  // is ordered by frequency and carries each transition's own numbers, so the
  // arrow and its hover text cannot get out of step.
  const row = selectedRow();
  if (row) {
    for (const member of row.members) {
      const from = lv.S[member.lower];
      const to = lv.P[member.upper];
      if (!from || !to) continue;
      annotations.push({
        x: to.mf, y: to.e, ax: from.mf, ay: from.e,
        xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
        showarrow: true, arrowhead: 2, arrowsize: 1,
        arrowwidth: 1.5, arrowcolor: colorForPol(row.polarization),
      });

      const text = `<b>${member.name}</b>`
        + `<br>${member.frequency} GHz`
        + `<br>${member.wavelength} nm`
        + `<br>intensity ${member.intensity} (${member.share}% of peak)`
        + `<br>${member.offset} GHz from the peak centroid`;
      for (let s = 0; s < SHAFT_SAMPLES; s++) {
        const t = 0.12 + (0.76 * s) / (SHAFT_SAMPLES - 1);
        hoverTargets.x.push(from.mf + (to.mf - from.mf) * t);
        hoverTargets.y.push(from.e + (to.e - from.e) * t);
        hoverTargets.text.push(text);
      }
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
      title: { text: 'Magnetic Quantum Number m<sub>F</sub>' },
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

  Plotly.react('levels-plot', [
    manifold(lv.P, THEME.levelUpper, lv.p_offset),
    manifold(lv.S, THEME.levelLower, 0, pumpExtra),
    hoverTargets,
  ], layout, PLOT_CONFIG);
}

function drawTable() {
  const tbody = document.querySelector('#transitions tbody');
  tbody.replaceChildren();

  // The threshold is fixed in frequency while the line width follows the
  // temperature and the pressure, so which lines get grouped is not purely
  // physical. Showing both lets the two be compared.
  const widths = Number(state.data.lorentz) > 0
    ? `Doppler ${state.data.doppler} + collisional ${state.data.lorentz}`
      + ` = Voigt ${state.data.voigt} GHz FWHM`
    : `Doppler width ${state.data.doppler} GHz FWHM`;
  document.getElementById('grouping-note').textContent =
    ` Lines are grouped into a peak when within ${state.data.group_threshold} GHz`
    + ` of its intensity-weighted centroid; ${widths}.`;

  state.data.table.forEach((row, index) => {
    const tr = document.createElement('tr');
    tr.dataset.index = String(index);

    const cells = [
      [row.polarization, 'pol'],
      [row.frequency, ''],
      [row.wavelength, ''],
      // Flagged when the group reaches wider than the line width, i.e. when
      // its members do not actually merge into a single peak. Pressure widens
      // the lines, so a group flagged at low pressure can stop being flagged.
      [row.span, Number(row.span) > Number(state.data.voigt) ? 'span wide' : 'span'],
      [row.transitions, ''],
      [row.intensity, ''],
    ];
    for (const [text, cls] of cells) {
      const td = document.createElement('td');
      td.textContent = text;
      if (cls) td.className = cls;
      if (cls === 'pol') td.style.color = colorForPol(row.polarization);
      tr.appendChild(td);
    }

    tr.addEventListener('click', () => {
      state.selected = state.selected === index ? null : index;  // click again to clear
      fetchPumping();
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
      ? `inset 3px 0 0 ${colorForPol(state.data.table[state.selected].polarization)}`
      : '';
  }
}

boot();
