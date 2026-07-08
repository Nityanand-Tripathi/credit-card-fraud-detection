/**
 * main.js — talks to /api/predict and /api/health, validates the form,
 * and renders the verdict panel + session history.
 */
(function () {
  'use strict';

  const form = document.getElementById('predict-form');
  const timeInput = document.getElementById('time');
  const amountInput = document.getElementById('amount');
  const modelSelect = document.getElementById('model-select');
  const vFeaturesInput = document.getElementById('vfeatures');
  const analyzeBtn = document.getElementById('btn-analyze');

  const verdictEmpty = document.getElementById('verdict-empty');
  const verdictContent = document.getElementById('verdict-content');
  const verdictError = document.getElementById('verdict-error');
  const verdictBadge = document.getElementById('verdict-badge');
  const scoreBarFill = document.getElementById('score-bar-fill');
  const scoreValue = document.getElementById('score-value');
  const metaModel = document.getElementById('meta-model');
  const metaPred = document.getElementById('meta-pred');
  const metaTime = document.getElementById('meta-time');

  const historyBody = document.getElementById('history-body');
  const history = [];

  /* ---------------------------------------------------------
   * Health check on load
   * ------------------------------------------------------- */
  const healthPill = document.getElementById('health-pill');
  fetch('/api/health')
    .then((r) => r.json())
    .then((data) => {
      if (data.models_loaded && data.models_loaded.length > 0) {
        healthPill.textContent = `${data.models_loaded.length} model(s) loaded`;
        healthPill.classList.add('ok');
      } else {
        healthPill.textContent = 'No models loaded';
        healthPill.classList.add('error');
      }
    })
    .catch(() => {
      healthPill.textContent = 'Backend unreachable';
      healthPill.classList.add('error');
    });

  /* ---------------------------------------------------------
   * Validation helpers
   * ------------------------------------------------------- */
  function setError(fieldName, hasError) {
    const wrap = form.querySelector(`[data-field="${fieldName}"]`);
    if (wrap) wrap.classList.toggle('has-error', hasError);
  }

  function parseVFeatures(raw) {
    const trimmed = raw.trim();
    if (trimmed === '') return { ok: true, values: new Array(28).fill(0) };
    const parts = trimmed.split(',').map((s) => s.trim());
    if (parts.length !== 28) return { ok: false, values: null };
    const nums = parts.map(Number);
    if (nums.some((n) => Number.isNaN(n))) return { ok: false, values: null };
    return { ok: true, values: nums };
  }

  function validate() {
    let valid = true;

    const time = parseFloat(timeInput.value);
    const timeOk = !Number.isNaN(time);
    setError('time', !timeOk);
    valid = valid && timeOk;

    const amount = parseFloat(amountInput.value);
    const amountOk = !Number.isNaN(amount) && amount >= 0;
    setError('amount', !amountOk);
    valid = valid && amountOk;

    const vResult = parseVFeatures(vFeaturesInput.value);
    setError('vfeatures', !vResult.ok);
    valid = valid && vResult.ok;

    const model = modelSelect.value;
    const modelOk = model !== '';
    if (!modelOk) {
      showToast('No model available', 'Check that at least one .pkl file loaded on the server.', 'error');
    }
    valid = valid && modelOk;

    return { valid, time, amount, vFeatures: vResult.values, model };
  }

  [timeInput, amountInput, vFeaturesInput].forEach((el) => {
    el.addEventListener('input', () => {
      const field = el.closest('[data-field]');
      if (field) field.classList.remove('has-error');
    });
  });

  /* ---------------------------------------------------------
   * Toasts
   * ------------------------------------------------------- */
  const toastRegion = document.getElementById('toast-region');
  function showToast(title, message, type) {
    const el = document.createElement('div');
    el.className = `toast ${type === 'error' ? 'error' : ''}`;
    el.innerHTML = `<strong>${escapeHtml(title)}</strong>${message ? `<br><span>${escapeHtml(message)}</span>` : ''}`;
    toastRegion.appendChild(el);
    setTimeout(() => el.remove(), 5000);
  }
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /* ---------------------------------------------------------
   * Rendering
   * ------------------------------------------------------- */
  function renderVerdict(data) {
    verdictError.hidden = true;
    verdictEmpty.hidden = true;
    verdictContent.hidden = false;

    const prob = data.probability != null ? data.probability : (data.prediction === 1 ? 1 : 0);
    const pct = Math.round(prob * 100);
    const isFraud = data.prediction === 1;

    verdictBadge.textContent = isFraud ? 'Likely fraud' : 'Looks legitimate';
    verdictBadge.className = `badge ${isFraud ? 'risk' : 'safe'}`;
    scoreBarFill.style.width = `${pct}%`;
    scoreValue.textContent = `${pct}%`;
    metaModel.textContent = data.model_used;
    metaPred.textContent = data.prediction === 1 ? '1 (fraud)' : '0 (not fraud)';
    metaTime.textContent = new Date(data.timestamp).toLocaleString();

    return { pct, isFraud };
  }

  function renderError(message) {
    verdictEmpty.hidden = true;
    verdictContent.hidden = true;
    verdictError.hidden = false;
    verdictError.textContent = message;
  }

  function addHistoryRow(amount, time, modelUsed, pct, isFraud) {
    history.unshift({ amount, time, modelUsed, pct, isFraud });
    const rows = history
      .slice(0, 15)
      .map(
        (h) => `
        <tr>
          <td>${h.time}</td>
          <td>$${h.amount.toFixed(2)}</td>
          <td>${escapeHtml(h.modelUsed)}</td>
          <td>${h.pct}%</td>
          <td><span class="badge ${h.isFraud ? 'risk' : 'safe'}">${h.isFraud ? 'Fraud' : 'Legit'}</span></td>
        </tr>`
      )
      .join('');
    historyBody.innerHTML = rows || '<tr class="empty-row"><td colspan="5">No transactions scored yet this session.</td></tr>';
  }

  document.getElementById('btn-clear-history').addEventListener('click', () => {
    history.length = 0;
    historyBody.innerHTML = '<tr class="empty-row"><td colspan="5">No transactions scored yet this session.</td></tr>';
  });

  /* ---------------------------------------------------------
   * Sample fillers
   * ------------------------------------------------------- */
  document.getElementById('btn-sample').addEventListener('click', () => {
    timeInput.value = '406';
    amountInput.value = '84.12';
    vFeaturesInput.value =
      '-1.36,-0.07,2.54,1.38,-0.34,0.46,0.24,0.10,0.36,0.09,-0.55,-0.62,-0.99,-0.31,1.47,-0.47,0.21,0.03,0.40,0.25,-0.02,0.28,-0.11,0.07,0.13,-0.19,0.13,-0.02';
    ['time', 'amount', 'vfeatures'].forEach((f) => setError(f, false));
  });

  document.getElementById('btn-sample-fraud').addEventListener('click', () => {
    // A shape more typical of flagged fraud rows in this dataset: larger
    // absolute values on the most fraud-correlated components (V14, V4, V12, V10).
    timeInput.value = '80712';
    amountInput.value = '1.00';
    vFeaturesInput.value =
      '-3.04,3.85,-6.66,5.98,-4.65,-1.36,-6.34,1.14,-3.11,-6.87,4.75,-8.19,0.55,-9.43,0.24,-5.31,-9.83,-3.02,1.01,0.15,0.83,0.34,-0.36,0.02,-0.11,0.36,0.28,-0.15';
    ['time', 'amount', 'vfeatures'].forEach((f) => setError(f, false));
    showToast('Risky sample loaded', 'These V-feature magnitudes resemble historically flagged fraud rows.', 'success');
  });

  /* ---------------------------------------------------------
   * Submit
   * ------------------------------------------------------- */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const { valid, time, amount, vFeatures, model } = validate();

    if (!valid) {
      showToast('Check the highlighted fields', 'Some values are missing or invalid.', 'error');
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing…';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, time, amount, v_features: vFeatures }),
      });
      const data = await res.json();

      if (!data.ok) {
        renderError(data.error || 'Unknown error scoring this transaction.');
        showToast('Prediction failed', data.error || '', 'error');
        return;
      }

      const { pct, isFraud } = renderVerdict(data);
      addHistoryRow(amount, time, data.model_used, pct, isFraud);
      showToast(isFraud ? 'Flagged as fraud' : 'Looks legitimate', `${pct}% estimated fraud probability.`, isFraud ? 'error' : 'success');
    } catch (err) {
      renderError('Could not reach the server. Is the Flask app running?');
      showToast('Network error', 'Could not reach the server.', 'error');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze transaction';
    }
  });
})();