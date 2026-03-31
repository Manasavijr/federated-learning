"""
Privacy-Preserving Federated Learning System — FastAPI Backend + Dashboard
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes.experiments import router as exp_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Federated Learning Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e2e8f0;font-family:Inter,system-ui,sans-serif;min-height:100vh}
.header{background:linear-gradient(135deg,#1a1f2e,#0f1117);border-bottom:1px solid #2d3748;padding:24px 40px}
h1{font-size:24px;font-weight:700}
.subtitle{color:#718096;font-size:14px;margin-top:6px}
.badge{display:inline-block;background:#1a1f2e;color:#b794f4;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;margin-top:10px;margin-right:6px;border:1px solid #2d3748}
.container{max-width:1200px;margin:0 auto;padding:32px 40px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:24px}
.card{background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:24px}
.card-title{font-size:14px;font-weight:600;color:#718096;margin-bottom:16px;text-transform:uppercase;letter-spacing:.05em}
label{display:block;font-size:11px;font-weight:700;color:#718096;margin-bottom:6px;text-transform:uppercase}
input,select{width:100%;background:#0f1117;border:1px solid #2d3748;border-radius:8px;padding:9px 12px;color:#e2e8f0;font-size:14px}
.toggle{display:flex;align-items:center;gap:8px;margin-top:8px}
.toggle input[type=checkbox]{width:auto}
.btn{width:100%;padding:13px;border-radius:8px;font-size:15px;font-weight:700;cursor:pointer;border:none;background:#b794f4;color:#1a1f2e;margin-top:16px}
.btn:disabled{background:#2d3748;color:#718096;cursor:not-allowed}
.btn-sm{padding:7px 14px;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;border:none;width:auto;margin-top:0}
.stat{text-align:center}
.stat-num{font-size:40px;font-weight:700}
.stat-label{color:#718096;font-size:12px;margin-top:4px}
.green{color:#68d391}.red{color:#fc8181}.yellow{color:#f6e05e}.purple{color:#b794f4}.blue{color:#63b3ed}
.progress-bar{height:8px;background:#2d3748;border-radius:4px;overflow:hidden;margin-top:8px}
.progress-fill{height:100%;background:#b794f4;border-radius:4px;transition:width .5s ease}
.round-row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#141822;border-radius:8px;margin-bottom:6px;font-size:13px}
.tag{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.tag-green{background:#1a3a2a;color:#68d391}
.tag-red{background:#3a1010;color:#fc8181}
.tag-yellow{background:#3a3010;color:#f6e05e}
.alert{padding:12px 16px;border-radius:8px;margin-bottom:16px;font-size:13px}
.alert-error{background:#3a1010;border:1px solid #fc8181;color:#fc8181}
.alert-success{background:#1a3a2a;border:1px solid #68d391;color:#68d391}
.dp-section{border-top:1px solid #2d3748;margin-top:16px;padding-top:16px}
canvas{max-height:260px}
</style>
</head>
<body>
<div class="header">
  <h1>🔒 Privacy-Preserving Federated Learning</h1>
  <p class="subtitle">FedAvg · Differential Privacy · Non-IID · Centralized vs Federated Comparison · MNIST</p>
  <div>
    <span class="badge">FedAvg</span><span class="badge">Differential Privacy</span>
    <span class="badge">PyTorch</span><span class="badge">Non-IID</span><span class="badge">10 Clients</span>
  </div>
</div>

<div class="container">
  <div id="alert" style="display:none" class="alert"></div>

  <div class="grid2">
    <!-- Config Panel -->
    <div class="card">
      <div class="card-title">Experiment Configuration</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div><label>Clients</label><input type="number" id="num_clients" value="10" min="2" max="50"/></div>
        <div><label>Rounds</label><input type="number" id="num_rounds" value="10" min="1" max="50"/></div>
        <div><label>Local Epochs</label><input type="number" id="local_epochs" value="2" min="1" max="10"/></div>
        <div><label>Learning Rate</label><input type="number" id="lr" value="0.01" step="0.001"/></div>
        <div><label>Client Fraction</label><input type="number" id="fraction" value="1.0" step="0.1" min="0.1" max="1.0"/></div>
        <div><label>Data Split</label>
          <select id="iid">
            <option value="true">IID (uniform)</option>
            <option value="false">Non-IID (Dirichlet)</option>
          </select>
        </div>
      </div>
      <div id="alpha_row" style="margin-top:12px;display:none">
        <label>Dirichlet α (heterogeneity)</label>
        <input type="number" id="alpha" value="0.5" step="0.1" min="0.05"/>
        <div style="color:#718096;font-size:11px;margin-top:3px">Lower = more heterogeneous (harder for FL)</div>
      </div>
      <div class="dp-section">
        <div class="toggle"><input type="checkbox" id="dp_enabled" onchange="toggleDP()"/> <label style="margin:0;text-transform:none;font-size:13px;color:#e2e8f0">🔒 Enable Differential Privacy</label></div>
        <div id="dp_options" style="display:none;margin-top:12px;display:none">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div><label>Noise Multiplier (σ)</label><input type="number" id="noise" value="1.0" step="0.1" min="0.1"/></div>
            <div><label>Grad Clip Norm</label><input type="number" id="clip" value="1.0" step="0.1" min="0.1"/></div>
          </div>
          <div id="privacy_estimate" style="margin-top:10px;font-size:12px;color:#718096"></div>
        </div>
      </div>
      <button class="btn" id="run_btn" onclick="runExperiment()">🚀 Run Federated Learning</button>
    </div>

    <!-- Live Status -->
    <div class="card">
      <div class="card-title">Live Training Status</div>
      <div class="grid3" style="margin-bottom:20px">
        <div class="stat"><div class="stat-num green" id="stat_acc">—</div><div class="stat-label">Accuracy</div></div>
        <div class="stat"><div class="stat-num blue" id="stat_round">0/0</div><div class="stat-label">Round</div></div>
        <div class="stat"><div class="stat-num yellow" id="stat_eps">—</div><div class="stat-label">ε (privacy)</div></div>
      </div>
      <div id="status_text" style="color:#718096;font-size:13px;margin-bottom:12px">No experiment running</div>
      <div class="progress-bar"><div class="progress-fill" id="progress_fill" style="width:0%"></div></div>
      <div style="font-size:11px;color:#718096;margin-top:6px" id="progress_pct">0%</div>

      <div style="margin-top:20px">
        <div class="card-title">Round History</div>
        <div id="round_history" style="max-height:200px;overflow-y:auto"></div>
      </div>
    </div>
  </div>

  <!-- Charts -->
  <div class="grid2">
    <div class="card">
      <div class="card-title">Accuracy Convergence</div>
      <canvas id="acc_chart"></canvas>
    </div>
    <div class="card">
      <div class="card-title">Loss Convergence</div>
      <canvas id="loss_chart"></canvas>
    </div>
  </div>

  <!-- Summary -->
  <div id="summary_card" style="display:none" class="card">
    <div class="card-title">Experiment Summary</div>
    <div id="summary_content"></div>
  </div>
</div>

<script>
let currentId = null;
let pollTimer = null;
let accChart = null, lossChart = null;

const chartDefaults = {
  responsive: true,
  plugins: { legend: { labels: { color: '#e2e8f0' } } },
  scales: {
    x: { ticks: { color: '#718096' }, grid: { color: '#2d3748' } },
    y: { ticks: { color: '#718096' }, grid: { color: '#2d3748' } }
  }
};

function initCharts() {
  if (accChart) accChart.destroy();
  if (lossChart) lossChart.destroy();
  accChart = new Chart(document.getElementById('acc_chart'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Test Accuracy', data: [], borderColor: '#68d391', backgroundColor: 'rgba(104,211,145,0.1)', tension: 0.3, fill: true }] },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0, max: 1 } } }
  });
  lossChart = new Chart(document.getElementById('loss_chart'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Test Loss', data: [], borderColor: '#fc8181', backgroundColor: 'rgba(252,129,129,0.1)', tension: 0.3, fill: true }] },
    options: chartDefaults
  });
}

function toggleDP() {
  const enabled = document.getElementById('dp_enabled').checked;
  document.getElementById('dp_options').style.display = enabled ? 'block' : 'none';
  if (enabled) updatePrivacyEstimate();
}

document.getElementById('iid').addEventListener('change', function() {
  document.getElementById('alpha_row').style.display = this.value === 'false' ? 'block' : 'none';
});

async function updatePrivacyEstimate() {
  const noise = document.getElementById('noise').value;
  const clients = document.getElementById('num_clients').value;
  const rounds = document.getElementById('num_rounds').value;
  try {
    const r = await fetch(`/api/v1/privacy/estimate?noise_multiplier=${noise}&num_clients=${clients}&rounds=${rounds}`);
    const d = await r.json();
    document.getElementById('privacy_estimate').innerHTML =
      `Estimated guarantee: <span style="color:#b794f4;font-weight:600">(ε=${d.epsilon}, δ=${d.delta})-DP</span> — lower ε = stronger privacy`;
  } catch {}
}

async function runExperiment() {
  const btn = document.getElementById('run_btn');
  btn.disabled = true;
  btn.textContent = '⏳ Starting...';
  initCharts();
  document.getElementById('round_history').innerHTML = '';
  document.getElementById('summary_card').style.display = 'none';
  document.getElementById('stat_acc').textContent = '—';
  document.getElementById('stat_round').textContent = '0/0';
  document.getElementById('stat_eps').textContent = '—';

  const config = {
    num_clients: parseInt(document.getElementById('num_clients').value),
    num_rounds: parseInt(document.getElementById('num_rounds').value),
    local_epochs: parseInt(document.getElementById('local_epochs').value),
    local_lr: parseFloat(document.getElementById('lr').value),
    fraction_fit: parseFloat(document.getElementById('fraction').value),
    iid: document.getElementById('iid').value === 'true',
    dirichlet_alpha: parseFloat(document.getElementById('alpha').value),
    dp_enabled: document.getElementById('dp_enabled').checked,
    noise_multiplier: parseFloat(document.getElementById('noise').value),
    max_grad_norm: parseFloat(document.getElementById('clip').value),
    experiment_name: 'dashboard_experiment',
  };

  try {
    const r = await fetch('/api/v1/experiments', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(config) });
    const d = await r.json();
    currentId = d.experiment_id;
    showAlert(`Experiment ${currentId} started!`, 'success');
    pollTimer = setInterval(pollStatus, 2000);
  } catch(e) {
    showAlert('Failed to start experiment: ' + e.message, 'error');
    btn.disabled = false;
    btn.textContent = '🚀 Run Federated Learning';
  }
}

async function pollStatus() {
  if (!currentId) return;
  try {
    const r = await fetch(`/api/v1/experiments/${currentId}`);
    const d = await r.json();
    updateUI(d);
    if (d.status === 'completed' || d.status === 'failed') {
      clearInterval(pollTimer);
      document.getElementById('run_btn').disabled = false;
      document.getElementById('run_btn').textContent = '🚀 Run Federated Learning';
      if (d.status === 'completed') showSummary(d.summary);
      if (d.status === 'failed') showAlert('Experiment failed: ' + d.error, 'error');
    }
  } catch {}
}

function updateUI(d) {
  const rounds = d.metrics || [];
  document.getElementById('stat_round').textContent = `${d.current_round}/${d.total_rounds}`;
  document.getElementById('status_text').textContent = `Status: ${d.status} — Round ${d.current_round} of ${d.total_rounds}`;
  document.getElementById('progress_fill').style.width = d.progress_pct + '%';
  document.getElementById('progress_pct').textContent = d.progress_pct + '%';

  if (d.current_accuracy !== null && d.current_accuracy !== undefined) {
    document.getElementById('stat_acc').textContent = (d.current_accuracy * 100).toFixed(1) + '%';
  }

  if (rounds.length > 0) {
    const last = rounds[rounds.length - 1];
    if (last.epsilon !== null && last.epsilon !== undefined) {
      document.getElementById('stat_eps').textContent = 'ε=' + last.epsilon.toFixed(3);
    }
    // Update charts
    const labels = rounds.map(r => `R${r.round_num}`);
    accChart.data.labels = labels;
    accChart.data.datasets[0].data = rounds.map(r => r.test_accuracy);
    accChart.update('none');
    lossChart.data.labels = labels;
    lossChart.data.datasets[0].data = rounds.map(r => r.test_loss);
    lossChart.update('none');
    // Round history (last 5)
    const hist = document.getElementById('round_history');
    hist.innerHTML = rounds.slice(-8).reverse().map(r => `
      <div class="round-row">
        <span>Round ${r.round_num}</span>
        <span class="${r.test_accuracy > 0.9 ? 'green' : r.test_accuracy > 0.7 ? 'yellow' : 'red'}">${(r.test_accuracy*100).toFixed(1)}%</span>
        <span style="color:#718096">loss ${r.test_loss.toFixed(4)}</span>
        ${r.epsilon !== null && r.epsilon !== undefined ? `<span class="tag tag-yellow">ε=${r.epsilon.toFixed(2)}</span>` : ''}
        <span style="color:#718096">${r.duration_s}s</span>
      </div>`).join('');
  }
}

function showSummary(s) {
  if (!s) return;
  document.getElementById('summary_card').style.display = 'block';
  const dp = s.privacy ? `<span class="green">ε=${s.privacy.epsilon_spent?.toFixed(3)}, δ=${s.privacy.delta}</span>` : '<span style="color:#718096">Disabled</span>';
  document.getElementById('summary_content').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px">
      <div class="stat"><div class="stat-num green">${(s.best_accuracy*100).toFixed(1)}%</div><div class="stat-label">Best Accuracy</div></div>
      <div class="stat"><div class="stat-num blue">${s.best_round}</div><div class="stat-label">Best Round</div></div>
      <div class="stat"><div class="stat-num purple">${(s.final_accuracy*100).toFixed(1)}%</div><div class="stat-label">Final Accuracy</div></div>
      <div class="stat"><div class="stat-num yellow">${s.num_rounds_completed}</div><div class="stat-label">Rounds</div></div>
    </div>
    <div style="margin-top:20px;padding:14px;background:#141822;border-radius:8px;font-size:13px">
      <div style="margin-bottom:6px">Config: ${s.config.num_clients} clients · ${s.config.local_epochs} local epochs · ${s.config.iid ? 'IID' : 'Non-IID'}</div>
      <div>Privacy: ${dp}</div>
    </div>`;
  showAlert(`✅ Training complete! Best accuracy: ${(s.best_accuracy*100).toFixed(1)}%`, 'success');
}

function showAlert(msg, type) {
  const el = document.getElementById('alert');
  el.className = 'alert alert-' + type;
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 5000);
}

initCharts();
</script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Federated Learning API...")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Privacy-Preserving Federated Learning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(exp_router, prefix="/api/v1", tags=["Experiments"])


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "federated-learning"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8082, reload=True)
