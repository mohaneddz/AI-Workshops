const map = L.map('map', { worldCopyJump: true, zoomControl: true }).setView([22, 10], 2);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
}).addTo(map);

const modelSelect = document.getElementById('modelSelect');
const thresholdCard = document.getElementById('thresholdCard');
const resetMetricsBtn = document.getElementById('resetMetricsBtn');

let activeLayers = [];
let totalEvents = 0;
let totalAnomalies = 0;

function currentModel() {
  return modelSelect?.value || 'isolation_forest';
}

function clearLayers() {
  for (const layer of activeLayers) {
    map.removeLayer(layer);
  }
  activeLayers = [];
}

function formatPath(ev) {
  return `${ev.source.name} -> ${ev.target.name}`;
}

function addEventToMap(ev) {
  const src = [ev.source.lat, ev.source.lon];
  const dst = [ev.target.lat, ev.target.lon];
  const color = ev.is_anomaly ? '#ff7a59' : '#27c2ff';
  const weight = ev.is_anomaly ? 3 : 1.8;
  const opacity = ev.is_anomaly ? 0.9 : 0.45;

  const line = L.polyline([src, dst], { color, weight, opacity }).addTo(map);
  const marker = L.circleMarker(src, {
    radius: ev.is_anomaly ? 6 : 4,
    color,
    weight: 1,
    fillColor: color,
    fillOpacity: 0.8,
  }).addTo(map);

  marker.bindPopup(`
    <strong>${ev.source.name}</strong> (${ev.source.country})<br/>
    to <strong>${ev.target.name}</strong> (${ev.target.country})<br/>
    model: ${ev.model_name}<br/>
    score: ${ev.model_score.toFixed(4)}<br/>
    flag: ${ev.is_anomaly ? 'anomaly' : 'normal'}
  `);

  activeLayers.push(line, marker);
}

function renderTable(events) {
  const body = document.getElementById('eventsBody');
  body.innerHTML = '';

  for (const ev of events) {
    const tr = document.createElement('tr');
    const dt = new Date(ev.timestamp);

    tr.innerHTML = `
      <td>${dt.toLocaleTimeString()}</td>
      <td>${formatPath(ev)}</td>
      <td>${ev.model_score.toFixed(4)}</td>
      <td class="${ev.is_anomaly ? 'flag-high' : 'flag-low'}">${ev.is_anomaly ? ev.severity.toUpperCase() : 'NORMAL'}</td>
    `;
    body.appendChild(tr);
  }
}

function renderMetrics(metrics) {
  document.getElementById('passedRequests').textContent = metrics.passed_requests ?? 0;
  document.getElementById('anomaliesClassified').textContent = metrics.anomalies_classified ?? 0;
  document.getElementById('unclassified').textContent = metrics.unclassified ?? 0;
  document.getElementById('goodMisclassified').textContent = metrics.good_connections_misclassified ?? 0;
}

async function fetchEvents() {
  const model = currentModel();
  const res = await fetch(`/api/events?batch=35&model=${encodeURIComponent(model)}`);
  const payload = await res.json();
  const events = payload.events || [];

  clearLayers();
  events.forEach(addEventToMap);
  renderTable(events);

  const anomalies = events.filter(e => e.is_anomaly).length;
  totalEvents += events.length;
  totalAnomalies += anomalies;

  document.getElementById('totalEvents').textContent = totalEvents;
  document.getElementById('anomalyCount').textContent = totalAnomalies;
  document.getElementById('lastPoll').textContent = new Date().toLocaleTimeString();
  if (payload.threshold !== null && payload.threshold !== undefined) {
    thresholdCard.textContent = Number(payload.threshold).toFixed(4);
  }
}

async function fetchMetrics() {
  const model = currentModel();
  const res = await fetch(`/api/metrics?model=${encodeURIComponent(model)}`);
  const payload = await res.json();
  if (payload.ok && payload.metrics) {
    renderMetrics(payload.metrics);
  }
}

async function refreshAll() {
  await fetchEvents();
  await fetchMetrics();
}

document.getElementById('refreshBtn').addEventListener('click', () => {
  refreshAll().catch(console.error);
});

if (modelSelect) {
  modelSelect.addEventListener('change', () => {
    refreshAll().catch(console.error);
  });
}

if (resetMetricsBtn) {
  resetMetricsBtn.addEventListener('click', async () => {
    await fetch('/api/metrics/reset', { method: 'POST' });
    await fetchMetrics();
  });
}

refreshAll().catch(console.error);
setInterval(() => {
  refreshAll().catch(console.error);
}, 4000);
