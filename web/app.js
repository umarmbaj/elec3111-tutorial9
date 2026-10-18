'use strict';
const pathname = window.location.pathname;
const mode = pathname === '/mlp' ? 'mlp' : pathname === '/mlr' ? 'mlr' : 'dashboard';
const $ = (id) => document.getElementById(id);
let fetchedRows = [], plotReady = false;
let lastAnalysisResult = null;
const num = (v, n=2) => v === null || v === undefined ? '—' : Number(v).toFixed(n);
const date = (ms) => new Date(ms).toLocaleString();
const banner = (message, cls='') => { $('modeLabel').className = `banner ${cls}`; $('modeLabel').textContent = message; };
const escapes = (s) => String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

for (const el of document.querySelectorAll('nav a')) {
  if (new URL(el.href).pathname === pathname) el.classList.add('active');
}
if (mode !== 'dashboard') {
  $('dashboard').classList.add('hidden'); $('analysis').classList.remove('hidden');
  $('analysisTitle').textContent = mode === 'mlr' ? 'Multiple Linear Regression (MLR)' : 'Multilayer Perceptron (MLP)';
  $('mlpParams').classList.toggle('hidden', mode !== 'mlp');
}

async function json(url, opts) {
  const r = await fetch(url, {cache:'no-store', ...opts});
  let payload;
  try {payload = await r.json();} catch {throw Error(`Unexpected server response (HTTP ${r.status})`);}
  if (!r.ok || !payload.ok) throw Error(payload.error || `HTTP ${r.status}`);
  return payload;
}

async function refresh() {
  try {
    const d = await json('/api/readings?limit=300');
    fetchedRows = d.rows || [];
    banner(d.mode === 'DEMO_DATA' ? '⚠ DEMO DATA — simulated measurements. Firebase and ESP32 are NOT verified.' :
      '● FIREBASE LIVE — Sensor measurements from the cloud database.', d.mode === 'DEMO_DATA' ? 'demo' : '');
    const last = d.latest;
    if (last) {
      $('temperature').textContent = num(last.temperature);
      $('humidity').textContent = num(last.humidity);
      $('lastSeen').textContent = 'Last recorded: ' + date(last.time) + (d.mode === 'DEMO_DATA' ? ' (simulated)' : '');
    } else {
      $('temperature').textContent = '—'; $('humidity').textContent = '—';
      $('lastSeen').textContent = 'No readings yet. Check ESP32 and Firebase.';
    }
    const recent = fetchedRows.slice(-20).reverse();
    $('readingsBody').innerHTML = recent.map(r => `<tr><td>${escapes(date(r.time))}</td><td>${num(r.temperature)}</td><td>${num(r.humidity)}</td></tr>`).join('') || '<tr><td colspan="3">No data available.</td></tr>';
    if (mode === 'dashboard' && window.Plotly) {
      await Plotly.react('livePlot', [
        {x:fetchedRows.map(r=>new Date(r.time)), y:fetchedRows.map(r=>r.temperature),type:'scatter',mode:'lines',name:'Temperature (°C)'},
        {x:fetchedRows.map(r=>new Date(r.time)), y:fetchedRows.map(r=>r.humidity),type:'scatter',mode:'lines',name:'Humidity (%)',yaxis:'y2'}
      ], {margin:{t:25,l:55,r:60,b:60},legend:{orientation:'h'}, xaxis:{title:'Time'}, yaxis:{title:'°C'}, yaxis2:{title:'% RH',overlaying:'y',side:'right'},autosize:true}, {responsive:true,displaylogo:false});
    }
  } catch (e) {
    banner('ERROR loading readings: ' + e.message, 'error');
  }
}
$('refresh').addEventListener('click', refresh);
refresh();
// Tabs not visible are not aggressively polled. Keep bandwidth reasonable on free hosting.
if (mode === 'dashboard') setInterval(() => {if (!document.hidden) refresh();}, 10000);

// Only graph measurements actually used by the Python calculation. Predictions remain
// distinct from sensor readings; the first forecast point is anchored to the last
// measured point so that even a one-step prediction makes a proper line segment.
function drawForecastGraph(result) {
  if (!window.Plotly || !result) return;
  const unit = result.target === 'temperature' ? '°C' : '% RH';
  const limitValue = $('historyWindow').value;
  const selected = Array.isArray(result.history) ? result.history : [];
  const limit = limitValue === 'all' ? selected.length : Number(limitValue);
  const history = selected
    .slice(-limit)
    .map(row => ({t: Number(row.time), y: Number(row[result.target])}))
    .filter(row => Number.isFinite(row.t) && Number.isFinite(row.y))
    .sort((a,b) => a.t - b.t);
  if (!history.length) {
    $('forecastExplanation').textContent = 'No valid sensor history was returned. Rebuild Tutorial 9 with the included analysis_python.py.';
    Plotly.purge('forecastPlot');
    return;
  }
  const futureValues = (Array.isArray(result.forecast) ? result.forecast : [])
    .map(Number).filter(Number.isFinite);
  if (!futureValues.length) {
    $('forecastExplanation').textContent = 'No forecast points were returned.';
    Plotly.purge('forecastPlot');
    return;
  }
  // Prefer actual timing; ESP32 usually writes every 5 seconds.
  const intervals = [];
  for (let i=Math.max(1,history.length-60); i<history.length; i++) {
    const delta = history[i].t-history[i-1].t;
    if (delta > 0 && delta <= 3600000) intervals.push(delta);
  }
  intervals.sort((a,b) => a-b);
  const period = intervals.length ? Math.max(1000, intervals[Math.floor(intervals.length/2)]) : 5000;
  const last = history[history.length-1];
  const forecastDates = futureValues.map((_,i)=>new Date(last.t + (i+1)*period));
  const measurementX = history.map(row=>new Date(row.t));
  const forecastLineX = [new Date(last.t),...forecastDates];
  const forecastLineY = [last.y,...futureValues];
  const label = result.target === 'temperature' ? 'Temperature' : 'Humidity';
  $('forecastExplanation').textContent =
    `Showing ${history.length} of ${selected.length} actual sensor readings used in the ${result.target.toUpperCase()} analysis. ` +
    `The dashed orange line starts at the last measurement and extends ${futureValues.length} predicted step(s). ` +
    'Forecast values are predictions, not new sensor measurements.';
  const traces = [
    {x:measurementX,y:history.map(row=>row.y),type:'scatter',mode:'lines+markers',
      name:'Actual Firebase readings',line:{color:'#167aa3',width:2.5},marker:{size:4},
      hovertemplate:'Measured: %{y:.3f} '+unit+'<br>%{x|%d %b %Y, %H:%M:%S}<extra></extra>'},
    {x:forecastLineX,y:forecastLineY,type:'scatter',mode:'lines+markers',
      name:'Model forecast',line:{color:'#e47936',width:3,dash:'dash'},marker:{color:'#e47936',size:9,symbol:'diamond'},
      hovertemplate:'Forecast line: %{y:.4f} '+unit+'<br>%{x|%d %b %Y, %H:%M:%S}<extra></extra>'}
  ];
  const labelX = new Date(last.t);
  const layout = {
    title:{text:`${label}: actual sensor history + ${result.steps}-step prediction`,font:{size:17}},
    margin:{t:75,l:65,r:30,b:75},height:430,autosize:true,
    paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',
    legend:{orientation:'h',x:0,y:1.14},
    xaxis:{title:'Time (measured and estimated future)',type:'date',showgrid:true,gridcolor:'#eaf0f4'},
    yaxis:{title:`${label} (${unit})`,showgrid:true,gridcolor:'#eaf0f4',zeroline:false},
    shapes:[{type:'line',x0:labelX,x1:labelX,y0:0,y1:1,xref:'x',yref:'paper',
      line:{color:'#71879a',width:1.5,dash:'dot'}}],
    annotations:[{x:forecastDates[forecastDates.length-1],y:futureValues[futureValues.length-1],
      xref:'x',yref:'y',text:`Prediction: ${num(futureValues[futureValues.length-1],3)} ${unit}`,
      showarrow:true,arrowhead:2,ax:0,ay:-46,bgcolor:'#fff',bordercolor:'#d4e1e8',borderpad:5,font:{size:11}}]
  };
  Plotly.react('forecastPlot',traces,layout,{responsive:true,displaylogo:false});
}
$('historyWindow').addEventListener('change',()=>drawForecastGraph(lastAnalysisResult));

$('analysisForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const f = new FormData(event.currentTarget);
  const options = {
    target:f.get('target'), steps:Number(f.get('steps')), rows:Number(f.get('rows')),
    scale:f.get('scale'), removeOutliers:f.has('removeOutliers'),
    window:Number(f.get('window')), hidden:Number(f.get('hidden')),
    hiddenLayers:Number(f.get('hiddenLayers')), epochs:Number(f.get('epochs')),
    lr:Number(f.get('lr')), momentum:Number(f.get('momentum'))
  };
  $('analyzeBtn').disabled = true;
  lastAnalysisResult = null;
  $('analysisResult').textContent = 'Cloud Python server is analyzing Firebase readings…';
  try {
    const d = await json('/api/' + mode, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(options)});
    const r = d.result;
    const metrics = mode === 'mlr'
      ? [`Equation: ${r.equation}`,`Correlation: ${num(r.correlation,5)}`,`MAPE: ${num(r.mape,3)}%`, `SSX: ${num(r.ssX,3)} · SSY: ${num(r.ssY,3)}`]
      : [`Epochs: ${r.epochs} · Window: ${r.window} · Hidden layers: ${r.hiddenLayers}`,`Validation MAPE: ${num(r.validationMAPE,3)}%`,`Train MSE (scaled): ${num(r.trainMSEScaled,5)}`];
    $('analysisResult').textContent = [r.algorithm,`Target: ${r.target}`,`Readings used: ${r.count}`, ...metrics,
      `PREDICTION: ${num(r.prediction,4)} ${r.target === 'temperature'?'°C':'% RH'}`,
      `Steps ahead: ${r.steps}`,r.notes || '', d.mode === 'DEMO_DATA' ? '⚠ SIMULATED DATA ONLY' : 'Data source: Firebase RTDB'].join('\n');
    lastAnalysisResult = r;
    drawForecastGraph(r);
  } catch (e) {$('analysisResult').textContent = 'Analysis error: ' + e.message;}
  finally {$('analyzeBtn').disabled = false;}
});
