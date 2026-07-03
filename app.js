// ====== Neural Capital Research · app.js (compartido por todas las páginas internas) ======
const NOMBRE   = "Neural Capital Research";
const TELEGRAM = "https://t.me/neural_capital_research";

// ---------- Barra de navegación ----------
function navHTML(active){
  const modelos = [
    ['par_oro_plata', 'Par oro-plata'],
    ['par_platino_paladio', 'Par platino-paladio'],
    ['oro_bh',        'Oro'],
    ['plata_bh',      'Plata'],
    ['koncorde',      'KONCORDE (S&P 500)'],
    ['figuras_tecnicas', 'Figuras técnicas (S&P 500)'],
    ['figuras_intradia', 'Figuras intradía (1h)'],
  ];
  const mods = modelos.map(([id,n]) =>
    `<a href="lab.html?id=${id}" class="${id===active?'active':''}">${n}</a>`).join('');
  const metales = [
    ['garch_oro','Oro'],['garch_plata','Plata'],['garch_platino','Platino'],
    ['garch_paladio','Paladio'],['garch_cobre','Cobre'],
  ];
  const vols = `<a href="lab.html?id=panel_metales" class="${active==='panel_metales'?'active':''}"><b>Panel de metales</b></a>`
    + metales.map(([id,n]) =>
      `<a href="lab.html?id=${id}" class="${id===active?'active':''}">${n}</a>`).join('');
  return `<div class="nav-inner">
    <a class="nav-logo" href="lab.html?id=par_oro_plata">Neural <b>Capital</b> Research</a>
    <div class="nav-menu">
      <div class="nav-item"><button class="nav-trigger" type="button">Modelos ▾</button>
        <div class="nav-drop">${mods}</div></div>
      <div class="nav-item"><button class="nav-trigger" type="button">Volatilidad ▾</button>
        <div class="nav-drop">${vols}</div></div>
      <div class="nav-item"><button class="nav-trigger" type="button">Operaciones ▾</button>
        <div class="nav-drop">
          <a href="lab.html?id=koncorde#operaciones">Acciones (S&P 500)</a>
          <a href="lab.html?id=par_oro_plata#operaciones">Oro</a>
          <a href="lab.html?id=par_oro_plata#operaciones">Plata</a>
        </div></div>
      <a class="nav-link ${active==='visor'?'active':''}" href="visor.html">Visor</a>
      <div class="nav-item"><button class="nav-trigger" type="button">Metodología ▾</button>
        <div class="nav-drop">
          <a href="metodologia.html#como-se-lee" class="${active==='metodologia'?'active':''}">Cómo se lee</a>
          <a href="metodologia.html#los-modelos">Los modelos</a>
          <a href="metodologia.html#motores">Motores del oro y la plata</a>
          <a href="metodologia.html#fallos">Por qué casi nada funciona</a>
        </div></div>
      <a class="nav-link" href="${TELEGRAM}" target="_blank" rel="noopener">El canal ↗</a>
    </div>
  </div>`;
}
function mountNav(active){
  const el = document.getElementById('topnav');
  if(el){ el.className = 'topnav'; el.innerHTML = navHTML(active); }
}

// ---------- Revelado al hacer scroll ----------
const _io = ('IntersectionObserver' in window)
  ? new IntersectionObserver((entries)=>{
      entries.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); _io.unobserve(e.target); } });
    }, {threshold:0.12, rootMargin:'0px 0px -8% 0px'})
  : null;
function observeReveals(){
  document.querySelectorAll('.sreveal:not(.in)').forEach(el=>{ if(_io) _io.observe(el); else el.classList.add('in'); });
}

// ---------- Render de un modelo ----------
let chart = null;
let hzChart = null;
let pvChart = null;
let pvPanelChart = null;
let histPanelChart = null;
let figCharts = [];
function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }

function render(exp){
  const h = exp.headline, s = exp.significancia, d = exp.diagnostico || {};
  const app = document.getElementById('app');
  const titulo = `<h1 class="model-title">${exp.etiqueta || ''}</h1>`;

  if(exp.panel){
    app.innerHTML = titulo + `<div class="tipo">▸ ${exp.tipo}</div>${panelHTML(exp)}`;
    drawPanelPrev(exp);
    drawPanelHist(exp);
    requestAnimationFrame(()=>document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in')));
    return;
  }

  if(exp.figuras_panel){
    app.innerHTML = titulo + `<div class="tipo">▸ ${exp.tipo}</div>${figurasHTML(exp)}`;
    drawFiguras(exp);
    requestAnimationFrame(()=>document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in')));
    return;
  }

  if(exp.sin_datos){
    app.innerHTML = titulo + `<div class="tipo">▸ ${exp.tipo}</div>
      <div class="empty"><b>Sin datos suficientes todavía</b>${exp.sin_datos_txt || 'El forward-test necesita operaciones cerradas; se irá llenando con el tiempo.'}</div>
      ${(exp.cards&&exp.cards.length)?`<div class="grid" style="margin-top:18px">${exp.cards.map(c=>`<div class="card"><div class="k">${c.k}</div><div class="v ${c.tono||''}">${c.v}</div></div>`).join("")}</div>`:''}
      ${opsTable(exp)}
      ${horizonteHTML(exp)}`;
    drawHorizonte(exp);
    requestAnimationFrame(()=>document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in')));
    return;
  }

  const sig = s.ic90[0] > 0;
  const pts = [0, s.ic90[0], s.ic90[1], h.valor];
  let lo = Math.min(...pts), hi = Math.max(...pts);
  const pad = (hi - lo) * 0.18 || 1; lo -= pad; hi += pad;
  const X = v => clamp((v - lo)/(hi - lo)*100, 0, 100);
  const dec = Math.abs(hi - lo) > 6 ? 0 : 1;
  let ticks = "";
  for(let i=0;i<=4;i++){ const t = lo + (hi-lo)*i/4; ticks += `<span style="left:${X(t)}%">${t.toFixed(dec)}</span>`; }
  const sfx = h.sufijo || "";

  app.innerHTML = titulo + `
   <div class="tipo">▸ ${exp.tipo}</div>
   <div class="verdict reveal">
     <div class="verdict-top">
       <div>
         <div class="sharpe-big">${h.valor.toFixed(h.decimales)}${sfx}</div>
         <div class="sharpe-lab">${h.etiqueta}</div>
       </div>
       <div class="verdict-tag ${sig?'v-good':'v-bad'}">
         ${sig ? '✓ DISTINGUIBLE DEL AZAR' : '✕ NO DISTINGUIBLE DEL AZAR'}
       </div>
     </div>
     <div class="numline">
       <div class="nl-track">
         <div class="nl-axis"></div>
         <div class="nl-ci" style="left:${X(s.ic90[0])}%; width:${X(s.ic90[1])-X(s.ic90[0])}%; background:${sig?'var(--good)':'var(--warn)'}"></div>
         <div class="nl-pt" style="left:${X(h.valor)}%"></div>
         <div class="nl-zero" style="left:${X(0)}%"><span>0</span></div>
       </div>
       <div class="nl-labels">${ticks}</div>
       <div class="nl-cap">Intervalo de confianza al 90% (${s.etiqueta}): <b>[${s.ic90[0]}, ${s.ic90[1]}]</b> · p-valor (≤ 0): <b>${s.p_valor}</b>. La barra es el intervalo; la línea blanca, el valor observado; la marca "0", la frontera del azar.</div>
     </div>
   </div>

   <div class="grid reveal">${(exp.cards||[]).map(c=>`<div class="card"><div class="k">${c.k}</div><div class="v ${c.tono||''}">${c.v}</div></div>`).join("")}</div>

   ${Object.keys(d).length ? `<div class="diag reveal">${diagChips(d)}</div>` : ''}

   ${previsionHTML(exp)}

   <div class="chartbox reveal">
     <h3>${exp.curva_titulo || 'Curva de capital fuera de muestra'}</h3>
     <div class="ch-sub">${exp.curva_sub || 'Crecimiento de 1 unidad invertida según el modelo. La línea base es no hacer nada.'}</div>
     <div class="canvas-h"><canvas id="eq"></canvas></div>
   </div>

   ${opsTable(exp)}
   ${horizonteHTML(exp)}
  `;

  drawChart(exp);
  drawPrevision(exp);
  drawHorizonte(exp);
  requestAnimationFrame(()=>document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in')));
}

function opsTable(exp){
  if(!exp.op_cols) return '';
  const cols = exp.op_cols, ops = exp.operaciones || [];
  const cell = (o,c)=>{
    let v = o[c.k]; let cls = '';
    if(v===null||v===undefined||v==='') return `<td class="est-obs">—</td>`;
    if(c.k==='retorno'){ const n=parseFloat(v); cls=n>=0?'pos':'neg'; v=(n>=0?'+':'')+v+(c.sufijo||''); }
    else if(c.k==='estado'){ cls = (''+v).indexOf('cerr')>=0?'est-cerr':'est-obs'; }
    else if(c.k==='accion_plata'||c.k==='accion_oro'){ cls=(''+v).indexOf('COMPRAR')>=0?'pos':((''+v).indexOf('VENDER')>=0?'neg':''); }
    else if(c.sufijo){ v = v+c.sufijo; }
    return `<td class="${cls}">${v}</td>`;
  };
  const head = cols.map(c=>`<th>${c.t}</th>`).join('');
  const body = ops.length
    ? ops.map(o=>`<tr>${cols.map(c=>cell(o,c)).join('')}</tr>`).join('')
    : `<tr><td colspan="${cols.length}" class="ops-empty">Aún no hay operaciones registradas. Aparecerán aquí en cuanto el modelo dispare una señal en vivo.</td></tr>`;
  const par = exp.id==='par_oro_plata';
  const titulo = par ? 'Operaciones en vivo · par oro-plata' : 'Operaciones · forward-test del S&P 500';
  const sub = par
    ? 'Cada fila es una señal del par, con su pata de plata y su pata de oro. Solo dispara cuando el par está cointegrado y el ratio se desvía ≥2σ.'
    : 'Cada señal (decidida al cierre) compra en la apertura siguiente y vende cuando el indicador gira a la baja (su propia señal de venta). "En observación" = aún sin cerrar.';
  return `<div class="opsbox reveal" id="operaciones"><h3>${titulo}</h3><div class="ch-sub">${sub}</div>
     <div class="ops-scroll"><table class="ops"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div></div>`;
}

function diagChips(d){
  const lab = {cointegrated:'Cointegrado', coint_pvalue:'p-valor cointegración',
    beta:'Ratio de cobertura β', half_life:'Vida media reversión',
    horizonte:'Horizonte', salida:'Salida', coste:'Coste por operación', operaciones_abiertas:'En observación'};
  let out = "";
  for(const k in d){
    let v = d[k];
    if(v === null || v === undefined) continue;
    if(k === 'cointegrated') v = v ? 'sí' : 'no';
    if(k === 'half_life') v = Math.round(v) + ' días';
    out += `<div class="chip">${lab[k] || k}: <b>${v}</b></div>`;
  }
  return out;
}

function drawChart(exp){
  const ctx = document.getElementById('eq');
  if(chart) chart.destroy();
  const curva = exp.curva || [];
  const labels = curva.map(p=>p.fecha);
  const unidad = exp.curva_unidad || '×';
  const base = (exp.curva_base !== undefined) ? exp.curva_base : 1;
  const col = exp.curva_color || '#e8b23a';
  const rgb = col.replace('#','').match(/.{2}/g).map(x=>parseInt(x,16)).join(',');
  const datasets = [
    {label:'Modelo', data:curva.map(p=>p.valor), borderColor:col, borderWidth:2, pointRadius:0, tension:.12,
     fill:true, backgroundColor:(c)=>{const g=c.chart.ctx.createLinearGradient(0,0,0,300); g.addColorStop(0,`rgba(${rgb},.18)`); g.addColorStop(1,`rgba(${rgb},0)`); return g;}}
  ];
  if(exp.curva2 && exp.curva2.datos){
    datasets.push({label: exp.curva2.nombre || 'Referencia', data: exp.curva2.datos.map(p=>p.valor),
      borderColor:'#5fb7c4', borderWidth:2, pointRadius:0, tension:.12});
  }
  datasets.push({label:'base', data:labels.map(()=>base), borderColor:'#5c6775', borderWidth:1, borderDash:[5,5], pointRadius:0});
  const showLegend = !!(exp.curva2 && exp.curva2.datos);
  chart = new Chart(ctx,{
    type:'line',
    data:{labels, datasets},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{display:showLegend, labels:{color:'#8a97a6', filter:i=>i.text!=='base', font:{family:'JetBrains Mono'}}},
        tooltip:{callbacks:{label:c=>c.dataset.label==='base'?null:'  '+c.dataset.label+': '+c.parsed.y.toFixed(2)+unidad, title:i=>i[0].label}}
      },
      scales:{
        x:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', font:{family:'JetBrains Mono'}, autoSkip:false, maxRotation:0,
            callback:function(value,index){
              const n=this.chart.data.labels.length;
              const keep=new Set([0, Math.round(n*0.2), Math.round(n*0.4), Math.round(n*0.6), Math.round(n*0.8), n-1]);
              return keep.has(index)? this.getLabelForValue(value) : '';
            }}},
        y:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', font:{family:'JetBrains Mono'}, callback:v=>v+unidad}}
      }
    }
  });
}

// ---------- Figuras técnicas (event study con FDR) ----------
function figurasHTML(exp){
  const u = '%';
  const bloques = (exp.figuras||[]).map((f,idx)=>{
    const rows = f.puntos.map(p=>{
      const cruda = p.sig_cruda ? '<b>✓</b>' : '<span class="est-obs">—</span>';
      const fdr = p.sig_fdr ? '<b class="pos">✓ real</b>' : '<span class="est-obs">—</span>';
      return `<tr><td>${p.etiqueta}</td>
        <td class="${p.valor>=0?'pos':'neg'}">${p.valor>=0?'+':''}${p.valor}${u}</td>
        <td class="est-obs">[${p.ic_lo}, ${p.ic_hi}]</td>
        <td style="text-align:center">${cruda}</td><td style="text-align:center">${fdr}</td>
        <td class="est-obs">${p.n}</td></tr>`;
    }).join('');
    return `<div class="chartbox reveal">
      <h3><span class="dot" style="background:${f.color}"></span>${f.nombre}
        <span class="est-obs" style="font-size:13px">· ${f.n_eventos} eventos</span></h3>
      <div class="canvas-h" style="height:180px"><canvas id="fig${idx}"></canvas></div>
      <div class="ops-scroll" style="margin-top:12px"><table class="ops">
        <thead><tr><th>Horizonte</th><th>Ventaja media</th><th>IC 90%</th>
          <th style="text-align:center">Crudo</th><th style="text-align:center">Tras FDR</th><th>n</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </div>`;
  }).join('');
  return `
    <div class="chartbox reveal"><h3>Cómo leer esto</h3>
      <div class="ch-sub">${exp.intro}</div>
      <div class="ch-sub" style="margin-top:12px"><b>${exp.nota_fdr}</b></div>
      ${(exp.tickers&&exp.tickers.length)?`<details style="margin-top:12px"><summary class="ch-sub" style="cursor:pointer">▸ Valores analizados (${exp.tickers.length}) — transparencia total</summary><div class="ch-sub" style="margin-top:8px;font-family:'JetBrains Mono',monospace;line-height:1.7">${exp.tickers.join(' · ')}</div></details>`:''}
    </div>
    ${bloques}
    <div class="chartbox reveal"><div class="ch-sub">${exp.nota}</div></div>`;
}

function drawFiguras(exp){
  figCharts.forEach(c=>{ try{c.destroy();}catch(e){} });
  figCharts = [];
  (exp.figuras||[]).forEach((f,idx)=>{
    const ctx = document.getElementById('fig'+idx); if(!ctx) return;
    const labels = f.puntos.map(p=>p.etiqueta);
    const central = f.puntos.map(p=>p.valor);
    const hi = f.puntos.map(p=>p.ic_hi), lo = f.puntos.map(p=>p.ic_lo);
    const ch = new Chart(ctx,{type:'line', data:{labels, datasets:[
      {label:'IC sup', data:hi, borderColor:'transparent', pointRadius:0, fill:'+1', backgroundColor:'rgba(150,160,175,.12)'},
      {label:'IC inf', data:lo, borderColor:'transparent', pointRadius:0, fill:false},
      {label:'Ventaja', data:central, borderColor:f.color, borderWidth:2.4, pointRadius:3, pointBackgroundColor:f.color, tension:.1},
      {label:'cero', data:labels.map(()=>0), borderColor:'#5c6775', borderWidth:1, borderDash:[5,5], pointRadius:0}
    ]}, options:{responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>['IC sup','IC inf','cero'].includes(c.dataset.label)?null:'  '+(c.parsed.y>=0?'+':'')+c.parsed.y.toFixed(2)+'%'}}},
      scales:{x:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#8a97a6', font:{family:'JetBrains Mono'}}},
        y:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', font:{family:'JetBrains Mono'}, callback:v=>v+'%'}}}}});
    figCharts.push(ch);
  });
}

// ---------- Panel conjunto de volatilidad de metales ----------
function panelHTML(exp){
  const u = exp.unidad || '%';
  const ms = exp.metales || [];
  const leyenda = ms.map(m=>`<span class="lg"><i style="background:${m.color}"></i>${m.nombre}</span>`).join('');
  // tabla resumen: previsión a 1 día y media de largo plazo por metal
  const filas = ms.map(m=>`<tr>
     <td><span class="dot" style="background:${m.color}"></span>${m.nombre}</td>
     <td class="pos">${m.actual!=null?m.actual+u:'—'}</td>
     <td class="est-obs">${m.largo_plazo!=null?m.largo_plazo+u:'—'}</td></tr>`).join('');
  return `
    <div class="chartbox reveal" id="prevision">
      <h3>${exp.prev_titulo}</h3>
      <div class="ch-sub">${exp.prev_sub}</div>
      <div class="lg-row">${leyenda}</div>
      <div class="canvas-h"><canvas id="pvpanel"></canvas></div>
      <div class="ops-scroll" style="margin-top:14px"><table class="ops">
        <thead><tr><th>Metal</th><th>Espera ahora · 1 día</th><th>Media largo plazo</th></tr></thead>
        <tbody>${filas}</tbody></table></div>
    </div>
    <div class="chartbox reveal">
      <h3>${exp.hist_titulo}</h3>
      <div class="ch-sub">${exp.hist_sub}</div>
      <div class="lg-row">${leyenda}</div>
      <div class="canvas-h"><canvas id="histpanel"></canvas></div>
      ${exp.nota?`<div class="ch-sub" style="margin-top:14px">${exp.nota}</div>`:''}
    </div>`;
}

function drawPanelPrev(exp){
  if(pvPanelChart){ pvPanelChart.destroy(); pvPanelChart = null; }
  const ms = exp.metales || [];
  const ctx = document.getElementById('pvpanel'); if(!ctx) return;
  const labels = exp.plazos || (ms[0] ? ms[0].prev.map(p=>p.etiqueta) : []);
  const u = exp.unidad || '%';
  const datasets = ms.map(m=>({
    label:m.nombre, data:m.prev.map(p=>p.vol), borderColor:m.color, backgroundColor:m.color,
    borderWidth:2.4, pointRadius:3, pointBackgroundColor:m.color, tension:.1
  }));
  pvPanelChart = new Chart(ctx,{type:'line', data:{labels, datasets}, options:{
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>'  '+c.dataset.label+': '+c.parsed.y.toFixed(1)+u}}},
    scales:{x:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#8a97a6', font:{family:'JetBrains Mono'}}},
      y:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', font:{family:'JetBrains Mono'}, callback:v=>v+u}}}
  }});
}

function drawPanelHist(exp){
  if(histPanelChart){ histPanelChart.destroy(); histPanelChart = null; }
  const ms = exp.metales || [];
  const ctx = document.getElementById('histpanel'); if(!ctx) return;
  const u = exp.unidad || '%';
  // eje X común: fechas del metal con más historia
  let base = []; ms.forEach(m=>{ if(m.hist && m.hist.length>base.length) base = m.hist.map(p=>p.fecha); });
  const datasets = ms.map(m=>{
    const mp = new Map((m.hist||[]).map(p=>[p.fecha, p.valor]));
    return {label:m.nombre, data:base.map(f=>mp.has(f)?mp.get(f):null), borderColor:m.color,
      backgroundColor:m.color, borderWidth:1.4, pointRadius:0, tension:.15, spanGaps:true};
  });
  histPanelChart = new Chart(ctx,{type:'line', data:{labels:base, datasets}, options:{
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}, tooltip:{mode:'index', intersect:false,
      callbacks:{label:c=>c.parsed.y==null?null:'  '+c.dataset.label+': '+c.parsed.y.toFixed(1)+u}}},
    scales:{x:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', maxTicksLimit:6, font:{family:'JetBrains Mono'}}},
      y:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', font:{family:'JetBrains Mono'}, callback:v=>v+u}}}
  }});
}

// ---------- Previsión actual de volatilidad (estructura de plazos) ----------
function previsionHTML(exp){
  const pv = exp.prevision;
  if(!pv || !pv.puntos || !pv.puntos.length) return '';
  const u = pv.unidad || '%';
  const rows = pv.puntos.map(p=>`<tr><td>${p.etiqueta}</td><td class="pos">${p.vol}${u}</td>
     <td class="est-obs">[${p.lo}, ${p.hi}]</td></tr>`).join('');
  return `<div class="chartbox reveal" id="prevision">
     <h3>${pv.titulo}</h3>
     <div class="ch-sub">${pv.sub}${pv.fecha?` · última sesión: <b>${pv.fecha}</b>`:''}</div>
     <div class="grid" style="margin:16px 0 4px">
       <div class="card"><div class="k">Espera ahora · 1 día</div><div class="v pos">${pv.actual}${u}</div></div>
       <div class="card"><div class="k">Media de largo plazo</div><div class="v">${pv.largo_plazo}${u}</div></div>
     </div>
     <div class="ch-sub" style="margin:8px 0 14px"><b>Régimen actual:</b> ${pv.regimen}</div>
     <div class="canvas-h"><canvas id="pv"></canvas></div>
     <div class="ops-scroll" style="margin-top:14px"><table class="ops">
       <thead><tr><th>Horizonte</th><th>Volatilidad esperada</th><th>Banda (5–95%)</th></tr></thead>
       <tbody>${rows}</tbody></table></div>
     ${pv.nota?`<div class="ch-sub" style="margin-top:14px">${pv.nota}</div>`:''}
   </div>`;
}

function drawPrevision(exp){
  const pv = exp.prevision;
  if(pvChart){ pvChart.destroy(); pvChart = null; }
  if(!pv || !pv.puntos || !pv.puntos.length) return;
  const ctx = document.getElementById('pv');
  if(!ctx) return;
  const labels = pv.puntos.map(p=>p.etiqueta);
  const vol = pv.puntos.map(p=>p.vol);
  const hi = pv.puntos.map(p=>p.hi);
  const lo = pv.puntos.map(p=>p.lo);
  const u = pv.unidad || '%';
  const lr = pv.largo_plazo;
  const col = pv.color || '#e8b23a';
  pvChart = new Chart(ctx,{
    type:'line',
    data:{labels, datasets:[
      {label:'banda sup', data:hi, borderColor:'transparent', pointRadius:0, fill:'+1',
       backgroundColor:'rgba(232,178,58,.13)'},
      {label:'banda inf', data:lo, borderColor:'transparent', pointRadius:0, fill:false},
      {label:'Volatilidad esperada', data:vol, borderColor:col, borderWidth:2.5,
       pointRadius:4, pointBackgroundColor:col, tension:.1},
      {label:'media de largo plazo', data:labels.map(()=>lr), borderColor:'#5fb7c4',
       borderWidth:1, borderDash:[5,5], pointRadius:0}
    ]},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{callbacks:{label:c=>['banda sup','banda inf'].includes(c.dataset.label)?null:
          '  '+c.dataset.label+': '+c.parsed.y.toFixed(1)+u, title:i=>'Horizonte: '+i[0].label}}
      },
      scales:{
        x:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#8a97a6', font:{family:'JetBrains Mono'}}},
        y:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', font:{family:'JetBrains Mono'}, callback:v=>v+u}}
      }
    }
  });
}

// ---------- Ventaja según el horizonte ----------
function horizonteHTML(exp){
  if(exp.horizonte_na){
    return `<div class="chartbox reveal" id="horizonte"><h3>Ventaja según el horizonte</h3>
      <div class="ch-sub">${exp.horizonte_na}</div></div>`;
  }
  const hz = exp.horizonte;
  if(!hz || !hz.puntos || !hz.puntos.length) return '';
  const u = hz.unidad || '%';
  const rows = hz.puntos.map(p=>{
    const sig = (p.ic_lo > 0) || (p.ic_hi < 0);
    const tag = sig ? '<b>✓ distinguible</b>' : '<span class="est-obs">— abraza el 0</span>';
    return `<tr><td>${p.etiqueta}</td><td class="${p.valor>=0?'pos':'neg'}">${p.valor>=0?'+':''}${p.valor}${u}</td>
      <td class="est-obs">[${p.ic_lo}, ${p.ic_hi}]</td><td>${tag}</td><td class="est-obs">${p.n}</td></tr>`;
  }).join('');
  return `<div class="chartbox reveal" id="horizonte">
     <h3>${hz.titulo}</h3>
     <div class="ch-sub">${hz.sub}</div>
     <div class="canvas-h"><canvas id="hz"></canvas></div>
     <div class="ops-scroll" style="margin-top:14px"><table class="ops">
       <thead><tr><th>Horizonte</th><th>Ventaja media</th><th>IC 90%</th><th>Veredicto</th><th>n</th></tr></thead>
       <tbody>${rows}</tbody></table></div>
     ${hz.nota?`<div class="ch-sub" style="margin-top:14px">${hz.nota}</div>`:''}
     ${(hz.tickers&&hz.tickers.length)?`<details style="margin-top:10px"><summary class="ch-sub" style="cursor:pointer">▸ Valores analizados (${hz.tickers.length})</summary><div class="ch-sub" style="margin-top:8px;font-family:'JetBrains Mono',monospace;line-height:1.7">${hz.tickers.join(' · ')}</div></details>`:''}
   </div>`;
}

function drawHorizonte(exp){
  const hz = exp.horizonte;
  if(hzChart){ hzChart.destroy(); hzChart = null; }
  if(!hz || !hz.puntos || !hz.puntos.length) return;
  const ctx = document.getElementById('hz');
  if(!ctx) return;
  const labels = hz.puntos.map(p=>p.etiqueta);
  const central = hz.puntos.map(p=>p.valor);
  const hi = hz.puntos.map(p=>p.ic_hi);
  const lo = hz.puntos.map(p=>p.ic_lo);
  const u = hz.unidad || '%';
  const col = exp.color || '#e8b23a';
  hzChart = new Chart(ctx,{
    type:'line',
    data:{labels, datasets:[
      {label:'IC sup', data:hi, borderColor:'transparent', pointRadius:0, fill:'+1',
       backgroundColor:'rgba(232,178,58,.13)'},
      {label:'IC inf', data:lo, borderColor:'transparent', pointRadius:0, fill:false},
      {label:'Ventaja media', data:central, borderColor:col, borderWidth:2.5,
       pointRadius:4, pointBackgroundColor:col, tension:.1},
      {label:'cero', data:labels.map(()=>0), borderColor:'#5c6775', borderWidth:1,
       borderDash:[5,5], pointRadius:0}
    ]},
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{callbacks:{label:c=>['IC sup','IC inf','cero'].includes(c.dataset.label)?null:
          '  '+c.dataset.label+': '+(c.parsed.y>=0?'+':'')+c.parsed.y.toFixed(2)+u, title:i=>'Horizonte: '+i[0].label}}
      },
      scales:{
        x:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#8a97a6', font:{family:'JetBrains Mono'}}},
        y:{grid:{color:'rgba(38,48,61,.4)'}, ticks:{color:'#5c6775', font:{family:'JetBrains Mono'}, callback:v=>v+u}}
      }
    }
  });
}

// ---------- Arranque del laboratorio ----------
async function initLab(){
  let id0 = new URLSearchParams(location.search).get('id') || 'par_oro_plata';
  if(id0 === 'garch_vol') id0 = 'garch_oro';   // compatibilidad con el enlace antiguo
  mountNav(id0);
  let doc;
  try{
    const r = await fetch('resultados.json?ts='+Date.now());
    if(!r.ok) throw new Error();
    doc = await r.json();
  }catch(e){
    document.getElementById('meta').textContent = '';
    document.getElementById('aviso').style.display = 'none';
    document.getElementById('app').innerHTML =
      `<div class="empty"><b>Aún no hay resultados</b>Lanza el laboratorio desde la pestaña <b>Actions</b> del repositorio para generar el primer <code>resultados.json</code>.</div>`;
    return;
  }
  document.getElementById('meta').textContent = 'Última corrida: ' + doc.generado;
  document.getElementById('aviso-txt').textContent = doc.aviso;
  if(doc.sintetico){
    const s = document.getElementById('syn');
    s.style.display = 'block';
    s.innerHTML = `<span class="badge-syn">⚠ datos de verificación · no reales</span>`;
  }
  const exp = doc.experimentos.find(e=>e.id===id0);
  if(!exp){
    mountNav(id0);
    document.getElementById('meta').textContent = 'Última corrida: ' + doc.generado;
    document.getElementById('aviso').style.display = 'none';
    document.getElementById('app').innerHTML =
      `<div class="empty"><b>Aún no disponible</b>Este experimento todavía no está en la última corrida del laboratorio. Puede estar calculándose ahora mismo, o no haber tenido datos suficientes hoy —algo habitual en el intradía si Yahoo limita las descargas—. Vuelve a probar cuando termine la corrida (pestaña <b>Actions</b> del repositorio).</div>`;
    return;
  }
  mountNav(exp.id);
  render(exp);
  if(location.hash === '#operaciones'){
    setTimeout(()=>{ const o = document.getElementById('operaciones'); if(o) o.scrollIntoView({behavior:'smooth', block:'start'}); }, 350);
  }
}

// ---------- Visor de gráficos (velas + figuras dibujadas) ----------
let VISOR = { graf:null, verd:{diario:{}, intradia:{}}, tf:'diario', tk:null };

async function initVisor(){
  mountNav('visor');
  document.getElementById('meta').textContent = '';
  const aviso = document.getElementById('aviso'); if(aviso) aviso.style.display = 'none';
  const app = document.getElementById('app');
  let graf, res;
  try{
    graf = await fetch('graficos.json?ts='+Date.now()).then(r=>{ if(!r.ok) throw 0; return r.json(); });
    res  = await fetch('resultados.json?ts='+Date.now()).then(r=>r.ok?r.json():null).catch(()=>null);
  }catch(e){
    app.innerHTML = `<div class="empty"><b>Aún no disponible</b>El visor necesita que el laboratorio genere <code>graficos.json</code>. Vuelve cuando termine la próxima corrida (pestaña <b>Actions</b>).</div>`;
    return;
  }
  VISOR.graf = graf;
  const mapaVerd = (idExp, etiqHz)=>{
    const m = {}; if(!res) return m;
    const e = (res.experimentos||[]).find(x=>x.id===idExp); if(!e) return m;
    (e.figuras||[]).forEach(f=>{
      const p = (f.puntos||[]).find(p=>p.etiqueta===etiqHz) || (f.puntos&&f.puntos[f.puntos.length-1]);
      if(p) m[f.tipo] = {valor:p.valor, fdr:p.sig_fdr};
    });
    return m;
  };
  VISOR.verd.diario   = mapaVerd('figuras_tecnicas','3 meses');
  VISOR.verd.intradia = mapaVerd('figuras_intradia','2 sem');

  app.innerHTML = `
    <div class="visor-head">
      <h2 class="ch-title">Visor de gráficos</h2>
      <div class="ch-sub">Las figuras que detecta el laboratorio, dibujadas sobre el precio real con reglas fijas. Cada figura lleva su veredicto histórico —lo que valió, no lo que promete—. Datos del último cierre, no en vivo.</div>
    </div>
    <div class="visor-controls">
      <div class="tf-toggle">
        <button data-tf="diario" class="tf-btn active">Diario</button>
        <button data-tf="intradia" class="tf-btn">Intradía 1h</button>
      </div>
      <div class="tk-combo">
        <input id="tk-input" type="text" placeholder="Buscar valor…" autocomplete="off" spellcheck="false" aria-label="Buscar valor">
        <div id="tk-list" class="tk-list" hidden></div>
      </div>
    </div>
    <div class="chartbox" style="padding:14px"><div class="visor-canvas-wrap"><canvas id="velas"></canvas></div></div>
    <div id="visor-leyenda"></div>
  `;
  app.querySelectorAll('.tf-btn').forEach(b=>b.addEventListener('click',()=>{
    VISOR.tf = b.dataset.tf;
    app.querySelectorAll('.tf-btn').forEach(x=>x.classList.toggle('active', x===b));
    poblarTickers(); renderVisor();
  }));
  const inp = document.getElementById('tk-input');
  const lista = document.getElementById('tk-list');
  inp.addEventListener('focus', comboAbrir);
  inp.addEventListener('input', ()=>comboRender(inp.value));
  inp.addEventListener('keydown', e=>{
    const opts = [...lista.querySelectorAll('.tk-opt')];
    if(e.key==='ArrowDown'){ e.preventDefault(); if(lista.hidden) comboAbrir(); VISOR._hl = Math.min((VISOR._hl ?? -1)+1, opts.length-1); }
    else if(e.key==='ArrowUp'){ e.preventDefault(); VISOR._hl = Math.max((VISOR._hl ?? 0)-1, 0); }
    else if(e.key==='Enter'){ e.preventDefault(); const o = opts[VISOR._hl] || opts[0]; if(o) comboElegir(o.dataset.tk); return; }
    else if(e.key==='Escape'){ comboCerrar(); inp.blur(); return; }
    else return;
    [...lista.querySelectorAll('.tk-opt')].forEach((o,i)=>o.classList.toggle('hl', i===VISOR._hl));
    if(opts[VISOR._hl]) opts[VISOR._hl].scrollIntoView({block:'nearest'});
  });
  lista.addEventListener('mousedown', e=>{ const o = e.target.closest('.tk-opt'); if(o){ e.preventDefault(); comboElegir(o.dataset.tk); } });
  document.addEventListener('click', e=>{ if(!e.target.closest('.tk-combo')) comboCerrar(); });
  poblarTickers(); renderVisor();
  let rt; window.addEventListener('resize', ()=>{ clearTimeout(rt); rt=setTimeout(renderVisor,150); });
}

function poblarTickers(){
  const tickers = Object.keys(VISOR.graf[VISOR.tf]||{}).sort();
  if(!VISOR.tk || !tickers.includes(VISOR.tk)) VISOR.tk = tickers[0] || null;
  const inp = document.getElementById('tk-input');
  if(inp) inp.value = VISOR.tk || '';
  comboCerrar();
}

function comboRender(filtro){
  const lista = document.getElementById('tk-list'); if(!lista) return;
  const all = Object.keys(VISOR.graf[VISOR.tf]||{}).sort();
  const q = (filtro||'').trim().toUpperCase();
  const items = all.filter(t=>t.toUpperCase().includes(q));
  lista.innerHTML = items.length
    ? items.map(t=>`<div class="tk-opt${t===VISOR.tk?' sel':''}" data-tk="${t}">${t}</div>`).join('')
    : '<div class="tk-empty">Sin coincidencias</div>';
  VISOR._hl = -1;
}
function comboAbrir(){ const l=document.getElementById('tk-list'); if(!l) return; l.hidden=false; comboRender(document.getElementById('tk-input').value); }
function comboCerrar(){ const l=document.getElementById('tk-list'); if(l) l.hidden=true; }
function comboElegir(tk){
  if(!tk) return;
  VISOR.tk = tk;
  const inp = document.getElementById('tk-input'); if(inp) inp.value = tk;
  comboCerrar(); renderVisor();
}

function renderVisor(){
  const ley = document.getElementById('visor-leyenda');
  const hayTickers = Object.keys(VISOR.graf[VISOR.tf]||{}).length > 0;
  if(!hayTickers){
    dibujarVelas(null);
    if(ley) ley.innerHTML = `<div class="chartbox reveal in"><div class="ch-sub">Esta temporalidad todavía no tiene datos en la última corrida —habitual en el intradía si Yahoo limitó las descargas—. Prueba la otra temporalidad o vuelve tras la próxima corrida.</div></div>`;
    return;
  }
  const datos = (VISOR.graf[VISOR.tf]||{})[VISOR.tk];
  if(!datos){ dibujarVelas(null); if(ley) ley.innerHTML=''; return; }
  dibujarVelas(datos);
  const verd = VISOR.verd[VISOR.tf] || {};
  const vistos = {}; (datos.figuras||[]).forEach(f=>{ vistos[f.tipo]=f; });
  const filas = Object.values(vistos).map(f=>{
    const v = verd[f.tipo]; let txt = 'aún sin backtest';
    if(v){
      if(v.fdr && v.valor<0) txt = `tiende a fallar (revierte) · ${v.valor>0?'+':''}${v.valor}%`;
      else if(v.fdr && v.valor>0) txt = `ventaja a favor (rara) · +${v.valor}%`;
      else txt = `sin ventaja fiable · ${v.valor>0?'+':''}${v.valor}%`;
    }
    return `<div class="vf-row"><span class="dot" style="background:${f.color}"></span><b>${f.nombre}</b><span class="est-obs"> — ${txt}${f.fuerza?` · 💪${f.fuerza}`:''}</span></div>`;
  }).join('');
  if(ley) ley.innerHTML = `<div class="chartbox reveal in"><h3>Figuras en este gráfico</h3>${filas || '<div class="ch-sub">No hay figuras en la ventana visible de este valor.</div>'}</div>`;
}

function dibujarVelas(datos){
  const cv = document.getElementById('velas'); if(!cv) return;
  const cssW = (cv.parentElement.clientWidth||800), cssH = 420;
  const dpr = window.devicePixelRatio||1;
  cv.width = cssW*dpr; cv.height = cssH*dpr; cv.style.width = cssW+'px'; cv.style.height = cssH+'px';
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,cssW,cssH);
  if(!datos || !datos.velas || !datos.velas.length) return;
  const velas = datos.velas, figs = datos.figuras||[];
  const padL=8, padR=60, padT=12, padB=22;
  const plotW = cssW-padL-padR, plotH = cssH-padT-padB;
  let lo=Infinity, hi=-Infinity;
  velas.forEach(v=>{ if(v[2]<lo)lo=v[2]; if(v[1]>hi)hi=v[1]; });
  figs.forEach(f=>f.trazos.forEach(t=>['y','y0','y1'].forEach(k=>{ if(k in t){ if(t[k]<lo)lo=t[k]; if(t[k]>hi)hi=t[k]; } })));
  if(!isFinite(lo)||!isFinite(hi)||hi<=lo){ lo=Math.min(...velas.map(v=>v[2])); hi=Math.max(...velas.map(v=>v[1])); }
  const pad=(hi-lo)*0.06||1; lo-=pad; hi+=pad;
  const n=velas.length, step=plotW/n;
  const X=i=>padL+(i+0.5)*step, Y=p=>padT+(1-(p-lo)/(hi-lo))*plotH;
  ctx.strokeStyle='rgba(38,48,61,.55)'; ctx.fillStyle='#5c6775'; ctx.font="11px 'JetBrains Mono',monospace"; ctx.lineWidth=1;
  for(let t=0;t<=5;t++){ const p=lo+(hi-lo)*t/5, y=Y(p);
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+plotW,y); ctx.stroke();
    ctx.fillText(p.toFixed(2), padL+plotW+6, y+3);
  }
  const bw=Math.max(1, step*0.62);
  velas.forEach((v,i)=>{ const o=v[0],h=v[1],l=v[2],c=v[3]; const col=c>=o?'#6ec08a':'#d2566a'; const x=X(i);
    ctx.strokeStyle=col; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x,Y(h)); ctx.lineTo(x,Y(l)); ctx.stroke();
    ctx.fillStyle=col; const yo=Y(o),yc=Y(c); ctx.fillRect(x-bw/2, Math.min(yo,yc), bw, Math.max(1,Math.abs(yc-yo)));
  });
  figs.forEach(f=>{ const col=f.color||'#e8b23a';
    f.trazos.forEach(t=>{ ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=1.7;
      if(t.k==='hline'){ ctx.setLineDash([5,4]); ctx.beginPath(); ctx.moveTo(X(t.x0),Y(t.y)); ctx.lineTo(X(t.x1),Y(t.y)); ctx.stroke(); ctx.setLineDash([]); }
      else if(t.k==='line'){ ctx.beginPath(); ctx.moveTo(X(t.x0),Y(t.y0)); ctx.lineTo(X(t.x1),Y(t.y1)); ctx.stroke(); }
      else if(t.k==='pico'){ ctx.beginPath(); ctx.arc(X(t.x),Y(t.y),3.2,0,7); ctx.fill(); }
      else if(t.k==='break'){ ctx.beginPath(); ctx.arc(X(t.x),Y(t.y),4.8,0,7); ctx.fill();
        ctx.globalAlpha=.9; ctx.lineWidth=1.6; ctx.strokeStyle='#0b0f14'; ctx.stroke(); ctx.globalAlpha=1; }
    });
  });
}

// ---------- Arranque según la página ----------
document.addEventListener('DOMContentLoaded', ()=>{
  const page = document.body.dataset.page;
  if(page === 'lab'){
    initLab();
  }else if(page === 'metodologia'){
    mountNav('metodologia');
    observeReveals();
  }else if(page === 'visor'){
    initVisor();
  }
});
