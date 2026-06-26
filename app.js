// ====== Neural Capital Research · app.js (compartido por todas las páginas internas) ======
const NOMBRE   = "Neural Capital Research";
const TELEGRAM = "https://t.me/neural_capital_research";

// ---------- Barra de navegación ----------
function navHTML(active){
  const modelos = [
    ['par_oro_plata', 'Par oro-plata'],
    ['oro_bh',        'Oro'],
    ['plata_bh',      'Plata'],
    ['koncorde',      'KONCORDE (S&P 500)'],
  ];
  const mods = modelos.map(([id,n]) =>
    `<a href="lab.html?id=${id}" class="${id===active?'active':''}">${n}</a>`).join('')
    + `<a href="lab.html?id=garch_vol" class="${active==='garch_vol'?'active':''}">GARCH (volatilidad)</a>`;
  return `<div class="nav-inner">
    <a class="nav-logo" href="lab.html?id=par_oro_plata">Neural <b>Capital</b> Research</a>
    <div class="nav-menu">
      <div class="nav-item"><button class="nav-trigger" type="button">Modelos ▾</button>
        <div class="nav-drop">${mods}</div></div>
      <div class="nav-item"><button class="nav-trigger" type="button">Operaciones ▾</button>
        <div class="nav-drop">
          <a href="lab.html?id=koncorde#operaciones">Acciones (S&P 500)</a>
          <a href="lab.html?id=par_oro_plata#operaciones">Oro</a>
          <a href="lab.html?id=par_oro_plata#operaciones">Plata</a>
        </div></div>
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
function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }

function render(exp){
  const h = exp.headline, s = exp.significancia, d = exp.diagnostico || {};
  const app = document.getElementById('app');
  const titulo = `<h1 class="model-title">${exp.etiqueta || ''}</h1>`;

  if(exp.sin_datos){
    app.innerHTML = titulo + `<div class="tipo">▸ ${exp.tipo}</div>
      <div class="empty"><b>Sin datos suficientes todavía</b>${exp.sin_datos_txt || 'El forward-test necesita operaciones cerradas; se irá llenando con el tiempo.'}</div>
      ${(exp.cards&&exp.cards.length)?`<div class="grid" style="margin-top:18px">${exp.cards.map(c=>`<div class="card"><div class="k">${c.k}</div><div class="v ${c.tono||''}">${c.v}</div></div>`).join("")}</div>`:''}
      ${opsTable(exp)}`;
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

   <div class="chartbox reveal">
     <h3>${exp.curva_titulo || 'Curva de capital fuera de muestra'}</h3>
     <div class="ch-sub">${exp.curva_sub || 'Crecimiento de 1 unidad invertida según el modelo. La línea base es no hacer nada.'}</div>
     <div class="canvas-h"><canvas id="eq"></canvas></div>
   </div>

   ${opsTable(exp)}
  `;

  drawChart(exp);
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
  const datasets = [
    {label:'Modelo', data:curva.map(p=>p.valor), borderColor:'#e8b23a', borderWidth:2, pointRadius:0, tension:.12,
     fill:true, backgroundColor:(c)=>{const g=c.chart.ctx.createLinearGradient(0,0,0,300); g.addColorStop(0,'rgba(232,178,58,.18)'); g.addColorStop(1,'rgba(232,178,58,0)'); return g;}}
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

// ---------- Arranque del laboratorio ----------
async function initLab(){
  const id0 = new URLSearchParams(location.search).get('id') || 'par_oro_plata';
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
  const exp = doc.experimentos.find(e=>e.id===id0) || doc.experimentos[0];
  mountNav(exp.id);
  render(exp);
  if(location.hash === '#operaciones'){
    setTimeout(()=>{ const o = document.getElementById('operaciones'); if(o) o.scrollIntoView({behavior:'smooth', block:'start'}); }, 350);
  }
}

// ---------- Arranque según la página ----------
document.addEventListener('DOMContentLoaded', ()=>{
  const page = document.body.dataset.page;
  if(page === 'lab'){
    initLab();
  }else if(page === 'metodologia'){
    mountNav('metodologia');
    observeReveals();
  }
});
