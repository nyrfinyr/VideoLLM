// Pagina interattiva del singolo video: selezione token (entity) → riordino
// dei frame per massa di attenzione + overlay rigenerati lato server.
const stem = document.body.dataset.stem;
const grid = document.getElementById('grid');
const selinfo = document.getElementById('selinfo');
const chips = Array.from(document.querySelectorAll('.chip'));

// Righe-query attualmente selezionate (data-row dei chip con classe .sel).
function selectedRows() {
  return chips.filter(c => c.classList.contains('sel')).map(c => +c.dataset.row);
}

function tokensParam(rows) {
  return rows.length ? `&tokens=${rows.join(',')}` : '';
}

// View sink corrente ('all' | 'sink' | 'nonsink'); assente se il capture
// non ha una sink_map (la combobox non viene renderizzata).
const sinkSelect = document.getElementById('sinkview-select');
function sinkView() {
  return sinkSelect ? sinkSelect.value : 'all';
}
function sinkParam() {
  return sinkSelect ? `&sink_view=${sinkView()}` : '';
}

// Costruisce una card frame (overlay sopra, originale sotto: toggle via CSS).
function card(cell, rows) {
  const tp = tokensParam(rows) + sinkParam();
  const fig = document.createElement('figure');
  fig.className = 'card' + (cell.is_top ? ' top' : '');
  fig.innerHTML = `
    <div class="imgwrap">
      <img class="over" src="/api/${stem}/overlay?cell=${cell.cell}${tp}" alt="overlay cella ${cell.cell}">
      <img class="orig" src="/api/${stem}/frame?cell=${cell.cell}" alt="frame cella ${cell.cell}">
    </div>
    <figcaption>
      <span class="rank">#${cell.rank}</span>
      <b>frame ${cell.rep_frame}</b> · ${cell.t_sec.toFixed(1)}s
      <span class="pct">${cell.pct.toFixed(1)}% massa</span>
      <small>cella ${cell.cell} · score ${cell.score.toFixed(4)}</small>
    </figcaption>`;
  fig.querySelector('.imgwrap').addEventListener('click', () => openLightbox(fig));
  return fig;
}

// Lightbox: mostra l'immagine attualmente visibile (overlay o originale).
const lb = document.getElementById('lightbox');
const lbImg = lb.querySelector('img');
function openLightbox(fig) {
  const showOrig = document.body.classList.contains('show-orig');
  lbImg.src = fig.querySelector(showOrig ? '.orig' : '.over').src;
  lb.classList.add('open');
}
lb.addEventListener('click', () => lb.classList.remove('open'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.classList.remove('open'); });

// Modale dati raw del capture.pt (debug/presentazione): il JSON è già
// renderizzato server-side nel <pre>, qui solo apertura/chiusura + copia.
const rawModal = document.getElementById('raw-modal');
const rawBtn = document.getElementById('raw-btn');
if (rawBtn) {
  rawBtn.addEventListener('click', () => rawModal.classList.add('open'));
  document.getElementById('raw-close').addEventListener('click', () => rawModal.classList.remove('open'));
  rawModal.addEventListener('click', () => rawModal.classList.remove('open'));
  document.getElementById('raw-copy').addEventListener('click', (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(document.getElementById('raw-pre').textContent);
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') rawModal.classList.remove('open'); });
}

// Modale di riproduzione del video completo: aperta dal bottone, il player
// viene messo in pausa alla chiusura (backdrop, bottone o Escape).
const videoModal = document.getElementById('video-modal');
const videoBtn = document.getElementById('video-btn');
if (videoBtn) {
  const videoPlayer = document.getElementById('video-player');
  const closeVideoModal = () => { videoModal.classList.remove('open'); videoPlayer.pause(); };
  videoBtn.addEventListener('click', () => videoModal.classList.add('open'));
  document.getElementById('video-close').addEventListener('click', closeVideoModal);
  videoModal.addEventListener('click', closeVideoModal);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeVideoModal(); });
}

// Toggle overlay/originale su tutte le card.
document.getElementById('tgl').addEventListener('change', e =>
  document.body.classList.toggle('show-orig', e.target.checked));

// Fetch del ranking per la selezione corrente e ridisegno della grid.
let reqToken = 0;
async function refresh() {
  const rows = selectedRows();
  const txt = rows.length
    ? `<b>${rows.length}</b> token (${chips.filter(c => c.classList.contains('sel')).map(c => c.textContent.trim()).join(' ')})`
    : '<b>nessuna</b> (media di tutti i token)';
  selinfo.innerHTML = 'selezione: ' + txt;

  const my = ++reqToken;
  const res = await fetch(`/api/${stem}/rank?tokens=${rows.join(',')}${sinkParam()}`);
  if (my !== reqToken) return;             // risposta obsoleta: scartala
  const data = await res.json();
  grid.replaceChildren(...data.cells.map(c => card(c, rows)));
}

chips.forEach(c => c.addEventListener('click', () => {
  c.classList.toggle('sel');
  refresh();
}));

if (sinkSelect) sinkSelect.addEventListener('change', refresh);

refresh();   // ordinamento iniziale = media di tutti i token
