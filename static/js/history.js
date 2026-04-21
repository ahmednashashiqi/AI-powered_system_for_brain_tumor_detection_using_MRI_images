const table = document.getElementById('histTable');
const statusEl = document.getElementById('histStatus');
const refreshBtn = document.getElementById('refreshBtn');
const clearBtn = document.getElementById('clearBtn');

async function loadHistory(){
  statusEl.textContent = 'Loading...';
  table.innerHTML = '';
  try{
    const res = await fetch('/api/history');
    const data = await res.json();
    if(!data.ok) throw new Error('Server error');

    const items = data.items || [];
    if(items.length === 0){
      table.innerHTML = '<div class="muted">No history yet.</div>';
    }else{
      const rows = items.map((it, idx) => {
        const conf = (it.confidence == null) ? '-' : it.confidence.toFixed(3);
        const rag = it.rag_used ? 'Yes' : 'No';
        return `
          <div style="padding:8px;border-bottom:1px solid var(--border)">
            <div><b>#${items.length - idx}</b> • <b>${it.ts}</b> • <span class="muted">${it.filename || '-'}</span></div>
            <div>Label: <b>${it.label || '-'}</b> • Confidence: <b>${conf}</b> • RAG: <b>${rag}</b> • Adv: <b>${it.adv_enabled ? 'Yes' : 'No'}</b></div>
            <div class="muted">${(it.text_preview || '').replace(/</g,'&lt;')}</div>
          </div>
        `;
      }).join('');
      table.innerHTML = rows;
    }
    statusEl.textContent = '';
  }catch(e){
    statusEl.textContent = 'Error loading history: ' + e.message;
  }
}

refreshBtn.addEventListener('click', loadHistory);
clearBtn.addEventListener('click', async ()=>{
  if(!confirm('Clear all history?')) return;
  statusEl.textContent = 'Clearing...';
  try{
    const res = await fetch('/api/history/clear', {method:'POST'});
    const data = await res.json();
    if(!data.ok) throw new Error('Server error');
    await loadHistory();
  }catch(e){
    statusEl.textContent = 'Error: ' + e.message;
  }
});

loadHistory();
