// static/js/app.js

// ------- عناصر الواجهة -------
const fileInput  = document.getElementById('file');
const preview    = document.getElementById('preview');
const question   = document.getElementById('question');
const analyzeBtn = document.getElementById('analyzeBtn');
const output     = document.getElementById('output');
const statusEl   = document.getElementById('status');
const useRag     = document.getElementById('useRag');
const advReport  = document.getElementById('advReport');
const ragList    = document.getElementById('ragList');
const heatDiv    = document.getElementById('heat');
const copyBtn    = document.getElementById('copyBtn');
const threshEl   = document.getElementById('thresh');

let selectedFile = null;

// ------- تحميل تفضيلات المستخدم عند فتح الصفحة -------
(function loadPrefs(){
  try{
    const prefs = JSON.parse(localStorage.getItem('bma_prefs') || '{}');
    if (typeof prefs.default_rag  === 'boolean' && useRag)    useRag.checked    = prefs.default_rag;
    if (typeof prefs.default_adv  === 'boolean' && advReport) advReport.checked = prefs.default_adv;
    if (typeof prefs.default_thresh === 'number' && threshEl) threshEl.value = prefs.default_thresh.toFixed(2);
  }catch(_){}
})();

// ------- معاينة الصورة -------
function updatePreview(file){
  if(!file){ if (preview) preview.src=''; return; }
  const reader = new FileReader();
  reader.onload = e => { if (preview) preview.src = e.target.result; };
  reader.readAsDataURL(file);
}

if (fileInput){
  fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0] || null;
    updatePreview(selectedFile);
    if (analyzeBtn) analyzeBtn.disabled = !selectedFile;
  });
}

// دعم السحب والإفلات
const dropZone = document.querySelector('.drop');
if (dropZone){
  dropZone.addEventListener('dragover', e => { e.preventDefault(); e.stopPropagation(); });
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if(file){
      if (fileInput) fileInput.files = e.dataTransfer.files;
      selectedFile = file;
      updatePreview(selectedFile);
      if (analyzeBtn) analyzeBtn.disabled = !selectedFile;
    }
  });
}

// ------- تلوين درجات RAG -------
function scoreColor(score){
  if (score >= 0.75) return "color:#00ff9d";
  if (score >= 0.60) return "color:#ffc107";
  return "color:#ff5c5c";
}

// ------- نسخ النص الناتج -------
if (copyBtn){
  copyBtn.addEventListener('click', async () => {
    try{
      await navigator.clipboard.writeText((output && output.textContent) ? output.textContent : '');
      copyBtn.textContent = "Copied ✓";
      setTimeout(()=>copyBtn.textContent="Copy", 1200);
    }catch(e){
      copyBtn.textContent = "Copy error";
      setTimeout(()=>copyBtn.textContent="Copy", 1200);
    }
  });
}

// ------- طباعة التقرير -------
const printBtn = document.getElementById('printBtn');
if (printBtn && output){
  printBtn.addEventListener('click', () => {
    const text = output.textContent || '';
    if (!text.trim()) return;
    const w = window.open('', '_blank');
    w.document.write('<html><head><title>Brain MRI Report</title><style>body{font-family:monospace;white-space:pre-wrap;padding:20px;max-width:800px;margin:0 auto;} h1{font-size:1.1em;}</style></head><body><h1>Brain MRI Analyzer — Report</h1><pre>' + text.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</pre></body></html>');
    w.document.close();
    w.focus();
    w.print();
    w.close();
  });
}

// ------- تحميل التقرير كملف نصي -------
const downloadBtn = document.getElementById('downloadBtn');
if (downloadBtn && output){
  downloadBtn.addEventListener('click', () => {
    const text = output.textContent || '';
    if (!text.trim()) return;
    const name = 'MRI_report_' + new Date().toISOString().slice(0,10) + '_' + Date.now().toString(36).slice(-6) + '.txt';
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

// ------- حدث التحليل -------
if (analyzeBtn){
  analyzeBtn.addEventListener('click', async () => {
    if(!selectedFile){ return; }

    if (output)   output.textContent = '';
    if (ragList)  ragList.innerHTML  = '';
    if (heatDiv)  heatDiv.innerHTML  = '';
    if (statusEl) statusEl.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;

    const form = new FormData();
    form.append('file', selectedFile);
    form.append('question', (question && question.value) ? question.value : 'Describe this image in simple medical terms.');
    form.append('use_rag', (useRag && useRag.checked) ? 'true' : 'false');
    form.append('adv_report', (advReport && advReport.checked) ? 'true' : 'false');
    const ragModeEl = document.getElementById('ragMode');
    form.append('rag_mode', (ragModeEl && ragModeEl.value) ? ragModeEl.value : 'auto');

    try{
      const res = await fetch('/analyze', { method:'POST', body: form });
      if(!res.ok){ throw new Error('HTTP ' + res.status); }
      const data = await res.json();

      if (output) output.textContent = data.text || '(No output)';

      // عدم تكرار قائمة المراجع — تظهر مرة واحدة داخل التقرير تحت ▸ REFERENCES (RAG)
      if (data.rag && data.rag.used && data.rag.sources && ragList && !(data.text && data.text.includes('REFERENCES'))){
        ragList.innerHTML = '<div style="margin-top:8px;font-weight:700">RAG Sources</div>';
        ragList.innerHTML += data.rag.sources.map(
          s => `<div style="${scoreColor(s.score)}">(${s.score}) ${s.source}#${s.chunk_id}</div>`
        ).join('');
      }

      if (data.heatmap && heatDiv){
        const img = new Image();
        img.src = "data:image/png;base64," + data.heatmap;
        img.style.maxWidth    = "240px";
        img.style.borderRadius= "12px";
        img.style.display     = "block";
        img.style.marginTop   = "10px";
        heatDiv.appendChild(img);
      }
    }catch(err){
      if (output) output.textContent = 'Error: ' + err.message;
    }finally{
      analyzeBtn.disabled = false;
      if (statusEl) statusEl.textContent = '';
    }
  });
}
