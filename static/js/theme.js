(function(){
  const body = document.getElementById('body');

  function apply(theme){
    // theme: 'dark' | 'light' | 'system'
    localStorage.setItem('bma_theme', theme);
    body.classList.remove('light');
    body.classList.remove('dark');
    if(theme === 'dark') body.classList.add('dark');
    else if(theme === 'light') body.classList.add('light');
    else {
      // system
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      body.classList.add(prefersDark ? 'dark' : 'light');
    }
  }

  // initial
  apply(localStorage.getItem('bma_theme') || 'dark');

  // watch system change if in system mode
  try{
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e=>{
      if((localStorage.getItem('bma_theme') || 'dark') === 'system'){
        apply('system');
      }
    });
  }catch(_){}

  // buttons
  document.getElementById('darkBtn').addEventListener('click', ()=>apply('dark'));
  document.getElementById('lightBtn').addEventListener('click', ()=>apply('light'));
  document.getElementById('systemBtn').addEventListener('click', ()=>apply('system'));
})();
