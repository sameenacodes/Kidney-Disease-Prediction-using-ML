(function(){
  // reveal on scroll
  const io = new IntersectionObserver(es => es.forEach(e => { if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } }), {threshold:.12});
  document.querySelectorAll('.rv').forEach((el,i)=>{ if(!el.style.getPropertyValue('--d')) el.style.setProperty('--d',(i%6)*0.07+'s'); io.observe(el); });

  // count-up numbers
  function countUp(el){
    const end = parseFloat(el.dataset.count), dec = parseInt(el.dataset.dec||'0'), dur = 1500, t0 = performance.now();
    (function f(t){ const p = Math.max(0,Math.min((t-t0)/dur,1)), e = 1-Math.pow(1-p,3); el.textContent = (end*e).toFixed(dec)+(el.dataset.suffix||''); if(p<1) requestAnimationFrame(f); })(t0);
  }
  const io2 = new IntersectionObserver(es => es.forEach(e => { if(e.isIntersecting){ countUp(e.target); io2.unobserve(e.target); } }), {threshold:.4});
  document.querySelectorAll('[data-count]').forEach(el=>io2.observe(el));

  // ripple on buttons
  document.addEventListener('click', e => {
    const b = e.target.closest('.btn'); if(!b) return;
    const r = b.getBoundingClientRect(), s = document.createElement('span'), d = Math.max(r.width,r.height);
    s.className='ripple'; s.style.cssText = `width:${d}px;height:${d}px;left:${e.clientX-r.left-d/2}px;top:${e.clientY-r.top-d/2}px`;
    b.appendChild(s); setTimeout(()=>s.remove(),600);
  });

  // mobile nav
  const bg = document.getElementById('burger'); if(bg) bg.onclick = () => document.querySelector('.links').classList.toggle('open');

  // animated rings / bars
  window.addEventListener('load', () => {
    document.querySelectorAll('.gauge .fg').forEach(c => { const len = 2*Math.PI*parseFloat(c.getAttribute('r')); setTimeout(()=>{ c.style.strokeDasharray = (len*parseFloat(c.dataset.p)/100)+' '+len; },250); });
    document.querySelectorAll('.bar i[data-w]').forEach((b,i) => setTimeout(()=>{ b.style.width = b.dataset.w+'%'; }, 300+i*120));
  });

  // language toggle (English / Tamil) for key UI text
  const T = {
    ta:{nav_home:'முகப்பு',nav_predict:'மதிப்பீடு',nav_history:'வரலாறு',nav_analytics:'பகுப்பாய்வு',nav_validation:'சரிபார்ப்பு',nav_about:'பற்றி',
      res_title:'சிறுநீரக ஆபத்து மதிப்பீடு',res_prob:'CKD மாதிரி நிகழ்தகவு',res_risk:'ஆபத்து நிலை',res_egfr:'eGFR மற்றும் நிலை',res_why:'இந்த முடிவுக்கு காரணங்கள்',
      res_params:'உள்ளிட்ட மதிப்புகள்',res_pdf:'PDF அறிக்கை',res_trend:'முன்னேற்றம் காண்க',res_new:'புதிய மதிப்பீடு',
      res_disc:'இது ஒரு இயந்திரக் கற்றல் அடிப்படையிலான முடிவு-ஆதரவு முன்மாதிரி மட்டுமே. இது மருத்துவர் மற்றும் ஆய்வக சோதனைக்கு மாற்று அல்ல. எந்த மருத்துவ முடிவுக்கும் தகுதியான மருத்துவரின் உறுதிப்படுத்தல் அவசியம்.'}
  };
  const L = {};
  document.querySelectorAll('[data-i]').forEach(el => L[el.dataset.i] = el.textContent);
  function setLang(l){
    document.querySelectorAll('[data-i]').forEach(el => { el.textContent = (l==='ta' && T.ta[el.dataset.i]) ? T.ta[el.dataset.i] : L[el.dataset.i]; });
    const b = document.getElementById('lang'); if(b) b.textContent = l==='ta' ? 'English' : 'தமிழ்';
    try{ localStorage.setItem('lang',l); }catch(e){}
  }
  const lb = document.getElementById('lang'); let cur='en';
  try{ cur = localStorage.getItem('lang')||'en'; }catch(e){}
  if(lb){ lb.onclick = () => { cur = cur==='en'?'ta':'en'; setLang(cur); }; if(cur==='ta') setLang('ta'); }
})();
