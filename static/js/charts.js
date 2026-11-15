/* tiny dependency-free animated charts */
(function(){
  function setup(cv){ const dpr = window.devicePixelRatio||1, w = cv.clientWidth, h = cv.clientHeight; cv.width = w*dpr; cv.height = h*dpr; const c = cv.getContext('2d'); c.scale(dpr,dpr); return {c,w,h}; }
  const ease = t => 1-Math.pow(1-t,3);
  function animate(draw){ const t0 = performance.now(); (function f(t){ const p = Math.max(0,Math.min((t-t0)/1300,1)); draw(ease(p)); if(p<1) requestAnimationFrame(f); })(t0); }

  window.lineChart = function(id, labels, series){ // series: [{name,color,data[],unit}]
    const cv = document.getElementById(id); if(!cv) return; const {c,w,h} = setup(cv);
    const pad = {l:44,r:18,t:16,b:36}, all = series.flatMap(s=>s.data.filter(v=>v!=null));
    if(!all.length){ c.fillStyle='#93a3bb'; c.font='14px Inter,sans-serif'; c.fillText('No data yet',20,40); return; }
    let mn = Math.min(...all), mx = Math.max(...all); if(mn===mx){ mn-=1; mx+=1; } const span = mx-mn; mn -= span*.1; mx += span*.1;
    const X = i => pad.l + (labels.length===1 ? (w-pad.l-pad.r)/2 : i*(w-pad.l-pad.r)/(labels.length-1)), Y = v => pad.t + (1-(v-mn)/(mx-mn))*(h-pad.t-pad.b);
    animate(p => {
      c.clearRect(0,0,w,h); c.strokeStyle='rgba(255,255,255,.08)'; c.fillStyle='#93a3bb'; c.font='11px Inter,sans-serif'; c.lineWidth=1;
      for(let i=0;i<=4;i++){ const y = pad.t + i*(h-pad.t-pad.b)/4; c.beginPath(); c.moveTo(pad.l,y); c.lineTo(w-pad.r,y); c.stroke(); c.fillText((mx-(mx-mn)*i/4).toFixed(0),6,y+4); }
      labels.forEach((l,i)=>{ if(labels.length<9 || i%Math.ceil(labels.length/8)===0) c.fillText(l,Math.max(2,X(i)-22),h-12); });
      series.forEach(s => {
        const pts = s.data.map((v,i)=>v==null?null:[X(i),Y(v)]).filter(Boolean); if(!pts.length) return;
        const grd = c.createLinearGradient(0,pad.t,0,h-pad.b); grd.addColorStop(0,s.color+'55'); grd.addColorStop(1,s.color+'00');
        const n = (pts.length-1)*p, k = Math.floor(n), f = n-k, cur = pts.slice(0,k+1); if(k<pts.length-1) cur.push([pts[k][0]+(pts[k+1][0]-pts[k][0])*f, pts[k][1]+(pts[k+1][1]-pts[k][1])*f]);
        c.beginPath(); cur.forEach((q,i)=> i?c.lineTo(q[0],q[1]):c.moveTo(q[0],q[1])); c.strokeStyle=s.color; c.lineWidth=3; c.lineJoin='round'; c.shadowColor=s.color; c.shadowBlur=12; c.stroke(); c.shadowBlur=0;
        if(cur.length>1){ c.lineTo(cur[cur.length-1][0],h-pad.b); c.lineTo(cur[0][0],h-pad.b); c.closePath(); c.fillStyle=grd; c.fill(); }
        cur.forEach((q,i)=>{ if(i<=k){ c.beginPath(); c.arc(q[0],q[1],5,0,7); c.fillStyle='#070b14'; c.fill(); c.lineWidth=2.5; c.strokeStyle=s.color; c.stroke(); } });
      });
    });
  };

  window.barChart = function(id, labels, data, color){
    const cv = document.getElementById(id); if(!cv) return; const {c,w,h} = setup(cv); const pad={l:34,r:10,t:14,b:34}, mx = Math.max(1,...data), bw = (w-pad.l-pad.r)/Math.max(labels.length,1);
    animate(p => { c.clearRect(0,0,w,h); c.fillStyle='#93a3bb'; c.font='11px Inter,sans-serif';
      data.forEach((v,i)=>{ const bh = (h-pad.t-pad.b)*v/mx*p, x = pad.l+i*bw+bw*.18, y = h-pad.b-bh; const g = c.createLinearGradient(0,y,0,h-pad.b); g.addColorStop(0,color||'#2dd4bf'); g.addColorStop(1,'#8b5cf6');
        c.fillStyle=g; c.beginPath(); c.roundRect(x,y,bw*.64,Math.max(bh,1),6); c.fill(); c.fillStyle='#93a3bb'; c.fillText(labels[i],x,h-14); c.fillText(v,x+bw*.2,y-5); }); });
  };

  window.donut = function(id, parts){ // [{label,value,color}]
    const cv = document.getElementById(id); if(!cv) return; const {c,w,h} = setup(cv), tot = parts.reduce((a,b)=>a+b.value,0), r = Math.min(w,h)/2-18, cx = w/2, cy = h/2;
    animate(p => { c.clearRect(0,0,w,h); let a = -Math.PI/2; if(!tot){ c.fillStyle='#93a3bb'; c.fillText('No data yet',cx-30,cy); return; }
      parts.forEach(s => { const sw = s.value/tot*Math.PI*2*p; c.beginPath(); c.arc(cx,cy,r,a,a+sw); c.strokeStyle=s.color; c.lineWidth=26; c.shadowColor=s.color; c.shadowBlur=14; c.stroke(); a += sw; });
      c.shadowBlur=0; c.fillStyle='#e8eef8'; c.font='700 30px Inter,sans-serif'; c.textAlign='center'; c.fillText(Math.round(tot*p),cx,cy+8); c.font='12px Inter,sans-serif'; c.fillStyle='#93a3bb'; c.fillText('assessments',cx,cy+28); c.textAlign='start'; });
  };
})();
