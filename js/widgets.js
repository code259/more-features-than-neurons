import{RNG,ToyModel,SAE,packScore,packGradStep}from'./core.js';

// ---------- shared ----------
const css=getComputedStyle(document.documentElement);
const FC=[...Array(8)].map((_,i)=>css.getPropertyValue(`--f${i}`).trim());
const INK='#0d0d0c',INK2='#84847f',RULE='#d6d6d2',ACC='#0d0d0c',GOOD='#0d0d0c';
// machine-world canvas palette (dark instrument panels)
const M={ink:'#f2f2f0',mut:'#8a8a86',rule:'#2b2b29',grid:'#191918'};
const SANS='11px "IBM Plex Mono",ui-monospace,Menlo,monospace';

function setupCanvas(c,w,h){
  const d=Math.min(devicePixelRatio||1,2);
  c.width=w*d;c.height=h*d;c.style.width=w+'px';c.style.height=h+'px';
  const x=c.getContext('2d');x.scale(d,d);return x;
}
function vecPanel(ctx,W,H,r,pal){
  const cx=W/2,cy=H/2,R=r||Math.min(W,H)/2-26,P=pal||{mut:INK2,rule:RULE,grid:null};
  return{
    cx,cy,R,px:v=>[cx+v[0]*R,cy-v[1]*R],
    base(){
      ctx.clearRect(0,0,W,H);
      if(P.grid){ctx.strokeStyle=P.grid;ctx.lineWidth=1;ctx.beginPath();
        for(let g=30;g<W;g+=30){ctx.moveTo(g,0);ctx.lineTo(g,H)}
        for(let g=30;g<H;g+=30){ctx.moveTo(0,g);ctx.lineTo(W,g)}ctx.stroke()}
      ctx.strokeStyle=P.rule;ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(cx-R-12,cy);ctx.lineTo(cx+R+12,cy);ctx.moveTo(cx,cy-R-12);ctx.lineTo(cx,cy+R+12);ctx.stroke();
      ctx.setLineDash([3,4]);ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=P.mut;ctx.font=SANS;ctx.textAlign='left';
      ctx.fillText('neuron 1 →',cx+R-64,cy+16);
      ctx.save();ctx.translate(cx-8,cy-R+62);ctx.rotate(-Math.PI/2);ctx.fillText('neuron 2 →',0,0);ctx.restore();
    },
    arrow(v,color,o={}){
      const[x,y]=this.px(v),lw=o.lw||3;
      ctx.strokeStyle=ctx.fillStyle=color;ctx.lineWidth=lw;ctx.globalAlpha=o.alpha??1;
      if(o.dash)ctx.setLineDash(o.dash);
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.stroke();ctx.setLineDash([]);
      const a=Math.atan2(y-cy,x-cx),hs=4+lw;
      ctx.beginPath();ctx.moveTo(x,y);
      ctx.lineTo(x-hs*Math.cos(a-.45),y-hs*Math.sin(a-.45));
      ctx.lineTo(x-hs*Math.cos(a+.45),y-hs*Math.sin(a+.45));
      ctx.closePath();ctx.fill();ctx.globalAlpha=1;
    }
  };
}
function drawBars(ctx,W,H,vals,colors,o={}){
  ctx.clearRect(0,0,W,H);
  const P=o.pal||{mut:INK2,rule:RULE};
  const max=o.max||1.3,pad=26,bw=(W-pad-10)/vals.length,base=H-22;
  ctx.font=SANS;
  if(o.target!=null){const y=base-(o.target/max)*(H-46);
    ctx.strokeStyle=P.mut;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(pad-6,y);ctx.lineTo(W-6,y);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=P.mut;ctx.textAlign='left';ctx.fillText(o.targetLabel||'1.0',2,y+4);}
  vals.forEach((v,i)=>{
    const h=Math.max(0,Math.min(v,max))/max*(H-46),x=pad+i*bw;
    ctx.fillStyle=colors[i];ctx.globalAlpha=.9;
    ctx.fillRect(x+3,base-h,bw-6,h);ctx.globalAlpha=1;
    ctx.fillStyle=P.mut;ctx.textAlign='center';ctx.fillText(o.labels?o.labels[i]:i+1,x+bw/2,base+14);
  });
  ctx.strokeStyle=P.rule;ctx.beginPath();ctx.moveTo(pad-6,base);ctx.lineTo(W-6,base);ctx.stroke();
}
// loop manager: widgets tick only when on-screen & tab visible
const ACTIVE=new Set(),ALL=[];
function register(el,w){
  ALL.push(w);
  new IntersectionObserver(e=>{e[0].isIntersecting?ACTIVE.add(w):ACTIVE.delete(w)},{rootMargin:'80px'}).observe(el);
}
window.__step=(n=1)=>{for(let k=0;k<n;k++)ALL.forEach(w=>w.tick())};   // headless testing
window.__dbg={};
(function loop(){
  if(!document.hidden)for(const w of ACTIVE){
    try{w.tick()}catch(e){console.error('widget tick failed:',e);ACTIVE.delete(w)}
  }
  requestAnimationFrame(loop);
})();
const $=s=>document.querySelector(s);
const fmt=(x,d=3)=>x.toFixed(d);
// Display smoothing (Polyak-style): figures render a light EMA of the weights
// so the reader sees the trajectory, not per-batch sampling noise. Training
// itself always runs on the raw weights.
function ema(dst,src,a){
  if(!dst||dst.length!==src.length)return Float64Array.from(src);
  for(let i=0;i<src.length;i++)dst[i]+=a*(src[i]-dst[i]);
  return dst;
}
const EMA_W=.12;   // weight-display smoothing per frame

// ---------- real-model record browsers (raw neurons at the top, dictionary atoms at the end) ----------
function recordBrowser(root,items,label){
  const box=root.querySelector('.snips'),nameEl=root.querySelector('.n-name'),valEl=root.querySelector('.n-val');
  let cur=0;
  const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  function show(){
    const it=items[cur];
    nameEl.innerHTML=label(it,cur,items.length);
    box.innerHTML=it.recs.map(r=>{
      const mx=Math.max(...r.v,1e-6);
      const spans=r.t.map((t,i)=>{
        const a=r.v[i]/mx;
        if(a<.03)return esc(t);
        const cls=a>=.45?'tk hi':'tk';
        return `<span class="${cls}" style="--a:${a.toFixed(2)}" data-v="${r.v[i]}" title="activation ${r.v[i]}">${esc(t)}</span>`;
      }).join('');
      return `<div class="snip"><span class="tag">peak activation ${r.m}</span>${spans}</div>`;
    }).join('');
    if(valEl)valEl.textContent='hover a lit token';
  }
  box.addEventListener('pointerover',e=>{
    const t=e.target.closest('.tk');
    if(t&&valEl)valEl.textContent='activation '+t.dataset.v;
  });
  root.querySelector('.n-prev').onclick=()=>{cur=(cur+items.length-1)%items.length;show()};
  root.querySelector('.n-next').onclick=()=>{cur=(cur+1)%items.length;show()};
  show();
}
(function(){const r=$('#w-hook');if(!r)return;
  recordBrowser(r,REALNEURONS,(it,i,n)=>`neuron ${it.id} · ${i+1} of ${n}`);
})();
(function(){const r=$('#w-real');if(!r)return;
  recordBrowser(r,REALFEATURES,(it,i,n)=>`atom <a href="${it.url}">${it.id}</a> “${it.label}” · ${i+1} of ${n}`);
})();

// ---------- 1. packing game ----------
(function(){
  const root=$('#w-game');if(!root)return;
  const N=6,I=[1,.9,.8,.7,.6,.5],C=.25,SEED=21;
  const cv=root.querySelector('canvas'),ctx=setupCanvas(cv,340,340),P=vecPanel(ctx,340,340);
  const youEl=root.querySelector('.s-you'),sgdEl=root.querySelector('.s-sgd');
  let V,sgdV=null,anim=0,drag=-1,dirty=true;
  function reset(){const r=new RNG(SEED);V=[...Array(N)].map(()=>{const a=r.range(0,6.28),m=r.range(.15,.35);return[m*Math.cos(a),m*Math.sin(a)]});sgdV=null;anim=0;dirty=true;update()}
  function update(){
    const s=packScore(V,I,C);youEl.textContent=fmt(s);
    if(sgdV){const g=packScore(sgdV,I,C);sgdEl.textContent=fmt(g);
      sgdEl.parentElement.classList.toggle('win',g<s-1e-4);
      youEl.parentElement.classList.toggle('win',s<g-1e-4);
    }else{sgdEl.textContent='—';sgdEl.parentElement.classList.remove('win');youEl.parentElement.classList.remove('win')}
  }
  cv.addEventListener('pointerdown',e=>{
    const b=cv.getBoundingClientRect(),mx=e.clientX-b.left,my=e.clientY-b.top;
    let best=20,bi=-1;
    V.forEach((v,i)=>{const[x,y]=P.px(v),d=Math.hypot(mx-x,my-y);if(d<best){best=d;bi=i}});
    drag=bi;if(bi>=0)cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove',e=>{
    if(drag<0)return;
    const b=cv.getBoundingClientRect();
    let vx=(e.clientX-b.left-P.cx)/P.R,vy=-(e.clientY-b.top-P.cy)/P.R;
    const n=Math.hypot(vx,vy);if(n>1.08){vx*=1.08/n;vy*=1.08/n}
    V[drag]=[vx,vy];dirty=true;update();
  });
  cv.addEventListener('pointerup',()=>drag=-1);
  root.querySelector('.b-sgd').addEventListener('click',()=>{sgdV=V.map(v=>[...v]);anim=420;dirty=true});
  root.querySelector('.b-reset').addEventListener('click',reset);
  const w={tick(){
    if(anim>0){for(let i=0;i<7&&anim>0;i++,anim--)packGradStep(sgdV,I,C,.02);update();dirty=true}
    if(!dirty)return;dirty=false;
    P.base();
    if(sgdV)V.forEach((v,i)=>P.arrow(v,FC[i],{alpha:.25,lw:2+2.5*I[i]}));
    else V.forEach((v,i)=>P.arrow(v,FC[i],{lw:2+2.5*I[i]}));
    if(sgdV)sgdV.forEach((v,i)=>P.arrow(v,FC[i],{lw:2+2.5*I[i]}));
  }};
  reset();register(root,w);
})();

// ---------- 2. toy model: linear vs relu ----------
(function(){
  const root=$('#w-toymodel');if(!root)return;
  const I6=[1,.7,.49,.34,.24,.17];let seed=3,relu=true,model;
  const cv=root.querySelector('.c-vec'),ctx=setupCanvas(cv,300,300),P=vecPanel(ctx,300,300,null,M);
  const cb=root.querySelector('.c-bars'),bx=setupCanvas(cb,250,300);
  const lossEl=root.querySelector('.r-loss'),stepEl=root.querySelector('.r-step');
  const ind=root.querySelector('.ind'),indB=ind&&ind.querySelector('b');
  const bL=root.querySelector('.b-lin'),bR=root.querySelector('.b-relu');
  let S=null,lossD=null;
  function build(){model=new ToyModel({n:6,m:2,relu,p:.05,importance:I6,seed});S=null;lossD=null;bL.classList.toggle('on',!relu);bR.classList.toggle('on',relu)}
  bL.onclick=()=>{relu=false;build()};bR.onclick=()=>{relu=true;build()};
  root.querySelector('.b-re').onclick=()=>{seed+=13;build()};
  const w={tick(){
    if(model.steps<12000)for(let i=0;i<25;i++)model.step(256);
    S=ema(S,model.theta,EMA_W);
    lossD=lossD==null?model.loss:lossD+.05*(model.loss-lossD);
    P.base();
    for(let i=0;i<6;i++)P.arrow(model.col(i,S),FC[i],{lw:2.5});
    drawBars(bx,250,300,[...Array(6)].map((_,i)=>model.colNorm(i,S)),FC,{target:1,targetLabel:'goal',pal:M});
    lossEl.textContent=fmt(lossD,4);stepEl.textContent=model.steps.toLocaleString();
    if(ind){const live=model.steps<12000;ind.classList.toggle('frozen',!live);indB.textContent=live?'LIVE':'CONVERGED'}
  }};
  build();register(root,w);
})();

// ---------- 3. sparsity dial ----------
// boundaries measured: 9k-step training, 3 seeds per point (see test sweep)
const REGIONS=[[0,.07,'2'],[.07,.17,'3'],[.17,.37,'4'],[.37,1,'5 — pentagon']];
(function(){
  const root=$('#w-sparsity');if(!root)return;
  const Id=[1,.93,.86,.8,.74,.69];let seed=5;
  const u2p=u=>Math.pow(10,(1-u)*0+u*Math.log10(.02));   // p: 1 → 0.02, log
  const slider=root.querySelector('input');
  let model=new ToyModel({n:6,m:2,p:u2p(+slider.value),importance:Id,seed});
  const cv=root.querySelector('.c-vec'),ctx=setupCanvas(cv,300,300),P=vecPanel(ctx,300,300,null,M);
  const cb=root.querySelector('.c-bars'),bx=setupCanvas(cb,250,170);
  const cs=root.querySelector('.c-strip'),sx=setupCanvas(cs,250,96);
  const spEl=root.querySelector('.r-sp'),repEl=root.querySelector('.r-rep');
  let S=null;
  slider.addEventListener('input',()=>{model.p=u2p(+slider.value)});
  root.querySelector('.b-re').onclick=()=>{seed+=13;model=new ToyModel({n:6,m:2,p:u2p(+slider.value),importance:Id,seed});S=null};
  function strip(u){
    sx.clearRect(0,0,250,96);sx.font=SANS;sx.textAlign='center';
    REGIONS.forEach(([a,b,t],i)=>{
      sx.fillStyle=['#161615','#1e1e1c','#262624','#2f2f2c'][i];
      sx.fillRect(8+a*234,14,(b-a)*234,30);
      sx.fillStyle=M.mut;sx.fillText(t,8+(a+b)/2*234,34);
    });
    sx.fillStyle=M.mut;sx.fillText('features represented (measured)',125,76);
    sx.strokeStyle=M.ink;sx.lineWidth=2.5;const x=8+u*234;
    sx.beginPath();sx.moveTo(x,8);sx.lineTo(x,50);sx.stroke();
    sx.fillStyle=M.ink;sx.beginPath();sx.moveTo(x,58);sx.lineTo(x-5,50);sx.lineTo(x+5,50);sx.fill();
  }
  const w={tick(){
    for(let i=0;i<25;i++)model.step(256);
    S=ema(S,model.theta,EMA_W);
    P.base();for(let i=0;i<6;i++)P.arrow(model.col(i,S),FC[i],{lw:2.5});
    const norms=[...Array(6)].map((_,i)=>model.colNorm(i,S));
    drawBars(bx,250,170,norms,FC,{target:1,targetLabel:'',pal:M});
    strip(+slider.value);
    const p=model.p;
    spEl.textContent=(100*(1-p)).toFixed(0)+'%';
    repEl.textContent=norms.filter(x=>x>.5).length;
  }};
  window.__dbg.sp3=()=>({raw:model.col(0),disp:S?model.col(0,S):null});
  register(root,w);
})();

// ---------- 4. geometry m=3 ----------
(function(){
  const root=$('#w-geom');if(!root)return;
  let seed=11,model=new ToyModel({n:8,m:3,p:.03,seed});
  const cv=root.querySelector('.c-3d'),ctx=setupCanvas(cv,300,300);
  const ch=root.querySelector('.c-dims'),hx=setupCanvas(ch,330,300);
  let yaw=.6,pitch=.35,auto=true,hist=[],S=null;
  const slider=root.querySelector('input');
  const u2p=u=>Math.pow(10,Math.log10(.5)*(1-u)+Math.log10(.015)*u);
  slider.addEventListener('input',()=>{model.p=u2p(+slider.value)});
  root.querySelector('.b-re').onclick=()=>{seed+=13;model=new ToyModel({n:8,m:3,p:u2p(+slider.value),seed});hist=[];S=null};
  cv.classList.add('grabbable');
  let dr=null;
  cv.addEventListener('pointerdown',e=>{dr=[e.clientX,e.clientY];auto=false;cv.setPointerCapture(e.pointerId)});
  cv.addEventListener('pointermove',e=>{if(!dr)return;yaw+=(e.clientX-dr[0])*.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-dr[1])*.01));dr=[e.clientX,e.clientY]});
  cv.addEventListener('pointerup',()=>dr=null);
  function proj(v){
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const x=v[0]*cy+v[2]*sy,z=-v[0]*sy+v[2]*cy,y=v[1]*cp-z*sp,zz=v[1]*sp+z*cp;
    return[x,y,zz];
  }
  function draw3d(){
    ctx.clearRect(0,0,300,300);
    const cx=150,cy=150,R=105;
    ctx.strokeStyle=M.rule;ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.stroke();
    ctx.setLineDash([3,4]);ctx.beginPath();                       // equator
    for(let t=0;t<=64;t++){const a=t/64*6.2832,p=proj([Math.cos(a),0,Math.sin(a)]);
      const X=cx+p[0]*R,Y=cy-p[1]*R;t?ctx.lineTo(X,Y):ctx.moveTo(X,Y)}
    ctx.stroke();ctx.setLineDash([]);
    const arrs=[...Array(8)].map((_,i)=>{
      const c=model.col(i,S),n=Math.hypot(...c)||1;
      return{i,n:Math.min(n,1.15),p:proj(c.map(x=>x/n))};
    }).sort((a,b)=>a.p[2]-b.p[2]);
    for(const a of arrs){
      const X=cx+a.p[0]*R*a.n,Y=cy-a.p[1]*R*a.n,back=a.p[2]<0;
      ctx.strokeStyle=ctx.fillStyle=FC[a.i];ctx.globalAlpha=back?.35:1;ctx.lineWidth=2.5;
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(X,Y);ctx.stroke();
      ctx.beginPath();ctx.arc(X,Y,4.5,0,7);ctx.fill();ctx.globalAlpha=1;
    }
  }
  const REF=[[1,'1 — own dimension'],[.5,'1/2 — antipodal pair'],[.375,'3/8 — antiprism'],[0,'0 — dropped']];
  function drawHist(){
    hx.clearRect(0,0,330,300);hx.font=SANS;
    const L=36,T=12,W2=330-L-10,H2=300-T-26,ymax=1.08;
    const Y=v=>T+H2-(v/ymax)*H2;
    for(const[v,t]of REF){hx.strokeStyle=M.rule;hx.setLineDash([3,3]);
      hx.beginPath();hx.moveTo(L,Y(v));hx.lineTo(L+W2,Y(v));hx.stroke();hx.setLineDash([]);
      hx.fillStyle=M.mut;hx.textAlign='left';hx.fillText(t,L+4,Y(v)-4)}
    hx.strokeStyle=M.mut;hx.beginPath();hx.moveTo(L,T);hx.lineTo(L,T+H2);hx.lineTo(L+W2,T+H2);hx.stroke();
    hx.save();hx.translate(12,T+H2/2);hx.rotate(-Math.PI/2);hx.textAlign='center';hx.fillText('dimensions per feature  Dᵢ',0,0);hx.restore();
    hx.textAlign='center';hx.fillText('training time →',L+W2/2,298);
    if(hist.length<2)return;
    const HN=hist.length;
    for(let i=0;i<8;i++){hx.strokeStyle=FC[i];hx.lineWidth=1.8;hx.globalAlpha=.85;hx.beginPath();
      for(let t=0;t<HN;t++){const X=L+t/(Math.max(HN-1,200))*W2;t?hx.lineTo(X,Y(hist[t][i])):hx.moveTo(X,Y(hist[t][i]))}
      hx.stroke()}
    hx.globalAlpha=1;
  }
  const w={tick(){
    for(let i=0;i<18;i++)model.step(256);
    S=ema(S,model.theta,EMA_W);
    if(model.steps%72<18){hist.push(model.dims(S));if(hist.length>200)hist.shift()}
    if(auto)yaw+=.004;
    draw3d();drawHist();
  }};
  register(root,w);
})();

// ---------- shared frozen base model (pentagon) for polysemanticity + SAE ----------
const BASE={model:new ToyModel({n:6,m:2,p:.03,importance:[1,.93,.86,.8,.74,.69],seed:5}),done:false,subs:[]};
window.__dbg.BASE=BASE;
BASE.tickTrain=()=>{ if(BASE.done)return;
  for(let i=0;i<120&&BASE.model.steps<8000;i++)BASE.model.step(256);
  if(BASE.model.steps>=8000){BASE.done=true;BASE.subs.forEach(f=>f())}
};

// ---------- 5. polysemanticity ----------
(function(){
  const root=$('#w-poly');if(!root)return;
  const cv=root.querySelector('canvas'),ctx=setupCanvas(cv,560,250);
  cv.style.maxWidth='100%';cv.style.height='auto';   // 560px logical width; shrink on phones
  const prog=root.querySelector('.r-prog');
  const w={tick(){
    BASE.tickTrain();
    const m=BASE.model;
    prog.textContent=BASE.done?'trained · frozen':`training… ${m.steps}/8000`;
    ctx.clearRect(0,0,560,250);ctx.font=SANS;
    for(let a=0;a<2;a++){
      const ox=a*285+10,mid=135;
      ctx.fillStyle=M.ink;ctx.textAlign='left';ctx.font='600 '+SANS;
      ctx.fillText(`neuron ${a+1} — weight per feature`,ox+8,18);ctx.font=SANS;
      ctx.strokeStyle=M.rule;ctx.beginPath();ctx.moveTo(ox+8,mid);ctx.lineTo(ox+262,mid);ctx.stroke();
      for(let i=0;i<6;i++){
        const v=m.theta[a*6+i],h=v*52,x=ox+18+i*41;
        ctx.fillStyle=FC[i];ctx.globalAlpha=.9;
        ctx.fillRect(x,h>0?mid-h:mid,30,Math.abs(h));ctx.globalAlpha=1;
        ctx.fillStyle=M.mut;ctx.textAlign='center';
        ctx.fillText(v.toFixed(2),x+15,h>0?mid-h-6:mid+Math.abs(h)+14);
        ctx.fillText('F'+(i+1),x+15,238);
      }
    }
  }};
  register(root,w);
})();

// ---------- 6. SAE lab ----------
(function(){
  const root=$('#w-sae');if(!root)return;
  const K=10;let sae=null,lam=.3;
  const slider=root.querySelector('input');
  const u2l=u=>Math.pow(10,Math.log10(5e-4)*(1-u)+Math.log10(2.5)*u);
  const l2u=l=>(Math.log10(l)-Math.log10(5e-4))/(Math.log10(2.5)-Math.log10(5e-4));
  slider.value=l2u(lam);
  const cv=root.querySelector('.c-vec'),ctx=setupCanvas(cv,320,320),P=vecPanel(ctx,320,320,null,M);
  const recEl=root.querySelector('.s-rec .v'),liveEl=root.querySelector('.s-live .v'),l0El=root.querySelector('.s-l0 .v'),lamEl=root.querySelector('.r-lam'),mseEl=root.querySelector('.s-mse .v');
  const ind=root.querySelector('.ind'),indB=ind&&ind.querySelector('b');
  const x6=new Float64Array(6);
  const sampler=h=>{BASE.model.sampleX(x6);BASE.model.hidden(x6,h)};
  const abl=new Uint8Array(K);let l0=0,tk=0,SW=null,recD=null,mseD=null;
  slider.addEventListener('input',()=>{lam=u2l(+slider.value);if(sae)sae.lam=lam});
  root.querySelector('.b-re').onclick=()=>{sae=new SAE({k:K,m:2,lam,seed:7+Math.floor(Math.random()*99)});SW=null;recD=null;mseD=null;abl.fill(0)};
  // the scalpel: click an atom to ablate it (forced silent in every forward pass)
  cv.style.cursor='pointer';
  cv.addEventListener('pointerdown',e=>{
    if(!sae||!SW)return;
    const b=cv.getBoundingClientRect(),mx=e.clientX-b.left,my=e.clientY-b.top;
    let best=20,bi=-1,bestLive=false;
    for(let j=0;j<K;j++){
      const a=sae.atom(j,SW),[X,Y]=P.px([a[0]*.82,a[1]*.82]);
      const d=Math.hypot(mx-X,my-Y),liveJ=sae.usage[j]>=.01||abl[j];
      if(d<20&&liveJ&&(!bestLive||d<best)){best=d;bi=j;bestLive=true}   // live (or ablated) atoms win ties
      else if(!bestLive&&d<best){best=d;bi=j}
    }
    if(bi>=0)abl[bi]^=1;
  });
  function estMSE(){
    const h=new Float64Array(2),f=new Float64Array(K);let tot=0,c=0;
    for(let t=0;t<100;t++){
      sampler(h);
      for(let j=0;j<K;j++){let z=sae.be[j];for(let a=0;a<2;a++)z+=sae.We[j*2+a]*h[a];f[j]=abl[j]?0:Math.max(0,z)}
      for(let a=0;a<2;a++){let y=0;for(let j=0;j<K;j++)y+=sae.Wd[a*K+j]*f[j];tot+=(y-h[a])**2}
      c++;
    }
    return tot/c;
  }
  function estL0(){
    const h=new Float64Array(2);let tot=0,cnt=0;
    for(let t=0;t<260;t++){
      BASE.model.sampleX(x6);if(!x6.some(v=>v>0))continue;BASE.model.hidden(x6,h);
      let c=0;for(let j=0;j<K;j++){let z=sae.be[j];for(let a=0;a<2;a++)z+=sae.We[j*2+a]*h[a];if(z>.02)c++}
      tot+=c;cnt++;
    }
    return cnt?tot/cnt:0;
  }
  const w={tick(){
    BASE.tickTrain();
    if(ind){ind.classList.toggle('frozen',!BASE.done);indB.textContent=BASE.done?'LIVE':'WAITING'}
    if(!BASE.done)return;
    if(!sae)sae=new SAE({k:K,m:2,lam,seed:7});
    const hot=abl.some(x=>x);          // scalpel in: slow motion, so the wound and the healing are visible
    for(let i=0;i<(hot?4:30);i++)sae.step(128,sampler,abl);
    SW=ema(SW,sae.Wd,EMA_W);
    const rec=sae.recovery(BASE.model);
    recD=recD==null?rec:recD+(hot?.2:.05)*(rec-recD);
    if(hot||++tk%5===0){const e2=estMSE();mseD=mseD==null?e2:mseD+(hot?.5:.25)*(e2-mseD)}
    if(tk%25===0)l0=estL0();
    P.base();
    for(let i=0;i<6;i++){const c=BASE.model.col(i);if(Math.hypot(...c)>.4)P.arrow(c.map(v=>v/1.18),FC[i],{lw:7,alpha:.30})}
    let live=0;
    for(let j=0;j<K;j++){
      const a=sae.atom(j,SW),dead=sae.usage[j]<.01;if(!dead&&!abl[j])live++;
      if(abl[j]){
        P.arrow(a.map(v=>v*.82),'#ff8a63',{lw:2,alpha:.9,dash:[5,3]});
        const[X,Y]=P.px([a[0]*.86,a[1]*.86]);
        ctx.strokeStyle='#ff8a63';ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(X-4,Y-4);ctx.lineTo(X+4,Y+4);ctx.moveTo(X+4,Y-4);ctx.lineTo(X-4,Y+4);ctx.stroke();
      }else P.arrow(a.map(v=>v*.82),dead?M.mut:M.ink,{lw:dead?1.5:2.5,alpha:dead?.45:.95,dash:dead?[4,4]:null});
    }
    recEl.textContent=(100*recD).toFixed(0)+'%';
    liveEl.textContent=live+' / '+K;l0El.textContent=l0.toFixed(1);
    if(mseEl)mseEl.textContent=mseD==null?'—':mseD.toFixed(3);
    lamEl.textContent='λ = '+lam.toPrecision(2)+(hot?' · scalpel in — slow motion':'');
  }};
  register(root,w);
})();

// ---------- 0. hero: the superposition machine (immersive scene) ----------
const NAMES=['citations','dialogue','HTTP','Korean','code','weather'];
const WORDS=('gravity sarcasm Seoul semicolons jazz enzymes irony April vowels debt chess typhoons haiku copper syntax tango glaciers probability garlic sonnets voltage camels metaphor Tuesday algebra perfume treaties whales sushi firewalls ballet neutrinos grief passports origami caffeine elections coral lullabies patents monsoons chemistry folklore spreadsheets tides opera bacteria currencies deserts nostalgia plumbing galaxies verbs insulin marathons puns vaccines poker lighthouses grammar meteors silk auctions pottery rivalry telescopes yeast contracts aurora dialects skyscrapers fossils recipes encryption harvest violins satire blueprints penguins calculus orchids radar proverbs cement butterflies inflation chopsticks thunder alibis kayaks melody quartz riddles subway honey equations parades wolves ledgers incense pixels anchors rumors cellos magnets dunes syllables barcodes embers').split(' ');
(function(){
  const root=$('#machine');if(!root)return;
  const stage=root.querySelector('.scene-stage');
  const BG='#0d0d0c',RL='#2b2b29',TX='#8a8a86',TRK='#1e1e1c',TICK='#f2f2f0';
  const S=280,OW=272;
  const ch=root.querySelector('.c-ch'),cx=setupCanvas(ch,S,S);
  const co=root.querySelector('.c-out'),ox=setupCanvas(co,OW,S);
  const errEl=root.querySelector('.r-err'),subEl=root.querySelector('.subtitle'),skipB=root.querySelector('.b-skip');
  const lamps=[...root.querySelectorAll('.lamps button')];
  const manual=new Set();let pulses=[],t0ms=null,demo=true,si=0,lastAmb=0;
  const SCRIPT=[                                  // wall-clock ms after training completes
    {at:500,sub:'Every word drifting around you is something a network must remember.'},
    {at:3200,sub:'This one has room for six — and only two neurons to keep them in. Watch:',fire:[0]},
    {at:6200,fire:[2]},
    {at:8200,sub:'…clean readout, every time. Six concepts, two neurons.',fire:[3]},
    {at:10900,sub:'Now fire two at once:',fire:[1,4]},
    {at:12500,sub:'✗ Crosstalk — they collided. This whole page is about that tradeoff.'},
    {at:15200,sub:'Your turn. Click the lamps — can you find a pair that survives?',end:true},
  ];
  function endDemo(){demo=false;skipB.style.display='none'}
  skipB.onclick=()=>{endDemo();pulses=[];subEl.textContent='Your turn. Click the lamps — can you find a pair that survives?'};
  lamps.forEach(b=>b.onclick=()=>{
    if(demo){endDemo();subEl.textContent=''}
    const i=+b.dataset.i;manual.has(i)?manual.delete(i):manual.add(i);b.classList.toggle('on');pulses=[]});
  root.querySelector('.b-clear').onclick=()=>{manual.clear();lamps.forEach(b=>b.classList.remove('on'))};
  // --- word field (the model's world, drifting behind the model) ---
  const wf=root.querySelector('.wordfield'),wx=wf.getContext('2d');
  let WW=0,WH=0,dpr=1,words=[];
  function sizeWF(){
    dpr=Math.min(devicePixelRatio||1,2);
    WW=root.clientWidth;WH=root.clientHeight;
    wf.width=WW*dpr;wf.height=WH*dpr;wx.setTransform(dpr,0,0,dpr,0,0);
  }
  function seedWords(){
    const r=new RNG(99),n=innerWidth<760?55:WORDS.length;
    words=[];
    for(let i=0;i<n;i++)words.push({w:WORDS[i],x:r.next()*WW,y:r.next()*WH,z:.25+.75*r.next(),f:-1});
    NAMES.forEach((w,i)=>words.push({w,x:r.next()*WW,y:r.next()*WH,z:.6+.4*r.next(),f:i}));
  }
  sizeWF();seedWords();
  addEventListener('resize',()=>{sizeWF();seedWords()});
  // --- pointer: parallax + stage tilt ---
  let mx=0,my=0,tx=0,ty=0;
  const fine=matchMedia('(pointer:fine)').matches&&!matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(fine)root.addEventListener('pointermove',e=>{
    const b=root.getBoundingClientRect();
    mx=(e.clientX-b.left)/b.width*2-1;my=(e.clientY-b.top)/b.height*2-1;
  });
  // --- scroll cue ---
  const cue=root.querySelector('.scroll-cue');
  addEventListener('scroll',()=>cue.classList.add('hide'),{once:true,passive:true});
  const x=new Float64Array(6),h=new Float64Array(2),z=new Float64Array(6);
  const rm=matchMedia('(prefers-reduced-motion: reduce)').matches;
  function drawWords(){
    wx.clearRect(0,0,WW,WH);
    for(const o of words){
      if(!rm){o.x+=.07*o.z;if(o.x>WW+60)o.x=-60}
      const px=o.x+mx*o.z*26,py=o.y+my*o.z*16;
      wx.font=(9.5+7*o.z)+'px -apple-system,sans-serif';
      if(o.f>=0){const a=x[o.f]||0;
        wx.fillStyle=FC[o.f];wx.globalAlpha=Math.min(1,.34+.3*o.z+.6*a);
        wx.font=(9.5+7*o.z+6*a)+'px -apple-system,sans-serif';
        if(a>.12){wx.shadowColor=FC[o.f];wx.shadowBlur=12}}
      else{wx.fillStyle='#7a7a76';wx.globalAlpha=.07+.13*o.z}
      wx.fillText(o.w,px,py);wx.shadowBlur=0;
    }
    wx.globalAlpha=1;
  }
  function chBase(){
    const c=S/2,R=S/2-26;
    cx.fillStyle=BG;cx.fillRect(0,0,S,S);
    cx.strokeStyle=RL;cx.lineWidth=1;
    cx.beginPath();cx.moveTo(c-R-12,c);cx.lineTo(c+R+12,c);cx.moveTo(c,c-R-12);cx.lineTo(c,c+R+12);cx.stroke();
    cx.setLineDash([3,4]);cx.beginPath();cx.arc(c,c,R,0,7);cx.stroke();cx.setLineDash([]);
    cx.fillStyle=TX;cx.font=SANS;cx.textAlign='left';
    cx.fillText('neuron 1 →',c+R-56,c+15);
    cx.save();cx.translate(c-7,c-R+58);cx.rotate(-Math.PI/2);cx.fillText('neuron 2 →',0,0);cx.restore();
  }
  function arrowD(vx,vy,color,lw,alpha){
    const c=S/2,R=S/2-26,X=c+vx*R,Y=c-vy*R;
    cx.strokeStyle=cx.fillStyle=color;cx.lineWidth=lw;cx.globalAlpha=alpha;
    cx.beginPath();cx.moveTo(c,c);cx.lineTo(X,Y);cx.stroke();
    const a=Math.atan2(Y-c,X-c),hs=4+lw;
    cx.beginPath();cx.moveTo(X,Y);
    cx.lineTo(X-hs*Math.cos(a-.45),Y-hs*Math.sin(a-.45));
    cx.lineTo(X-hs*Math.cos(a+.45),Y-hs*Math.sin(a+.45));
    cx.closePath();cx.fill();cx.globalAlpha=1;
  }
  const w={tick(){
    BASE.tickTrain();
    const m=BASE.model,now=performance.now();
    if(BASE.done&&t0ms===null)t0ms=now;
    const dms=t0ms===null?-1:now-t0ms;
    // demo script (wall-clock so it's identical at any frame rate)
    if(demo&&dms>=0&&si<SCRIPT.length&&dms>=SCRIPT[si].at){
      const st=SCRIPT[si++];
      if(st.sub)subEl.textContent=st.sub;
      if(st.fire)st.fire.forEach(i=>pulses.push({i,s:now,dur:1400}));
      if(st.end)endDemo();
    }
    // ambient auto-fire after demo
    if(!demo&&manual.size===0&&dms>=0&&now-lastAmb>1900){
      lastAmb=now;
      const two=Math.random()<.2,i=Math.floor(Math.random()*5);
      pulses.push({i,s:now,dur:1300});
      if(two){let j=Math.floor(Math.random()*5);if(j===i)j=(j+1)%5;pulses.push({i:j,s:now,dur:1300})}
    }
    x.fill(0);for(const i of manual)x[i]=.7;
    pulses=pulses.filter(p=>now-p.s<p.dur);
    for(const p of pulses){const ph=(now-p.s)/p.dur;x[p.i]=Math.min(1,x[p.i]+.7*Math.sin(Math.PI*ph))}
    lamps.forEach((b,i)=>b.classList.toggle('lit',x[i]>.12));
    m.hidden(x,h);
    for(let i=0;i<6;i++){let s2=m.theta[12+i];for(let a=0;a<2;a++)s2+=m.theta[a*6+i]*h[a];z[i]=Math.max(0,s2)}
    // world + tilt
    drawWords();
    if(fine){tx+=((-my*3.2)-tx)*.06;ty+=((mx*4)-ty)*.06;
      stage.style.transform=`perspective(1100px) rotateX(${tx.toFixed(2)}deg) rotateY(${ty.toFixed(2)}deg)`}
    // channel
    chBase();
    for(let i=0;i<6;i++){const c=m.col(i);arrowD(c[0]/1.25,c[1]/1.25,FC[i],2,.3)}
    let px=0,py=0;
    const c0=S/2,R0=S/2-26;
    for(let i=0;i<6;i++){
      if(x[i]<.03)continue;const c=m.col(i),nx=px+x[i]*c[0]/1.25,ny=py+x[i]*c[1]/1.25;
      cx.strokeStyle=FC[i];cx.lineWidth=6;cx.lineCap='round';
      cx.shadowColor=FC[i];cx.shadowBlur=14;
      cx.beginPath();cx.moveTo(c0+px*R0,c0-py*R0);cx.lineTo(c0+nx*R0,c0-ny*R0);cx.stroke();
      cx.shadowBlur=0;px=nx;py=ny;
    }
    if(px||py){cx.fillStyle='#fff';cx.shadowColor='#fff';cx.shadowBlur=10;
      cx.beginPath();cx.arc(c0+px*R0,c0-py*R0,4.5,0,7);cx.fill();cx.shadowBlur=0}
    // output
    ox.fillStyle=BG;ox.fillRect(0,0,OW,S);ox.font=SANS;
    let worst=0,lost=false;
    for(let i=0;i<6;i++){
      const y0=26+i*42,d=Math.abs(x[i]-z[i]);
      if(x[i]>.3&&m.colNorm(i)<.5)lost=true;
      if(x[i]>.03||z[i]>.03)worst=Math.max(worst,d);
      ox.fillStyle=TX;ox.textAlign='left';ox.fillText(NAMES[i],8,y0+10);
      ox.fillStyle=TRK;ox.fillRect(76,y0,164,12);
      if(z[i]>.005){ox.fillStyle=FC[i];ox.shadowColor=FC[i];ox.shadowBlur=7;
        ox.fillRect(76,y0,164*Math.min(z[i],1.1),12);ox.shadowBlur=0}
      ox.strokeStyle=TICK;ox.lineWidth=2;
      const tx2=76+164*Math.min(x[i],1.1);
      ox.beginPath();ox.moveTo(tx2,y0-3);ox.lineTo(tx2,y0+15);ox.stroke();
      if(d>.28){ox.fillStyle='#ff6b4a';ox.beginPath();ox.arc(254,y0+6,3.5,0,7);ox.fill()}
    }
    if(!BASE.done){errEl.textContent='assembling the model… '+Math.round(100*m.steps/8000)+'%';errEl.style.color=''}
    else if(lost){errEl.textContent='✗ lost — the model never stored this concept';errEl.style.color='#ff8a63'}
    else if(worst>.28){errEl.textContent='✗ crosstalk — concepts collided';errEl.style.color='#ff8a63'}
    else{errEl.textContent='✓ clean readout';errEl.style.color='#f2f2f0'}
  }};
  window.__dbg.hero=()=>({t0ms,si,demo,pulses:pulses.length,words:words.length,skipShown:skipB.style.display!=='none'});
  register(root,w);
})();

// ---------- scroll reveal + progress nav ----------
(function(){
  const rd=matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(!rd){
    const els=[...document.querySelectorAll('figure.widget,.takeaway')];
    els.forEach(e=>e.classList.add('reveal'));
    const ro=new IntersectionObserver(es=>es.forEach(en=>{
      if(en.isIntersecting){en.target.classList.add('in');ro.unobserve(en.target)}
    }),{rootMargin:'0px 0px -6% 0px'});
    els.forEach(e=>ro.observe(e));
  }
  const pa=document.querySelector('#primer-all');
  if(pa)pa.onclick=()=>{
    const ds=[...document.querySelectorAll('details.primer')],open=ds.some(d=>!d.open);
    ds.forEach(d=>d.open=open);pa.textContent=open?'collapse all':'expand all';
  };
  const map=document.querySelector('#map');if(!map)return;
  // specimen thumbnail: the model (BASE), animated only while it trains, then frozen.
  // The map also drives BASE training so the later sections are ready even if the reader skips the hero.
  const scv=map.querySelector('canvas'),sx=setupCanvas(scv,118,118),sst=map.querySelector('.st');
  function drawSpec(){
    const S=118,c=S/2,R=S/2-9,m=BASE.model;
    sx.clearRect(0,0,S,S);
    sx.strokeStyle=M.rule;sx.lineWidth=1;
    sx.beginPath();sx.moveTo(c-R,c);sx.lineTo(c+R,c);sx.moveTo(c,c-R);sx.lineTo(c,c+R);sx.stroke();
    sx.setLineDash([2,3]);sx.beginPath();sx.arc(c,c,R,0,7);sx.stroke();sx.setLineDash([]);
    for(let i=0;i<6;i++){
      const v=m.col(i),X=c+v[0]/1.2*R,Y=c-v[1]/1.2*R;
      sx.strokeStyle=sx.fillStyle=FC[i];sx.lineWidth=2;
      sx.beginPath();sx.moveTo(c,c);sx.lineTo(X,Y);sx.stroke();
      sx.beginPath();sx.arc(X,Y,2.5,0,7);sx.fill();
    }
  }
  let specDone=false;
  register(map,{tick(){
    if(specDone)return;
    BASE.tickTrain();drawSpec();
    if(BASE.done){sst.textContent='trained · frozen';specDone=true}
    else sst.textContent='training '+Math.round(100*BASE.model.steps/8000)+'%';
  }});
  // the map is redundant (and illegible) while the model itself fills the screen
  const heroSec=document.querySelector('#machine');
  if(heroSec)new IntersectionObserver(e=>map.classList.toggle('off',e[0].isIntersecting),{threshold:.25}).observe(heroSec);
  const links=[...map.querySelectorAll('nav a')];
  const secs=links.map(a=>document.querySelector(a.getAttribute('href')));
  const so=new IntersectionObserver(es=>es.forEach(en=>{
    if(en.isIntersecting){const i=secs.indexOf(en.target);
      links.forEach((l,j)=>l.classList.toggle('cur',j===i))}
  }),{rootMargin:'-40% 0px -50% 0px'});
  secs.forEach(s=>s&&so.observe(s));
})();
