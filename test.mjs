import{ToyModel,SAE,RNG}from'./js/core.js';
const norms=m=>Array.from({length:m.n},(_,i)=>+m.colNorm(i).toFixed(2));
const train=(m,s)=>{for(let i=0;i<s;i++)m.step(256);return m};

// 1. Linear, sparse, decaying importance -> learns only top-2 (PCA-like)
const I6=[1,.7,.49,.34,.24,.17];
const lin=train(new ToyModel({n:6,m:2,relu:false,p:.05,importance:I6,seed:3}),3000);
console.log('linear norms',norms(lin),'loss',lin.loss.toFixed(4));

// 2. ReLU, same data -> superposition (most/all features represented)
const rl=train(new ToyModel({n:6,m:2,relu:true,p:.05,importance:I6,seed:3}),6000);
console.log('relu   norms',norms(rl),'loss',rl.loss.toFixed(4));

// 3. Sparsity sweep, equal importance: # represented features (norm>.5)
for(const p of[1,.6,.3,.1,.03]){
  const m=train(new ToyModel({n:6,m:2,p,seed:5}),6000);
  console.log(`p=${p}`.padEnd(7),'rep=',norms(m).filter(x=>x>.5).length,norms(m));
}

// 4. n=8,m=3 high sparsity -> dims near 3/8=0.375 (antiprism)
const g=train(new ToyModel({n:8,m:3,p:.03,seed:11}),12000);
console.log('m=3 dims',g.dims().map(d=>+d.toFixed(2)),'norms',norms(g));

// 5. SAE on the relu/equal-importance pentagon-ish model
const base=train(new ToyModel({n:6,m:2,p:.03,seed:5}),8000);
console.log('base norms',norms(base));
const x=new Float64Array(6);
const sampler=h=>{base.sampleX(x);base.hidden(x,h)};
for(const lam of[.005,.05,.25]){
  const s=new SAE({k:10,m:2,lam,seed:7});
  for(let i=0;i<6000;i++)s.step(128,sampler);
  const live=Array.from(s.usage).filter(u=>u>.005).length;
  console.log(`lam=${lam}`.padEnd(9),'recovery',s.recovery(base).toFixed(3),'live atoms',live,'loss',s.loss.toFixed(4));
}
