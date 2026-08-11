// Shared math/training engine. Tiny models, plain typed arrays, no deps.
export class RNG{
  constructor(seed){this.s=seed>>>0||1}
  next(){let t=this.s+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}
  range(a,b){return a+(b-a)*this.next()}
}
export class Adam{
  constructor(n,lr){this.m=new Float64Array(n);this.v=new Float64Array(n);this.t=0;this.lr=lr}
  step(p,g){const{m,v}=this,b1=.9,b2=.999;this.t++;
    const a=this.lr*Math.sqrt(1-Math.pow(b2,this.t))/(1-Math.pow(b1,this.t));
    for(let i=0;i<p.length;i++){m[i]=b1*m[i]+(1-b1)*g[i];v[i]=b2*v[i]+(1-b2)*g[i]*g[i];p[i]-=a*m[i]/(Math.sqrt(v[i])+1e-8)}}
}

// Anthropic toy model: x sparse in R^n, h=Wx in R^m, x' = ReLU(WᵀWx + b).
export class ToyModel{
  constructor(o){
    this.n=o.n;this.m=o.m;this.relu=o.relu!==false;this.p=o.p;       // p = P(feature active)
    this.I=o.importance||new Array(o.n).fill(1);
    this.rng=new RNG(o.seed||1);
    this.theta=new Float64Array(this.m*this.n+this.n);               // [W (m×n) | b (n)]
    for(let i=0;i<this.m*this.n;i++)this.theta[i]=this.rng.range(-.4,.4);
    this.grad=new Float64Array(this.theta.length);
    this.opt=new Adam(this.theta.length,o.lr||0.01);
    this.steps=0;this.loss=0;
    this._x=new Float64Array(this.n);this._h=new Float64Array(this.m);
    this._z=new Float64Array(this.n);this._dz=new Float64Array(this.n);this._dh=new Float64Array(this.m);
  }
  col(i,th){const W=th||this.theta,c=new Array(this.m);for(let a=0;a<this.m;a++)c[a]=W[a*this.n+i];return c}
  colNorm(i,th){const W=th||this.theta;let s=0;for(let a=0;a<this.m;a++){const w=W[a*this.n+i];s+=w*w}return Math.sqrt(s)}
  sampleX(x){for(let i=0;i<this.n;i++)x[i]=this.rng.next()<this.p?this.rng.next():0}
  hidden(x,h){const{n,m,theta:W}=this;for(let a=0;a<m;a++){let t=0;for(let i=0;i<n;i++)t+=W[a*n+i]*x[i];h[a]=t}}
  step(B){
    const{n,m,theta:W,grad:g,I}=this,x=this._x,h=this._h,z=this._z,dz=this._dz,dh=this._dh;
    g.fill(0);let L=0;
    for(let s=0;s<B;s++){
      this.sampleX(x);this.hidden(x,h);
      for(let i=0;i<n;i++){let t=this.relu?W[m*n+i]:0;for(let a=0;a<m;a++)t+=W[a*n+i]*h[a];z[i]=t}
      dh.fill(0);
      for(let i=0;i<n;i++){
        const y=this.relu?Math.max(0,z[i]):z[i],d=x[i]-y;L+=I[i]*d*d;
        const dzi=(this.relu&&z[i]<=0)?0:-2*I[i]*d;dz[i]=dzi;
        if(this.relu)g[m*n+i]+=dzi;
        for(let a=0;a<m;a++)dh[a]+=W[a*n+i]*dzi;
      }
      for(let a=0;a<m;a++){const da=dh[a],ha=h[a];for(let i=0;i<n;i++)g[a*n+i]+=dz[i]*ha+da*x[i]}
    }
    for(let i=0;i<g.length;i++)g[i]/=B;
    this.opt.step(this.theta,g);this.steps++;this.loss=L/B;return this.loss;
  }
  // feature dimensionality D_i = ||W_i||^2 / Σ_j (Ŵ_i·W_j)^2
  dims(th){
    const n=this.n,D=new Array(n);
    for(let i=0;i<n;i++){
      const wi=this.col(i,th),ni=Math.hypot(...wi);
      if(ni<1e-3){D[i]=0;continue}
      let den=0;
      for(let j=0;j<n;j++){const wj=this.col(j,th);let dot=0;for(let a=0;a<this.m;a++)dot+=wi[a]/ni*wj[a];den+=dot*dot}
      D[i]=ni*ni/den;
    }
    return D;
  }
}

// Sparse autoencoder on the toy model's hidden activations h ∈ R^m.
// f = ReLU(We h + be), ĥ = Wd f (decoder columns unit-norm). Loss = ||h−ĥ||² + λ||f||₁
export class SAE{
  constructor(o){
    this.k=o.k;this.m=o.m;this.lam=o.lam;
    this.rng=new RNG(o.seed||7);
    const{k,m}=this;
    this.We=new Float64Array(k*m);this.be=new Float64Array(k);this.Wd=new Float64Array(m*k);
    for(let i=0;i<k*m;i++)this.We[i]=this.rng.range(-.5,.5);
    for(let i=0;i<m*k;i++)this.Wd[i]=this.rng.range(-.5,.5);
    this.normalize();
    this.opt=new Adam(k*m+k+m*k,5e-3);
    this._g=new Float64Array(k*m+k+m*k);
    this.usage=new Float64Array(k);this.steps=0;this.loss=0;
    this._f=new Float64Array(k);this._r=new Float64Array(m);
  }
  atom(j,Wd0){const Wd=Wd0||this.Wd,c=new Array(this.m);for(let a=0;a<this.m;a++)c[a]=Wd[a*this.k+j];return c}
  normalize(){const{m,k,Wd}=this;for(let j=0;j<k;j++){let s=0;for(let a=0;a<m;a++)s+=Wd[a*k+j]**2;s=Math.sqrt(s)||1;for(let a=0;a<m;a++)Wd[a*k+j]/=s}}
  step(B,sampleH,abl){
    const{k,m,We,be,Wd,lam,_g:g,_f:f,_r:r}=this;g.fill(0);let L=0;
    const oWe=0,obe=k*m,oWd=k*m+k,h=new Float64Array(m);
    for(let s=0;s<B;s++){
      sampleH(h);
      for(let j=0;j<k;j++){let t=be[j];for(let a=0;a<m;a++)t+=We[j*m+a]*h[a];f[j]=(abl&&abl[j])?0:Math.max(0,t)}
      for(let a=0;a<m;a++){let t=0;for(let j=0;j<k;j++)t+=Wd[a*k+j]*f[j];r[a]=t-h[a];L+=r[a]*r[a]}
      for(let j=0;j<k;j++){
        if(f[j]>0){this.usage[j]=.999*this.usage[j]+.001;L+=lam*f[j];
          let df=lam;for(let a=0;a<m;a++)df+=2*r[a]*Wd[a*k+j];
          g[obe+j]+=df;for(let a=0;a<m;a++)g[oWe+j*m+a]+=df*h[a];
        }else this.usage[j]*=.999;
        for(let a=0;a<m;a++)g[oWd+a*k+j]+=2*r[a]*f[j];
      }
    }
    for(let i=0;i<g.length;i++)g[i]/=B;
    const theta=new Float64Array(g.length);theta.set(We,0);theta.set(be,obe);theta.set(Wd,oWd);
    this.opt.step(theta,g);
    We.set(theta.subarray(0,obe));be.set(theta.subarray(obe,oWd));Wd.set(theta.subarray(oWd));
    this.normalize();this.steps++;this.loss=L/B;return this.loss;
  }
  // mean over represented true features of best (signed, clamped) cosine match among live atoms
  recovery(model,minNorm=.4){
    let tot=0,cnt=0;
    for(let i=0;i<model.n;i++){
      const w=model.col(i),nw=Math.hypot(...w);if(nw<minNorm)continue;
      let best=0;
      for(let j=0;j<this.k;j++){
        if(this.usage[j]<.005)continue;
        const a=this.atom(j);let dot=0;for(let q=0;q<this.m;q++)dot+=w[q]/nw*a[q];
        if(dot>best)best=dot;
      }
      tot+=best;cnt++;
    }
    return cnt?tot/cnt:0;
  }
}

// Analytic packing score for the hand game (paper's loss approximation, b=0):
// L = Σ_i I_i(1−|v_i|²)² + c·Σ_{i≠j} I_i (v_i·v_j)²
export function packScore(V,I,c){
  let L=0;const n=V.length;
  for(let i=0;i<n;i++){
    const ni=V[i][0]**2+V[i][1]**2;L+=I[i]*(1-ni)**2;
    for(let j=0;j<n;j++){if(i===j)continue;const d=V[i][0]*V[j][0]+V[i][1]*V[j][1];L+=c*I[i]*d*d}
  }
  return L;
}
export function packGradStep(V,I,c,lr){
  const n=V.length,G=V.map(()=>[0,0]);
  for(let i=0;i<n;i++){
    const ni=V[i][0]**2+V[i][1]**2,s=-4*I[i]*(1-ni);
    G[i][0]+=s*V[i][0];G[i][1]+=s*V[i][1];
    for(let j=0;j<n;j++){if(i===j)continue;
      const d=V[i][0]*V[j][0]+V[i][1]*V[j][1],t=2*c*(I[i]+I[j])*d;
      G[i][0]+=t*V[j][0];G[i][1]+=t*V[j][1];
    }
  }
  for(let i=0;i<n;i++){V[i][0]-=lr*G[i][0];V[i][1]-=lr*G[i][1]}
}
