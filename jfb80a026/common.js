(function(){
const PAGES=[["index.html","Home"],["16aefa39.html","Animations"],["03454c9f.html","Search explorer"],["c54da44b.html","Results tables"],["d97f3d8b.html","Source code"]];
window.SI={};
SI.nav=function(){
  const cur=location.pathname.split("/").pop()||"index.html";
  const h=document.createElement("header");h.className="site";
  h.innerHTML='<div class="in"><a class="brand" href="index.html">Review materials</a><nav>'+PAGES.map(p=>'<a href="'+p[0]+'"'+(p[0]===cur?' class="on"':'')+'>'+p[1]+'</a>').join("")+'</nav><button class="theme" title="Switch light/dark">◐</button></div>';
  document.body.prepend(h);
  h.querySelector(".theme").onclick=()=>{const r=document.documentElement;const d=getComputedStyle(r).getPropertyValue("--bg").trim()==="#15181b";r.dataset.theme=d?"light":"dark";try{localStorage.setItem("si-theme",r.dataset.theme)}catch(e){};document.dispatchEvent(new Event("themechange"))};
  try{const t=localStorage.getItem("si-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}
  const f=document.createElement("footer");f.innerHTML='Review materials for a computational study of reaction barriers (Skala-1.1, vinyl alcohol and formamide). Static pages; they run from a folder or from GitHub Pages and need no server.';
  document.body.append(f);
};
SI.colab=function(file){const c=window.SI_CONFIG;return "https://colab.research.google.com/github/"+c.repo+"/blob/"+c.branch+"/"+c.folder+c.src+file};
SI.cfgOk=function(){return window.SI_CONFIG&&!/YOUR-/.test(window.SI_CONFIG.repo)};
// colour scale (viridis-like)
const VS=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
SI.viridis=function(t){t=Math.max(0,Math.min(1,t));const x=t*(VS.length-1),i=Math.min(VS.length-2,Math.floor(x)),f=x-i;const a=VS[i],b=VS[i+1];return "rgb("+[0,1,2].map(k=>Math.round(a[k]+(b[k]-a[k])*f)).join(",")+")"};
SI.css=function(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()};
SI.median=function(a){if(!a.length)return NaN;const s=[...a].sort((x,y)=>x-y),m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2};
SI.fmt=function(x,d){return (x===null||x===undefined||isNaN(x))?"–":Number(x).toFixed(d===undefined?2:d)};
SI.armName=function(a){return ({"agent":"Menu agent","twin":"Frozen twin","twin_wells_first":"Reordered twin","control":"Anchor-first control","saddle_exploit+delta":"Best classical rule (saddle exploit + Δ)","saddle_exploit":"Saddle exploit","random":"Random","random+delta":"Random + Δ","uniform":"Uniform","uniform+delta":"Uniform + Δ","uncertainty":"Uncertainty","uncertainty+delta":"Uncertainty + Δ","path_uncertainty":"Path uncertainty","path_uncertainty+delta":"Path uncertainty + Δ"})[a]||a};
SI.molName=function(m){return m==="v"?"Vinyl alcohol → acetaldehyde":"Formamide → formimidic acid"};
SI.designName=function(c){return ({e:"Edge anchors",s:"Scout's two wells",r:"Four random points"})[c]};
SI.molKey=function(m){return m==="v"?"vinyl":"formamide"};
// heat-map of a surface with sample points. opts: {vmax, picks:[[i,j]..], upto, start:4, marks:[{ij,shape,label}], cell}
SI.heat=function(cv,G,Z,o){
  o=o||{};const nt=G.theta.length,nr=G.r.length,cw=o.cell||26,ml=44,mb=34,mt=10,mr=10;
  const W=ml+nt*cw+mr,H=mt+nr*cw+mb;const dpr=window.devicePixelRatio||1;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+"px";cv.style.height=H+"px";const x=cv.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);
  x.clearRect(0,0,W,H);const vmax=o.vmax||120;
  const X=i=>ml+i*cw,Y=j=>mt+(nr-1-j)*cw;
  for(let i=0;i<nt;i++)for(let j=0;j<nr;j++){x.fillStyle=SI.viridis(1-Math.min(Math.max(Z[i][j],0),vmax)/vmax);x.fillRect(X(i),Y(j),cw,cw)}
  x.fillStyle=SI.css("--mut");x.font="11px sans-serif";x.textAlign="center";
  for(let i=0;i<nt;i+=3){x.fillText(G.theta[i],X(i)+cw/2,H-18)}
  x.fillText("θ (degrees)",ml+nt*cw/2,H-4);x.textAlign="right";
  for(let j=0;j<nr;j+=2){x.fillText(G.r[j].toFixed(1),ml-5,Y(j)+cw/2+4)}
  x.save();x.translate(11,mt+nr*cw/2);x.rotate(-Math.PI/2);x.textAlign="center";x.fillText("r (Å)",0,0);x.restore();
  (o.marks||[]).forEach(m=>{const cx=X(m.ij[0])+cw/2,cy=Y(m.ij[1])+cw/2;x.lineWidth=2;x.strokeStyle=m.color||"#fff";x.fillStyle=m.color||"#fff";
    if(m.shape==="star"){x.beginPath();for(let k=0;k<10;k++){const r=k%2?cw*.22:cw*.5,a=-Math.PI/2+k*Math.PI/5;x.lineTo(cx+r*Math.cos(a),cy+r*Math.sin(a))}x.closePath();x.stroke()}
    else if(m.shape==="diamond"){x.beginPath();x.moveTo(cx,cy-cw*.5);x.lineTo(cx+cw*.5,cy);x.lineTo(cx,cy+cw*.5);x.lineTo(cx-cw*.5,cy);x.closePath();x.stroke()}
    else if(m.shape==="x"){x.beginPath();x.moveTo(cx-cw*.35,cy-cw*.35);x.lineTo(cx+cw*.35,cy+cw*.35);x.moveTo(cx+cw*.35,cy-cw*.35);x.lineTo(cx-cw*.35,cy+cw*.35);x.stroke()}
    else{x.strokeRect(cx-cw*.45,cy-cw*.45,cw*.9,cw*.9)}});
  const P=o.picks||[],upto=o.upto===undefined?P.length:o.upto,ns=o.start===undefined?4:o.start;
  for(let k=0;k<Math.min(upto,P.length);k++){const cx=X(P[k][0])+cw/2,cy=Y(P[k][1])+cw/2;const last=k===upto-1;
    x.beginPath();if(k<ns){x.rect(cx-9,cy-9,18,18)}else{x.arc(cx,cy,9,0,7)}
    x.fillStyle=last?"#ffd23f":"#ffffff";x.fill();x.lineWidth=last?2.5:1.2;x.strokeStyle="#111";x.stroke();
    x.fillStyle="#111";x.font="bold 10px sans-serif";x.textAlign="center";x.fillText(k+1,cx,cy+3.5)}
};
SI.lightbox=function(){
  if(document.getElementById("lb"))return;const d=document.createElement("div");d.id="lb";d.innerHTML='<img alt=""><p></p><div><button class="btn alt" data-k="p" style="color:#fff;border-color:#fff">‹ previous</button><button class="btn alt" data-k="c" style="color:#fff;border-color:#fff">close</button><button class="btn alt" data-k="n" style="color:#fff;border-color:#fff">next ›</button></div>';document.body.append(d);
  let L=[],i=0;const show=()=>{d.querySelector("img").src=L[i].src;d.querySelector("p").textContent=L[i].cap;d.classList.add("on")};
  d.onclick=e=>{const k=e.target.dataset&&e.target.dataset.k;if(k==="p"){i=(i-1+L.length)%L.length;show()}else if(k==="n"){i=(i+1)%L.length;show()}else if(k==="c"||e.target===d)d.classList.remove("on")};
  document.addEventListener("keydown",e=>{if(!d.classList.contains("on"))return;if(e.key==="Escape")d.classList.remove("on");if(e.key==="ArrowRight"){i=(i+1)%L.length;show()}if(e.key==="ArrowLeft"){i=(i-1+L.length)%L.length;show()}});
  SI.openLB=function(list,idx){L=list;i=idx;show()};
};
document.addEventListener("DOMContentLoaded",()=>SI.nav());
})();
