// Presentation only: consumes existing RIC states and never advances physics.
const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
const vector = p => [p.r,p.i,p.c];
export const supportsSandbox3D = (mode, view, finePointer, landscape = false) => ['sandbox','operatorSandbox'].includes(mode) && ((view === 'desktop' && finePointer) || (view === 'mobile' && landscape));
export class RicCamera3D {
  constructor() {
    this.yaw=-0.65;
    this.elevation=0.5;
    this.focus=[0,0,0];
    this.span=1;
    this.inTrackSign=1;
    this.initialized=false;
  }
  basis() {
    const y=this.yaw,e=this.elevation;
    return [[0,this.inTrackSign*Math.cos(y),Math.sin(y)],[Math.cos(e),this.inTrackSign*Math.sin(y)*Math.sin(e),-Math.cos(y)*Math.sin(e)]];
  }
  project(p,w,h) {
    const b=this.basis(),d=vector(p).map((v,i)=>v-this.focus[i]),s=Math.min(w,h)/this.span;
    return {
      x:w/2+dot(d,b[0])*s,y:h/2-dot(d,b[1])*s
    };
  }
  fit(points,w,h,lockTarget=false) {
    this.focus=lockTarget ? [0,0,0] : [0,1,2].map(i=>points.reduce((s,p)=>s+vector(p)[i],0)/points.length);
    this.span=0.00001;
    this.keepVisible(points,w,h);
    this.initialized=true;
  }
  keepVisible(points,w,h) {
    const b=this.basis(),half=[Math.max(1,w*0.4-12),Math.max(1,h*0.4-12)];
    for(const p of points) {
      const d=vector(p).map((v,i)=>v-this.focus[i]);
      this.span=Math.max(this.span,...b.map((axis,i)=>Math.abs(dot(d,axis))*Math.min(w,h)/half[i]),0.01);
    }
  }
  pan(dx,dy,w,h) {
    const b=this.basis(),scale=this.span/Math.min(w,h);
    this.focus=this.focus.map((v,i)=>v+(-dx*b[0][i]+dy*b[1][i])*scale);
  }
}
// Track only touches that begin on the camera canvas. Rebase naturally when
// fingers enter or leave so pinch-to-orbit transitions never jump.
export class CameraTouches {
  constructor(camera) { this.camera = camera; this.points = new Map(); }
  clear() { this.points.clear(); }
  start(id, x, y) { this.points.set(id, {x, y}); }
  end(id) { this.points.delete(id); }
  move(id, x, y) {
    if (!this.points.has(id)) return;
    const previous = this.points.get(id);
    const distance = () => {
      const [a, b] = [...this.points.values()];
      return Math.hypot(a.x - b.x, a.y - b.y);
    };
    const before = this.points.size === 2 ? distance() : 0;
    this.points.set(id, {x, y});
    if (this.points.size === 1) {
      this.camera.yaw += (x - previous.x) * 0.008;
      this.camera.elevation = Math.max(-Math.PI/2, Math.min(Math.PI/2,
        this.camera.elevation + (y - previous.y) * 0.008));
    } else if (this.points.size === 2) {
      const after = distance();
      if (before > 1 && after > 1) this.camera.span = Math.max(0.01,
        Math.min(1e8, this.camera.span * before / after));
    }
    this.camera.focus = [0,0,0];
  }
}
export function createSandbox3D({
  getState,fitCanvas,drawMarker,drawPath,onChange
}) {
  const camera=new RicCamera3D();
  const touches=new CameraTouches(camera);
  const mobile=()=>getState().activeView === 'mobile';
  const landscapeQuery=matchMedia('(orientation: landscape)');
  const panel=document.createElement('section');
  panel.className='plot-panel sandbox-3d-panel';
  panel.hidden=true;
  panel.innerHTML='<div class="plot-title"><span>RIC · 3D</span><div class="camera-3d-toolbar"><button type="button" data-camera="toggle">2D</button><button type="button" data-camera="center">Recenter</button><button type="button" data-camera="fit">Fit both</button><button type="button" data-camera="RI">RI</button><button type="button" data-camera="RC">RC</button><button type="button" data-camera="IC">IC</button></div></div><canvas aria-label="3D RIC view. Drag to orbit, Shift-drag or middle-drag to pan, scroll to zoom."></canvas><div class="camera-3d-help">Drag: orbit · Shift / middle-drag: pan · Scroll: zoom · Equal scale</div>';
  document.querySelector('.plot-stack').append(panel);
  const canvas=panel.querySelector('canvas'), panels=[document.querySelector('#riPanel'),document.querySelector('#rcPanel')];
  const toggle=document.createElement('button');
  toggle.type='button';
  toggle.textContent='3D';
  toggle.hidden=true;
  toggle.className='camera-3d-toggle';
  toggle.setAttribute('aria-label','Switch Sandbox to 3D');
  panels[0].querySelector('.plot-title').append(toggle);
  const pointerQuery=matchMedia('(hover: hover) and (pointer: fine)');
  let enabled=false,drag=null,zoomUntil=0,eligible=false;
  const available=()=>{
    const s=getState();
    return supportsSandbox3D(s.mode,s.activeView,pointerQuery.matches,landscapeQuery.matches);
  };
  const active=()=>enabled&&available();
  const cancel=()=>{
    const ids = [...touches.points.keys(), ...(drag ? [drag.id] : [])];
    touches.clear();
    drag=null;
    for (const id of ids) if(canvas.hasPointerCapture(id))canvas.releasePointerCapture(id);
    zoomUntil=0;
  };
  const points=()=>[{
    r:0,i:0,c:0
  },getState().sim];
  function sync(){
    const sign=getState().frameConvention==='space_force'?-1:1;
    if(camera.inTrackSign!==sign){camera.inTrackSign=sign;camera.initialized=false;}
    eligible=available();
    if(!eligible){
      cancel();
      if (!['sandbox','operatorSandbox'].includes(getState().mode)) {
        enabled=false;
        camera.initialized=false;
      }
    }
    if (mobile()) camera.focus=[0,0,0];
    canvas.setAttribute('aria-label', mobile()
      ? '3D RIC view. Drag to rotate around the target; pinch to zoom.'
      : '3D RIC view. Drag to orbit, Shift-drag or middle-drag to pan, scroll to zoom.');
    panel.querySelector('[data-camera="center"]').hidden=mobile();
    panel.querySelector('.camera-3d-help').textContent=mobile()
      ? 'Drag: rotate · Pinch: zoom · Target locked'
      : 'Drag: orbit · Shift / middle-drag: pan · Scroll: zoom · Equal scale';
    const toggleParent=document.querySelector('.top-bar');
    if(toggle.parentElement!==toggleParent)toggleParent.append(toggle);
    toggle.textContent=active()?'2D':'3D';
    toggle.setAttribute('aria-label',active()?'Switch Sandbox to 2D':'Switch Sandbox to 3D');
    toggle.hidden=!eligible;
    panel.querySelector('[data-camera="toggle"]').hidden=true;
    panel.hidden=!active();
    panels.forEach(p=>p.style.display=active()?'none':'');
    return active();
  }
  function flip(){
    if(!available())return;
    enabled=!enabled;
    cancel();
    sync();
    onChange();
  }
  for (const name of ['pointerdown','mousedown','click']) toggle.addEventListener(name,event=>event.stopPropagation());
  toggle.addEventListener('click',flip);
  panel.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{
    const command=button.dataset.camera;
    if(command==='toggle')return flip();
    const r=canvas.getBoundingClientRect();
    if(command==='fit')camera.fit(points(),r.width,r.height,mobile());
    else if(command==='center')camera.focus=[0,0,0];
    else [camera.yaw,camera.elevation]=({
      RI:[0,0],RC:[Math.PI/2,0],IC:[0,-Math.PI/2]
    })[command];
    onChange();
  }));
  canvas.addEventListener('pointerdown',event=>{
    if(!active())return;
    if (mobile() && event.pointerType === 'touch') {
      event.preventDefault();
      touches.start(event.pointerId,event.clientX,event.clientY);
      canvas.setPointerCapture(event.pointerId);
      onChange();
      return;
    }
    if(event.pointerType!=='mouse'||![0,1].includes(event.button))return;
    event.preventDefault();
    drag={
      id:event.pointerId,x:event.clientX,y:event.clientY,pan:!mobile()&&(event.shiftKey||event.button===1)
    };
    canvas.setPointerCapture(event.pointerId);
    onChange();
  });
  canvas.addEventListener('pointermove',event=>{
    if (touches.points.has(event.pointerId)) {
      touches.move(event.pointerId,event.clientX,event.clientY);
      onChange();
      return;
    }
    if(!drag||event.pointerId!==drag.id)return;
    const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
    drag.x=event.clientX;
    drag.y=event.clientY;
    const r=canvas.getBoundingClientRect();
    if(drag.pan)camera.pan(dx,dy,r.width,r.height);
    else{
      camera.yaw+=dx*0.008;
      camera.elevation=Math.max(-Math.PI/2,Math.min(Math.PI/2,camera.elevation+dy*0.008));
    }
    onChange();
  });
  for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,event=>{
    touches.end(event.pointerId);
    if (drag?.id === event.pointerId) drag=null;
    onChange();
  });
  canvas.addEventListener('wheel',event=>{
    if(!active())return;
    event.preventDefault();
    const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?300:1);
    camera.span=Math.max(0.01,Math.min(1e8,camera.span*Math.exp(Math.max(-2,Math.min(2,delta*0.0015)))));
    zoomUntil=performance.now()+250;
    onChange();
  },{
    passive:false
  });
  window.addEventListener('blur',()=>{
    cancel();
    onChange();
  });
  document.addEventListener('visibilitychange',()=>{ if(document.hidden) { cancel(); onChange(); } });
  landscapeQuery.addEventListener('change',()=>{ cancel(); sync(); onChange(); });
  pointerQuery.addEventListener('change',()=>{
    sync();
    onChange();
  });
  function render(){
    const sign=getState().frameConvention==='space_force'?-1:1;
    if(camera.inTrackSign!==sign){camera.inTrackSign=sign;camera.initialized=false;}
    const {
      width:w,height:h
    }
    =fitCanvas(canvas),ctx=canvas.getContext('2d');
    if(w<1||h<1)return;
    const top=0,bottom=0,H=Math.max(1,h-top-bottom);
    if(!camera.initialized)camera.fit(points(),w,H,mobile());
    camera.keepVisible(points(),w,H);
    ctx.clearRect(0,0,w,h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0,top,w,H);
    ctx.clip();
    ctx.translate(0,top);
    const project=p=>camera.project(p,w,H),scale=Math.min(w,H)/camera.span;
    const line=(a,b,color)=>{
      ctx.strokeStyle=color;
      ctx.lineWidth=1;
      ctx.beginPath();
      ctx.moveTo(a.x,a.y);
      ctx.lineTo(b.x,b.y);
      ctx.stroke();
    };
    const extent=Math.max(w,H)/scale*4+Math.hypot(...camera.focus);
    const basis=camera.basis();
    const determinant=Math.abs(basis[0][0]*basis[1][1]-basis[0][1]*basis[1][0]);
    const step=Math.max(camera.span/8,extent/45,36/(scale*Math.max(determinant,0.025)));
    if (determinant > 0.025) {
      for(let v=Math.floor(-extent/step)*step;
      v<=extent;
      v+=step){
        line(project({
          r:v,i:-extent,c:0
        }),project({
          r:v,i:extent,c:0
        }),'#263440');
        line(project({
          r:-extent,i:v,c:0
        }),project({
          r:extent,i:v,c:0
        }),'#263440');
      }
    }
    const origin=project({
      r:0,i:0,c:0
    });
    ctx.font='12px monospace';
    ['r','i','c'].forEach((axis,index)=>{
      const color=['#f57d73','#78dc96','#78afff'][index],end=project({
        r:0,i:0,c:0,[axis]:extent
      }),start=project({
        r:0,i:0,c:0,[axis]:-extent
      });
      line(start,end,color);
      const dx=end.x-origin.x,dy=end.y-origin.y;
      let t=1;
      for(const [d,o,max] of [[dx,origin.x,w],[dy,origin.y,H]])if(Math.abs(d)>1e-8)t=Math.min(t,((d>0?max-25:20)-o)/d);
      if(t>=0){
        ctx.fillStyle=color;
        ctx.fillText('+'+axis.toUpperCase(),origin.x+dx*t,origin.y+dy*t);
      }
    });
    const s=getState();
    drawPath(ctx,s.mode==='operatorSandbox'?s.operatorPlanPath:s.ghost,project,'rgba(135,150,172,.95)',true,2);
    drawPath(ctx,s.trail,project,'rgba(245,205,92,.95)',false,2);
    for(const [p,role,color] of [[points()[0],'target','#f55c5c'],[s.sim,'chaser','#f5cd5c']]){
      const point=project(p);
      drawMarker(ctx,point,role,{
        scale,fallbackRadius:role==='target'?6:7
      });
      ctx.fillStyle=color;
      ctx.fillText(role==='target'?'Target':'Chaser',point.x+10,point.y-12);
    }
    line({
      x:20,y:H-15
    },{
      x:100,y:H-15
    },'#bbcbdc');
    ctx.fillStyle='#bbcbdc';
    ctx.fillText((80/scale).toPrecision(3)+' km',20,H-23);
    ctx.restore();
  }
  return {
    sync,render,active,interacting:()=>active()&&(!!drag||touches.points.size>0||performance.now()<zoomUntil),debug:()=>({
      enabled:active(),yaw:camera.yaw,elevation:camera.elevation,span:camera.span,focus:[...camera.focus]
    })
  };
}
