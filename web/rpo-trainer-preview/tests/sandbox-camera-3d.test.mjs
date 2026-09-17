import test from 'node:test';
import assert from 'node:assert/strict';
import {RicCamera3D,supportsSandbox3D} from '../src/sandbox-camera-3d.js';

test('3D eligibility is limited to desktop mouse Sandbox',()=>{
  for(const mode of ['sandbox','operatorSandbox','tutorial','operatorTutorial','arcade','selector','sandboxSetup','primer']) {
    assert.equal(supportsSandbox3D(mode,'desktop',true),['sandbox','operatorSandbox'].includes(mode));
    assert.equal(supportsSandbox3D(mode,'mobile',true),false);
    assert.equal(supportsSandbox3D(mode,'desktop',false),false);
  }
});
test('orthographic RIC camera uses R up and an orthonormal equal-scale basis',()=>{
 const camera=new RicCamera3D();
 const b=camera.basis();
 assert.equal(b[0][0],0);assert.ok(b[1][0]>0);
 for(let i=0;i<2;i++)for(let j=0;j<2;j++)assert.ok(Math.abs(b[i].reduce((s,v,k)=>s+v*b[j][k],0)-(i===j?1:0))<1e-12);
});
test('camera keeps both satellites within view after zoom, pan and rotation',()=>{
 const points=[{r:0,i:0,c:0},{r:2,i:-3,c:1}];
 for(const [w,h] of [[1200,400],[600,700]]) {
  const camera=new RicCamera3D();camera.fit(points,w,h);
  for(const zoom of [0.001,0.5,2]) {
   camera.span*=zoom;camera.pan(400,-200,w,h);camera.yaw+=0.4;camera.elevation-=0.3;
   camera.keepVisible(points,w,h);
   for(const p of points){const q=camera.project(p,w,h);assert.ok(q.x>0&&q.x<w&&q.y>0&&q.y<h);}
   const span=camera.span;camera.keepVisible(points,w,h);assert.equal(camera.span,span);
  }
 }
});

test('mobile 3D requires landscape and remains Sandbox-only',()=>{
  for (const mode of ['sandbox','operatorSandbox','tutorial','operatorTutorial','arcade','selector']) {
    assert.equal(supportsSandbox3D(mode,'mobile',false,true),['sandbox','operatorSandbox'].includes(mode));
    assert.equal(supportsSandbox3D(mode,'mobile',false,false),false);
  }
});

test('touch orbit and pinch keep target locked, with no translation or transition jump',async()=>{
  const {CameraTouches}=await import('../src/sandbox-camera-3d.js');
  const camera=new RicCamera3D();
  camera.fit([{r:0,i:0,c:0},{r:2,i:-3,c:1}],600,250,true);
  const touch=new CameraTouches(camera);
  touch.start(1,100,100);
  const yaw=camera.yaw;
  touch.move(1,120,100);
  assert.ok(camera.yaw>yaw);
  touch.start(2,220,100);
  const span=camera.span,angle=camera.yaw;
  touch.move(2,320,100);
  assert.equal(camera.span,span/2);
  assert.equal(camera.yaw,angle);
  assert.deepEqual(camera.focus,[0,0,0]);
  touch.end(2);
  touch.move(1,120,100);
  assert.equal(camera.yaw,angle);
  camera.keepVisible([{r:0,i:0,c:0},{r:2,i:-3,c:1}],600,250);
  assert.deepEqual(camera.project({r:0,i:0,c:0},600,250),{x:300,y:125});
  touch.clear();
  touch.move(1,500,500);
  assert.equal(camera.yaw,angle);
  assert.equal(touch.points.size,0);
});
