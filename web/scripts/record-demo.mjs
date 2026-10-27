// Record the actual local dashboard tour. No generated or simulated UI frames.
import {mkdir,writeFile,readFile} from 'node:fs/promises';
const base=process.env.TYRETRACE_URL??'http://127.0.0.1:5179';
const root=new URL('../../',import.meta.url),out=new URL('out/demo/',root);
await mkdir(new URL('frames/',out),{recursive:true});
const pages=await(await fetch('http://127.0.0.1:9223/json/list')).json();
const page=pages.find(p=>p.url.startsWith(base))??pages.find(p=>p.url==='about:blank');
if(!page)throw Error('Open a local browser tab with debugging port 9223');
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0,started=0;const pending=new Map(),frames=[],writes=[],checks=[],errors=[];
function call(method,params={}){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});}
ws.addEventListener('message',event=>{const m=JSON.parse(event.data);
 if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.exception?.description??m.params.exceptionDetails.text);
 if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}
 if(m.method==='Page.screencastFrame'){
  const filename=`frames/frame-${String(frames.length).padStart(6,'0')}.jpg`;
  const time=Math.max(0,(performance.now()-started)/1000);frames.push({filename,time});
  const write=writeFile(new URL(filename,out),Buffer.from(m.params.data,'base64')).then(()=>call('Page.screencastFrameAck',{sessionId:m.params.sessionId}));
  writes.push(write);
 }
});
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text);return r.result.value;}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
await call('Page.enable');await call('Runtime.enable');
await call('Page.bringToFront');
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:`${base}/demo.html`});
for(let i=0;i<100;i++){await sleep(100);if(await evaluate(`typeof toggle==='function'`))break;}
await evaluate('if(playing)toggle();position=0;render();');
for(let i=0;i<100;i++){await sleep(150);if(await evaluate(`Boolean(frame.contentDocument?.querySelector('h1'))&&!frame.contentDocument.body.innerText.includes('Loading')`))break;}
const scenes=await evaluate('scenes');const narration=JSON.parse(await readFile(new URL('docs/demo-narration.json',root),'utf8'));
if(narration.length!==scenes.length)throw Error('Narration does not match tour scenes');
await writeFile(new URL('narration.json',out),JSON.stringify(scenes.map(([route,seconds,caption],i)=>({route,seconds,caption,text:narration[i]})),null,2));
// Screencast events arrive only when pixels change. Seed the static opening
// explicitly, otherwise the first event can be eight seconds into the tour.
const initial=await call('Page.captureScreenshot',{format:'jpeg',quality:82});
await writeFile(new URL('frames/frame-000000.jpg',out),Buffer.from(initial.data,'base64'));
frames.push({filename:'frames/frame-000000.jpg',time:0});
const total=scenes.reduce((n,s)=>n+s[1],0);started=performance.now();
await call('Page.startScreencast',{format:'jpeg',quality:82,maxWidth:1440,maxHeight:1000,everyNthFrame:2});
await evaluate('toggle()');
let position=-1,sceneStarted=0,phase=0,verified=false;
const targets={overview:['Predictive intervals','The identification chain'],races:['One race','Source'],track:['Qualifying track','Physical tyre'],ladder:['Rolling practice models','Rolling predictive calibration'],predictions:['forecasts and race','Predictive intervals'],backtest:['All baseline families','Unseen Austin'],rig:['hidden fuel mass','Live']};
while((performance.now()-started)/1000<total+.4){
 const current=await evaluate('position');const elapsed=(performance.now()-started)/1000;
 if(current!==position){position=current;sceneStarted=elapsed;phase=0;verified=false;console.log(`Recording ${position+1}/${scenes.length}: ${scenes[position][0]}`);}
 const inScene=elapsed-sceneStarted;const next=inScene>=16?2:inScene>=8?1:0;
 if(inScene>=3&&!verified){
  const state=await evaluate(`({heading:frame.contentDocument?.querySelector('h1')?.textContent,text:frame.contentDocument?.body.innerText??'',route:frame.contentWindow?.location.hash})`);
  if(!state.heading||/Loading|Could not load|not available yet/i.test(state.text)||state.route!==`#/${scenes[position][0]}`)throw Error(`Scene ${position+1} did not render: ${JSON.stringify(state)}`);
  checks.push({scene:position+1,route:scenes[position][0],heading:state.heading,checked_at_seconds:elapsed});verified=true;
 }
 if(errors.length)throw Error(`Dashboard exception while recording: ${errors.join('\n')}`);
 if(next>phase){phase=next;const route=scenes[position][0],needle=targets[route]?.[phase-1];
  if(position!==scenes.length-1)await evaluate(`(()=>{const d=frame.contentDocument,w=frame.contentWindow;const title=${JSON.stringify(needle??'')};const card=title&&Array.from(d.querySelectorAll('.card-title')).find(h=>h.textContent.toLowerCase().includes(title.toLowerCase()))?.closest('.card');if(card)card.scrollIntoView({behavior:'smooth',block:'start'});else w.scrollTo({top:${phase===1?400:800},behavior:'smooth'});})()`);
 }
 await sleep(200);
}
await call('Page.stopScreencast');await Promise.all(writes);
if(frames.length<20||checks.length!==scenes.length)throw Error('Browser supplied an incomplete tour');
await writeFile(new URL('capture.json',out),JSON.stringify({recorded_at:new Date().toISOString(),duration_seconds:(performance.now()-started)/1000,base,scenes,checks,errors,frames},null,2));
ws.close();console.log(`Recorded ${frames.length} actual browser frames`);
