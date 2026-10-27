// Real-browser checks without a third-party driver. Start a local static server
// and Edge/Chrome with --remote-debugging-port=9223 before running this script.
import {mkdir,writeFile} from 'node:fs/promises';
const base=process.env.TYRETRACE_URL??'http://127.0.0.1:5179';
const output=new URL('../../out/browser-check/',import.meta.url);
await mkdir(output,{recursive:true});
const list=await (await fetch('http://127.0.0.1:9223/json/list')).json();
const page=list.find(p=>p.url==='about:blank')??list.find(p=>p.url.startsWith(base));
if(!page)throw Error('No local browser tab available');
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0;const pending=new Map();const errors=[];
ws.addEventListener('message',event=>{const m=JSON.parse(event.data);
 if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}
 if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text+': '+(m.params.exceptionDetails.exception?.description??''));
 if(m.method==='Log.entryAdded'&&m.params.entry.level==='error')errors.push(m.params.entry.text);
});
function call(method,params={}){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});}
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text);return r.result.value;}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function ready(route){for(let i=0;i<100;i++){await sleep(150);const state=await evaluate(`({heading:document.querySelector('h1')?.textContent,text:document.body.innerText,overflow:document.documentElement.scrollWidth>innerWidth+1,svg:document.querySelectorAll('main svg').length,tables:document.querySelectorAll('main table').length})`);if(state.heading&&!/Loading/.test(state.text)&&state.text.length>600)return state;}throw Error(`${route}: loading did not finish`);}
await call('Page.enable');await call('Runtime.enable');await call('Log.enable');
await call('Page.bringToFront');
await call('Network.enable');await call('Network.setCacheDisabled',{cacheDisabled:true});
const routes=['overview','problem','races','track','ladder','predictions','backtest','planner','rig'];const report=[];
for(const theme of ['dark','light'])for(const mobile of [false,true]){
 await call('Emulation.setDeviceMetricsOverride',{width:mobile?390:1440,height:mobile?844:1100,deviceScaleFactor:1,mobile});
 for(const route of routes){
  const before=errors.length;
  await call('Page.navigate',{url:`${base}/?theme=${theme}#/${route}`});
  let state=await ready(route);
  if(route==='ladder'&&!state.heading.includes('Complexity'))throw Error('Models route did not render');
  if(!(await evaluate(`document.querySelector('.nav-item[aria-current="page"]')?.getAttribute('href')==='#/${route}'`)))throw Error(`${route}: navigation selected the wrong page`);
  if(!state.heading||/Could not load|not available yet|coming soon/i.test(state.text))throw Error(`${route}: missing page content`);
  if(state.overflow)throw Error(`${route}: horizontal page overflow (${mobile?'mobile':'desktop'})`);
  if(await evaluate(`Array.from(document.querySelectorAll('svg path[d]')).some(p=>/NaN|Infinity/.test(p.getAttribute('d')))`))throw Error(`${route}: non-finite chart coordinates`);
  if(await evaluate('document.documentElement.dataset.theme')!==theme)throw Error(`${route}: theme did not apply`);
  if(errors.length>before)throw Error(`${route}: ${errors.slice(before).join('\n')}`);
  // Exercise every dropdown and make sure a selection can render without a
  // runtime error, including switching weekend datasets and candidate methods.
  const count=await evaluate('document.querySelectorAll("main select").length');
  for(let i=0;i<count;i++){
   await evaluate(`(()=>{const s=document.querySelectorAll('main select')[${i}];if(s&&s.options.length>1){s.selectedIndex=s.options.length-1;s.dispatchEvent(new Event('change',{bubbles:true}));}})()`);
   await ready(route);
  }
  if(errors.length>before)throw Error(`${route} filters: ${errors.slice(before).join('\n')}`);
  if(route==='ladder'&&theme==='dark'&&!mobile){
   const csv=await evaluate(`(async()=>{const w=Array.from(document.querySelectorAll('.data-table-wrap')).at(-1);const d=w.closest('details');if(d)d.open=true;const create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;let blob;URL.createObjectURL=b=>{blob=b;return create.call(URL,b);};HTMLAnchorElement.prototype.click=function(){};try{w.querySelector('button').click();await new Promise(r=>setTimeout(r,20));if(!blob)throw Error('CSV button created no data');const text=await blob.text();return {rows:text.split('\\r\\n').length-1,expected:Number(w.querySelector('.table-toolbar span').textContent.split('rows')[0].replace(/[^0-9]/g,'')),bytes:text.length};}finally{URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;}})()`);
   if(csv.rows!==csv.expected||csv.rows<1)throw Error('CSV omitted source rows');
   console.log(`CSV: ${csv.rows} complete source rows`);
  }
  // Reset for a representative screenshot with the original selection.
  await call('Page.reload');await ready(route);
  const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await writeFile(new URL(`${route}-${mobile?'mobile':'desktop'}-${theme}.png`,output),Buffer.from(shot.data,'base64'));
  report.push({route,mobile,theme,heading:state.heading,charts:state.svg,tables:state.tables,errors:errors.slice(before)});
  console.log(`${theme} ${mobile?'mobile':'desktop'} ${route}: OK`);
 }
}
await call('Page.navigate',{url:`${base}/demo.html`});await sleep(600);
const tour=await evaluate(`(()=>{document.getElementById('play').click();document.getElementById('next').click();return {caption:document.getElementById('caption').textContent,route:document.getElementById('dashboard').getAttribute('src')};})()`);
if(!tour.route.endsWith('#/problem'))throw Error('Demo tour navigation failed');
await writeFile(new URL('report.json',output),JSON.stringify({checked_at:new Date().toISOString(),pages:report,tour},null,2));
ws.close();console.log(`${report.length} browser views passed`);
