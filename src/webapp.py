"""
Road Surface Monitor - dashboard.
Run:  python src/webapp.py      Open: http://127.0.0.1:5000
Upload images in capture order (drag & drop). Sliders re-run the finite automaton live.
"""
import io
import numpy as np
import torch
from flask import Flask, request, jsonify, Response
from PIL import Image, UnidentifiedImageError
from config import MODEL_DIR, OUT_DIR, CONFIDENCE_THRESHOLD, CONFIRM_FRAMES
from data import eval_tf
from model import build_model
from explain import gradcam, segment_damage, overlay_cam, overlay_mask, to_data_uri, DAMAGE_CLASS
from fsm import RoadStateFSM, UNKNOWN, INIT, WARNINGS, SPEED_LIMIT

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
MAX_FILES = 300
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(MODEL_DIR / "best.pt", map_location=dev)
CLASSES = ck["classes"]
model = build_model(len(CLASSES), pretrained=False).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()
tf = eval_tf()


def clamp(v, lo, hi, cast):
    try:
        return max(lo, min(hi, cast(v)))
    except (TypeError, ValueError):
        return lo


@app.get("/api/info")
def info():
    rep = OUT_DIR / "classification_report.txt"
    return jsonify(classes=CLASSES, device=str(dev), k=CONFIRM_FRAMES, thr=CONFIDENCE_THRESHOLD,
                   report=rep.read_text() if rep.exists() else "",
                   warnings=WARNINGS, limits=SPEED_LIMIT)


@app.post("/api/analyze")
def analyze():
    files = [f for f in request.files.getlist("images") if f and f.filename][:MAX_FILES]
    if not files:
        return jsonify(error="No images received."), 400
    k = clamp(request.form.get("k"), 1, 30, int)
    thr = clamp(request.form.get("thr"), 0.0, 1.0, float)
    fsm = RoadStateFSM(CLASSES, k)
    rows = []
    for i, f in enumerate(files):
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify(error=f"Could not read '{f.filename}' as an image."), 400
        with torch.no_grad():
            p = torch.softmax(model(tf(img).unsqueeze(0).to(dev)), 1)[0]
        conf, idx = p.max(0)
        raw = CLASSES[idx.item()]
        sym = raw if conf.item() >= thr else UNKNOWN
        rows.append(dict(i=i + 1, name=f.filename, raw=raw, conf=round(conf.item(), 3), sym=sym,
                         state=fsm.step(sym), probs=[round(v, 3) for v in p.tolist()]))
    chg = lambda key: sum(1 for a, b in zip(rows, rows[1:]) if a[key] != b[key] and a[key] != INIT)
    final = fsm.state.current
    return jsonify(rows=rows, final=final, warning=WARNINGS.get(final, ""), limit=SPEED_LIMIT.get(final),
                   raw_changes=chg("raw"), fsm_changes=chg("state"), k=k, thr=thr,
                   unk=sum(r["sym"] == UNKNOWN for r in rows),
                   avg_conf=round(sum(r["conf"] for r in rows) / len(rows), 3))


@app.post("/api/explain")
def explain():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image received."), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="Could not read the image."), 400
    cls = request.form.get("cls")
    cls = clamp(cls, 0, len(CLASSES) - 1, int) if cls not in (None, "") else None
    x = tf(img).unsqueeze(0).to(dev)
    probs, cam, idx = gradcam(model, x, cls)
    rgb = np.array(img.resize((x.shape[-1], x.shape[-2])))
    seg, area = None, 0.0
    if CLASSES[idx] == DAMAGE_CLASS:
        mask, area = segment_damage(rgb, cam)
        seg = to_data_uri(overlay_mask(rgb, mask))
    return jsonify(orig=to_data_uri(rgb), cam=to_data_uri(overlay_cam(rgb, cam)), seg=seg, area=area,
                   probs=probs, cls=idx,
                   note="Grad-CAM shows the regions the CNN used for the selected class. Segmentation is a heuristic "
                        "estimate (dark-region and crack filters inside the Grad-CAM area), not a trained segmentation model.")


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Road Surface Monitor</title>
<style>
:root{--bg:#f4f6fa;--card:#fff;--tx:#1b2330;--mu:#667085;--bd:#e3e8ef;--ac:#2563eb;--wa:#fff7e6}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f141c;--card:#171e29;--tx:#e6ebf2;--mu:#98a2b3;--bd:#2a3342;--ac:#6ea0ff;--wa:#2b2411}}
:root[data-theme=dark]{--bg:#0f141c;--card:#171e29;--tx:#e6ebf2;--mu:#98a2b3;--bd:#2a3342;--ac:#6ea0ff;--wa:#2b2411}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,Segoe UI,Arial;background:var(--bg);color:var(--tx)}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;background:var(--card);border-bottom:1px solid var(--bd)}
h1{font-size:18px;margin:0}h2{font-size:14px;margin:0 0 10px}.mu{color:var(--mu)}main{max-width:1200px;margin:0 auto;padding:20px}
.grid{display:grid;gap:14px}.kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}.two{grid-template-columns:2fr 1fr}
@media(max-width:860px){.two{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
.kpi .v{font-size:24px;font-weight:700;margin-top:4px}.kpi .l{color:var(--mu);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
#drop{border:2px dashed var(--bd);border-radius:12px;padding:22px;text-align:center;cursor:pointer}
#drop.on{border-color:var(--ac);background:rgba(37,99,235,.06)}
button,.btn{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer;font:inherit}
button.g{background:transparent;color:var(--tx);border:1px solid var(--bd)}
.ctl{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;margin:8px 0}
#banner{border-radius:12px;padding:14px 16px;font-weight:600;color:#fff;margin:14px 0}
canvas{width:100%;height:240px;display:block}
table{border-collapse:collapse;width:100%}th,td{padding:6px 8px;border-bottom:1px solid var(--bd);text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);font-size:12px;color:var(--mu)}.scr{max-height:360px;overflow:auto}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;color:#fff;font-size:12px}
img.t{width:38px;height:38px;object-fit:cover;border-radius:6px;vertical-align:middle}
.bar{display:flex;align-items:center;gap:8px;margin:5px 0}.bar i{display:block;height:10px;border-radius:5px}
pre{white-space:pre-wrap;font-size:12px;margin:0}#err{color:#d92d20;margin-top:8px}.hid{display:none}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;z-index:9;padding:16px}#modal.hid{display:none}
#mbox{max-width:780px;width:100%;max-height:90vh;overflow:auto}figure{margin:0}figure img{width:100%;border-radius:8px}figcaption{font-size:12px;color:var(--mu);margin-top:4px}
tr.rw{cursor:pointer}tr.rw:hover{background:rgba(37,99,235,.07)}
@media print{header button,#drop,.ctls{display:none}body{background:#fff}}
</style></head><body>
<header><div><h1>Road Surface Condition Monitor</h1><span class="mu">CNN classifier + finite automaton state tracker</span></div>
<div><button class="g" onclick="window.print()">Print report</button> <button class="g" id="theme">Theme</button></div></header>
<main>
<div class="grid two">
 <div class="card"><div id="drop"><b>Drop road images here</b> or click to choose<div class="mu">Select several in capture order (sorted by file name)</div>
  <input id="file" type="file" accept="image/*" multiple class="hid"></div><div id="err"></div></div>
 <div class="card ctls"><h2>Automaton settings</h2>
  <div class="ctl"><span>K frames</span><input id="k" type="range" min="1" max="15" value="5"><b id="kv">5</b></div>
  <div class="ctl"><span>Confidence</span><input id="thr" type="range" min="0" max="0.95" step="0.05" value="0.6"><b id="tv">0.60</b></div>
  <div class="mu">K = consecutive frames needed to change state. Predictions below the confidence threshold are ignored (unk).</div></div>
</div>
<div id="out" class="hid">
 <div id="banner"></div>
 <div class="grid kpis" id="kpis"></div>
 <div class="grid two" style="margin-top:14px">
  <div class="card"><h2>State timeline: raw CNN vs automaton</h2><canvas id="c1"></canvas></div>
  <div class="card"><h2>Class distribution (raw CNN)</h2><div id="dist"></div></div></div>
 <div class="card" style="margin-top:14px"><h2>Prediction confidence per frame</h2><canvas id="c2"></canvas></div>
 <div class="card" style="margin-top:14px"><div style="display:flex;justify-content:space-between"><h2>Frame results <span class="mu" style="font-weight:400">- click a row for Grad-CAM and damage segmentation</span></h2><button class="g" id="csv">Export CSV</button></div>
  <div class="scr"><table><thead><tr><th>#</th><th></th><th>File</th><th>CNN</th><th>Conf.</th><th>Input</th><th>State</th></tr></thead><tbody id="rows"></tbody></table></div></div>
</div>
<div class="grid two" style="margin-top:14px">
 <div class="card"><h2>Model evaluation (test set)</h2><pre id="rep" class="mu">No report yet. Run python src/evaluate.py</pre></div>
 <div class="card"><h2>System</h2><div id="sys" class="mu"></div></div></div>
</main>
<div id="modal" class="hid"><div class="card" id="mbox"><div style="display:flex;justify-content:space-between"><h2 id="mt">Explainability</h2><button class="g" id="mx">Close</button></div>
<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))"><figure><img id="m1" alt=""><figcaption>Original</figcaption></figure><figure><img id="m2" alt=""><figcaption>Grad-CAM heatmap</figcaption></figure><figure><img id="m3" alt=""><figcaption id="m3c">Damage segmentation</figcaption></figure></div>
<div class="ctl"><span>Explain class</span><select id="mcls"></select><span></span></div><div id="mp"></div><p class="mu" id="mn"></p></div></div>
<script>
const $=s=>document.querySelector(s);let files=[],info={},last=null,urls=[],timer;
const PAL=['#2563eb','#0891b2','#7c3aed','#d97706','#dc2626','#059669','#db2777'];
const col=s=>s==='INIT'||s==='unk'?'#8a94a6':PAL[info.classes.indexOf(s)%PAL.length];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
async function init(){info=await(await fetch('/api/info')).json();
 $('#k').value=info.k;$('#kv').textContent=info.k;$('#thr').value=info.thr;$('#tv').textContent=info.thr.toFixed(2);
 if(info.report)$('#rep').textContent=info.report;
 $('#sys').innerHTML='Device: <b>'+info.device+'</b><br>Classes: '+info.classes.map(c=>'<span class="pill" style="background:'+col(c)+'">'+c+'</span>').join(' ')}
const drop=$('#drop');drop.onclick=()=>$('#file').click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('on')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('on')}));
drop.addEventListener('drop',ev=>add(ev.dataTransfer.files));$('#file').onchange=e=>add(e.target.files);
function add(fl){files=[...fl].filter(f=>f.type.startsWith('image/')).sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));
 urls.forEach(URL.revokeObjectURL);urls=files.map(f=>URL.createObjectURL(f));run()}
['k','thr'].forEach(id=>$('#'+id).oninput=()=>{$('#kv').textContent=$('#k').value;$('#tv').textContent((+$('#thr').value).toFixed(2));clearTimeout(timer);timer=setTimeout(run,350)});
async function run(){if(!files.length)return;$('#err').textContent='Analysing '+files.length+' image(s)...';
 const fd=new FormData();files.forEach(f=>fd.append('images',f));fd.append('k',$('#k').value);fd.append('thr',$('#thr').value);
 try{const r=await fetch('/api/analyze',{method:'POST',body:fd});const d=await r.json();
  if(!r.ok){$('#err').textContent=d.error||'Error';return}$('#err').textContent='';last=d;render(d)}catch(e){$('#err').textContent='Request failed: '+e}}
function kpi(l,v,s){return '<div class="card kpi"><div class="l">'+l+'</div><div class="v">'+v+'</div><div class="mu">'+(s||'')+'</div></div>'}
function render(d){$('#out').classList.remove('hid');const b=$('#banner');b.style.background=col(d.final);
 b.textContent=d.final.toUpperCase()+' - '+d.warning+(d.limit?' Recommended speed: '+d.limit+' km/h.':'');
 const gain=d.raw_changes>0?Math.round((1-d.fsm_changes/d.raw_changes)*100)+'% fewer':'n/a';
 $('#kpis').innerHTML=kpi('Current state',d.final.toUpperCase(),'after '+d.rows.length+' frames')+kpi('Speed limit',d.limit?d.limit+' km/h':'-','recommended')
  +kpi('State changes',d.raw_changes+' to '+d.fsm_changes,'raw CNN to automaton ('+gain+')')+kpi('Avg confidence',(d.avg_conf*100).toFixed(0)+'%','CNN softmax')
  +kpi('Ignored (unk)',d.unk,'below threshold '+d.thr.toFixed(2))+kpi('Confirm window','K = '+d.k,'frames to switch');
 const cnt={};d.rows.forEach(r=>cnt[r.raw]=(cnt[r.raw]||0)+1);
 $('#dist').innerHTML=info.classes.map(c=>{const n=cnt[c]||0,w=Math.round(100*n/d.rows.length);return '<div class="bar"><span style="width:64px">'+c+'</span><i style="width:'+w*1.4+'px;background:'+col(c)+'"></i><span class="mu">'+n+'</span></div>'}).join('');
 $('#rows').innerHTML=d.rows.map((r,i)=>'<tr class="rw" onclick="explain('+i+')"><td>'+r.i+'</td><td><img class="t" src="'+urls[i]+'"></td><td>'+r.name+'</td><td><span class="pill" style="background:'+col(r.raw)+'">'+r.raw+'</span></td><td>'+r.conf.toFixed(2)+'</td><td>'+r.sym+'</td><td><span class="pill" style="background:'+col(r.state)+'">'+r.state+'</span></td></tr>').join('');
 timeline(d);confChart(d)}
function cv(id){const c=$(id),r=c.getBoundingClientRect(),dp=devicePixelRatio||1;c.width=r.width*dp;c.height=r.height*dp;const x=c.getContext('2d');x.scale(dp,dp);return[x,r.width,r.height]}
function timeline(d){const[x,w,h]=cv('#c1'),cats=['INIT',...info.classes],P=70,n=d.rows.length,mu=css('--mu'),bd=css('--bd');
 const X=i=>P+(w-P-12)*(i+.5)/n,Y=c=>14+(h-34)*(cats.indexOf(c)+.5)/cats.length;x.font='12px system-ui';
 cats.forEach(c=>{x.fillStyle=mu;x.fillText(c,6,Y(c)+4);x.strokeStyle=bd;x.beginPath();x.moveTo(P,Y(c));x.lineTo(w-8,Y(c));x.stroke()});
 d.rows.forEach((r,i)=>{x.fillStyle=col(r.raw);x.globalAlpha=.55;x.beginPath();x.arc(X(i),Y(r.raw),3.5,0,7);x.fill()});x.globalAlpha=1;
 x.strokeStyle=css('--ac');x.lineWidth=3;x.beginPath();d.rows.forEach((r,i)=>{i?x.lineTo(X(i),Y(r.state)):x.moveTo(X(i),Y(r.state))});x.stroke();
 x.fillStyle=mu;x.fillText('dots = raw CNN, line = automaton state',P,h-4)}
function confChart(d){const[x,w,h]=cv('#c2'),P=36,n=d.rows.length,mu=css('--mu'),X=i=>P+(w-P-10)*(i+.5)/n,Y=v=>10+(h-30)*(1-v);x.font='12px system-ui';
 [0,.5,1].forEach(v=>{x.fillStyle=mu;x.fillText(v.toFixed(1),4,Y(v)+4);x.strokeStyle=css('--bd');x.beginPath();x.moveTo(P,Y(v));x.lineTo(w-8,Y(v));x.stroke()});
 x.setLineDash([5,4]);x.strokeStyle='#d92d20';x.beginPath();x.moveTo(P,Y(d.thr));x.lineTo(w-8,Y(d.thr));x.stroke();x.setLineDash([]);
 x.strokeStyle=css('--ac');x.lineWidth=2;x.beginPath();d.rows.forEach((r,i)=>{i?x.lineTo(X(i),Y(r.conf)):x.moveTo(X(i),Y(r.conf))});x.stroke();
 x.fillStyle='#d92d20';x.fillText('threshold',w-70,Y(d.thr)-5)}
$('#csv').onclick=()=>{if(!last)return;const q=v=>'"'+String(v).replace(/"/g,'""')+'"';
 const t=['frame,file,cnn_raw,confidence,automaton_input,automaton_state',...last.rows.map(r=>[r.i,q(r.name),r.raw,r.conf,r.sym,r.state].join(','))].join('\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{type:'text/csv'}));a.download='road_surface_results.csv';a.click()};
$('#theme').onclick=()=>{const r=document.documentElement,d=(r.dataset.theme||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'))==='dark';r.dataset.theme=d?'light':'dark';if(last)render(last)};
let cur=-1;
async function explain(i,cls){cur=i;const fd=new FormData();fd.append('image',files[i]);if(cls!==undefined)fd.append('cls',cls);
 $('#modal').classList.remove('hid');$('#mt').textContent='Analysing '+files[i].name+'...';
 try{const r=await fetch('/api/explain',{method:'POST',body:fd});const d=await r.json();
 if(!r.ok){$('#mt').textContent=d.error||'Error';return}
 $('#mt').textContent=files[i].name;$('#m1').src=d.orig;$('#m2').src=d.cam;$('#m3').src=d.seg||d.orig;
 $('#m3c').textContent=d.seg?'Damage segmentation ('+d.area.toFixed(1)+'% of image)':'Segmentation runs only for the damaged class';
 $('#mcls').innerHTML=info.classes.map((c,j)=>'<option value="'+j+'"'+(j===d.cls?' selected':'')+'>'+c+'</option>').join('');
 $('#mcls').onchange=e=>explain(cur,e.target.value);
 $('#mp').innerHTML=info.classes.map((c,j)=>'<div class="bar"><span style="width:64px">'+c+'</span><i style="width:'+Math.round(d.probs[j]*200)+'px;background:'+col(c)+'"></i><span class="mu">'+(d.probs[j]*100).toFixed(1)+'%</span></div>').join('');
 $('#mn').textContent=d.note}catch(e){$('#mt').textContent='Request failed: '+e}}
$('#mx').onclick=()=>$('#modal').classList.add('hid');$('#modal').onclick=e=>{if(e.target.id==='modal')$('#modal').classList.add('hid')};
addEventListener('keydown',e=>{if(e.key==='Escape')$('#modal').classList.add('hid')});
addEventListener('resize',()=>{if(last){timeline(last);confChart(last)}});init();
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
