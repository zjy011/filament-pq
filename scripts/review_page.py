"""Offline, configuration-blind E5 review page with local-only CSV export."""

import json


def make_review_page(rows, reviewer):
    data = json.dumps([{"sample_id": r[0], "crop": r[1]} for r in rows])
    return """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>暗区样本独立评审</title><style>
body{font:16px system-ui;margin:30px auto;max-width:1050px;padding:0 20px;background:#f6f7f9;color:#14202b}
header{position:sticky;top:0;background:#f6f7f9;padding:12px 0;border-bottom:1px solid #cbd2d8}
article{background:white;padding:20px;margin:20px 0;border:1px solid #d9dfe5;border-radius:10px}
img{display:block;max-width:100%;max-height:550px;margin:12px 0;image-rendering:auto}
select,input,button{font:inherit;padding:10px;margin:5px}button{background:#125a72;color:white;border:0;cursor:pointer}
input{width:45%}small{color:#53616d}h1{font-size:24px}
</style><header><h1>暗区样本独立评审</h1><p>左图为原图，右图黄色轮廓为待判断对象。
请独立完成所有样本；不确定时在备注中说明。完成后导出 CSV。</p>
<button id="export">导出我的 CSV</button><strong id="progress"></strong></header>
<p>可能的暗条：细长的暗色太阳结构。太阳黑子：紧凑的暗斑。日面边缘伪影：边缘或环状伪影。
其余暗区、噪声归入“其他”。这些判断仅用于敏感性分析，不是专家真值。</p><main id="samples"></main>
<script>const rows=DATA; const reviewer=REVIEWER;
const key='filament-blind-review-'+reviewer+'-'+rows.length;
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const choices=[['','请选择'],['possible_filament','可能是漏标暗条'],['sunspot','太阳黑子'],
['limb_artifact','日面边缘伪影'],['other','其他暗区或噪声']];
function update(){document.getElementById('progress').textContent=rows.filter(r=>saved[r.sample_id]?.label).length+' / '+rows.length;
try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){}}
for(const row of rows){const article=document.createElement('article');const h=document.createElement('strong');h.textContent=row.sample_id;
const img=document.createElement('img');img.src=row.crop;img.alt=row.sample_id+'：原图和候选轮廓';img.loading='lazy';
const select=document.createElement('select');select.setAttribute('aria-label',row.sample_id+' 类别');
for(const [v,t] of choices){const o=document.createElement('option');o.value=v;o.textContent=t;select.append(o)}
select.value=saved[row.sample_id]?.label||'';const input=document.createElement('input');input.placeholder='备注（可选）';
input.setAttribute('aria-label',row.sample_id+' 备注');input.value=saved[row.sample_id]?.notes||'';
function change(){saved[row.sample_id]={label:select.value,notes:input.value};update()}
select.onchange=change;input.oninput=change;article.append(h,img,select,input);document.getElementById('samples').append(article)}
update();document.getElementById('export').onclick=()=>{
if(rows.some(r=>!saved[r.sample_id]?.label)){alert('请先完成全部样本。');return}
const quote=s=>'"'+String(s).replaceAll('"','""')+'"';
const csv=[['sample_id','crop','label','notes'],...rows.map(r=>[r.sample_id,r.crop,saved[r.sample_id].label,saved[r.sample_id].notes])]
.map(r=>r.map(quote).join(',')).join('\\r\\n');
const url=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}));
const a=document.createElement('a');a.href=url;a.download='reviewer_'+reviewer+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>""".replace("DATA", data).replace("REVIEWER", json.dumps(reviewer))
