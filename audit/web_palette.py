"""Offline contrast and colour-vision screening for the dashboard tokens.

Machado severity-1 matrices are screening approximations, not a claim that all
eight colours are distinguishable for every viewer. Labels and dash patterns
remain required. Adjacent ordered series must keep CIELAB distance >= 8.
"""
import json
import re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MATRICES={
 'normal':np.eye(3),
 'protan':np.array([[.152286,1.052583,-.204868],[.114503,.786281,.099216],[-.003882,-.048116,1.051998]]),
 'deutan':np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]]),
 'tritan':np.array([[1.255528,-.076749,-.178779],[-.078411,.930809,.147602],[.004733,.691367,.303900]])}

def rgb(c):
    a=np.array([int(c[i:i+2],16)/255 for i in (1,3,5)])
    return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)

def lum(c):return float(rgb(c)@np.array([.2126,.7152,.0722]))
def contrast(a,b):
    x,y=sorted((lum(a),lum(b)));return (y+.05)/(x+.05)
def lab(linear):
    xyz=linear@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
    v=xyz/np.array([.95047,1,1.08883]);f=np.where(v>.008856,np.cbrt(v),7.787*v+16/116)
    return np.column_stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])])

def main():
    css=(ROOT/'web/src/styles/tokens.css').read_text();blocks=css.split(':root[data-theme="light"]')
    rows=[];fail=[]
    for theme,block in zip(('dark','light'),blocks):
        values=dict(re.findall(r'--([\w-]+):\s*(#[0-9a-f]{6})',block))
        for role in ('ink','ink-2','ink-3'):
            for surface in ('page','surface','surface-2','surface-3'):
                ratio=contrast(values[role],values[surface]);rows.append({'theme':theme,'kind':'text','role':role,'surface':surface,'ratio':ratio})
                if ratio<4.5:fail.append(rows[-1])
        for role in [f's{i}' for i in range(1,9)]:
            ratio=contrast(values[role],values['surface']);rows.append({'theme':theme,'kind':'series','role':role,'ratio':ratio})
            if ratio<3:fail.append(rows[-1])
        colors=np.array([rgb(values[f's{i}']) for i in range(1,9)])
        for name,m in MATRICES.items():
            labs=lab(np.clip(colors@m.T,0,1));distance=np.linalg.norm(np.diff(labs,axis=0),axis=1)
            rows.append({'theme':theme,'kind':'cvd_adjacent','vision':name,'minimum_delta_e76':float(distance.min())})
            if distance.min()<8:fail.append(rows[-1])
    out=ROOT/'out/browser-check';out.mkdir(parents=True,exist_ok=True)
    (out/'palette.json').write_text(json.dumps({'checks':rows,'failures':fail},indent=2))
    print(f'Palette: {len(rows)} checks, {len(fail)} failures',fail)
    return int(bool(fail))

if __name__=='__main__':raise SystemExit(main())
