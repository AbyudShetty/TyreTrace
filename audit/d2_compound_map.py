"""Coverage audit; unresolved mappings are incomplete, never inferred as C3."""
from pathlib import Path
import sys
import hashlib
import polars as pl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tyretrace import artifacts,results
from tyretrace.data import sources
from tyretrace.data.compounds import nominations,resolve,REGISTRY


def main():
    d=sources.load('lake');reg=nominations();rows=[]
    for _,g in d['race'].group_by('session_uid'):
        year=int(g['year'][0]);circuit=str(g['circuit'][0])
        row={'weekend':str(g['session_uid'][0])[:7],'year':year,'circuit':circuit,'laps':g.height}
        row.update({c:resolve(year,circuit,c,reg) for c in ('HARD','MEDIUM','SOFT')})
        rows.append(row)
    t=pl.DataFrame(rows).sort('weekend');mapped=t.filter(pl.all_horizontal(pl.col('HARD','MEDIUM','SOFT').is_not_null()))
    artifacts.save('compound_map','coverage',t)
    missing=t.join(mapped.select('weekend'),on='weekend',how='anti').to_dicts()
    results.record('compound_map',{'weekends':t.height,'mapped':mapped.height,'missing':missing,
        'registry_sha256':hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        'scope':'nomination lookup; prototype sets and within-season construction changes are not identified per lap'},
        gate='PASS' if not missing else 'HOLD')
    print(f'COMPOUND MAP {mapped.height}/{t.height}',missing,flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
