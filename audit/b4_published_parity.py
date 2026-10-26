"""Source-verified published curve, keeping the old adaptation's scores intact."""
from pathlib import Path
import sys
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit.l2_tau_scale import build
from audit.l3_dirty_air import split,subset
from audit.l4_calibration import lap_labels
from tyretrace import artifacts,results
from tyretrace.eval import metrics
from tyretrace.models.baselines import published_verified


def main():
    d=build(pl.read_parquet(ROOT/'out/audit_race.parquet'),pl.read_parquet(ROOT/'out/audit_quali.parquet'))
    tr,te=split(d);train,test=subset(d,tr),subset(d,te)
    draws=published_verified(train,test)
    score=metrics.score(draws,test.y)
    original=results.get('baselines').get('metrics',{}).get('crps_b4')
    l3=results.get('l3').get('metrics',{}).get('crps_l3')
    artifacts.save('published_verified','predictive',artifacts.predictive(draws,test.y,**lap_labels(test)))
    artifacts.save('published_verified','calibration',artifacts.calibration(draws,test.y))
    results.record('published_verified',{'crps':score['crps'],'mae':score['mae'],
        'cov90':score['coverage_90'],'calibration':score['calibration'],
        'legacy_crps':original,'l3_crps':l3,'l3_beats_verified':l3<score['crps'] if l3 is not None else None,
        'scope':'primary-code logarithmic tyre curve with k0/start-age; fitted baseline adaptation, not the full race simulator',
        'source':'https://github.com/TUMFTM/race-simulation/blob/master/helper_funcs/src/calc_tire_degradation.py'},gate='PASS')
    print(score)
    print(f'Legacy B4 CRPS {original}; source-verified curve {score["crps"]:.6f}; L3 {l3}')
    return 0


if __name__=='__main__':raise SystemExit(main())
