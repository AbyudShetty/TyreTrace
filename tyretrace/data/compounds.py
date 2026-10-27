"""Dated primary-source nomination lookup, with explicit unresolved results.

Never reinterpret the old canonical C3 placeholder as observed construction.
Lookup keys are season and circuit, not round numbering (cancelled races move
rounds). Pooling keys include season because same-named rubber changes by era.
"""
from __future__ import annotations
import json
from pathlib import Path
import unicodedata
import polars as pl

REGISTRY=Path(__file__).with_name('nominations.json')
ALIASES={'sakhir':'bahrain','jeddah':'saudiarabia','melbourne':'australia','baku':'azerbaijan',
 'miami':'miami','miamigardens':'miami','monaco':'monaco','montecarlo':'monaco',
 'barcelona':'spainbarcelona','montreal':'canada','spielberg':'austria','silverstone':'britain',
 'budapest':'hungary','spafrancorchamps':'belgium','zandvoort':'netherlands','monza':'italy',
 'marinabay':'singapore','suzuka':'japan','lusail':'qatar','austin':'usa','mexicocity':'mexico',
 'saopaulo':'brazil','yasisland':'abudhabi','yasmarina':'abudhabi','shanghai':'china',
 'imola':'imola','madrid':'spainmadrid','kualalumpur':'sepang','sepang':'sepang','lasvegas':'lasvegas'}


def event_key(circuit: str) -> str:
    normalized=''.join(c for c in unicodedata.normalize('NFKD',str(circuit)).lower() if c.isascii() and c.isalpha())
    return ALIASES.get(normalized,normalized)


def nominations(path: Path=REGISTRY):
    data=json.loads(path.read_text(encoding='utf-8'))
    out={}
    for r in data['nominations']:
        key=(int(r['year']),event_key(r['event']))
        if key in out:raise ValueError(f'duplicate nomination: {key}')
        values=[r[k] for k in ('hard','medium','soft')]
        if (len(set(values))!=3 or any(v not in [f'C{i}' for i in range(7)] for v in values)
            or sorted(values,key=lambda x:int(x[1:]))!=values or not r.get('source') or not r.get('source_date')):
            raise ValueError(f'invalid sourced nomination: {key}')
        out[key]=r
    return out


def resolve(year: int,circuit: str,relative: str,registry=None):
    if relative in ('WET','INTERMEDIATE'):return 'WET' if relative=='WET' else 'INT'
    row=(registry if registry is not None else nominations()).get((int(year),event_key(circuit)))
    return row.get(relative.lower()) if row is not None and relative in ('SOFT','MEDIUM','HARD') else None


def construction_key(year: int,circuit: str,relative: str,registry=None):
    value=resolve(year,circuit,relative,registry)
    return f'{year}:{value}' if value else None


def annotate(laps: pl.DataFrame,registry=None):
    """An analysis view: actual construction is nullable until a source exists.

    The canonical source files are not rewritten and keep their schema. Extra
    analysis columns explicitly distinguish a verified mapping from a legacy
    placeholder, so unresolved tyres cannot accidentally enter physical pooling.
    """
    reg=nominations() if registry is None else registry
    keys=laps.select('year','circuit','compound_rel').unique()
    mapped=[resolve(y,c,str(cmp),reg) for y,c,cmp in keys.iter_rows()]
    keys=keys.with_columns(pl.Series('verified_compound',mapped,dtype=pl.Utf8))
    return laps.join(keys,on=['year','circuit','compound_rel'],how='left')
