"""Sourced nomination lookup must never trust the historic C3 placeholder."""
import json
import sys
from pathlib import Path
import polars as pl
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tyretrace.data.compounds import resolve,nominations,annotate,construction_key,event_key


def test_skipped_compounds_and_season_changes_are_preserved():
    assert [resolve(2025,'Austin',c) for c in ('HARD','MEDIUM','SOFT')]==['C1','C3','C4']
    assert [resolve(2025,'Mexico City',c) for c in ('HARD','MEDIUM','SOFT')]==['C2','C4','C5']
    assert resolve(2024,'Miami Gardens','HARD')=='C2'
    assert resolve(2025,'Miami Gardens','HARD')=='C3'
    assert resolve(2025,'Imola','SOFT')=='C6'


def test_unresolved_is_nullable_and_legacy_placeholder_cannot_enter_pooling():
    t=pl.DataFrame({'year':[2022,2025],'circuit':['unknown','Imola'],
        'compound_rel':['SOFT','SOFT'],'compound_abs':['C3','C3']})
    out=annotate(t)
    assert out['verified_compound'].to_list()==[None,'C6']
    assert out['compound_abs'].to_list()==['C3','C3']
    assert construction_key(2022,'Imola','SOFT') is None
    assert construction_key(2024,'Monza','SOFT')!=construction_key(2025,'Monza','SOFT')
    assert resolve(2025,'unknown','INTERMEDIATE')=='INT'


def test_aliases_preserve_circuit_identity():
    assert event_key('São Paulo')=='brazil'
    assert event_key('Spa-Francorchamps')=='belgium'
    assert event_key('Madrid')!=event_key('Barcelona')
    assert event_key('Kuala Lumpur')=='sepang'


def test_registry_requires_unique_dated_sources(tmp_path):
    row={'year':2025,'event':'miami','hard':'C1','medium':'C3','soft':'C4',
         'source':'https://press.pirelli.com/test','source_date':'2025-01-01'}
    p=tmp_path/'nominations.json'
    p.write_text(json.dumps({'nominations':[row,row]}))
    with pytest.raises(ValueError,match='duplicate'):nominations(p)
    row['soft']='C1';p.write_text(json.dumps({'nominations':[row]}))
    with pytest.raises(ValueError,match='invalid'):nominations(p)


@pytest.mark.parametrize('year,compound,expected',[(2024,'SOFT','C4'),(2025,'SOFT','C5'),(2022,'SOFT',None),(2025,'INTERMEDIATE','INT')])
def test_new_openf1_ingest_has_sourced_construction_and_exact_schema(monkeypatch,year,compound,expected):
    from tyretrace.data import openf1
    from tyretrace.schema import LAPS_SCHEMA
    payloads={
        'laps':[{'driver_number':1,'lap_number':1,'lap_duration':90.,'date_start':'2025-05-04T15:00:00+00:00'}],
        'stints':[{'driver_number':1,'stint_number':1,'lap_start':1,'lap_end':1,'compound':compound}],
        'drivers':[{'driver_number':1,'name_acronym':'VER','team_name':'Red Bull'}],
        'race_control':[],
    }
    monkeypatch.setattr(openf1,'get',lambda endpoint,**kwargs:payloads[endpoint])
    t=openf1.load_session({'session_key':99,'year':year,'round':6,'uid':f'{year}_06_R','code':'R','location':'Miami Gardens'})
    assert dict(t.schema)==LAPS_SCHEMA
    assert t['compound_abs'].cast(pl.Utf8).to_list()==[expected]
