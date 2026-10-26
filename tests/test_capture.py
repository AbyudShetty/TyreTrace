"""Independent spec fixtures, raw evidence integrity and blind recovery boundaries."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tyretrace.rig import packets as P
from tyretrace.rig import capture as C
from tyretrace.rig.record import LapAccumulator
from tyretrace.rig.validation import blind_estimate,recovery_report


@pytest.mark.parametrize('year',[2023,2024,2025])
def test_lap_data_matches_independently_packed_ea_field_order(year):
    # Literal full C-struct layout. Do not use decoder offset constants here.
    leading='IIHBHBHH' if year==2023 else 'IIHBHBHBHB'
    values=[91234,12345,30123,0,30111,0,500,900] if year==2023 else [91234,12345,30123,0,30111,0,500,0,900,0]
    # 15 uint8: position,current lap,pit,stops,sector,invalid,penalties,
    # warnings,corner warnings,drive-through,stop-go,grid,driver,result,timer.
    values += [1234.5,98765.,0.0, 4,17,2,1,0,1,0,0,0,0,0,8,4,2,0, 0,0,0]
    fmt='<'+leading+'fff15BHHB'+('' if year==2023 else 'fB')
    if year!=2023:values += [320.,15]
    block=struct.pack(fmt,*values)
    assert len(block)==(50 if year==2023 else 57)
    header=struct.pack('<HBBBBBQfIIBB',year,year%100,1,0,1,2,123,12.,1,1,0,255)
    payload=header+block*22+b'\xff\xff'
    assert len(payload)==(1131 if year==2023 else 1285)
    d=P.decode_lap_data(payload)[3]
    assert d['current_lap_num']==17 and d['car_position']==4
    assert d['pit_status']==2 and d['num_pit_stops']==1
    assert d['current_lap_invalid']==1 and d['driver_status']==4
    assert d['lap_distance_m']==pytest.approx(1234.5)


@pytest.mark.parametrize('year',[2023,2024,2025])
def test_capture_replay_keeps_provenance_and_correct_completed_lap(year,tmp_path):
    p=tmp_path/f'fake{year}.udp'
    C.write_archive(p,C.fake_records(laps=40,cars=4,fmt=year),origin='synthetic')
    report=C.validate_archive(p)
    assert report['layout_gate']=='PASS'
    assert report['origin']=='synthetic' and not report['tier2_established']
    assert report['multi_car_truth'] and report['cars_with_fuel_and_wear']==[0,1,2,3]
    assert report['laps']==160
    assert report['recovery']['gate']=='SYNTHETIC_ONLY'
    assert report['recovery']['deg_difference_mae']<1e-5
    import polars as pl
    laps=pl.read_parquet(p.with_suffix('.parquet'))
    assert laps['lap_in_session'].min()==1 and laps['lap_in_session'].max()==40
    assert laps['compound_rel'].cast(pl.Utf8).unique().to_list()==['MEDIUM']
    assert recovery_report(laps,origin='live_udp')['gate']=='REFERENCE_COMPARISON'


def test_archive_detects_truncation_and_checksum_mismatch(tmp_path):
    p=tmp_path/'raw.udp'
    C.write_archive(p,C.fake_records(laps=4,cars=2),origin='synthetic')
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(C.CaptureError,match='checksum'):
        C.validate_archive(p)
    with pytest.raises(C.CaptureError,match='truncated'):
        list(C.read_archive(p))


def test_existing_evidence_is_never_overwritten(tmp_path):
    p=tmp_path/'raw.udp'
    C.write_archive(p,C.fake_records(laps=1,cars=1),origin='synthetic')
    original=p.read_bytes()
    with pytest.raises(FileExistsError):
        C.write_archive(p,[],origin='synthetic')
    assert p.read_bytes()==original


def test_missing_packet_kind_and_no_stream_do_not_pass():
    from tyretrace.rig.fake import FakeSession
    records=[(float(i),lap) for i,(lap,_,_) in enumerate(FakeSession(n_laps=4).stream())]
    report,_=C.analyse(records,origin='synthetic')
    assert report['layout_gate']=='HOLD' and report['missing_or_sparse_packet_ids']
    empty,_=C.analyse([],origin='live_udp')
    assert empty['layout_gate']=='HOLD' and not empty['tier2_established']


def test_mixed_sessions_and_bad_bytes_are_rejected():
    a=list(C.fake_records(laps=3,cars=2))
    bad=bytearray(a[-1][1]);struct.pack_into('<Q',bad,7,999)
    report,_=C.analyse(a+[(100.,bytes(bad))],origin='synthetic')
    assert report['layout_gate']=='FAIL' and any('multiple' in p for p in report['problems'])
    report,_=C.analyse(a+[(100.,b'bad')],origin='synthetic')
    assert report['layout_gate']=='FAIL'


def test_unsupported_or_truncated_real_packet_cannot_look_plausible():
    payload=list(C.fake_records(laps=1,cars=2))[-1][1]
    with pytest.raises(P.PacketError,match='partial'):
        P.decode_lap_data(payload[:-1])


def test_blind_estimator_refuses_truth_columns():
    _,laps=C.analyse(C.fake_records(laps=40,cars=2),origin='synthetic')
    with pytest.raises(ValueError,match='stripped'):
        blind_estimate(laps)


def test_sparse_capture_does_not_claim_recovery():
    _,laps=C.analyse(C.fake_records(laps=10,cars=2),origin='live_udp')
    r=recovery_report(laps,origin='live_udp')
    assert r['gate']=='HOLD' and not r['tier2_established']


def test_actual_udp_loopback_receives_original_datagrams():
    import socket
    import threading
    import time
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    packets=[p for _,p in list(C.fake_records(laps=1,cars=1))[:3]]
    def send():
        time.sleep(.1)
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
            for p in packets:sender.sendto(p,('127.0.0.1',port))
    thread=threading.Thread(target=send,daemon=True);thread.start()
    received=[p for _,p in C.live_records(port,.4)]
    thread.join(timeout=1.)
    assert received==packets


def test_stale_truth_and_unobserved_lap_gaps_are_not_fabricated():
    from tyretrace.rig.fake import build_lap_data,build_car_status,build_car_damage
    acc=LapAccumulator('rig_test')
    base=dict(session_uid=1,session_time=0.,frame=0)
    status=P.decode_car_status(build_car_status([{'fuel':100.,'age':1}],**base))
    damage=P.decode_car_damage(build_car_damage([{'wear':3.}],**base))
    acc.update(7,status,0.);acc.update(10,damage,0.)
    first=P.decode_lap_data(build_lap_data([{'lap_num':1}],**base))
    acc.update(2,first,0.)
    next_lap=P.decode_lap_data(build_lap_data([{'lap_num':2,'last_lap_ms':90000}],**base))
    acc.update(2,next_lap,90.)
    assert not acc.rows and acc.skipped_stale_truth==1
    gap=P.decode_lap_data(build_lap_data([{'lap_num':5,'last_lap_ms':90000}],**base))
    acc.update(2,gap,91.)
    assert not acc.rows and acc.skipped_gaps>=1
