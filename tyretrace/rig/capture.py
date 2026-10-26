"""Record raw UDP evidence, replay it and emit a machine-readable rig report.

    python -m tyretrace.rig.capture --seconds 900 --out out/rig/live.udp
    python -m tyretrace.rig.capture --replay out/rig/live.udp
    python -m tyretrace.rig.capture --fake --out out/rig/synthetic.udp

Archives are never overwritten. Synthetic provenance survives replay and never
turns into live game proof. No firewall/game settings are changed by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import struct
import time

from tyretrace.rig import packets as P
from tyretrace.rig.record import LapAccumulator

MAGIC = b'TTUDP001'
RECORD = struct.Struct('<dI')
MAX_DATAGRAM = 65507


class CaptureError(ValueError):
    pass


def write_archive(path: Path, records, *, origin: str) -> dict:
    if origin not in ('live_udp', 'synthetic'):
        raise ValueError('origin must be live_udp or synthetic')
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_suffix(path.suffix+'.json')
    if any(p.exists() for p in (path,manifest,path.with_suffix('.parquet'),path.with_suffix('.report.json'))):
        raise FileExistsError(f'capture already exists: {path}; choose a new name')
    meta = {'format':'TTUDP001','origin':origin,'complete':False,'packets':0}
    manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    digest = hashlib.sha256()
    n=0
    previous=-1.0
    with path.open('xb') as f:
        f.write(MAGIC);digest.update(MAGIC)
        for elapsed, payload in records:
            if not math.isfinite(elapsed) or elapsed<previous or not 0<len(payload)<=MAX_DATAGRAM:
                raise CaptureError('invalid capture record')
            previous=elapsed
            raw=RECORD.pack(elapsed,len(payload))+payload
            f.write(raw);digest.update(raw);n+=1
    meta.update(complete=True,packets=n,sha256=digest.hexdigest())
    manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


def read_archive(path: Path):
    with path.open('rb') as f:
        if f.read(len(MAGIC))!=MAGIC:
            raise CaptureError('not a Tyre Trace UDP archive')
        previous=-1.0
        while raw:=f.read(RECORD.size):
            if len(raw)!=RECORD.size:
                raise CaptureError('truncated archive record header')
            elapsed,n=RECORD.unpack(raw)
            if not math.isfinite(elapsed) or elapsed<previous or not 0<n<=MAX_DATAGRAM:
                raise CaptureError('invalid archive timestamp or datagram length')
            payload=f.read(n)
            if len(payload)!=n:
                raise CaptureError('truncated archive datagram')
            previous=elapsed
            yield elapsed,payload


def fake_records(laps: int=40,cars: int=20,fmt: int=2024):
    from tyretrace.rig.fake import FakeSession
    for lap,status,damage in FakeSession(n_laps=laps,n_cars=cars,fmt=fmt,seed=11).stream():
        # Status/damage precede the crossing packet, as in the recorder tests.
        for payload in (status,damage,lap):
            yield float(P.parse_header(payload).session_time),payload


def live_records(port: int,seconds: float):
    if not 1<=port<=65535 or seconds<=0:
        raise ValueError('positive duration and a valid UDP port are required')
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.bind(('',port));sock.settimeout(min(seconds,1.0))
        start=time.monotonic()
        try:
            while time.monotonic()-start<seconds:
                try:
                    payload,_=sock.recvfrom(MAX_DATAGRAM)
                except socket.timeout:
                    continue
                yield time.monotonic()-start,payload
        except KeyboardInterrupt:
            return


def analyse(records,*,origin: str) -> tuple[dict,object]:
    counts={pid:0 for pid in P.DECODERS}
    problems=set();uids=set();formats=set();fuel=set();wear=set()
    seen_times={};duplicates=0;acc=None;total=0
    for _,payload in records:
        total+=1
        try:
            hdr=P.parse_header(payload)
            uids.add(hdr.session_uid);formats.add(hdr.packet_format)
            if hdr.packet_id not in P.DECODERS:
                continue
            rows=P.DECODERS[hdr.packet_id](payload)
            issues=P.validate(hdr.packet_id,rows)
            if issues:
                problems.update(issues);continue
            stamp=(hdr.session_uid,hdr.packet_id)
            old=seen_times.get(stamp)
            if old is not None and hdr.session_time<old:
                problems.add('session clock moved backwards: flashback or stale datagram')
                continue
            if old==hdr.session_time:
                duplicates+=1;continue
            seen_times[stamp]=hdr.session_time
            counts[hdr.packet_id]+=1
            if hdr.packet_id==P.PACKET_CAR_STATUS:
                fuel.update(i for i,r in enumerate(rows) if r['fuel_in_tank_kg']>.01)
            if hdr.packet_id==P.PACKET_CAR_DAMAGE:
                wear.update(i for i,r in enumerate(rows) if max(r['wear_'+w] for w in ('fl','fr','rl','rr'))>0)
            if acc is None:
                acc=LapAccumulator(f'rig_{hdr.session_uid:016x}',year=hdr.packet_format)
            if len(uids)==1:
                acc.update(hdr.packet_id,rows,hdr.session_time)
        except P.PacketError as e:
            problems.add(str(e))
    if len(uids)>1:
        problems.add('capture mixes multiple game sessions; record one session per archive')
    if len(formats)>1:
        problems.add('capture mixes packet formats')
    missing=[str(pid) for pid,n in counts.items() if n<3]
    truth_cars=sorted(fuel & wear)
    from tyretrace import schema
    laps=acc.frame() if acc is not None else schema.empty()
    gate='FAIL' if problems else ('HOLD' if missing or not truth_cars or laps.height<10 else 'PASS')
    report={'origin':origin,'layout_gate':gate,'packets':total,
        'packet_counts':{str(k):v for k,v in counts.items()},'formats':sorted(formats),
        'sessions':[f'{u:016x}' for u in sorted(uids)],'problems':sorted(problems),
        'missing_or_sparse_packet_ids':missing,'fuel_cars':sorted(fuel),'wear_cars':sorted(wear),
        'cars_with_fuel_and_wear':truth_cars,'multi_car_truth':len(truth_cars)>1,
        'laps':laps.height,'skipped_lap_gaps':acc.skipped_gaps if acc else 0,
        'skipped_stale_truth':acc.skipped_stale_truth if acc else 0,
        'duplicate_packets':duplicates,'tier2_established':False,
        'scope':'packet/data readiness only; a replay or fake stream is not live hidden-state proof'}
    return report,laps


def validate_archive(path: Path) -> dict:
    manifest=path.with_suffix(path.suffix+'.json')
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not meta.get('complete') or meta.get('sha256')!=digest:
        raise CaptureError('capture is incomplete or its checksum does not match')
    if meta.get('origin') not in ('synthetic','live_udp'):
        raise CaptureError('capture has no recognised provenance')
    report,laps=analyse(read_archive(path),origin=meta['origin'])
    report['sha256']=digest
    if report['layout_gate']=='PASS':
        laps.write_parquet(path.with_suffix('.parquet'))
        from tyretrace.rig.validation import recovery_report
        report['recovery']=recovery_report(laps,origin=meta['origin'])
    path.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('out/rig/live.udp'))
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--replay',type=Path)
    mode.add_argument('--fake',action='store_true')
    ap.add_argument('--format',type=int,choices=P.SUPPORTED_FORMATS,default=2024)
    ap.add_argument('--seconds',type=float,default=900.)
    ap.add_argument('--port',type=int,default=20777)
    args=ap.parse_args(argv)
    if args.replay:
        path=args.replay
    else:
        path=args.out
        print(f'Capturing {"synthetic" if args.fake else "live UDP"} evidence to {path}',flush=True)
        records=fake_records(fmt=args.format) if args.fake else live_records(args.port,args.seconds)
        write_archive(path,records,origin='synthetic' if args.fake else 'live_udp')
    report=validate_archive(path)
    print(json.dumps(report,indent=2))
    return 0 if report['layout_gate']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
