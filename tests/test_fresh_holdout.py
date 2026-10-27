"""An accessed holdout cannot silently validate a changed algorithm."""
import importlib.util
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('fresh_holdout_audit',ROOT/'audit/m3_fresh_holdout.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_frozen_sources_are_checked_before_scoring(tmp_path,monkeypatch):
    p=tmp_path/'algorithm.py';p.write_text('frozen predictor')
    frozen={'sha256':{'algorithm.py':audit.digest(p)}}
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    audit.verify(frozen)
    p.write_text('retuned after seeing the holdout')
    with pytest.raises(ValueError,match='changed after holdout access'):audit.verify(frozen)


def test_current_isolated_holdout_still_matches_its_frozen_sources():
    # Optional local evidence check; a clean clone need not contain ignored data.
    import json
    p=ROOT/'out/fresh_holdout/freeze.json'
    if not p.exists():pytest.skip('local holdout has not been explicitly fetched')
    audit.verify(json.loads(p.read_text()))
