"""The research controller must reject partial/stale promotions without touching releases."""
import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_promote import check

def test_incomplete_trial_cannot_be_promoted():
 with pytest.raises(ValueError,match='qualification'):
  check({'status':'training'},{'id':'baseline'})

def test_stale_parent_cannot_be_promoted():
 with pytest.raises(ValueError,match='stale incumbent'):
  check({'status':'quality-qualified','quality_parent':'old'},{'id':'current'})

def test_required_replication_cannot_be_skipped():
 with pytest.raises(ValueError,match='replication'):
  check({'status':'quality-qualified','quality_parent':'base','quality_gates':{'primary':True},
   'replication_requirement':'predeclared second seed','replication_passed':False},{'id':'base'})

def test_failed_or_missing_evidence_cannot_be_promoted():
 with pytest.raises(ValueError,match='quality gates failed'):
  check({'status':'quality-qualified','quality_parent':'base','quality_gates':{'primary':False}},{'id':'base'})
 with pytest.raises(ValueError,match='missing evidence'):
  check({'status':'quality-qualified','quality_parent':'base','quality_gates':{'primary':True}},{'id':'base'})
