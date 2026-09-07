import numpy as np
import pytest

from eot.audio import PauseSpan, Span
from eot.labeling import apptek
from eot.labeling.apptek import Decision, conservative_decision


def decision(reason="hold_backchannel", label=0):
    return Decision(5.0, 9.0, 9.0, 5.3, 0.3, 1, label, reason, "high", 1.0)


def test_short_reply_is_not_proven_backchannel():
    d = conservative_decision(decision(), [Span(5.3, 5.6)], [])
    assert d.label is None and d.reason == "ambiguous_short_listener_event"
    assert d.confidence == "high"  # acoustic agreement does not resolve label ambiguity


def test_brief_ack_then_agent_turn_is_not_hold():
    d = conservative_decision(decision(), [Span(5.3, 5.6), Span(6.0, 8.0)], [])
    assert d.label is None and d.reason == "ambiguous_listener_event_chain"


def test_listener_turn_after_customer_resume_is_irrelevant():
    assert conservative_decision(decision(), [Span(5.3, 5.6), Span(10.0, 12.0)], []).reason == "ambiguous_short_listener_event"


def test_preserves_unaffected_decisions():
    for reason, label in [("hold", 0), ("eot", 1), ("collision", None)]:
        d = decision(reason, label)
        assert conservative_decision(d, [], []) is d


def test_even_eot_crops_stop_before_speaker_resumes(monkeypatch, tmp_path):
    pause = PauseSpan(1.0, 4.0, 4.0, 1.0, "high", Span(1.0, 4.0), Span(1.0, 4.0), 1.0)

    class Detector:
        detector = "fixture"

        def __call__(self, *args):
            return [pause]

    monkeypatch.setattr(apptek, "load_wav", lambda path: np.zeros(5 * 16000, dtype=np.float32))
    monkeypatch.setattr(apptek, "decide", lambda *args: Decision(1.0, 4.0, 4.0, 1.4, 0.7, 3, 1, "eot", "high", 1.0))
    conv = apptek.Conversation("fixture", "en-AU", "unit", 5.0, [], tmp_path / "a.wav", tmp_path / "c.wav")
    samples, _, _ = apptek.label_conversation(conv, Detector(), grid=[0.2, 3.2], rule_version="v2")
    assert len(samples) == 1 and samples[0].cut_time == 1.2
    assert samples[0].time_to_onset > 0
    assert samples[0].meta()["label_confidence"] == "heuristic_unreviewed"


def test_existing_label_run_cannot_be_overwritten(tmp_path):
    (tmp_path / "conversations.jsonl").write_text("preserve me\n")
    with pytest.raises(FileExistsError, match="already exists"):
        apptek.main(["--root", str(tmp_path / "input"), "--out", str(tmp_path)])
    assert (tmp_path / "conversations.jsonl").read_text() == "preserve me\n"
