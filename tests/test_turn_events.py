import pytest

from eot.labeling.turn_events import Event, label_pause, parse_srt


def label(after=(), prior="Normal Turn", onset=9.0, observed=12.0, cut=5.2):
    return label_pause(pause_start=5.0, cut_time=cut, speaker="s1", events=[Event(3.0, 5.0, "s1", prior)] + list(after),
                       observed_until=observed, next_own_onset=onset)


def test_short_bounded_response_can_take_floor():
    assert label([Event(5.4, 5.6, "s2", "Bounded Response", "Yes")]).label == 1


def test_explicit_backchannel_does_not_take_floor():
    r = label([Event(5.4, 5.6, "s2", "Continuer Backchannel", "Mhm")])
    assert r.label == 0 and r.reason == "own_continuation_after_backchannel"


def test_ack_then_real_turn_is_transfer():
    assert label([Event(5.3, 5.5, "s2", "Acknowledgement Backchannel"), Event(6.0, 8.0, "s2", "Normal Turn")]).label == 1


@pytest.mark.parametrize("kind", ["Floor-taking Competitive Interruption", "Overlap", "Channel Bleed"])
def test_ambiguous_events_abstain(kind):
    assert label([Event(5.4, 6.0, "s2", kind)]).label is None


def test_annotation_end_alone_does_not_prove_eot():
    assert label(onset=None).label is None


def test_insufficient_observation_does_not_confirm_transfer():
    assert label([Event(5.4, 5.6, "s2", "Bounded Response")], onset=None, observed=6.0).reason == "censored_transfer_confirmation"


def test_hold_and_floor_transfer_conflict_abstains():
    assert label([Event(5.4, 7.0, "s2", "Normal Turn")], prior="Strong Floor Hold").label is None


def test_backchanneler_is_not_assumed_floor_holder():
    assert label(prior="Continuer Backchannel").label is None


def test_cut_cannot_use_audio_past_own_onset():
    with pytest.raises(ValueError, match="onset"):
        label(onset=5.1)


def test_parser_multiline_and_bom():
    e = parse_srt("﻿1\r\n00:00:12,340 --> 00:00:13,020\r\n[Normal Turn] Hello\r\nthere.\r\n", "s1")[0]
    assert e.start == 12.34 and e.end == 13.02 and e.text == "Hello there."


@pytest.mark.parametrize("text", ["[Unknown] hi", "hi"])
def test_parser_rejects_unrecognized_labels(text):
    with pytest.raises(ValueError, match="event"):
        parse_srt("1\n00:00:01,000 --> 00:00:02,000\n" + text, "s1")


def test_parser_rejects_backwards_timing():
    with pytest.raises(ValueError, match="interval"):
        parse_srt("1\n00:00:02,000 --> 00:00:01,000\n[Normal Turn] hi", "s1")
