from eval.stream_diar_metrics import evaluate_events


def test_stream_metrics_count_revisions_tts_and_order():
    events = [
        {"event": "si_speaker_update", "data": {"revision": 1, "segments": [{"segment_id": "s1", "start_time": 0, "end_time": 1000}], "segment_assignments": [{"segment_key": "k1", "speaker_id": 1}]}},
        {"event": "si_text_done", "data": {"segment_id": 1, "segment_key": "k1", "received_ms": 120}},
        {"event": "si_tts_start", "data": {"segment_id": 1, "segment_key": "k1", "speaker_id": 1, "voice": "v1"}},
        {"event": "si_tts_end", "data": {"segment_id": 1, "segment_key": "k1"}},
        {"event": "si_segment_done", "data": {"segment_id": 1, "segment_key": "k1"}},
        {"event": "si_speaker_update", "data": {"revision": 2, "segments": [{"segment_id": "s1", "start_time": 0, "end_time": 1000}], "segment_assignments": [{"segment_key": "k1", "speaker_id": 2}]}},
    ]
    result = evaluate_events(events)
    assert result["published_label_revision_count"] == 1
    assert result["published_label_revision_rate"] == 1.0
    assert result["tts_attribution_consistency"] == 1.0
    assert result["event_order_correct_rate"] == 1.0
    assert result["speaker_result_latency_ms"]["p95"] == 120.0
