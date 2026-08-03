import json
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import LipSight


def _result(text='hello world'):
    return {
        'start': 1.25,
        'end': 3.5,
        'text': text,
        'confidence': 1.2,
        'words': [
            {'text': 'hello', 'start': 1.25, 'end': 2.0, 'confidence': 0.8},
        ],
    }


def _video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 4.0, (32, 24))
    assert writer.isOpened()
    for value in range(8):
        frame = np.full((24, 32, 3), value * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_normalize_result_produces_stable_schema_and_clamps_confidence():
    normalized = LipSight.normalize_result(_result())

    assert list(normalized) == ['speaker', 'start', 'end', 'text', 'confidence', 'words', 'segment']
    assert normalized['speaker'] == 'A'
    assert normalized['confidence'] == 1.0
    assert normalized['words'][0]['confidence'] == 0.8
    assert normalized['segment'] == 1


def test_normalize_result_rejects_inverted_ranges():
    with pytest.raises(ValueError, match='end'):
        LipSight.normalize_result({'start': 4, 'end': 2, 'text': 'bad'})


def test_srt_vtt_and_json_exports(tmp_path):
    results = [_result()]
    srt = tmp_path / 'captions.srt'
    vtt = tmp_path / 'captions.vtt'
    payload = tmp_path / 'captions.json'

    LipSight.export_srt(results, srt)
    LipSight.export_vtt(results, vtt)
    LipSight.export_json(results, payload, metadata={'backend': 'test'})

    assert '00:00:01,250 --> 00:00:03,500' in srt.read_text(encoding='utf-8')
    assert 'WEBVTT' in vtt.read_text(encoding='utf-8')
    assert '00:00:01.250 --> 00:00:03.500' in vtt.read_text(encoding='utf-8')
    document = json.loads(payload.read_text(encoding='utf-8'))
    assert document['schema_version'] == LipSight.RESULT_SCHEMA_VERSION
    assert document['metadata'] == {'backend': 'test'}
    assert document['results'][0]['speaker'] == 'A'


def test_project_round_trip_extracts_embedded_video(tmp_path):
    video = tmp_path / 'source.mp4'
    _video(video)
    project = tmp_path / 'session.lipsight'
    results = [_result()]

    saved = LipSight.save_project(
        project,
        video_path=video,
        segments=[(1.25, 3.5)],
        results=results,
        edits=[{'word': 'world', 'replacement': 'there'}],
        include_video=True,
    )
    loaded = LipSight.load_project(saved, extract_dir=tmp_path / 'media')

    assert loaded['schema_version'] == LipSight.PROJECT_SCHEMA_VERSION
    assert loaded['segments'] == [{'start': 1.25, 'end': 3.5}]
    assert loaded['edits'][0]['replacement'] == 'there'
    assert loaded['video']['embedded'] is True
    assert (tmp_path / 'media' / 'source.mp4').is_file()


def test_session_archive_searches_transcript(tmp_path):
    archive = LipSight.SessionArchive(tmp_path / 'sessions.jsonl')
    archive.record('clip.mp4', [_result('unique phrase')], backend='test')

    matches = archive.search('UNIQUE PHRASE')

    assert len(matches) == 1
    assert matches[0]['backend'] == 'test'


def test_process_video_uses_headless_backend(tmp_path):
    video = tmp_path / 'source.mp4'
    _video(video)

    class Backend:
        def transcribe(self, path, log_cb=None):
            assert path == str(video)
            return {'text': 'offline result', 'confidence': 0.9}

    results = LipSight.process_video(video, Backend())

    assert results == [{
        'speaker': 'A',
        'start': 0.0,
        'end': 2.0,
        'text': 'offline result',
        'confidence': 0.9,
        'words': [],
        'segment': 1,
    }]


def test_cli_parser_supports_headless_options():
    args = LipSight.build_cli_parser().parse_args([
        '--input', 'clip.mp4', '--backend', 'custom', '--endpoint-url', 'http://localhost:8000',
        '--output', 'captions.vtt', '--no-segmentation',
    ])

    assert args.input == 'clip.mp4'
    assert args.backend == 'custom'
    assert args.no_segmentation is True


def test_mouth_preprocessor_aligns_and_normalizes_roi():
    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    frame[60:105, 70:140] = (40, 80, 140)
    points = np.array([[75, 80], [95, 68], [120, 70], [138, 83], [112, 100], [88, 101]], dtype=np.float32)

    crop = LipSight.MouthPreprocessor(output_size=(64, 64)).process(frame, points=points)

    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8


def test_super_resolution_fallback_upscales_small_crop():
    crop = np.zeros((12, 20, 3), dtype=np.uint8)

    result = LipSight.SuperResolutionProcessor().enhance(crop, minimum_size=48)

    assert min(result.shape[:2]) >= 48


def test_apply_review_text_preserves_timing_and_records_edits():
    results = [
        {'start': 0, 'end': 1, 'text': 'hello'},
        {'start': 1, 'end': 2, 'text': 'world'},
    ]

    updated, edits = LipSight.apply_review_text(results, 'hi\nthere')

    assert [result['text'] for result in updated] == ['hi', 'there']
    assert updated[0]['start'] == 0.0 and updated[1]['end'] == 2.0
    assert edits == [
        {'segment': 1, 'before': 'hello', 'after': 'hi'},
        {'segment': 2, 'before': 'world', 'after': 'there'},
    ]


def test_onnx_backend_adapts_channel_first_and_time_first_shapes():
    frames = np.zeros((5, 96, 96), dtype=np.float32)

    time_first = LipSight.LocalONNXBackend._adapt_input(frames, [1, 'T', 1, 96, 96])
    channel_first = LipSight.LocalONNXBackend._adapt_input(frames, [1, 1, 'T', 96, 96])

    assert time_first.shape == (1, 5, 1, 96, 96)
    assert channel_first.shape == (1, 1, 5, 96, 96)


def test_command_backend_parses_json_runner_output(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({'text': 'runner result'}), stderr='')

    monkeypatch.setattr(LipSight.subprocess, 'run', fake_run)
    backend = LipSight.CommandModelBackend(['runner.exe'], extra_args=['--video', '{video}'], label='test runner')

    result = backend.transcribe(tmp_path / 'clip.mp4')

    assert result['text'] == 'runner result'
    assert calls[0][-1].endswith('clip.mp4')


def test_audio_visual_fusion_prefers_high_confidence_audio_word():
    class Backend:
        def __init__(self, payload):
            self.payload = payload

        def transcribe(self, _path, log_cb=None):
            return self.payload

    visual = {'text': 'hello world', 'confidence': 0.4, 'words': [
        {'text': 'hello', 'start': 0, 'end': 1, 'confidence': 0.4},
        {'text': 'world', 'start': 1, 'end': 2, 'confidence': 0.4},
    ]}
    audio = {'text': 'hello word', 'confidence': 0.9, 'words': [
        {'text': 'hello', 'start': 0, 'end': 1, 'confidence': 0.9},
        {'text': 'word', 'start': 1, 'end': 2, 'confidence': 0.9},
    ]}

    fused = LipSight.AudioVisualFusionBackend(Backend(visual), Backend(audio)).transcribe('clip.mp4')

    assert fused['text'] == 'hello word'
    assert fused['confidence'] == 0.9


def test_confidence_overlay_colors_each_word():
    overlay = LipSight.confidence_overlay_html([{
        'start': 0,
        'end': 1,
        'text': 'sure maybe',
        'words': [
            {'text': 'sure', 'confidence': 0.9},
            {'text': 'maybe', 'confidence': 0.2},
        ],
    }])

    assert 'sure' in overlay and LipSight.C['green'] in overlay
    assert 'maybe' in overlay and LipSight.C['red'] in overlay


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg is not installed')
def test_burn_in_subtitles_writes_video(tmp_path):
    video = tmp_path / 'source.mp4'
    _video(video)

    output = LipSight.burn_in_subtitles(video, [_result('burned')], tmp_path / 'burned.mp4')

    assert output.is_file()
    assert output.stat().st_size > 0
