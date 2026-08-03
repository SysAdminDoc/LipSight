import json
import sys
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
