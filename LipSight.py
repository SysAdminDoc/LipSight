#!/usr/bin/env python3
"""
LipSight v1.1 — AI Lip Reading Tool
Powered by Auto-AVSR (state-of-the-art visual speech recognition)
Supports: Local PyTorch inference (FREE), HuggingFace Spaces (FREE), Replicate API, Custom Endpoints
"""

import sys, os, subprocess, json, time, tempfile, threading, math, hashlib, random, shutil
import argparse
import uuid
import zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta


# codex-branding:start
def _branding_icon_path() -> Path:
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "icon.png")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "icon.png")
    current = Path(__file__).resolve()
    candidates.extend([current.parent / "icon.png", current.parent.parent / "icon.png", current.parent.parent.parent / "icon.png"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("icon.png")
# codex-branding:end


# ── Auto-Bootstrap ──────────────────────────────────────────────────────────
def _bootstrap():
    """Auto-install dependencies and configure prerequisites."""
    if sys.version_info < (3, 8):
        print("Python 3.8+ required"); sys.exit(1)

    required = ['PyQt6', 'opencv-python', 'requests', 'numpy']
    for pkg in required:
        mod = pkg.split('[')[0].replace('-', '_').lower()
        if mod == 'opencv_python': mod = 'cv2'
        try:
            __import__(mod)
        except ImportError:
            for flags in [[], ['--user'], ['--break-system-packages']]:
                try:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install', pkg, '-q'] + flags,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except subprocess.CalledProcessError:
                    continue

    for pkg in ['mediapipe']:
        try:
            __import__(pkg)
        except ImportError:
            for flags in [[], ['--user'], ['--break-system-packages']]:
                try:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install', pkg, '-q'] + flags,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except subprocess.CalledProcessError:
                    continue

_bootstrap()

import cv2
import numpy as np
try:
    import mediapipe as _mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False
    _mp = None
import requests
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import QBrush, QColor, QIcon, QImage, QPainter, QPen, QPixmap

APP_NAME = "LipSight"
APP_VERSION = "1.1.0"
RESULT_SCHEMA_VERSION = "1.0"
PROJECT_SCHEMA_VERSION = "1.0"
PROJECT_MANIFEST = "project.json"
VIDEO_EXTENSIONS = {'.avi', '.mkv', '.mov', '.mp4', '.webm'}

# ── Catppuccin Mocha ────────────────────────────────────────────────────────
C = {
    'base': '#1e1e2e', 'mantle': '#181825', 'crust': '#11111b',
    'surface0': '#313244', 'surface1': '#45475a', 'surface2': '#585b70',
    'overlay0': '#6c7086', 'overlay1': '#7f849c', 'text': '#cdd6f4',
    'subtext0': '#a6adc8', 'subtext1': '#bac2de',
    'blue': '#89b4fa', 'green': '#a6e3a1', 'red': '#f38ba8',
    'peach': '#fab387', 'yellow': '#f9e2af', 'mauve': '#cba6f7',
    'teal': '#94e2d5', 'sky': '#89dceb', 'lavender': '#b4befe',
    'flamingo': '#f2cdcd', 'rosewater': '#f5e0dc',
}

DARK_STYLE = f"""
QMainWindow, QWidget {{ background-color: {C['base']}; color: {C['text']}; }}
QMenuBar {{ background-color: {C['mantle']}; color: {C['text']}; }}
QPushButton {{
    background-color: {C['blue']}; color: {C['base']}; border: none;
    padding: 8px 18px; border-radius: 6px; font-weight: bold; font-size: 13px;
}}
QPushButton:hover {{ background-color: {C['sky']}; }}
QPushButton:pressed {{ background-color: {C['lavender']}; }}
QPushButton:disabled {{ background-color: {C['surface1']}; color: {C['overlay0']}; }}
QPushButton#dangerBtn {{ background-color: {C['red']}; }}
QPushButton#dangerBtn:hover {{ background-color: {C['flamingo']}; }}
QPushButton#secondaryBtn {{ background-color: {C['surface0']}; color: {C['text']}; }}
QPushButton#secondaryBtn:hover {{ background-color: {C['surface1']}; }}
QPushButton#accentBtn {{ background-color: {C['mauve']}; }}
QPushButton#accentBtn:hover {{ background-color: {C['lavender']}; }}
QPushButton#greenBtn {{ background-color: {C['green']}; color: {C['base']}; }}
QPushButton#greenBtn:hover {{ background-color: {C['teal']}; }}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {C['surface0']}; color: {C['text']};
    border: 1px solid {C['surface1']}; border-radius: 6px; padding: 8px;
    selection-background-color: {C['blue']}; selection-color: {C['base']}; font-size: 13px;
}}
QLineEdit:focus, QTextEdit:focus {{ border: 1px solid {C['blue']}; }}
QComboBox {{
    background-color: {C['surface0']}; color: {C['text']};
    border: 1px solid {C['surface1']}; border-radius: 6px; padding: 8px; font-size: 13px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background-color: {C['base']}; color: {C['text']};
    border: 1px solid {C['surface1']}; selection-background-color: {C['blue']};
}}
QLabel {{ color: {C['text']}; font-size: 13px; }}
QLabel#dimLabel {{ color: {C['overlay0']}; font-size: 12px; }}
QGroupBox {{
    border: 1px solid {C['surface1']}; border-radius: 10px;
    margin-top: 1.2em; padding: 16px 12px 12px 12px; color: {C['text']};
    font-weight: bold; font-size: 13px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 8px; color: {C['lavender']}; }}
QProgressBar {{
    background-color: {C['surface0']}; border: none; border-radius: 5px;
    text-align: center; color: {C['text']}; font-size: 12px; min-height: 10px;
}}
QProgressBar::chunk {{ background-color: {C['blue']}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 6px; background: {C['surface0']}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {C['blue']}; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px;
}}
QSlider::sub-page:horizontal {{ background: {C['blue']}; border-radius: 3px; }}
QScrollBar:vertical {{
    background: {C['mantle']}; width: 8px; border: none; border-radius: 4px;
}}
QScrollBar::handle:vertical {{ background: {C['surface1']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {C['surface2']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTabWidget::pane {{ border: 1px solid {C['surface1']}; background: {C['base']}; border-radius: 8px; }}
QTabBar::tab {{
    background: {C['mantle']}; color: {C['overlay0']}; padding: 10px 20px;
    border: none; font-size: 13px; font-weight: bold;
}}
QTabBar::tab:selected {{ color: {C['text']}; border-bottom: 2px solid {C['blue']}; }}
QTabBar::tab:hover {{ color: {C['subtext1']}; }}
QTableWidget {{
    background-color: {C['base']}; alternate-background-color: {C['mantle']};
    color: {C['text']}; border: 1px solid {C['surface1']};
    gridline-color: {C['surface0']}; font-size: 13px; border-radius: 6px;
}}
QTableWidget::item:selected {{ background-color: {C['blue']}; color: {C['base']}; }}
QHeaderView::section {{
    background-color: {C['mantle']}; color: {C['subtext0']};
    border: none; border-bottom: 1px solid {C['surface1']}; padding: 8px;
    font-weight: bold; font-size: 12px;
}}
QStatusBar {{ background-color: {C['mantle']}; color: {C['overlay0']}; font-size: 12px; }}
QToolTip {{
    background-color: {C['surface0']}; color: {C['text']};
    border: 1px solid {C['surface1']}; padding: 6px; border-radius: 6px;
}}
QCheckBox {{ color: {C['text']}; spacing: 8px; }}
"""

# ── Config ──────────────────────────────────────────────────────────────────
def get_config_dir():
    base = os.environ.get('APPDATA', os.path.expanduser('~'))
    path = os.path.join(base, '.lipsight')
    os.makedirs(path, exist_ok=True)
    return path

def load_config():
    try:
        with open(os.path.join(get_config_dir(), 'config.json')) as f: return json.load(f)
    except: return {}

def save_config(cfg):
    with open(os.path.join(get_config_dir(), 'config.json'), 'w') as f: json.dump(cfg, f, indent=2)


# ── Stable data contracts ──────────────────────────────────────────────────
def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _timestamp_value(value, field, default=None):
    if value is None and default is not None:
        value = default
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be a number') from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{field} must be a finite, non-negative number')
    return value


def _confidence_value(value):
    if value is None or value == '':
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('confidence must be a number or null') from exc
    if not math.isfinite(value):
        raise ValueError('confidence must be finite')
    return max(0.0, min(1.0, value))


def normalize_result(result, segment_number=None):
    """Return one transcription result in the versioned export shape."""
    if not isinstance(result, dict):
        raise TypeError('result must be a dictionary')

    start = _timestamp_value(result.get('start', 0.0), 'start')
    end = _timestamp_value(result.get('end', start), 'end')
    if end < start:
        raise ValueError('end must be greater than or equal to start')

    words = []
    for raw_word in result.get('words') or []:
        word = {'text': str(raw_word)} if isinstance(raw_word, str) else dict(raw_word)
        word_start = _timestamp_value(word.get('start', start), 'word.start')
        word_end = _timestamp_value(word.get('end', word_start), 'word.end')
        if word_end < word_start:
            raise ValueError('word.end must be greater than or equal to word.start')
        words.append({
            'text': str(word.get('text', '')).strip(),
            'start': word_start,
            'end': word_end,
            'confidence': _confidence_value(word.get('confidence')),
        })

    raw_segment = result.get('segment', segment_number or 1)
    try:
        raw_segment = int(raw_segment)
    except (TypeError, ValueError) as exc:
        raise ValueError('segment must be an integer') from exc

    return {
        'speaker': str(result.get('speaker') or 'A'),
        'start': start,
        'end': end,
        'text': str(result.get('text', '')).strip(),
        'confidence': _confidence_value(result.get('confidence')),
        'words': words,
        'segment': max(1, raw_segment),
    }


def normalize_results(results):
    return [normalize_result(result, i) for i, result in enumerate(results or [], 1)]


def apply_review_text(results, edited_text):
    """Apply reviewed transcript text while retaining segment timing metadata."""
    normalized = normalize_results(results)
    if not normalized:
        return [], []
    edited_text = str(edited_text or '').strip()
    lines = [line.strip() for line in edited_text.splitlines() if line.strip()]
    if len(lines) == len(normalized):
        replacements = lines
    elif len(normalized) == 1:
        replacements = [edited_text]
    else:
        words = edited_text.split()
        weights = [max(1, len(result['text'].split())) for result in normalized]
        total_weight = sum(weights)
        replacements = []
        cursor = 0
        for index, weight in enumerate(weights):
            if index == len(weights) - 1:
                stop = len(words)
            else:
                stop = cursor + max(0, round(len(words) * weight / total_weight))
            replacements.append(' '.join(words[cursor:stop]))
            cursor = stop

    edits = []
    updated = []
    for result, replacement in zip(normalized, replacements):
        replacement = replacement.strip()
        if replacement != result['text']:
            edits.append({'segment': result['segment'], 'before': result['text'], 'after': replacement})
        updated.append({**result, 'text': replacement})
    return updated, edits


def normalize_segments(segments):
    normalized = []
    for raw_segment in segments or []:
        if isinstance(raw_segment, dict):
            start = raw_segment.get('start', 0.0)
            end = raw_segment.get('end', start)
        else:
            try:
                start, end = raw_segment
            except (TypeError, ValueError) as exc:
                raise ValueError('segments must contain start/end pairs') from exc
        start = _timestamp_value(start, 'segment.start')
        end = _timestamp_value(end, 'segment.end')
        if end < start:
            raise ValueError('segment.end must be greater than or equal to segment.start')
        item = {'start': start, 'end': end}
        if isinstance(raw_segment, dict) and raw_segment.get('speaker'):
            item['speaker'] = str(raw_segment['speaker'])
        normalized.append(item)
    return normalized


def build_result_document(results, metadata=None):
    """Build the stable JSON document shared by GUI, CLI, and project exports."""
    return {
        'schema_version': RESULT_SCHEMA_VERSION,
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'generated_at': _utc_now(),
        'metadata': dict(metadata or {}),
        'results': normalize_results(results),
    }


def _safe_output_path(path):
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _project_member_path(root, member):
    root = Path(root).resolve()
    target = (root / Path(member).name).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise ValueError('project member escapes extraction directory')
    return target


def save_project(project_path, video_path=None, segments=None, results=None, edits=None,
                 include_video=False, metadata=None):
    """Save segmentation, transcription, edits, and video reference in a .lipsight zip."""
    project_path = _safe_output_path(project_path)
    if project_path.suffix.lower() != '.lipsight':
        project_path = project_path.with_suffix('.lipsight')

    video = None
    if video_path:
        source = Path(video_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'video not found: {source}')
        member = f'media/{source.name}' if include_video else None
        video = {'path': str(source), 'embedded': bool(member), 'member': member}

    manifest = {
        'schema_version': PROJECT_SCHEMA_VERSION,
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'created_at': _utc_now(),
        'video': video,
        'segments': normalize_segments(segments),
        'results': normalize_results(results),
        'edits': edits if edits is not None else [],
        'metadata': dict(metadata or {}),
    }

    temp_path = project_path.with_name(f'.{project_path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(PROJECT_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
            if video and video['embedded']:
                archive.write(video_path, video['member'])
        os.replace(temp_path, project_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return project_path


def load_project(project_path, extract_dir=None):
    """Load and validate a .lipsight project, extracting embedded media safely."""
    project_path = Path(project_path).expanduser().resolve()
    with zipfile.ZipFile(project_path, 'r') as archive:
        try:
            manifest = json.loads(archive.read(PROJECT_MANIFEST).decode('utf-8'))
        except KeyError as exc:
            raise ValueError('project is missing project.json') from exc
        if manifest.get('schema_version') != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema: {manifest.get('schema_version')}")
        manifest['segments'] = normalize_segments(manifest.get('segments'))
        manifest['results'] = normalize_results(manifest.get('results'))

        video = manifest.get('video')
        if video and video.get('embedded'):
            member = video.get('member')
            if not member or member not in archive.namelist():
                raise ValueError('embedded video is missing from project')
            target_root = Path(extract_dir or project_path.with_suffix('.media'))
            target_root.mkdir(parents=True, exist_ok=True)
            target = _project_member_path(target_root, member)
            with archive.open(member) as source, open(target, 'wb') as destination:
                shutil.copyfileobj(source, destination)
            video['path'] = str(target)
        return manifest


class SessionArchive:
    """Append-only local transcript archive with lightweight full-text search."""

    def __init__(self, path=None):
        self.path = Path(path or os.path.join(get_config_dir(), 'sessions.jsonl')).expanduser()

    def record(self, video_path, results, backend='', metadata=None):
        entry = {
            'id': uuid.uuid4().hex,
            'created_at': _utc_now(),
            'video_path': str(Path(video_path).expanduser()) if video_path else None,
            'backend': str(backend or ''),
            'results': normalize_results(results),
            'metadata': dict(metadata or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as archive:
            archive.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return entry

    def entries(self, limit=None):
        if not self.path.is_file():
            return []
        entries = []
        with self.path.open('r', encoding='utf-8') as archive:
            for line in archive:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        entries.reverse()
        return entries[:limit] if limit else entries

    def search(self, query, limit=100):
        needle = str(query or '').casefold().strip()
        if not needle:
            return self.entries(limit)
        matches = []
        for entry in self.entries():
            haystack = json.dumps(entry, ensure_ascii=False).casefold()
            if needle in haystack:
                matches.append(entry)
                if len(matches) >= limit:
                    break
        return matches


# ── Mouth preprocessing ────────────────────────────────────────────────────
def align_mouth_roi(frame, points, output_size=(96, 96)):
    """Rotate, crop, and resize a mouth landmark cloud to a canonical pose."""
    if frame is None or getattr(frame, 'size', 0) == 0:
        raise ValueError('frame is empty')
    if points is None:
        raise ValueError('mouth landmarks are required for alignment')
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        raise ValueError('at least four mouth landmarks are required')

    center = points.mean(axis=0)
    _, eigenvectors, _ = cv2.PCACompute2(points - center, mean=None)
    angle = math.degrees(math.atan2(float(eigenvectors[0, 1]), float(eigenvectors[0, 0])))
    rotation = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    rotated = cv2.warpAffine(frame, rotation, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    rotated_points = cv2.transform(points[None, :, :], rotation)[0]
    x1, y1 = np.floor(rotated_points.min(axis=0) - 8).astype(int)
    x2, y2 = np.ceil(rotated_points.max(axis=0) + 8).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(rotated.shape[1], x2), min(rotated.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('mouth landmarks do not define a valid ROI')
    crop = rotated[y1:y2, x1:x2]
    return cv2.resize(crop, tuple(output_size), interpolation=cv2.INTER_AREA)


def normalize_lighting(crop, clip_limit=2.0, tile_grid=(8, 8)):
    """Apply CLAHE to the luminance channel without changing mouth colors."""
    if crop is None or getattr(crop, 'size', 0) == 0:
        return crop
    if len(crop.shape) == 2:
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(crop)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


class MouthStabilizer:
    """Optical-flow stabilizer for sequential mouth crops."""

    def __init__(self):
        self._previous = None

    def reset(self):
        self._previous = None

    def stabilize(self, crop):
        if crop is None or getattr(crop, 'size', 0) == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        gray = gray.astype(np.float32) / 255.0
        if self._previous is None:
            self._previous = gray.copy()
            return crop
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 1e-4)
            cv2.findTransformECC(self._previous, gray, warp, cv2.MOTION_AFFINE, criteria)
            stabilized = cv2.warpAffine(
                crop, warp, (crop.shape[1], crop.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error:
            stabilized = crop
        self._previous = gray.copy()
        return stabilized


class SuperResolutionProcessor:
    """Optional OpenCV DNN super-resolution for small face crops."""

    def __init__(self, model_path='', model_name='edsr', scale=4):
        self.model_path = str(model_path or '')
        self.model_name = model_name
        self.scale = int(scale)
        self._model = None

    def _load(self):
        if self._model is not None or not self.model_path:
            return self._model
        dnn_superres = getattr(cv2, 'dnn_superres', None)
        if dnn_superres is None or not os.path.isfile(self.model_path):
            return None
        try:
            model = dnn_superres.DnnSuperResImpl_create()
            model.readModel(self.model_path)
            model.setModel(self.model_name, self.scale)
            self._model = model
        except (cv2.error, OSError):
            self._model = None
        return self._model

    def enhance(self, crop, minimum_size=96):
        if crop is None or getattr(crop, 'size', 0) == 0:
            return crop
        if min(crop.shape[:2]) >= minimum_size:
            return crop
        model = self._load()
        if model is not None:
            try:
                return model.upsample(crop)
            except cv2.error:
                pass
        scale = max(1.0, minimum_size / min(crop.shape[:2]))
        size = (max(minimum_size, int(crop.shape[1] * scale)), max(minimum_size, int(crop.shape[0] * scale)))
        return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)


class MouthPreprocessor:
    """Canonical mouth crop pipeline: alignment, enhancement, stabilization, CLAHE."""

    def __init__(self, output_size=(96, 96), super_resolution_model='', stabilize=True, clahe=True):
        self.output_size = tuple(output_size)
        self.super_resolution = SuperResolutionProcessor(super_resolution_model)
        self.stabilizer = MouthStabilizer() if stabilize else None
        self.clahe = clahe

    def reset(self):
        if self.stabilizer:
            self.stabilizer.reset()

    def process(self, frame, roi=None, points=None):
        if points is not None:
            crop = align_mouth_roi(frame, points, self.output_size)
        elif roi is not None:
            x1, y1, x2, y2 = [int(value) for value in roi]
            crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
            crop = self.super_resolution.enhance(crop)
            crop = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_AREA)
        else:
            raise ValueError('roi or points are required')
        if self.stabilizer:
            crop = self.stabilizer.stabilize(crop)
        return normalize_lighting(crop) if self.clahe else crop


# ── Face Detection ──────────────────────────────────────────────────────────
class FaceAnalyzer:
    """Face/mouth detection: MediaPipe preferred, OpenCV Haar cascade fallback."""

    MOUTH_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
    MOUTH_INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]

    def __init__(self):
        self.face_mesh = None
        self._backend = 'none'
        self._cascade = None

        if _HAS_MEDIAPIPE:
            try:
                self.face_mesh = _mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=5,
                    refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                self._backend = 'mediapipe'
            except Exception:
                pass

        if self._backend == 'none':
            try:
                p = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                self._cascade = cv2.CascadeClassifier(p)
                if not self._cascade.empty():
                    self._backend = 'opencv'
            except Exception:
                pass

    @property
    def available(self): return self._backend != 'none'
    @property
    def backend_name(self): return self._backend

    def analyze_frame(self, frame):
        if self._backend == 'mediapipe': return self._mp(frame)
        if self._backend == 'opencv': return self._cv(frame)
        return frame.copy(), None, 0.0, None

    def _mp(self, frame):
        h, w = frame.shape[:2]
        out = frame.copy()
        observations = []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            for face_index, lm in enumerate(res.multi_face_landmarks or []):
                pts = [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in self.MOUTH_OUTER]
                if pts:
                    arr = np.array(pts)
                    x1, y1 = arr.min(0) - 20; x2, y2 = arr.max(0) + 20
                    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
                    roi = (int(x1), int(y1), int(x2), int(y2))
                    cv2.rectangle(out, (x1,y1), (x2,y2), (137,180,250), 2)
                    for i in self.MOUTH_OUTER:
                        cv2.circle(out, (int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)), 2, (166,227,161), -1)
                    for i in self.MOUTH_INNER:
                        cv2.circle(out, (int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)), 2, (180,190,254), -1)
                    t, b = lm.landmark[13], lm.landmark[14]
                    l, r = lm.landmark[61], lm.landmark[291]
                    mh = math.sqrt((t.x-b.x)**2+(t.y-b.y)**2)
                    mw = math.sqrt((l.x-r.x)**2+(l.y-r.y)**2)
                    ratio = mh / max(mw, 0.001)
                    observations.append({
                        'index': face_index,
                        'roi': roi,
                        'points': pts,
                        'open_ratio': ratio,
                        'center': (float((x1 + x2) / 2), float((y1 + y2) / 2)),
                    })
                    col = (166,227,161) if ratio > 0.06 else (108,112,134)
                    cv2.putText(out, "SPEAKING" if ratio>0.06 else "SILENT", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        except: pass
        if observations:
            primary = observations[0]
            return out, primary['roi'], primary['open_ratio'], {'faces': observations}
        return out, None, 0.0, {'faces': []}

    def _cv(self, frame):
        h, w = frame.shape[:2]
        out = frame.copy()
        observations = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
            for face_index, (fx,fy,fw,fh) in enumerate(faces):
                cv2.rectangle(out, (fx,fy), (fx+fw,fy+fh), (69,71,90), 1)
                mx1, my1 = max(0,fx+int(fw*0.2)), max(0,fy+int(fh*0.65))
                mx2, my2 = min(w,fx+int(fw*0.8)), min(h,fy+fh+5)
                roi = (int(mx1), int(my1), int(mx2), int(my2))
                cv2.rectangle(out, (mx1,my1), (mx2,my2), (137,180,250), 2)
                mg = gray[my1:my2, mx1:mx2]
                ratio = 0.0
                if mg.size > 0:
                    gx = cv2.Sobel(mg, cv2.CV_64F, 1, 0, ksize=3)
                    gy = cv2.Sobel(mg, cv2.CV_64F, 0, 1, ksize=3)
                    ratio = min(np.mean(np.sqrt(gx**2+gy**2)) / 50.0, 0.3)
                observations.append({
                    'index': face_index,
                    'roi': roi,
                    'points': None,
                    'open_ratio': ratio,
                    'center': (float(fx + fw / 2), float(fy + fh / 2)),
                })
                cv2.putText(out, "DETECTED", (mx1,my1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (137,180,250), 1, cv2.LINE_AA)
        except: pass
        if observations:
            primary = observations[0]
            return out, primary['roi'], primary['open_ratio'], {'faces': observations}
        return out, None, 0.0, {'faces': []}

    def close(self):
        if self.face_mesh:
            try: self.face_mesh.close()
            except: pass


# ── Video Segmenter ─────────────────────────────────────────────────────────
class VideoSegmenter:
    def __init__(self, threshold=0.06):
        self.threshold = threshold
        self.last_curve = []
        self.last_speaker_curves = {}

    def segment(self, video_path, progress_cb=None, log_cb=None, multi_speaker=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_curve = []
        self.last_speaker_curves = {}
        analyzer = FaceAnalyzer()
        if not analyzer.available:
            cap.release()
            if log_cb: log_cb("⚠️  No face detection — whole video as one segment")
            fallback = {'start': 0.0, 'end': total/fps, 'speaker': 'A'}
            return [fallback] if multi_speaker else [(0.0, total/fps)]

        if log_cb: log_cb(f"📐 Analyzing {total} frames ({analyzer.backend_name})...")
        ratios = []; speaker_ratios = {}; idx = 0; step = max(1, int(fps/10))
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % step == 0:
                _, _, r, info = analyzer.analyze_frame(frame)
                ratios.append((idx, r))
                self.last_curve.append((idx / fps, r))
                if multi_speaker:
                    faces = sorted((info or {}).get('faces', []), key=lambda face: face.get('center', (0, 0))[0])
                    for speaker_index, face in enumerate(faces):
                        speaker_ratios.setdefault(speaker_index, []).append((idx, face.get('open_ratio', 0.0)))
            idx += 1
            if progress_cb and idx % 50 == 0: progress_cb(int(idx/max(total,1)*100))
        cap.release(); analyzer.close()
        if not ratios: return [(0.0, total/fps)]

        def find_segments(samples):
            ms, ml = int(0.5*25), int(0.3*25)
            found, speech, start, silence = [], False, 0, 0
            for frame_number, ratio in samples:
                if ratio > self.threshold:
                    if not speech:
                        start = frame_number
                        speech = True
                    silence = 0
                elif speech:
                    silence += step
                    if silence >= ml:
                        end = frame_number - silence
                        if (end-start) >= ms:
                            found.append((start/fps, end/fps))
                        speech = False
                        silence = 0
            if speech:
                end = samples[-1][0]
                if (end-start) >= ms:
                    found.append((start/fps, end/fps))
            return found

        segs = find_segments(ratios)
        if multi_speaker and speaker_ratios:
            speaker_segments = []
            self.last_speaker_curves = {}
            for speaker_index, samples in sorted(speaker_ratios.items()):
                label = chr(ord('A') + speaker_index)
                self.last_speaker_curves[label] = [(frame / fps, ratio) for frame, ratio in samples]
                speaker_segments.extend({
                    'start': start, 'end': end, 'speaker': label,
                } for start, end in find_segments(samples))
            if speaker_segments:
                segs = sorted(speaker_segments, key=lambda segment: (segment['start'], segment['speaker']))
        if log_cb: log_cb(f"🔍 Found {len(segs)} speech segments")
        return segs if segs else [(0.0, total/fps)]


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class HuggingFaceSpaceBackend:
    """FREE inference via public HuggingFace Gradio Spaces — no token, no signup."""

    KNOWN_SPACES = [
        "https://mpc001-auto-avsr.hf.space",
        "https://vumichien-av-hubert.hf.space",
    ]

    def __init__(self, custom_url=""):
        self.custom_url = custom_url.strip().rstrip('/')

    def transcribe(self, video_path, log_cb=None):
        spaces = []
        if self.custom_url: spaces.append(self.custom_url)
        spaces.extend(self.KNOWN_SPACES)
        last_err = None

        for base in spaces:
            name = base.split("//")[1].split(".")[0] if "//" in base else base
            try:
                if log_cb: log_cb(f"🤗 Trying Space: {name}...")

                # Check reachability
                try:
                    requests.get(base, timeout=15)
                except requests.exceptions.ConnectionError:
                    if log_cb: log_cb(f"   ❌ Unreachable"); continue

                # Upload file
                if log_cb: log_cb(f"   📤 Uploading video...")
                with open(video_path, 'rb') as f:
                    up = requests.post(f"{base}/upload",
                        files={"files": (os.path.basename(video_path), f, "video/mp4")}, timeout=120)
                up.raise_for_status()
                uploaded = up.json()
                fpath = uploaded[0] if isinstance(uploaded, list) else uploaded

                # Predict (try multiple Gradio API versions)
                if log_cb: log_cb(f"   🧠 Running inference (may take 30-120s)...")
                session = hashlib.md5(str(random.random()).encode()).hexdigest()[:12]

                for api_path in ["/api/predict", "/run/predict"]:
                    for payload in [
                        {"data": [{"path": fpath, "orig_name": os.path.basename(video_path)}], "session_hash": session},
                        {"data": [fpath], "session_hash": session},
                    ]:
                        try:
                            r = requests.post(f"{base}{api_path}", json=payload, timeout=300)
                            if r.status_code in (404, 422): continue
                            r.raise_for_status()
                            data = r.json().get("data", [])
                            if data and data[0]:
                                if log_cb: log_cb(f"   ✅ Got result")
                                return str(data[0]).strip()
                        except (requests.exceptions.HTTPError, requests.exceptions.Timeout):
                            continue

                if log_cb: log_cb(f"   ⚠️  No valid response")
            except Exception as e:
                last_err = str(e)
                if log_cb: log_cb(f"   ❌ {e}")

        raise RuntimeError(
            f"All HuggingFace Spaces unavailable.\n"
            f"Last error: {last_err}\n\n"
            f"Options: Try again later, use Local backend, or set a custom Space URL.")


class LocalAutoAVSRBackend:
    """FREE local inference — auto-downloads PyTorch + Auto-AVSR."""

    MODEL_DIR = os.path.join(get_config_dir(), 'models', 'auto_avsr')
    REPO_URL = "https://github.com/mpc001/auto_avsr.git"

    def __init__(self):
        self._ready = False

    def _pip(self, pkgs, log_cb=None):
        for pkg in pkgs:
            mod = pkg.split('[')[0].split('=')[0].split('>')[0].split('<')[0].replace('-','_').lower()
            if mod == 'opencv_python': mod = 'cv2'
            try: __import__(mod); continue
            except ImportError: pass
            if log_cb: log_cb(f"   📦 Installing {pkg}...")
            for fl in [[], ['--user'], ['--break-system-packages']]:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'] + fl, timeout=600)
                    break
                except: continue

    def _ensure_setup(self, log_cb=None):
        if self._ready: return
        if log_cb: log_cb("🔧 Setting up local Auto-AVSR...")

        # PyTorch
        try:
            import torch
            if log_cb: log_cb(f"   ✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        except ImportError:
            if log_cb: log_cb("   📦 Installing PyTorch (may take several minutes)...")
            for cmd in [
                [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121', '-q'],
                [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '-q'],
            ]:
                try: subprocess.check_call(cmd, timeout=900); break
                except: continue
            import torch
            if log_cb: log_cb(f"   ✅ PyTorch {torch.__version__}")

        self._pip(['sentencepiece', 'pytorch-lightning', 'hydra-core', 'omegaconf'], log_cb)

        # Clone repo
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        repo = os.path.join(self.MODEL_DIR, 'repo')
        if not os.path.isdir(repo):
            if log_cb: log_cb("   📥 Cloning Auto-AVSR repository...")
            try:
                subprocess.check_call(['git', 'clone', '--depth', '1', self.REPO_URL, repo], timeout=120)
            except FileNotFoundError:
                raise RuntimeError("Git not installed. Install from https://git-scm.com/download/win")
            except Exception as e:
                raise RuntimeError(f"Clone failed: {e}")

        # Install package
        if log_cb: log_cb("   📦 Installing Auto-AVSR package...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', repo, '-q'], timeout=300)
        except:
            if repo not in sys.path: sys.path.insert(0, repo)

        self._ready = True
        if log_cb: log_cb("   ✅ Local setup complete")

    def transcribe(self, video_path, log_cb=None):
        self._ensure_setup(log_cb)
        import torch

        repo = os.path.join(self.MODEL_DIR, 'repo')
        if repo not in sys.path: sys.path.insert(0, repo)

        if log_cb: log_cb("🧠 Running local inference...")

        # Try CLI inference
        for script in ['infer.py', 'eval.py', 'predict.py', 'demo.py']:
            sp = os.path.join(repo, script)
            if os.path.exists(sp):
                try:
                    result = subprocess.run(
                        [sys.executable, sp, '--video_path', video_path, '--modality', 'video'],
                        capture_output=True, text=True, timeout=300, cwd=repo)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in reversed(lines):
                            l = line.strip()
                            if l and not l.startswith(('[','=','W','I','D')): return l
                        return result.stdout.strip()
                    else:
                        err = result.stderr.strip() or result.stdout.strip()
                        if log_cb: log_cb(f"   ⚠️  {script} error: {err[:200]}")
                except subprocess.TimeoutExpired:
                    raise RuntimeError("Inference timed out (>5min)")
                except Exception as e:
                    if log_cb: log_cb(f"   ⚠️  {script}: {e}")

        raise RuntimeError(
            f"Could not run inference. The Auto-AVSR repo may require additional setup.\n\n"
            f"Manual steps:\n  cd {repo}\n  python infer.py --video_path \"{video_path}\" --modality video")


class ReplicateBackend:
    """Cloud inference via Replicate API — requires token."""

    API = "https://api.replicate.com/v1"
    MODEL = "basord/lip-reading-ai-vsr"

    def __init__(self, token):
        self.token = token
        self.h = {"Authorization": f"Token {token}", "Content-Type": "application/json"}

    def transcribe(self, video_path, log_cb=None):
        if log_cb: log_cb("☁️  Uploading to Replicate...")
        try:
            # Upload
            with open(video_path, 'rb') as f:
                r = requests.post(f"{self.API}/files", headers={"Authorization": f"Token {self.token}"},
                    files={"content": (os.path.basename(video_path), f, "video/mp4")},
                    data={"content_type": "video/mp4"}, timeout=120)
            r.raise_for_status()
            url = r.json().get("urls", {}).get("get", "") or r.json().get("url", "")
            if not url: raise RuntimeError("No upload URL returned")

            # Version
            r = requests.get(f"{self.API}/models/{self.MODEL}/versions", headers=self.h, timeout=30)
            r.raise_for_status()
            ver = r.json()["results"][0]["id"]

            # Predict
            if log_cb: log_cb("☁️  Running prediction...")
            r = requests.post(f"{self.API}/predictions", headers=self.h,
                json={"version": ver, "input": {"video": url}}, timeout=30)
            r.raise_for_status()
            pred = r.json()
            get_url = pred.get("urls", {}).get("get", f"{self.API}/predictions/{pred['id']}")

            for elapsed in range(0, 300, 2):
                time.sleep(2)
                p = requests.get(get_url, headers=self.h, timeout=30).json()
                st = p.get("status", "")
                if st == "succeeded":
                    out = p.get("output", "")
                    if isinstance(out, dict): return out.get("text", str(out))
                    if isinstance(out, list): return " ".join(str(o) for o in out)
                    return str(out) if out else "(empty)"
                if st in ("failed", "canceled"):
                    raise RuntimeError(f"Prediction {st}: {p.get('error','?')}")
                if log_cb and elapsed % 10 == 0 and elapsed > 0:
                    log_cb(f"☁️  Waiting... ({elapsed}s)")
            raise RuntimeError("Timed out")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Replicate HTTP {e.response.status_code}: {e.response.text[:200]}")


class DirectAPIBackend:
    def __init__(self, url, key=""):
        self.url = url; self.key = key
    def transcribe(self, video_path, log_cb=None):
        if log_cb: log_cb(f"🌐 Sending to {self.url}...")
        h = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        with open(video_path, 'rb') as f:
            r = requests.post(self.url, files={'video': (os.path.basename(video_path), f, 'video/mp4')},
                headers=h, timeout=300)
        r.raise_for_status()
        d = r.json()
        return d.get('text', d.get('transcription', str(d)))


# ── Export ──────────────────────────────────────────────────────────────────
def _ts(s):
    milliseconds = int(round(_timestamp_value(s, 'timestamp') * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def _vtt_ts(s):
    return _ts(s).replace(',', '.')

def export_srt(res, fp):
    normalized = normalize_results(res)
    with open(_safe_output_path(fp), 'w', encoding='utf-8', newline='\n') as f:
        for i, result in enumerate(normalized, 1):
            f.write(f"{i}\n{_ts(result['start'])} --> {_ts(result['end'])}\n{result['text']}\n\n")


def export_vtt(res, fp):
    normalized = normalize_results(res)
    with open(_safe_output_path(fp), 'w', encoding='utf-8', newline='\n') as f:
        f.write('WEBVTT\n\n')
        for i, result in enumerate(normalized, 1):
            f.write(f"{i}\n{_vtt_ts(result['start'])} --> {_vtt_ts(result['end'])}\n{result['text']}\n\n")

def export_txt(res, fp):
    normalized = normalize_results(res)
    with open(_safe_output_path(fp), 'w', encoding='utf-8', newline='\n') as f:
        for result in normalized:
            speaker = f"{result['speaker']}: " if result['speaker'] else ''
            f.write(f"[{_ts(result['start'])} -> {_ts(result['end'])}] {speaker}{result['text']}\n")

def export_json(res, fp, metadata=None):
    with open(_safe_output_path(fp), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(build_result_document(res, metadata), f, ensure_ascii=False, indent=2)
        f.write('\n')


def export_results(res, fp, fmt=None, metadata=None):
    output = Path(fp)
    format_name = (fmt or output.suffix.lstrip('.')).lower()
    exporters = {'srt': export_srt, 'vtt': export_vtt, 'txt': export_txt, 'json': export_json}
    if format_name not in exporters:
        raise ValueError(f'unsupported export format: {format_name or "none"}')
    if format_name == 'json':
        exporters[format_name](res, output, metadata=metadata)
    else:
        exporters[format_name](res, output)
    return output


def extract_video_segment(video_path, start, end, output_path):
    """Extract a silent video segment with ffmpeg, falling back to OpenCV."""
    source = Path(video_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f'video not found: {source}')
    start = _timestamp_value(start, 'start')
    end = _timestamp_value(end, 'end')
    if end <= start:
        raise ValueError('segment end must be greater than start')
    output = _safe_output_path(output_path)

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        completed = subprocess.run(
            [ffmpeg, '-y', '-ss', str(start), '-i', str(source), '-t', str(end - start),
             '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', str(output)],
            capture_output=True, text=True, timeout=120)
        if completed.returncode == 0 and output.is_file() and output.stat().st_size > 0:
            return output

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'cannot create segment: {output}')
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        for _ in range(max(1, int((end - start) * fps))):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f'could not extract segment: {output}')
    return output


def export_segments(video_path, results, output_dir):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    exported = []
    for index, result in enumerate(normalize_results(results), 1):
        output = output_dir / f'{stem}_segment_{index:03d}.mp4'
        exported.append(extract_video_segment(video_path, result['start'], result['end'], output))
    return exported


def burn_in_subtitles(video_path, results, output_path):
    """Render SRT captions into a video through ffmpeg."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required for burn-in subtitles')
    source = Path(video_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f'video not found: {source}')
    output = _safe_output_path(output_path)
    subtitle_path = output.with_name(f'.{output.stem}.{uuid.uuid4().hex}.srt')
    try:
        export_srt(results, subtitle_path)
        escaped = subtitle_path.as_posix().replace(':', r'\:').replace("'", r"\'")
        filter_arg = f"subtitles=filename='{escaped}'"
        completed = subprocess.run(
            [ffmpeg, '-y', '-i', str(source), '-vf', filter_arg, '-c:v', 'libx264', '-c:a', 'aac', str(output)],
            capture_output=True, text=True, timeout=300)
        if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            detail = (completed.stderr or completed.stdout or 'ffmpeg failed').strip().splitlines()[-1]
            raise RuntimeError(detail)
        return output
    finally:
        if subtitle_path.exists():
            subtitle_path.unlink()


# ── Headless processing ────────────────────────────────────────────────────
def video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video: {video_path}')
    try:
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    finally:
        cap.release()


def _transcription_payload(value):
    if isinstance(value, dict):
        return {
            'text': str(value.get('text', value.get('transcription', ''))).strip(),
            'confidence': value.get('confidence'),
            'words': value.get('words') or [],
            'speaker': value.get('speaker') or 'A',
        }
    if isinstance(value, list):
        return {'text': ' '.join(str(item) for item in value).strip(), 'confidence': None, 'words': [], 'speaker': 'A'}
    return {'text': str(value or '').strip(), 'confidence': None, 'words': [], 'speaker': 'A'}


def _transcribe_batch(backend, paths, log_cb=None):
    batch = getattr(backend, 'transcribe_batch', None)
    if not callable(batch):
        return None
    values = batch(paths, log_cb=log_cb)
    if not isinstance(values, list) or len(values) != len(paths):
        raise RuntimeError('batch backend returned an unexpected number of results')
    return values


def process_video(video_path, backend, segments=None, progress_cb=None, log_cb=None,
                  segment_cb=None, should_stop=None):
    """Process a video without Qt so CLI, watch mode, and workers share one path."""
    progress_cb = progress_cb or (lambda _value: None)
    log_cb = log_cb or (lambda _message: None)
    segment_cb = segment_cb or (lambda _result: None)
    should_stop = should_stop or (lambda: False)
    normalized_segments = normalize_segments(segments)
    results = []

    if not normalized_segments or len(normalized_segments) == 1:
        log_cb('🎬 Processing entire video...')
        progress_cb(10)
        payload = _transcription_payload(backend.transcribe(str(video_path), log_cb=log_cb))
        result = normalize_result({
            **payload,
            'start': normalized_segments[0]['start'] if normalized_segments else 0.0,
            'end': normalized_segments[0]['end'] if normalized_segments else video_duration(video_path),
            'segment': 1,
        })
        results.append(result)
        segment_cb(result)
        progress_cb(100)
        log_cb('✅ Complete — 1 segment(s)')
        return results

    temp_dir = Path(tempfile.mkdtemp(prefix='lipsight_'))
    paths = []
    try:
        for index, segment in enumerate(normalized_segments, 1):
            if should_stop():
                break
            log_cb(f"🎬 Preparing segment {index}/{len(normalized_segments)} [{segment['start']:.1f}s-{segment['end']:.1f}s]")
            path = temp_dir / f'seg_{index:04d}.mp4'
            try:
                paths.append(extract_video_segment(video_path, segment['start'], segment['end'], path))
            except Exception as exc:
                log_cb(f'⚠️  Segment {index}: {exc}')
                paths.append(None)

        valid = [(index, segment, path) for index, (segment, path) in enumerate(zip(normalized_segments, paths), 1) if path]
        batch_values = None
        if valid and len(valid) == len(normalized_segments):
            batch_values = _transcribe_batch(backend, [path for _, _, path in valid], log_cb=log_cb)

        for position, (index, segment, path) in enumerate(valid):
            if should_stop():
                break
            log_cb(f"🧠 Transcribing segment {index}/{len(normalized_segments)}")
            value = batch_values[position] if batch_values is not None else backend.transcribe(str(path), log_cb=log_cb)
            payload = _transcription_payload(value)
            result = normalize_result({**payload, **segment, 'segment': index})
            results.append(result)
            segment_cb(result)
            progress_cb(int(index / max(len(normalized_segments), 1) * 100))
        log_cb(f'✅ Complete — {len(results)} segment(s)')
        return results
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_backend(name, config=None):
    config = config or {}
    normalized = str(name or 'hf').strip().lower().replace('_', '-').replace(' ', '-')
    if normalized in ('hf', 'huggingface', 'space'):
        return HuggingFaceSpaceBackend(config.get('hf_space_url', ''))
    if normalized in ('local', 'auto-avsr', 'autoavsr'):
        return LocalAutoAVSRBackend()
    if normalized in ('replicate', 'cloud'):
        token = config.get('replicate_api_token', '')
        if not token:
            raise ValueError('Replicate backend requires an API token')
        return ReplicateBackend(token)
    if normalized in ('custom', 'endpoint', 'api'):
        url = config.get('custom_endpoint', '')
        if not url:
            raise ValueError('custom backend requires an endpoint URL')
        return DirectAPIBackend(url, config.get('custom_endpoint_key', ''))
    raise ValueError(f'unknown backend: {name}')


def _watch_candidates(folder):
    return sorted(
        (path for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name.casefold())


def watch_folder(folder, backend, output_dir=None, auto_segment=True, threshold=0.06,
                 interval=5.0, once=False, stop_event=None, log_cb=None):
    """Poll a folder and process each new, stable video exactly once."""
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir or folder).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_cb = log_cb or (lambda _message: None)
    stop_event = stop_event or threading.Event()
    seen = set()
    completed = []

    while not stop_event.is_set():
        for video in _watch_candidates(folder):
            key = str(video.resolve())
            if key in seen or video.name.endswith('.lipsight-processing.mp4'):
                continue
            marker = video.with_name(video.name + '.lipsight-processing')
            try:
                before = video.stat()
                if before.st_size <= 0:
                    continue
                time.sleep(0.05)
                after = video.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    continue
                marker.write_text(_utc_now(), encoding='utf-8')
                segments = VideoSegmenter(threshold).segment(str(video), log_cb=log_cb) if auto_segment else None
                results = process_video(str(video), backend, segments=segments, log_cb=log_cb)
                output = output_dir / f'{video.stem}.srt'
                export_srt(results, output)
                SessionArchive().record(str(video), results, type(backend).__name__)
                completed.append(output)
                seen.add(key)
                log_cb(f'✅ Watch complete: {video.name}')
            except Exception as exc:
                log_cb(f'❌ Watch failed for {video.name}: {exc}')
                seen.add(key)
            finally:
                if marker.exists():
                    marker.unlink()
        if once:
            return completed
        stop_event.wait(max(0.1, float(interval)))
    return completed


# ── Workers ─────────────────────────────────────────────────────────────────
class ProcessingWorker(QThread):
    progress = pyqtSignal(int); log = pyqtSignal(str)
    segment_result = pyqtSignal(dict); finished = pyqtSignal(list); error = pyqtSignal(str)

    def __init__(self, vpath, backend, segs=None):
        super().__init__()
        self.vpath, self.backend, self.segs = vpath, backend, segs
        self._stop = False

    def cancel(self): self._stop = True

    def run(self):
        try:
            results = process_video(
                self.vpath,
                self.backend,
                segments=self.segs,
                progress_cb=self.progress.emit,
                log_cb=self.log.emit,
                segment_cb=self.segment_result.emit,
                should_stop=lambda: self._stop,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class SegmentWorker(QThread):
    progress = pyqtSignal(int); log = pyqtSignal(str)
    curve = pyqtSignal(list)
    finished = pyqtSignal(list); error = pyqtSignal(str)
    def __init__(self, vp, threshold=0.06, multi_speaker=False):
        super().__init__()
        self.vp = vp
        self.threshold = threshold
        self.multi_speaker = multi_speaker
    def run(self):
        try:
            segmenter = VideoSegmenter(self.threshold)
            segments = segmenter.segment(self.vp, self.progress.emit, self.log.emit, self.multi_speaker)
            self.curve.emit(segmenter.last_curve)
            self.finished.emit(segments)
        except Exception as e: self.error.emit(str(e))


class FrameWorker(QThread):
    frame_ready = pyqtSignal(QImage, float, dict, QImage); finished = pyqtSignal()
    def __init__(self, vp, n): super().__init__(); self.vp, self.n = vp, n
    def run(self):
        try:
            cap = cv2.VideoCapture(self.vp)
            if not cap.isOpened(): self.finished.emit(); return
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.n)
            ok, frame = cap.read(); cap.release()
            if not ok or frame is None: self.finished.emit(); return
            roi_image = QImage()
            roi = None
            ratio = 0.0
            details = {}
            try:
                a = FaceAnalyzer()
                out, roi, ratio, details = a.analyze_frame(frame) if a.available else (frame, None, 0.0, {})
                a.close()
                if roi:
                    first_face = ((details or {}).get('faces') or [{}])[0]
                    preprocessor = MouthPreprocessor(output_size=(160, 120), stabilize=False)
                    mouth = preprocessor.process(frame, roi=roi, points=first_face.get('points'))
                    mouth_rgb = np.ascontiguousarray(cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB))
                    mh, mw, mch = mouth_rgb.shape
                    roi_image = QImage(mouth_rgb.data, mw, mh, mch*mw, QImage.Format.Format_RGB888).copy()
            except: out, ratio, details = frame, 0.0, {}
            rgb = np.ascontiguousarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888).copy()
            info = {'open_ratio': ratio, 'face_count': len((details or {}).get('faces', []))}
            self.frame_ready.emit(img, self.n/fps, info, roi_image)
        except: pass
        self.finished.emit()


# ── Widgets ─────────────────────────────────────────────────────────────────
class MouthCurveWidget(QWidget):
    """Pointer-editable mouth-motion curve with draggable segment handles."""

    segments_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(96)
        self.setMouseTracking(True)
        self.curve = []
        self.segment_data = []
        self._drag = None

    def set_data(self, curve=None, segments=None):
        self.curve = [(float(time_value), float(ratio)) for time_value, ratio in (curve or [])]
        self.segment_data = normalize_segments(segments)
        self.update()

    def segments(self):
        return [dict(segment) for segment in self.segment_data]

    def _duration(self):
        values = [item[0] for item in self.curve]
        values.extend(segment['end'] for segment in self.segment_data)
        return max(max(values or [1.0]), 1.0)

    def _plot_rect(self):
        return (10, 10, max(1, self.width() - 20), max(1, self.height() - 28))

    def _x(self, time_value):
        left, _top, width, _height = self._plot_rect()
        return left + (time_value / self._duration()) * width

    def _time(self, x_value):
        left, _top, width, _height = self._plot_rect()
        return max(0.0, min(self._duration(), (x_value - left) / max(width, 1) * self._duration()))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(C['crust']))
        left, top, width, height = self._plot_rect()
        painter.setPen(QPen(QColor(C['surface1']), 1))
        painter.drawRect(left, top, width, height)

        for segment in self.segment_data:
            start_x = int(self._x(segment['start']))
            end_x = int(self._x(segment['end']))
            color = C['mauve'] if segment.get('speaker') == 'B' else C['blue']
            painter.fillRect(start_x, top, max(2, end_x - start_x), height, QColor(color + '22'))
            painter.setPen(QPen(QColor(color), 1))
            painter.drawLine(start_x, top, start_x, top + height)
            painter.drawLine(end_x, top, end_x, top + height)

        if self.curve:
            painter.setPen(QPen(QColor(C['green']), 2))
            previous = None
            for time_value, ratio in self.curve:
                point = QPointF(self._x(time_value), top + height - min(1.0, max(0.0, ratio) / 0.3) * height)
                if previous is not None:
                    painter.drawLine(previous, point)
                previous = point
        painter.setPen(QPen(QColor(C['overlay1']), 1))
        painter.drawText(left, self.height() - 6, f'0:00   Mouth movement   {LipSightWindow._ft(self._duration())}')

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.segment_data:
            return
        x_value = event.position().x()
        nearest = None
        distance = 13
        for index, segment in enumerate(self.segment_data):
            for side in ('start', 'end'):
                candidate = abs(self._x(segment[side]) - x_value)
                if candidate < distance:
                    nearest = (index, side)
                    distance = candidate
        self._drag = nearest

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        index, side = self._drag
        value = self._time(event.position().x())
        segment = self.segment_data[index]
        if side == 'start':
            segment['start'] = min(value, segment['end'] - 0.05)
        else:
            segment['end'] = max(value, segment['start'] + 0.05)
        self.update()
        self.segments_changed.emit(self.segments())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = None


class VideoPreview(QLabel):
    def __init__(self):
        super().__init__(); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480,270)
        self.setStyleSheet(f"background-color:{C['crust']};border:1px solid {C['surface1']};border-radius:8px;color:{C['overlay0']};font-size:16px;")
        self._pm = None; self.setText("📹  Load a video to begin")
    def set_frame(self, img):
        self._pm = QPixmap.fromImage(img); self._upd()
    def _upd(self):
        if self._pm: self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    def resizeEvent(self, e): super().resizeEvent(e); self._upd()

class Toast(QLabel):
    def __init__(self, p, msg, col=C['green'], ms=2500):
        super().__init__(msg, p)
        self.setStyleSheet(f"background-color:{C['surface0']};color:{col};border:1px solid {col};border-radius:8px;padding:10px 20px;font-size:13px;font-weight:bold;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter); self.adjustSize()
        self.move(p.width()//2-self.width()//2, 20); self.show(); self.raise_()
        QTimer.singleShot(ms, self.deleteLater)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
BACKENDS = ["🤗 HuggingFace Space (Free)", "💻 Local Auto-AVSR (Free)", "☁️ Replicate API (Token)", "🌐 Custom Endpoint"]

class LipSightWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}"); self.resize(1300,840); self.setMinimumSize(1000,650)
        self.cfg = load_config()
        self.video_path = None; self.video_info = {}; self.segments = []; self.results = []
        self.review_edits = []; self._curve_data = []
        self._fw = self._sw = self._pw = None
        self._build(); self._connect()
        fa = FaceAnalyzer()
        self._log(f"✅ Face detection: {fa.backend_name}" if fa.available else "⚠️  No face detection — segmentation disabled")
        fa.close()
        self._log(f"🔧 Backend: {BACKENDS[self.cfg.get('backend_index', 0)]}")

    def _build(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setSpacing(0); root.setContentsMargins(0,0,0,0)

        # Header
        hdr = QWidget(); hdr.setFixedHeight(56)
        hdr.setStyleSheet(f"background-color:{C['mantle']};border-bottom:1px solid {C['surface0']};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(16,0,16,0)
        lg = QLabel(f"👁️  {APP_NAME}"); lg.setStyleSheet(f"font-size:18px;font-weight:bold;color:{C['blue']};background:transparent;border:none;")
        hl.addWidget(lg)
        vl = QLabel(f"v{APP_VERSION}"); vl.setStyleSheet(f"font-size:11px;color:{C['overlay0']};background:transparent;border:none;")
        hl.addWidget(vl); hl.addStretch()
        self.badge = QLabel(BACKENDS[self.cfg.get('backend_index',0)])
        self.badge.setStyleSheet(f"background-color:{C['surface0']};color:{C['teal']};padding:4px 12px;border-radius:12px;font-size:12px;font-weight:bold;")
        hl.addWidget(self.badge); root.addWidget(hdr)

        body = QWidget(); bl = QHBoxLayout(body); bl.setContentsMargins(12,12,12,0); bl.setSpacing(12)

        # Left
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0); ll.setSpacing(8)
        views = QHBoxLayout(); views.setSpacing(8)
        self.preview = VideoPreview(); self.preview.setText("📹  Load a video to begin")
        self.preview.setToolTip("Annotated video preview")
        self.roi_preview = VideoPreview(); self.roi_preview.setText("👄  Mouth ROI")
        self.roi_preview.setToolTip("Aligned, normalized mouth crop")
        views.addWidget(self.preview, stretch=3); views.addWidget(self.roi_preview, stretch=2)
        ll.addLayout(views, stretch=1)

        sr = QHBoxLayout()
        self.t_lbl = QLabel("00:00.000"); self.t_lbl.setStyleSheet(f"color:{C['overlay1']};font-family:monospace;font-size:12px;")
        sr.addWidget(self.t_lbl)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setMinimum(0); self.slider.setMaximum(0)
        sr.addWidget(self.slider, stretch=1)
        self.d_lbl = QLabel("00:00.000"); self.d_lbl.setStyleSheet(f"color:{C['overlay1']};font-family:monospace;font-size:12px;")
        sr.addWidget(self.d_lbl); ll.addLayout(sr)
        self.curve = MouthCurveWidget(); self.curve.setToolTip("Mouth movement curve — drag segment edges to review timing")
        ll.addWidget(self.curve)

        br = QHBoxLayout(); br.setSpacing(8)
        self.b_load = QPushButton("📂  Load Video"); br.addWidget(self.b_load)
        self.b_analyze = QPushButton("🔍  Analyze"); self.b_analyze.setObjectName("accentBtn"); self.b_analyze.setEnabled(False); br.addWidget(self.b_analyze)
        self.b_process = QPushButton("🧠  Lip Read"); self.b_process.setObjectName("greenBtn"); self.b_process.setEnabled(False); br.addWidget(self.b_process)
        self.b_cancel = QPushButton("⏹"); self.b_cancel.setObjectName("dangerBtn"); self.b_cancel.setEnabled(False); self.b_cancel.setFixedWidth(50); br.addWidget(self.b_cancel)
        ll.addLayout(br)

        self.prog = QProgressBar(); self.prog.setFixedHeight(6); self.prog.setTextVisible(False); ll.addWidget(self.prog)

        sts = QHBoxLayout(); sts.setSpacing(16)
        self.sf = self._mk("FRAMES"); self.sfps = self._mk("FPS"); self.sres = self._mk("RES"); self.sseg = self._mk("SEGS"); self.smo = self._mk("MOUTH")
        for s in [self.sf,self.sfps,self.sres,self.sseg,self.smo]: sts.addWidget(s)
        sts.addStretch(); ll.addLayout(sts)
        bl.addWidget(left, stretch=6)

        # Right tabs
        tabs = QTabWidget(); tabs.setMinimumWidth(380)

        # Results
        rw = QWidget(); rl = QVBoxLayout(rw); rl.setContentsMargins(8,8,8,8)
        self.tbl = QTableWidget(); self.tbl.setColumnCount(5)
        self.tbl.setHorizontalHeaderLabels(["Time","Dur","Speaker","Conf.","Transcription"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.setAlternatingRowColors(True); self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.tbl.verticalHeader().setVisible(False)
        rl.addWidget(self.tbl)
        self.txte = QTextEdit(); self.txte.setReadOnly(False); self.txte.setMaximumHeight(120)
        self.txte.setPlaceholderText("Full transcript — edit here, then Apply Review..."); rl.addWidget(self.txte)
        er = QHBoxLayout()
        self.bsrt=QPushButton("💾 SRT"); self.bsrt.setObjectName("secondaryBtn"); self.bsrt.setEnabled(False)
        self.bvtt=QPushButton("💾 VTT"); self.bvtt.setObjectName("secondaryBtn"); self.bvtt.setEnabled(False)
        self.btxt=QPushButton("📄 TXT"); self.btxt.setObjectName("secondaryBtn"); self.btxt.setEnabled(False)
        self.bjsn=QPushButton("{ } JSON"); self.bjsn.setObjectName("secondaryBtn"); self.bjsn.setEnabled(False)
        self.bcpy=QPushButton("📋 Copy"); self.bcpy.setObjectName("secondaryBtn"); self.bcpy.setEnabled(False)
        self.breview=QPushButton("✎ Apply Review"); self.breview.setObjectName("accentBtn"); self.breview.setEnabled(False)
        self.bproject=QPushButton("📦 Project"); self.bproject.setObjectName("secondaryBtn"); self.bproject.setEnabled(False)
        for b in [self.bsrt,self.bvtt,self.btxt,self.bjsn,self.bcpy,self.breview,self.bproject]: er.addWidget(b)
        rl.addLayout(er); tabs.addTab(rw, "📝 Results")

        # Log
        lw = QWidget(); ll2 = QVBoxLayout(lw); ll2.setContentsMargins(8,8,8,8)
        self.logw = QPlainTextEdit(); self.logw.setReadOnly(True)
        self.logw.setStyleSheet(f"font-family:'Consolas','Cascadia Code',monospace;font-size:12px;background-color:{C['crust']};")
        ll2.addWidget(self.logw); tabs.addTab(lw, "📋 Log")

        # Settings
        sw = QWidget(); sl = QVBoxLayout(sw); sl.setContentsMargins(12,12,12,12); sl.setSpacing(10)

        bg = QGroupBox("Inference Backend"); bgl = QVBoxLayout(bg)
        self.be_combo = QComboBox(); self.be_combo.addItems(BACKENDS)
        self.be_combo.setCurrentIndex(self.cfg.get('backend_index',0))
        self.be_combo.currentIndexChanged.connect(self._on_be)
        bgl.addWidget(self.be_combo); sl.addWidget(bg)

        self.stack = QStackedWidget()

        # P0: HF
        p0 = QWidget(); p0l = QVBoxLayout(p0); p0l.setContentsMargins(0,4,0,0)
        h0 = QLabel("🤗 Free — no signup. Connects to public Gradio Spaces.\nSpaces may sleep; wake-up takes 30-60s.")
        h0.setObjectName("dimLabel"); h0.setWordWrap(True); p0l.addWidget(h0)
        p0l.addWidget(QLabel("Custom Space URL (optional):"))
        self.hf_url = QLineEdit(); self.hf_url.setPlaceholderText("https://user-space.hf.space")
        self.hf_url.setText(self.cfg.get('hf_space_url','')); p0l.addWidget(self.hf_url); p0l.addStretch()
        self.stack.addWidget(p0)

        # P1: Local
        p1 = QWidget(); p1l = QVBoxLayout(p1); p1l.setContentsMargins(0,4,0,0)
        h1 = QLabel("💻 Free & offline. Auto-downloads PyTorch + model (~4GB).\nGPU recommended. First run takes several minutes.")
        h1.setObjectName("dimLabel"); h1.setWordWrap(True); p1l.addWidget(h1)
        self.b_dl = QPushButton("📥  Pre-Download Model"); self.b_dl.setObjectName("accentBtn")
        self.b_dl.clicked.connect(self._pre_dl); p1l.addWidget(self.b_dl); p1l.addStretch()
        self.stack.addWidget(p1)

        # P2: Replicate
        p2 = QWidget(); p2l = QVBoxLayout(p2); p2l.setContentsMargins(0,4,0,0)
        p2l.addWidget(QLabel("API Token:"))
        self.rep_tok = QLineEdit(); self.rep_tok.setPlaceholderText("r8_xxx"); self.rep_tok.setEchoMode(QLineEdit.EchoMode.Password)
        self.rep_tok.setText(self.cfg.get('replicate_api_token','')); p2l.addWidget(self.rep_tok)
        rh = QLabel("Get token at replicate.com/account/api-tokens"); rh.setObjectName("dimLabel"); p2l.addWidget(rh); p2l.addStretch()
        self.stack.addWidget(p2)

        # P3: Custom
        p3 = QWidget(); p3l = QVBoxLayout(p3); p3l.setContentsMargins(0,4,0,0)
        p3l.addWidget(QLabel("URL:")); self.ep_url = QLineEdit(); self.ep_url.setPlaceholderText("https://..."); self.ep_url.setText(self.cfg.get('custom_endpoint','')); p3l.addWidget(self.ep_url)
        p3l.addWidget(QLabel("Key:")); self.ep_key = QLineEdit(); self.ep_key.setEchoMode(QLineEdit.EchoMode.Password); self.ep_key.setText(self.cfg.get('custom_endpoint_key','')); p3l.addWidget(self.ep_key); p3l.addStretch()
        self.stack.addWidget(p3)

        sl.addWidget(self.stack); self.stack.setCurrentIndex(self.cfg.get('backend_index',0))

        sg = QGroupBox("Segmentation"); sgl = QVBoxLayout(sg)
        self.chk_seg = QCheckBox("Auto-segment by mouth movement"); self.chk_seg.setChecked(self.cfg.get('auto_segment',True)); sgl.addWidget(self.chk_seg)
        self.chk_multi = QCheckBox("Label multiple speakers (A, B, ...)"); self.chk_multi.setChecked(self.cfg.get('multi_speaker', False)); sgl.addWidget(self.chk_multi)
        sl.addWidget(sg)

        self.b_save = QPushButton("💾  Save Settings"); sl.addWidget(self.b_save); sl.addStretch()
        tabs.addTab(sw, "⚙️ Settings")
        bl.addWidget(tabs, stretch=4); root.addWidget(body, stretch=1)
        self.statusBar().showMessage("Ready — load a video to begin")

    def _mk(self, lbl):
        w = QWidget(); w.setFixedWidth(80); l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(0)
        v = QLabel("—"); v.setAlignment(Qt.AlignmentFlag.AlignCenter); v.setStyleSheet(f"font-size:15px;font-weight:bold;color:{C['blue']};"); l.addWidget(v)
        b = QLabel(lbl); b.setAlignment(Qt.AlignmentFlag.AlignCenter); b.setStyleSheet(f"font-size:10px;color:{C['overlay1']};"); l.addWidget(b)
        w._v = v; return w

    def _sv(self, w, val): w._v.setText(str(val))

    def _connect(self):
        self.b_load.clicked.connect(self._load); self.b_analyze.clicked.connect(self._analyze)
        self.b_process.clicked.connect(self._process); self.b_cancel.clicked.connect(self._cancel)
        self.b_save.clicked.connect(self._save); self.slider.valueChanged.connect(self._scrub)
        self.bsrt.clicked.connect(lambda: self._export('srt')); self.btxt.clicked.connect(lambda: self._export('txt'))
        self.bvtt.clicked.connect(lambda: self._export('vtt')); self.bjsn.clicked.connect(lambda: self._export('json'))
        self.bcpy.clicked.connect(self._copy); self.breview.clicked.connect(self._apply_review)
        self.bproject.clicked.connect(self._save_project); self.curve.segments_changed.connect(self._segments_changed)

    def _on_be(self, i): self.stack.setCurrentIndex(i); self.badge.setText(BACKENDS[i])

    def _load(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video (*.mp4 *.mov *.avi *.mkv *.webm);;All (*)")
        if not p: return
        try:
            self.video_path = p; self.results = []; self.segments = []; self.review_edits = []; self._curve_data = []
            self.tbl.setRowCount(0); self.txte.clear(); self.curve.set_data(); self.roi_preview.clear(); self._exp(False)
            cap = cv2.VideoCapture(p)
            if not cap.isOpened(): self._log("❌ Can't open"); return
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); dur = frames/fps; cap.release()
            self.video_info = {'fps':fps,'frames':frames,'w':w,'h':h,'dur':dur}
            self.slider.setMaximum(max(0,frames-1)); self.slider.setValue(0); self.d_lbl.setText(self._ft(dur))
            self._sv(self.sf,f"{frames:,}"); self._sv(self.sfps,f"{fps:.1f}"); self._sv(self.sres,f"{w}x{h}"); self._sv(self.sseg,"—"); self._sv(self.smo,"—")
            self.b_analyze.setEnabled(True); self.b_process.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(p)}")
            self._log(f"📂 {os.path.basename(p)} — {w}x{h} @ {fps:.1f}fps — {frames:,} frames — {self._ft(dur)}")
            self._lf(0)
        except Exception as e: self._log(f"❌ {e}")

    def _lf(self, n):
        if not self.video_path: return
        if self._fw and self._fw.isRunning(): return
        self._fw = FrameWorker(self.video_path, n)
        self._fw.frame_ready.connect(self._of); self._fw.start()

    def _of(self, img, ts, info, roi_img):
        self.preview.set_frame(img)
        if not roi_img.isNull(): self.roi_preview.set_frame(roi_img)
        self.t_lbl.setText(self._ft(ts))
        self._sv(self.smo, f"{info.get('open_ratio',0):.2f}")

    def _scrub(self, v): self._lf(v)

    def _analyze(self):
        if not self.video_path: return
        self.b_analyze.setEnabled(False); self.b_process.setEnabled(False); self.prog.setValue(0)
        self._log("🔍 Analyzing...")
        self._curve_data = []
        self._sw = SegmentWorker(
            self.video_path,
            threshold=float(self.cfg.get('mouth_threshold', 0.06)),
            multi_speaker=self.chk_multi.isChecked(),
        )
        self._sw.progress.connect(self.prog.setValue); self._sw.log.connect(self._log)
        self._sw.curve.connect(self._oc)
        self._sw.finished.connect(self._os)
        self._sw.error.connect(lambda m: (self._log(f"❌ {m}"), self.b_analyze.setEnabled(True), self.b_process.setEnabled(True)))
        self._sw.start()

    def _os(self, segs):
        self.segments = normalize_segments(segs); self._sv(self.sseg, len(self.segments))
        self.curve.set_data(self._curve_data, self.segments)
        self.b_analyze.setEnabled(True); self.b_process.setEnabled(True); self.prog.setValue(100)
        for i, segment in enumerate(self.segments):
            speaker = f" {segment['speaker']}" if segment.get('speaker') else ''
            self._log(f"   [{i+1}{speaker}] {self._ft(segment['start'])} → {self._ft(segment['end'])} ({segment['end']-segment['start']:.1f}s)")
        Toast(self, f"  ✅  {len(self.segments)} segments  ", C['green'])

    def _process(self):
        if not self.video_path: return
        be = self._get_be()
        if not be: return
        self.b_process.setEnabled(False); self.b_analyze.setEnabled(False); self.b_cancel.setEnabled(True)
        self.prog.setValue(0); self.results=[]; self.tbl.setRowCount(0); self.txte.clear(); self._exp(False)
        segs = self.segments if (self.chk_seg.isChecked() and len(self.segments)>1) else None
        self._log(f"🧠 Lip reading via {BACKENDS[self.be_combo.currentIndex()]}...")
        self._pw = ProcessingWorker(self.video_path, be, segs)
        self._pw.progress.connect(self.prog.setValue); self._pw.log.connect(self._log)
        self._pw.segment_result.connect(self._or); self._pw.finished.connect(self._od)
        self._pw.error.connect(self._oe); self._pw.start()

    def _cancel(self):
        if self._pw: self._pw.cancel(); self._log("⏹ Cancelling...")
        self.b_cancel.setEnabled(False)

    def _get_be(self):
        i = self.be_combo.currentIndex()
        if i == 0: return HuggingFaceSpaceBackend(self.hf_url.text().strip())
        if i == 1: return LocalAutoAVSRBackend()
        if i == 2:
            t = self.rep_tok.text().strip()
            if not t: Toast(self,"  ⚠️  Set token in Settings  ",C['peach']); return None
            return ReplicateBackend(t)
        u = self.ep_url.text().strip()
        if not u: Toast(self,"  ⚠️  Set URL in Settings  ",C['peach']); return None
        return DirectAPIBackend(u, self.ep_key.text().strip())

    def _or(self, r):
        row = self.tbl.rowCount(); self.tbl.insertRow(row)
        self.tbl.setItem(row,0,QTableWidgetItem(f"{self._ft(r['start'])} → {self._ft(r['end'])}"))
        self.tbl.setItem(row,1,QTableWidgetItem(f"{r['end']-r['start']:.1f}s"))
        self.tbl.setItem(row,2,QTableWidgetItem(r.get('speaker','A')))
        confidence = r.get('confidence')
        confidence_item = QTableWidgetItem(f"{confidence:.0%}" if confidence is not None else "—")
        if confidence is not None:
            confidence_item.setForeground(QColor(C['green'] if confidence >= 0.75 else C['peach'] if confidence >= 0.5 else C['red']))
        self.tbl.setItem(row,3,confidence_item)
        self.tbl.setItem(row,4,QTableWidgetItem(r['text'])); self.tbl.scrollToBottom()

    def _od(self, res):
        self.results = res; self.b_process.setEnabled(True); self.b_analyze.setEnabled(True); self.b_cancel.setEnabled(False); self._exp(True)
        full = '\n'.join(r['text'] for r in res if r['text']); self.txte.setPlainText(full)
        self.statusBar().showMessage(f"Done — {len(res)} seg(s), {len(full.split())} words")
        Toast(self, f"  ✅  {len(res)} segments transcribed  ", C['green'])

    def _oe(self, msg):
        self._log(f"❌ {msg}"); self.b_process.setEnabled(True); self.b_analyze.setEnabled(True); self.b_cancel.setEnabled(False)
        Toast(self, f"  ❌  {msg[:80]}  ", C['red'])

    def _export(self, fmt):
        if not self.results: return
        base = Path(self.video_path).stem if self.video_path else "lipsight"
        p, _ = QFileDialog.getSaveFileName(self, f"Export", f"{base}_lipsight.{fmt}",
            {"srt":"SRT (*.srt)","vtt":"WebVTT (*.vtt)","txt":"Text (*.txt)","json":"JSON (*.json)"}.get(fmt,"*"))
        if not p: return
        try:
            export_results(self.results, p, fmt=fmt, metadata={'video_path': self.video_path})
            self._log(f"💾 {p}"); Toast(self, "  💾  Exported  ", C['green'])
        except Exception as e: self._log(f"❌ {e}"); Toast(self, "  ❌  Failed  ", C['red'])

    def _copy(self):
        t = self.txte.toPlainText()
        if t: QApplication.clipboard().setText(t); Toast(self, "  📋  Copied  ", C['green'])

    def _exp(self, on):
        for b in [self.bsrt, self.bvtt, self.btxt, self.bjsn, self.bcpy, self.breview, self.bproject]: b.setEnabled(on)

    def _save(self):
        self.cfg.update({
            'backend_index': self.be_combo.currentIndex(),
            'hf_space_url': self.hf_url.text().strip(),
            'replicate_api_token': self.rep_tok.text().strip(),
            'custom_endpoint': self.ep_url.text().strip(),
            'custom_endpoint_key': self.ep_key.text().strip(),
            'auto_segment': self.chk_seg.isChecked(),
            'multi_speaker': self.chk_multi.isChecked(),
        })
        save_config(self.cfg); Toast(self, "  ✅  Saved  ", C['green']); self._log("💾 Settings saved")

    def _oc(self, curve):
        self._curve_data = curve

    def _segments_changed(self, segments):
        self.segments = normalize_segments(segments)
        self._sv(self.sseg, len(self.segments))

    def _apply_review(self):
        if not self.results:
            return
        self.results, edits = apply_review_text(self.results, self.txte.toPlainText())
        self.review_edits.extend(edits)
        self.tbl.setRowCount(0)
        for result in self.results:
            self._or(result)
        self.txte.setPlainText('\n'.join(result['text'] for result in self.results if result['text']))
        self._log(f"✎ Applied {len(edits)} review edit(s)")
        Toast(self, f"  ✅  Review applied  ", C['green'])

    def _save_project(self):
        if not self.video_path:
            return
        base = Path(self.video_path).stem
        path, _ = QFileDialog.getSaveFileName(self, "Save LipSight Project", f"{base}.lipsight", "LipSight Project (*.lipsight)")
        if not path:
            return
        try:
            saved = save_project(path, self.video_path, self.segments, self.results, self.review_edits, metadata={'backend': BACKENDS[self.be_combo.currentIndex()]})
            self._log(f"📦 {saved}")
            Toast(self, "  ✅  Project saved  ", C['green'])
        except Exception as exc:
            self._log(f"❌ {exc}")
            Toast(self, "  ❌  Project failed  ", C['red'])

    def _pre_dl(self):
        self._log("📥 Pre-downloading local model..."); self.b_dl.setEnabled(False); self.b_dl.setText("⏳ Downloading...")
        class W(QThread):
            log=pyqtSignal(str); done=pyqtSignal(bool,str)
            def run(s):
                try: LocalAutoAVSRBackend()._ensure_setup(s.log.emit); s.done.emit(True,"")
                except Exception as e: s.done.emit(False,str(e))
        w = W(); w.log.connect(self._log)
        def fin(ok,msg):
            self.b_dl.setEnabled(True)
            if ok: self.b_dl.setText("✅  Model Ready"); Toast(self,"  ✅  Ready  ",C['green'])
            else: self.b_dl.setText("📥  Pre-Download Model"); self._log(f"❌ {msg}"); Toast(self,f"  ❌  {msg[:60]}  ",C['red'])
        w.done.connect(fin); w.start(); self._dlw = w

    def _log(self, msg): self.logw.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    @staticmethod
    def _ft(s): m=int(s)//60; sec=s-m*60; return f"{m:02d}:{sec:06.3f}"


def build_cli_parser():
    parser = argparse.ArgumentParser(
        prog='lipsight',
        description='Transcribe speech from silent video using LipSight.',
    )
    parser.add_argument('--input', help='input video path')
    parser.add_argument('--output', help='output SRT/VTT/TXT/JSON path')
    parser.add_argument('--output-format', choices=('srt', 'vtt', 'txt', 'json'), help='override output format')
    parser.add_argument('--backend', default='hf', help='hf, local, replicate, or custom')
    parser.add_argument('--hf-url', default='', help='custom HuggingFace Space URL')
    parser.add_argument('--replicate-token', default='', help='Replicate API token')
    parser.add_argument('--endpoint-url', default='', help='custom inference endpoint')
    parser.add_argument('--endpoint-key', default='', help='custom endpoint bearer key')
    parser.add_argument('--no-segmentation', action='store_true', help='process the full video as one segment')
    parser.add_argument('--threshold', type=float, default=0.06, help='mouth movement threshold for segmentation')
    parser.add_argument('--project', help='also save a .lipsight project')
    parser.add_argument('--embed-video', action='store_true', help='embed input video in the project bundle')
    parser.add_argument('--archive', help='JSONL session archive path')
    parser.add_argument('--watch', help='watch a folder for new videos')
    parser.add_argument('--watch-output', help='output folder for watch-mode subtitles')
    parser.add_argument('--interval', type=float, default=5.0, help='watch polling interval in seconds')
    parser.add_argument('--once', action='store_true', help='scan watch folder once and exit')
    return parser


def _cli_config(args):
    return {
        'hf_space_url': args.hf_url,
        'replicate_api_token': args.replicate_token,
        'custom_endpoint': args.endpoint_url,
        'custom_endpoint_key': args.endpoint_key,
    }


def run_cli(argv=None):
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if not args.input and not args.watch:
        parser.error('--input or --watch is required for headless mode')
    if args.input and args.watch:
        parser.error('--input and --watch cannot be used together')

    backend = create_backend(args.backend, _cli_config(args))
    log_cb = lambda message: print(message, flush=True)
    if args.watch:
        watch_folder(
            args.watch,
            backend,
            output_dir=args.watch_output,
            auto_segment=not args.no_segmentation,
            threshold=args.threshold,
            interval=args.interval,
            once=args.once,
            log_cb=log_cb,
        )
        return 0

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f'input video not found: {input_path}')
    segments = None
    if not args.no_segmentation:
        print('🔍 Analyzing mouth movement...', flush=True)
        segments = VideoSegmenter(args.threshold).segment(str(input_path), log_cb=log_cb)
    results = process_video(str(input_path), backend, segments=segments, log_cb=log_cb)
    metadata = {'video_path': str(input_path), 'backend': args.backend}

    if args.output:
        output = export_results(results, args.output, fmt=args.output_format, metadata=metadata)
        print(f'💾 Exported {output}', flush=True)
    else:
        print(json.dumps(build_result_document(results, metadata), ensure_ascii=False, indent=2), flush=True)
    if args.project:
        project = save_project(
            args.project,
            video_path=input_path,
            segments=segments,
            results=results,
            include_video=args.embed_video,
            metadata=metadata,
        )
        print(f'📦 Saved {project}', flush=True)
    if args.archive:
        SessionArchive(args.archive).record(str(input_path), results, args.backend, metadata)
    return 0


# ── Entry ───────────────────────────────────────────────────────────────────
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv:
        return run_cli(argv)

    import traceback as _tb
    def _exc(t,v,tb):
        msg=''.join(_tb.format_exception(t,v,tb))
        f=os.path.join(get_config_dir(),'crash.log')
        try:
            with open(f,'w') as fh: fh.write(msg)
        except: pass
        print(f"\n{'='*60}\n{APP_NAME} Crash\n{'='*60}\n{msg}")
        if sys.platform=='win32':
            try: import ctypes; ctypes.windll.user32.MessageBoxW(0,f"Log: {f}\n\n{msg[:500]}",f"{APP_NAME} Error",0x10)
            except: pass
        sys.__excepthook__(t,v,tb)
    sys.excepthook = _exc

    app = QApplication(sys.argv)

    branding_icon = QIcon(str(_branding_icon_path()))

    app.setWindowIcon(branding_icon)
    app.setStyle("Fusion"); app.setStyleSheet(DARK_STYLE)
    font = app.font(); font.setFamily("Segoe UI"); font.setPointSize(10); app.setFont(font)
    w = LipSightWindow(); w.show()
    w.setWindowIcon(branding_icon)
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
