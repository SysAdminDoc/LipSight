#!/usr/bin/env python3
"""
LipSight v1.0 — AI Lip Reading Tool
Powered by Auto-AVSR (state-of-the-art visual speech recognition)
Uses Replicate API for cloud inference with local face/mouth detection
"""

import sys, os, subprocess, json, time, tempfile, threading, math, struct, re
from pathlib import Path
from datetime import timedelta

# ── Auto-Bootstrap ──────────────────────────────────────────────────────────
def _bootstrap():
    if sys.version_info < (3, 8):
        print("Python 3.8+ required"); sys.exit(1)
    required = ['PyQt6', 'opencv-python', 'mediapipe', 'requests', 'numpy']
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

_bootstrap()

import cv2
import numpy as np
import mediapipe as mp
import requests
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# ── Version ─────────────────────────────────────────────────────────────────
APP_NAME = "LipSight"
APP_VERSION = "1.0.0"

# ── Catppuccin Mocha Theme ──────────────────────────────────────────────────
COLORS = {
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
QMainWindow, QWidget {{ background-color: {COLORS['base']}; color: {COLORS['text']}; }}
QMenuBar {{ background-color: {COLORS['mantle']}; color: {COLORS['text']}; }}
QMenuBar::item:selected {{ background-color: {COLORS['surface0']}; }}
QMenu {{ background-color: {COLORS['base']}; border: 1px solid {COLORS['surface1']}; }}
QMenu::item:selected {{ background-color: {COLORS['surface0']}; }}
QPushButton {{
    background-color: {COLORS['blue']}; color: {COLORS['base']}; border: none;
    padding: 8px 18px; border-radius: 6px; font-weight: bold; font-size: 13px;
}}
QPushButton:hover {{ background-color: {COLORS['sky']}; }}
QPushButton:pressed {{ background-color: {COLORS['lavender']}; }}
QPushButton:disabled {{ background-color: {COLORS['surface1']}; color: {COLORS['overlay0']}; }}
QPushButton#dangerBtn {{ background-color: {COLORS['red']}; }}
QPushButton#dangerBtn:hover {{ background-color: {COLORS['flamingo']}; }}
QPushButton#secondaryBtn {{ background-color: {COLORS['surface0']}; color: {COLORS['text']}; }}
QPushButton#secondaryBtn:hover {{ background-color: {COLORS['surface1']}; }}
QPushButton#accentBtn {{ background-color: {COLORS['mauve']}; }}
QPushButton#accentBtn:hover {{ background-color: {COLORS['lavender']}; }}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['surface0']}; color: {COLORS['text']};
    border: 1px solid {COLORS['surface1']}; border-radius: 6px; padding: 8px;
    selection-background-color: {COLORS['blue']}; selection-color: {COLORS['base']};
    font-size: 13px;
}}
QLineEdit:focus, QTextEdit:focus {{ border: 1px solid {COLORS['blue']}; }}
QComboBox {{
    background-color: {COLORS['surface0']}; color: {COLORS['text']};
    border: 1px solid {COLORS['surface1']}; border-radius: 6px; padding: 8px;
    font-size: 13px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background-color: {COLORS['base']}; color: {COLORS['text']};
    border: 1px solid {COLORS['surface1']}; selection-background-color: {COLORS['blue']};
}}
QLabel {{ color: {COLORS['text']}; font-size: 13px; }}
QLabel#header {{ font-size: 22px; font-weight: bold; color: {COLORS['blue']}; }}
QLabel#subheader {{ font-size: 14px; color: {COLORS['subtext0']}; }}
QLabel#sectionTitle {{ font-size: 15px; font-weight: bold; color: {COLORS['lavender']}; }}
QLabel#dimLabel {{ color: {COLORS['overlay0']}; font-size: 12px; }}
QLabel#statValue {{ font-size: 28px; font-weight: bold; color: {COLORS['blue']}; }}
QLabel#statLabel {{ font-size: 11px; color: {COLORS['overlay1']}; text-transform: uppercase; }}
QGroupBox {{
    border: 1px solid {COLORS['surface1']}; border-radius: 10px;
    margin-top: 1.2em; padding: 16px 12px 12px 12px; color: {COLORS['text']};
    font-weight: bold; font-size: 13px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 8px; color: {COLORS['lavender']}; }}
QProgressBar {{
    background-color: {COLORS['surface0']}; border: none; border-radius: 5px;
    text-align: center; color: {COLORS['text']}; font-size: 12px; min-height: 10px;
}}
QProgressBar::chunk {{ background-color: {COLORS['blue']}; border-radius: 5px; }}
QSlider::groove:horizontal {{
    height: 6px; background: {COLORS['surface0']}; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['blue']}; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {COLORS['sky']}; }}
QSlider::sub-page:horizontal {{ background: {COLORS['blue']}; border-radius: 3px; }}
QScrollBar:vertical {{
    background: {COLORS['mantle']}; width: 8px; border: none; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['surface1']}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {COLORS['surface2']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTabWidget::pane {{ border: 1px solid {COLORS['surface1']}; background: {COLORS['base']}; border-radius: 8px; }}
QTabBar::tab {{
    background: {COLORS['mantle']}; color: {COLORS['overlay0']}; padding: 10px 20px;
    border: none; font-size: 13px; font-weight: bold;
}}
QTabBar::tab:selected {{ color: {COLORS['text']}; border-bottom: 2px solid {COLORS['blue']}; }}
QTabBar::tab:hover {{ color: {COLORS['subtext1']}; }}
QTableWidget {{
    background-color: {COLORS['base']}; alternate-background-color: {COLORS['mantle']};
    color: {COLORS['text']}; border: 1px solid {COLORS['surface1']};
    gridline-color: {COLORS['surface0']}; font-size: 13px; border-radius: 6px;
}}
QTableWidget::item:selected {{ background-color: {COLORS['blue']}; color: {COLORS['base']}; }}
QHeaderView::section {{
    background-color: {COLORS['mantle']}; color: {COLORS['subtext0']};
    border: none; border-bottom: 1px solid {COLORS['surface1']}; padding: 8px;
    font-weight: bold; font-size: 12px;
}}
QStatusBar {{ background-color: {COLORS['mantle']}; color: {COLORS['overlay0']}; font-size: 12px; }}
QToolTip {{
    background-color: {COLORS['surface0']}; color: {COLORS['text']};
    border: 1px solid {COLORS['surface1']}; padding: 6px; border-radius: 6px;
}}
QSplitter::handle {{ background: {COLORS['surface1']}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
"""

# ── Config ──────────────────────────────────────────────────────────────────
def get_config_dir():
    base = os.environ.get('APPDATA', os.path.expanduser('~'))
    path = os.path.join(base, '.lipsight')
    os.makedirs(path, exist_ok=True)
    return path

def load_config():
    cfg_file = os.path.join(get_config_dir(), 'config.json')
    try:
        with open(cfg_file) as f: return json.load(f)
    except: return {}

def save_config(cfg):
    cfg_file = os.path.join(get_config_dir(), 'config.json')
    with open(cfg_file, 'w') as f: json.dump(cfg, f, indent=2)

# ── Face/Mouth Detection ───────────────────────────────────────────────────
class FaceAnalyzer:
    """MediaPipe-based face mesh for mouth ROI detection and visualization."""

    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    LIP_TOP = [13]
    LIP_BOTTOM = [14]

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def analyze_frame(self, frame):
        """Returns (annotated_frame, mouth_roi, mouth_open_ratio, landmarks)."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        annotated = frame.copy()
        mouth_roi = None
        open_ratio = 0.0
        landmarks = None

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]
            landmarks = face_lm

            # Extract mouth region
            mouth_pts = []
            for idx in self.MOUTH_OUTER:
                lm = face_lm.landmark[idx]
                mouth_pts.append((int(lm.x * w), int(lm.y * h)))

            if mouth_pts:
                pts = np.array(mouth_pts)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                pad = 20
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)
                mouth_roi = (x_min, y_min, x_max, y_max)

                # Draw mouth ROI
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (137, 180, 250), 2)

                # Draw mouth landmarks
                for idx in self.MOUTH_OUTER:
                    lm = face_lm.landmark[idx]
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (px, py), 2, (166, 227, 161), -1)

                for idx in self.MOUTH_INNER:
                    lm = face_lm.landmark[idx]
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (px, py), 2, (180, 190, 254), -1)

                # Calculate mouth open ratio
                top = face_lm.landmark[13]
                bottom = face_lm.landmark[14]
                left = face_lm.landmark[61]
                right = face_lm.landmark[291]
                mouth_height = math.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
                mouth_width = math.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
                open_ratio = mouth_height / max(mouth_width, 0.001)

                # Draw face outline (jawline)
                jaw_indices = list(range(0, 17)) + [234, 127, 162, 21, 54, 103, 67, 109]
                for idx in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]:
                    lm = face_lm.landmark[idx]
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (px, py), 1, (69, 71, 90), -1)

                # Status indicator
                status_color = (166, 227, 161) if open_ratio > 0.06 else (108, 112, 134)
                status_text = "SPEAKING" if open_ratio > 0.06 else "SILENT"
                cv2.putText(annotated, status_text, (x_min, y_min - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        return annotated, mouth_roi, open_ratio, landmarks

    def close(self):
        self.face_mesh.close()


# ── Video Segmenter ─────────────────────────────────────────────────────────
class VideoSegmenter:
    """Segments video into speech/silence regions based on mouth movement analysis."""

    def __init__(self, min_speech_duration=0.5, min_silence_duration=0.3, open_threshold=0.06):
        self.min_speech_frames = int(min_speech_duration * 25)
        self.min_silence_frames = int(min_silence_duration * 25)
        self.open_threshold = open_threshold

    def segment(self, video_path, progress_cb=None, log_cb=None):
        """Analyze video and return speech segments as [(start_sec, end_sec), ...]."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        analyzer = FaceAnalyzer()

        if log_cb: log_cb(f"📐 Analyzing {total_frames} frames at {fps:.1f} fps...")

        open_ratios = []
        frame_idx = 0
        sample_rate = max(1, int(fps / 10))  # Sample ~10 frames/sec for speed

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                _, _, ratio, _ = analyzer.analyze_frame(frame)
                open_ratios.append((frame_idx, ratio))
            frame_idx += 1
            if progress_cb and frame_idx % 50 == 0:
                progress_cb(int(frame_idx / max(total_frames, 1) * 100))

        cap.release()
        analyzer.close()

        if not open_ratios:
            return [(0.0, total_frames / fps)]

        # Detect speech segments
        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0

        for frame_num, ratio in open_ratios:
            if ratio > self.open_threshold:
                if not in_speech:
                    speech_start = frame_num
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += sample_rate
                    if silence_count >= self.min_silence_frames:
                        end_frame = frame_num - silence_count
                        if (end_frame - speech_start) >= self.min_speech_frames:
                            segments.append((speech_start / fps, end_frame / fps))
                        in_speech = False
                        silence_count = 0

        # Close final segment
        if in_speech:
            end_frame = open_ratios[-1][0]
            if (end_frame - speech_start) >= self.min_speech_frames:
                segments.append((speech_start / fps, end_frame / fps))

        if log_cb: log_cb(f"🔍 Found {len(segments)} speech segments")

        # If no segments detected, return whole video as one segment
        if not segments:
            segments = [(0.0, total_frames / fps)]

        return segments


# ── Inference Backends ──────────────────────────────────────────────────────
class ReplicateBackend:
    """Cloud inference via Replicate API (Auto-AVSR model) using direct HTTP requests."""

    API_BASE = "https://api.replicate.com/v1"
    MODEL_VERSION = "basord/lip-reading-ai-vsr"

    def __init__(self, api_token):
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
        }

    def _upload_file(self, file_path):
        """Upload file to Replicate's file hosting and return the serving URL."""
        # Create upload URL
        resp = requests.post(
            f"{self.API_BASE}/files",
            headers={
                "Authorization": f"Token {self.api_token}",
            },
            files={"content": (os.path.basename(file_path), open(file_path, "rb"), "video/mp4")},
            data={"content_type": "video/mp4"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        serving_url = data.get("urls", {}).get("get", "")
        if not serving_url:
            serving_url = data.get("url", "")
        if not serving_url:
            raise RuntimeError(f"Upload succeeded but no serving URL returned: {data}")
        return serving_url

    def _get_latest_version(self):
        """Resolve the latest version ID for the model."""
        resp = requests.get(
            f"{self.API_BASE}/models/{self.MODEL_VERSION}/versions",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        versions = resp.json().get("results", [])
        if not versions:
            raise RuntimeError(f"No versions found for model {self.MODEL_VERSION}")
        return versions[0]["id"]

    def transcribe(self, video_path, log_cb=None):
        """Send video to Replicate and return transcription text."""
        if log_cb: log_cb("☁️  Uploading video to Replicate...")

        try:
            # Upload the file first
            file_url = self._upload_file(video_path)
            if log_cb: log_cb(f"☁️  File uploaded, resolving model version...")

            # Get latest model version
            version_id = self._get_latest_version()
            if log_cb: log_cb(f"☁️  Creating prediction (version {version_id[:12]}...)...")

            # Create prediction
            resp = requests.post(
                f"{self.API_BASE}/predictions",
                headers=self.headers,
                json={
                    "version": version_id,
                    "input": {"video": file_url},
                },
                timeout=30,
            )
            resp.raise_for_status()
            prediction = resp.json()
            pred_id = prediction["id"]
            get_url = prediction.get("urls", {}).get("get", f"{self.API_BASE}/predictions/{pred_id}")

            if log_cb: log_cb(f"☁️  Prediction {pred_id} queued, polling for result...")

            # Poll for completion
            max_wait = 300  # 5 minutes
            poll_interval = 2
            elapsed = 0
            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                resp = requests.get(get_url, headers=self.headers, timeout=30)
                resp.raise_for_status()
                prediction = resp.json()
                status = prediction.get("status", "")

                if status == "succeeded":
                    output = prediction.get("output", "")
                    if isinstance(output, dict):
                        return output.get("text", output.get("transcription", str(output)))
                    elif isinstance(output, list):
                        return " ".join(str(o) for o in output)
                    return str(output) if output else "(no output)"

                elif status == "failed":
                    error_msg = prediction.get("error", "Unknown error")
                    raise RuntimeError(f"Prediction failed: {error_msg}")

                elif status == "canceled":
                    raise RuntimeError("Prediction was canceled")

                if log_cb and elapsed % 10 == 0:
                    log_cb(f"☁️  Still processing... ({elapsed}s elapsed, status: {status})")

            raise RuntimeError(f"Prediction timed out after {max_wait}s")

        except requests.exceptions.HTTPError as e:
            body = ""
            try: body = e.response.text[:300]
            except: pass
            raise RuntimeError(f"Replicate API HTTP {e.response.status_code}: {body}")
        except Exception as e:
            if "Replicate" in str(type(e).__name__) or "RuntimeError" in str(type(e).__name__):
                raise
            raise RuntimeError(f"Replicate API error: {e}")


class DirectAPIBackend:
    """Direct HTTP-based inference for custom endpoints."""

    def __init__(self, endpoint_url, api_key=""):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def transcribe(self, video_path, log_cb=None):
        if log_cb: log_cb(f"🌐 Sending to {self.endpoint_url}...")
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        with open(video_path, 'rb') as f:
            resp = requests.post(
                self.endpoint_url,
                files={'video': (os.path.basename(video_path), f, 'video/mp4')},
                headers=headers, timeout=300
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get('text', data.get('transcription', str(data)))


# ── Export Utilities ────────────────────────────────────────────────────────
def format_srt_time(seconds):
    td = timedelta(seconds=seconds)
    total_secs = int(td.total_seconds())
    ms = int((td.total_seconds() - total_secs) * 1000)
    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

def export_srt(results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, r in enumerate(results, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(r['start'])} --> {format_srt_time(r['end'])}\n")
            f.write(f"{r['text']}\n\n")

def export_txt(results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for r in results:
            ts = f"[{format_srt_time(r['start'])} -> {format_srt_time(r['end'])}]"
            f.write(f"{ts} {r['text']}\n")

def export_json(results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'version': APP_VERSION}, f, indent=2)


# ── Worker Threads ──────────────────────────────────────────────────────────
class ProcessingWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    segment_result = pyqtSignal(dict)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, video_path, backend, segments=None):
        super().__init__()
        self.video_path = video_path
        self.backend = backend
        self.segments = segments
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            results = []

            if self.segments and len(self.segments) > 1:
                # Process each segment separately
                temp_dir = tempfile.mkdtemp(prefix='lipsight_')
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                for i, (start, end) in enumerate(self.segments):
                    if self._cancelled:
                        break

                    self.log.emit(f"🎬 Processing segment {i+1}/{len(self.segments)} [{start:.1f}s - {end:.1f}s]")

                    # Extract segment using ffmpeg or opencv
                    seg_path = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
                    try:
                        subprocess.run([
                            'ffmpeg', '-y', '-i', self.video_path,
                            '-ss', str(start), '-t', str(end - start),
                            '-c:v', 'libx264', '-an', '-preset', 'ultrafast',
                            seg_path
                        ], capture_output=True, timeout=60)
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        # Fallback: use OpenCV to extract segment
                        seg_path = self._extract_segment_cv(self.video_path, start, end, seg_path, fps, w, h)

                    if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                        try:
                            text = self.backend.transcribe(seg_path, log_cb=self.log.emit)
                            result = {'start': start, 'end': end, 'text': text.strip(), 'segment': i+1}
                            results.append(result)
                            self.segment_result.emit(result)
                        except Exception as e:
                            self.log.emit(f"⚠️  Segment {i+1} failed: {e}")

                    self.progress.emit(int((i + 1) / len(self.segments) * 100))

                # Cleanup temp
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except: pass
            else:
                # Process entire video as one segment
                self.log.emit("🎬 Processing entire video...")
                self.progress.emit(10)
                text = self.backend.transcribe(self.video_path, log_cb=self.log.emit)
                duration = self._get_duration(self.video_path)
                result = {'start': 0.0, 'end': duration, 'text': text.strip(), 'segment': 1}
                results.append(result)
                self.segment_result.emit(result)
                self.progress.emit(100)

            self.log.emit(f"✅ Processing complete — {len(results)} segment(s) transcribed")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def _extract_segment_cv(self, video_path, start, end, out_path, fps, w, h):
        """Extract segment using OpenCV as fallback."""
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        start_frame = int(start * fps)
        end_frame = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret: break
            writer.write(frame)

        writer.release()
        cap.release()
        return out_path

    def _get_duration(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames / fps


class SegmentationWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            segmenter = VideoSegmenter()
            segments = segmenter.segment(
                self.video_path, progress_cb=self.progress.emit, log_cb=self.log.emit
            )
            self.finished.emit(segments)
        except Exception as e:
            self.error.emit(str(e))


class FrameLoaderWorker(QThread):
    frame_ready = pyqtSignal(QImage, float, dict)
    finished = pyqtSignal()

    def __init__(self, video_path, frame_num):
        super().__init__()
        self.video_path = video_path
        self.frame_num = frame_num

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        ret, frame = cap.read()
        cap.release()

        if ret:
            analyzer = FaceAnalyzer()
            annotated, mouth_roi, open_ratio, _ = analyzer.analyze_frame(frame)
            analyzer.close()

            h, w, ch = annotated.shape
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

            info = {
                'mouth_roi': mouth_roi,
                'open_ratio': open_ratio,
                'timestamp': self.frame_num / fps
            }
            self.frame_ready.emit(img, self.frame_num / fps, info)
        self.finished.emit()


# ── Video Preview Widget ────────────────────────────────────────────────────
class VideoPreview(QLabel):
    """Video frame display with aspect-ratio scaling."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setStyleSheet(f"""
            background-color: {COLORS['crust']};
            border: 1px solid {COLORS['surface1']};
            border-radius: 8px;
        """)
        self._pixmap = None
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText("📹  Load a video to begin")
        self.setStyleSheet(self.styleSheet() + f"color: {COLORS['overlay0']}; font-size: 16px;")

    def set_frame(self, qimage):
        self._pixmap = QPixmap.fromImage(qimage)
        self._update_display()

    def _update_display(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def clear_frame(self):
        self._pixmap = None
        self._show_placeholder()


# ── Toast Notification ──────────────────────────────────────────────────────
class Toast(QLabel):
    def __init__(self, parent, message, color=COLORS['green'], duration=2500):
        super().__init__(message, parent)
        self.setStyleSheet(f"""
            background-color: {COLORS['surface0']};
            color: {color};
            border: 1px solid {color};
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 13px;
            font-weight: bold;
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adjustSize()
        self.move(parent.width() // 2 - self.width() // 2, 20)
        self.show()
        self.raise_()
        QTimer.singleShot(duration, self.deleteLater)


# ── Main Window ─────────────────────────────────────────────────────────────
class LipSightWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 650)

        self.config = load_config()
        self.video_path = None
        self.video_info = {}
        self.segments = []
        self.results = []
        self._frame_worker = None
        self._seg_worker = None
        self._proc_worker = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── Header Bar ──────────────────────────────────────────────────
        header_bar = QWidget()
        header_bar.setFixedHeight(56)
        header_bar.setStyleSheet(f"background-color: {COLORS['mantle']}; border-bottom: 1px solid {COLORS['surface0']};")
        hbar_layout = QHBoxLayout(header_bar)
        hbar_layout.setContentsMargins(16, 0, 16, 0)

        logo = QLabel(f"👁️  {APP_NAME}")
        logo.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['blue']}; background: transparent; border: none;")
        hbar_layout.addWidget(logo)

        ver = QLabel(f"v{APP_VERSION}")
        ver.setStyleSheet(f"font-size: 11px; color: {COLORS['overlay0']}; background: transparent; border: none;")
        hbar_layout.addWidget(ver)

        hbar_layout.addStretch()

        model_label = QLabel("Model:")
        model_label.setStyleSheet(f"color: {COLORS['subtext0']}; font-size: 12px; background: transparent; border: none;")
        hbar_layout.addWidget(model_label)

        self.model_badge = QLabel("Auto-AVSR (Replicate)")
        self.model_badge.setStyleSheet(f"""
            background-color: {COLORS['surface0']}; color: {COLORS['teal']};
            padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;
        """)
        hbar_layout.addWidget(self.model_badge)

        main_layout.addWidget(header_bar)

        # ── Body ────────────────────────────────────────────────────────
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(12, 12, 12, 0)
        body_layout.setSpacing(12)

        # Left panel — video + controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Video preview
        self.video_preview = VideoPreview()
        left_layout.addWidget(self.video_preview, stretch=1)

        # Scrubber
        scrub_row = QHBoxLayout()
        self.time_label = QLabel("00:00.000")
        self.time_label.setStyleSheet(f"color: {COLORS['overlay1']}; font-family: monospace; font-size: 12px;")
        scrub_row.addWidget(self.time_label)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        scrub_row.addWidget(self.frame_slider, stretch=1)

        self.duration_label = QLabel("00:00.000")
        self.duration_label.setStyleSheet(f"color: {COLORS['overlay1']}; font-family: monospace; font-size: 12px;")
        scrub_row.addWidget(self.duration_label)
        left_layout.addLayout(scrub_row)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_load = QPushButton("📂  Load Video")
        btn_row.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("🔍  Analyze Segments")
        self.btn_analyze.setObjectName("accentBtn")
        self.btn_analyze.setEnabled(False)
        btn_row.addWidget(self.btn_analyze)

        self.btn_process = QPushButton("🧠  Lip Read")
        self.btn_process.setEnabled(False)
        btn_row.addWidget(self.btn_process)

        self.btn_cancel = QPushButton("⏹  Cancel")
        self.btn_cancel.setObjectName("dangerBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setFixedWidth(100)
        btn_row.addWidget(self.btn_cancel)

        left_layout.addLayout(btn_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self.stat_frames = self._make_stat("FRAMES", "—")
        self.stat_fps = self._make_stat("FPS", "—")
        self.stat_resolution = self._make_stat("RESOLUTION", "—")
        self.stat_segments = self._make_stat("SEGMENTS", "—")
        self.stat_mouth = self._make_stat("MOUTH", "—")
        stats_row.addWidget(self.stat_frames)
        stats_row.addWidget(self.stat_fps)
        stats_row.addWidget(self.stat_resolution)
        stats_row.addWidget(self.stat_segments)
        stats_row.addWidget(self.stat_mouth)
        stats_row.addStretch()
        left_layout.addLayout(stats_row)

        body_layout.addWidget(left_panel, stretch=6)

        # Right panel — tabs (Results, Log, Settings)
        right_panel = QTabWidget()
        right_panel.setMinimumWidth(360)

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(8, 8, 8, 8)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Time", "Duration", "Transcription"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        results_layout.addWidget(self.results_table)

        # Full transcript
        self.full_transcript = QTextEdit()
        self.full_transcript.setReadOnly(True)
        self.full_transcript.setMaximumHeight(120)
        self.full_transcript.setPlaceholderText("Full transcript will appear here...")
        results_layout.addWidget(self.full_transcript)

        # Export buttons
        export_row = QHBoxLayout()
        self.btn_export_srt = QPushButton("💾  SRT")
        self.btn_export_srt.setObjectName("secondaryBtn")
        self.btn_export_srt.setEnabled(False)
        export_row.addWidget(self.btn_export_srt)

        self.btn_export_txt = QPushButton("📄  TXT")
        self.btn_export_txt.setObjectName("secondaryBtn")
        self.btn_export_txt.setEnabled(False)
        export_row.addWidget(self.btn_export_txt)

        self.btn_export_json = QPushButton("{ }  JSON")
        self.btn_export_json.setObjectName("secondaryBtn")
        self.btn_export_json.setEnabled(False)
        export_row.addWidget(self.btn_export_json)

        self.btn_copy = QPushButton("📋  Copy")
        self.btn_copy.setObjectName("secondaryBtn")
        self.btn_copy.setEnabled(False)
        export_row.addWidget(self.btn_copy)

        results_layout.addLayout(export_row)
        right_panel.addTab(results_widget, "📝 Results")

        # Log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            font-family: 'Consolas', 'Cascadia Code', monospace;
            font-size: 12px;
            background-color: {COLORS['crust']};
        """)
        log_layout.addWidget(self.log_text)
        right_panel.addTab(log_widget, "📋 Log")

        # Settings tab
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(12)

        # API Settings
        api_group = QGroupBox("Replicate API")
        api_layout = QVBoxLayout(api_group)

        api_layout.addWidget(QLabel("API Token:"))
        self.api_token_input = QLineEdit()
        self.api_token_input.setPlaceholderText("r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.api_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_token_input.setText(self.config.get('replicate_api_token', ''))
        api_layout.addWidget(self.api_token_input)

        api_hint = QLabel("Get your token at replicate.com/account/api-tokens")
        api_hint.setObjectName("dimLabel")
        api_hint.setWordWrap(True)
        api_layout.addWidget(api_hint)

        settings_layout.addWidget(api_group)

        # Custom Endpoint
        endpoint_group = QGroupBox("Custom Endpoint (Optional)")
        endpoint_layout = QVBoxLayout(endpoint_group)

        endpoint_layout.addWidget(QLabel("URL:"))
        self.endpoint_input = QLineEdit()
        self.endpoint_input.setPlaceholderText("https://your-server.com/api/lip-read")
        self.endpoint_input.setText(self.config.get('custom_endpoint', ''))
        endpoint_layout.addWidget(self.endpoint_input)

        endpoint_layout.addWidget(QLabel("API Key:"))
        self.endpoint_key_input = QLineEdit()
        self.endpoint_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.endpoint_key_input.setText(self.config.get('custom_endpoint_key', ''))
        endpoint_layout.addWidget(self.endpoint_key_input)

        settings_layout.addWidget(endpoint_group)

        # Segmentation settings
        seg_group = QGroupBox("Segmentation")
        seg_layout = QVBoxLayout(seg_group)

        self.chk_auto_segment = QCheckBox("Auto-segment by mouth movement")
        self.chk_auto_segment.setChecked(self.config.get('auto_segment', True))
        self.chk_auto_segment.setStyleSheet(f"color: {COLORS['text']};")
        seg_layout.addWidget(self.chk_auto_segment)

        seg_layout.addWidget(QLabel("Mouth open threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(3)
        self.threshold_slider.setMaximum(15)
        self.threshold_slider.setValue(int(self.config.get('open_threshold', 6)))
        seg_layout.addWidget(self.threshold_slider)

        settings_layout.addWidget(seg_group)

        # Backend selector
        backend_group = QGroupBox("Inference Backend")
        backend_layout = QVBoxLayout(backend_group)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Replicate API (Cloud)", "Custom Endpoint"])
        self.backend_combo.setCurrentIndex(self.config.get('backend_index', 0))
        backend_layout.addWidget(self.backend_combo)
        settings_layout.addWidget(backend_group)

        self.btn_save_settings = QPushButton("💾  Save Settings")
        settings_layout.addWidget(self.btn_save_settings)

        settings_layout.addStretch()
        right_panel.addTab(settings_widget, "⚙️ Settings")

        body_layout.addWidget(right_panel, stretch=4)
        main_layout.addWidget(body, stretch=1)

        # ── Status Bar ──────────────────────────────────────────────────
        self.statusBar().showMessage("Ready — load a video to begin")

    def _make_stat(self, label_text, value_text):
        w = QWidget()
        w.setFixedWidth(90)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        val = QLabel(value_text)
        val.setObjectName("statValue")
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['blue']};")
        layout.addWidget(val)
        lbl = QLabel(label_text)
        lbl.setObjectName("statLabel")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"font-size: 10px; color: {COLORS['overlay1']}; text-transform: uppercase;")
        layout.addWidget(lbl)
        w._val_label = val
        return w

    def _update_stat(self, widget, value):
        widget._val_label.setText(str(value))

    def _connect_signals(self):
        self.btn_load.clicked.connect(self._load_video)
        self.btn_analyze.clicked.connect(self._analyze_segments)
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        self.btn_save_settings.clicked.connect(self._save_settings)
        self.frame_slider.valueChanged.connect(self._on_frame_scrub)
        self.btn_export_srt.clicked.connect(lambda: self._export('srt'))
        self.btn_export_txt.clicked.connect(lambda: self._export('txt'))
        self.btn_export_json.clicked.connect(lambda: self._export('json'))
        self.btn_copy.clicked.connect(self._copy_transcript)

    # ── Video Loading ───────────────────────────────────────────────────
    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        if not path:
            return

        self.video_path = path
        self.results = []
        self.segments = []
        self.results_table.setRowCount(0)
        self.full_transcript.clear()
        self._enable_exports(False)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._log("❌ Failed to open video file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()

        self.video_info = {'fps': fps, 'frames': total_frames, 'width': w, 'height': h, 'duration': duration}
        self.frame_slider.setMaximum(total_frames - 1)
        self.frame_slider.setValue(0)
        self.duration_label.setText(self._fmt_time(duration))

        self._update_stat(self.stat_frames, f"{total_frames:,}")
        self._update_stat(self.stat_fps, f"{fps:.1f}")
        self._update_stat(self.stat_resolution, f"{w}x{h}")
        self._update_stat(self.stat_segments, "—")
        self._update_stat(self.stat_mouth, "—")

        self.btn_analyze.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)} — {self._fmt_time(duration)}")
        self._log(f"📂 Loaded: {os.path.basename(path)}")
        self._log(f"   {w}x{h} @ {fps:.1f}fps — {total_frames:,} frames — {self._fmt_time(duration)}")

        # Load first frame
        self._load_frame(0)

    def _load_frame(self, frame_num):
        if not self.video_path:
            return
        self._frame_worker = FrameLoaderWorker(self.video_path, frame_num)
        self._frame_worker.frame_ready.connect(self._on_frame_loaded)
        self._frame_worker.start()

    def _on_frame_loaded(self, qimage, timestamp, info):
        self.video_preview.set_frame(qimage)
        self.time_label.setText(self._fmt_time(timestamp))
        ratio = info.get('open_ratio', 0)
        self._update_stat(self.stat_mouth, f"{ratio:.2f}")

    def _on_frame_scrub(self, value):
        self._load_frame(value)

    # ── Segmentation ────────────────────────────────────────────────────
    def _analyze_segments(self):
        if not self.video_path:
            return

        self.btn_analyze.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.progress_bar.setValue(0)
        self._log("🔍 Starting mouth movement analysis...")

        self._seg_worker = SegmentationWorker(self.video_path)
        self._seg_worker.progress.connect(self.progress_bar.setValue)
        self._seg_worker.log.connect(self._log)
        self._seg_worker.finished.connect(self._on_segments_ready)
        self._seg_worker.error.connect(self._on_segment_error)
        self._seg_worker.start()

    def _on_segments_ready(self, segments):
        self.segments = segments
        self._update_stat(self.stat_segments, str(len(segments)))
        self.btn_analyze.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(100)

        self._log(f"✅ Found {len(segments)} speech segment(s):")
        for i, (start, end) in enumerate(segments):
            self._log(f"   [{i+1}] {self._fmt_time(start)} → {self._fmt_time(end)} ({end-start:.1f}s)")

        Toast(self, f"  ✅  {len(segments)} speech segments detected  ", COLORS['green'])

    def _on_segment_error(self, msg):
        self._log(f"❌ Segmentation error: {msg}")
        self.btn_analyze.setEnabled(True)
        self.btn_process.setEnabled(True)
        Toast(self, f"  ❌  Segmentation failed  ", COLORS['red'])

    # ── Processing ──────────────────────────────────────────────────────
    def _start_processing(self):
        if not self.video_path:
            return

        # Validate backend
        backend = self._get_backend()
        if backend is None:
            return

        self.btn_process.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.results = []
        self.results_table.setRowCount(0)
        self.full_transcript.clear()
        self._enable_exports(False)

        use_segments = self.chk_auto_segment.isChecked() and len(self.segments) > 1
        segments = self.segments if use_segments else None

        self._log("🧠 Starting lip reading inference...")
        if segments:
            self._log(f"   Processing {len(segments)} segments individually")

        self._proc_worker = ProcessingWorker(self.video_path, backend, segments)
        self._proc_worker.progress.connect(self.progress_bar.setValue)
        self._proc_worker.log.connect(self._log)
        self._proc_worker.segment_result.connect(self._on_segment_result)
        self._proc_worker.finished.connect(self._on_processing_complete)
        self._proc_worker.error.connect(self._on_processing_error)
        self._proc_worker.start()

    def _cancel_processing(self):
        if self._proc_worker:
            self._proc_worker.cancel()
            self._log("⏹ Cancelling...")
        self.btn_cancel.setEnabled(False)

    def _get_backend(self):
        idx = self.backend_combo.currentIndex()
        if idx == 0:
            token = self.api_token_input.text().strip()
            if not token:
                Toast(self, "  ⚠️  Set Replicate API token in Settings  ", COLORS['peach'])
                return None
            return ReplicateBackend(token)
        else:
            url = self.endpoint_input.text().strip()
            if not url:
                Toast(self, "  ⚠️  Set custom endpoint URL in Settings  ", COLORS['peach'])
                return None
            return DirectAPIBackend(url, self.endpoint_key_input.text().strip())

    def _on_segment_result(self, result):
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        time_str = f"{self._fmt_time(result['start'])} → {self._fmt_time(result['end'])}"
        dur = result['end'] - result['start']

        self.results_table.setItem(row, 0, QTableWidgetItem(time_str))
        self.results_table.setItem(row, 1, QTableWidgetItem(f"{dur:.1f}s"))
        self.results_table.setItem(row, 2, QTableWidgetItem(result['text']))
        self.results_table.scrollToBottom()

    def _on_processing_complete(self, results):
        self.results = results
        self.btn_process.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._enable_exports(True)

        # Build full transcript
        full_text = ' '.join(r['text'] for r in results if r['text'])
        self.full_transcript.setPlainText(full_text)

        self.statusBar().showMessage(f"Complete — {len(results)} segment(s), {len(full_text.split())} words")
        Toast(self, f"  ✅  Lip reading complete — {len(results)} segments  ", COLORS['green'])

    def _on_processing_error(self, msg):
        self._log(f"❌ Processing error: {msg}")
        self.btn_process.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        Toast(self, f"  ❌  {msg[:80]}  ", COLORS['red'])

    # ── Export ──────────────────────────────────────────────────────────
    def _export(self, fmt):
        if not self.results:
            return

        base_name = Path(self.video_path).stem if self.video_path else "lipsight"
        filters = {
            'srt': "SRT Subtitles (*.srt)",
            'txt': "Text File (*.txt)",
            'json': "JSON File (*.json)",
        }

        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt.upper()}", f"{base_name}_lipsight.{fmt}",
            filters.get(fmt, "All Files (*)")
        )
        if not path:
            return

        try:
            if fmt == 'srt': export_srt(self.results, path)
            elif fmt == 'txt': export_txt(self.results, path)
            elif fmt == 'json': export_json(self.results, path)
            self._log(f"💾 Exported to {path}")
            Toast(self, f"  💾  Exported {fmt.upper()}  ", COLORS['green'])
        except Exception as e:
            self._log(f"❌ Export failed: {e}")
            Toast(self, f"  ❌  Export failed  ", COLORS['red'])

    def _copy_transcript(self):
        text = self.full_transcript.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            Toast(self, "  📋  Copied to clipboard  ", COLORS['green'])

    def _enable_exports(self, enabled):
        self.btn_export_srt.setEnabled(enabled)
        self.btn_export_txt.setEnabled(enabled)
        self.btn_export_json.setEnabled(enabled)
        self.btn_copy.setEnabled(enabled)

    # ── Settings ────────────────────────────────────────────────────────
    def _save_settings(self):
        self.config['replicate_api_token'] = self.api_token_input.text().strip()
        self.config['custom_endpoint'] = self.endpoint_input.text().strip()
        self.config['custom_endpoint_key'] = self.endpoint_key_input.text().strip()
        self.config['auto_segment'] = self.chk_auto_segment.isChecked()
        self.config['open_threshold'] = self.threshold_slider.value()
        self.config['backend_index'] = self.backend_combo.currentIndex()
        save_config(self.config)
        Toast(self, "  ✅  Settings saved  ", COLORS['green'])
        self._log("💾 Settings saved")

    # ── Utilities ───────────────────────────────────────────────────────
    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")

    @staticmethod
    def _fmt_time(seconds):
        m = int(seconds) // 60
        s = seconds - m * 60
        return f"{m:02d}:{s:06.3f}"


# ── Entry Point ─────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    # Font
    font = app.font()
    font.setFamily("Segoe UI, Roboto, Arial, sans-serif")
    font.setPointSize(10)
    app.setFont(font)

    window = LipSightWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
