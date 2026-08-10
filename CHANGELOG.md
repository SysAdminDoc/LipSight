# Changelog

All notable changes to LipSight will be documented in this file.

## [v1.2.0] - 2026-08-03

- Added versioned SRT/VTT/TXT/JSON exports, `.lipsight` project bundles, session archive search, headless processing, CLI mode, and watch-folder processing.
- Added canonical mouth alignment, lighting normalization, optical-flow stabilization, optional super-resolution, and multi-face observations.
- Added side-by-side annotated/ROI previews, mouth-motion curve editing, multi-speaker controls, confidence indicators, transcript review, and project actions in the GUI.
- Added optional local ONNX, VALLR, AV-HuBERT, faster-whisper, whisper.cpp, and confidence-aware audio/visual fusion adapters with headless configuration.
- Added correction-dataset/fine-tune hooks, selectable audio language codes, and large high-contrast accessibility captions.
- Added an opt-in always-on-top caption overlay and non-injected Ctrl+Alt+F8 toggle listener.

## [v1.0.0] - 2026-04-13

- Added: Add files via upload
- Changed: Update README.md
- Added: Add files via upload

## Roadmap archive — 2026-08-10 — ROADMAP.md

<details>
<summary>Original roadmap snapshot</summary>

```markdown
# LipSight Roadmap

AI-powered lip-reading from silent video using Auto-AVSR via Replicate. PyQt6 GUI with MediaPipe face detection, automatic segmentation, SRT/TXT/JSON export. Roadmap focuses on local inference, multi-speaker, and accuracy-boosting pre-processing.

## Planned Features

### Inference

### Preprocessing (big accuracy wins)

### Video UX

### Export

### Workflow

## Competitive Research
- **Auto-AVSR (upstream)** — Apache 2.0, ~20% WER on LRS3; already the current backend. Track new releases.
- **VALLR (ICCV 2025)** — 18.7% WER using LLaMA integration; top of the leaderboard as of 2025. High-priority port.
- **AV-HuBERT (Meta)** — strong self-supervised encoder; useful as a general feature extractor.
- **Commercial: SpeechMatics / Liopa** — closed-source medical/legal-focused; document strengths so we know where OSS tools underperform.
- **Whisper (OpenAI)** — audio-based, not visual, but fusion with Whisper dramatically improves noisy-audio scenarios.

## Nice-to-Haves

## Open-Source Research (Round 2)

### Related OSS Projects
- abb128/LiveCaptions — https://github.com/abb128/LiveCaptions — Linux desktop live captioning; local inference via aprilasr; Flatpak distribution; no network
- MidCamp/live-captioning — https://github.com/MidCamp/live-captioning — browser-based using Chrome Web Speech API; zero-install for event accessibility
- steveseguin/captionninja — https://github.com/steveseguin/captionninja — browser mic → STT → websocket → overlay, pairs with Electron Capture for desktop pinning
- botbahlul/Live-Subtitle — https://github.com/botbahlul/Live-Subtitle — Android app recognizing VLC streams, adds MLKit translate
- zats/SpeechRecognition — https://github.com/zats/SpeechRecognition — iOS SFSpeechRecognizer demo generating subtitles in real-time
- XR-Access-Initiative/chirp-captions — https://github.com/XR-Access-Initiative/chirp-captions — Unity VR captions system, paired with Whisperer
- livekit-examples/live-translated-captioning — https://github.com/livekit-examples/live-translated-captioning — LiveKit agent with Deepgram; swap-in any STT
- openai/whisper — https://github.com/openai/whisper — the STT baseline
- ggerganov/whisper.cpp — https://github.com/ggerganov/whisper.cpp — C++ Whisper port, offline-capable, CPU-friendly
- SYSTRAN/faster-whisper — https://github.com/SYSTRAN/faster-whisper — CTranslate2 Whisper; 4x faster, lower VRAM
- mpc001/auto_avsr — https://github.com/mpc001/auto_avsr — audio-visual speech recognition (lip reading + audio fusion); closest to the project's name
- facebookresearch/av_hubert — https://github.com/facebookresearch/av_hubert — audio-visual HuBERT; lip-reading research baseline
- rizkiarm/LipNet — https://github.com/rizkiarm/LipNet — end-to-end sentence-level lip reading (Keras); classic reference

### Features to Borrow

### Patterns & Architectures Worth Studying
- abb128/LiveCaptions **loopback audio capture on Linux via PipeWire/PulseAudio** — direct read-from-monitor-sink; on Windows this is WASAPI loopback
- whisper.cpp's **streaming decode** — progressive partials that stabilize as more audio arrives; render "faded" partial text, commit when stable
- Auto-AVSR's **two-stream encoder + late fusion** — audio path + lip path, fused only at final layers; tolerates mic noise / masked faces individually
- CaptionNinja's **WebSocket fan-out** — one source, N overlay clients (great for streamers)
- **Face/mouth ROI detection** (mediapipe face-mesh) for lip-reading path — landmarks 78/95/308/317/13/14/80/310 bound the lip region tightly
```

</details>
