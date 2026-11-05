# 🎭 Emotion-Aware Text-to-Speech System

An advanced, modular TTS system that understands and controls emotional expression dynamically. Unlike typical TTS systems that apply a fixed emotion label to an entire sentence, this system models emotion as a **continuous, controllable signal** that can evolve over time — ideal for stories, audiobooks, and expressive dialogue.

## Architecture

```
Input Text ──→ Text Processor ──→ Emotion Analyzer ──→ Emotion Curve
                    │                                        │
                    ▼                                        ▼
              Phoneme/Text ──→ FastSpeech2 ──→ Emotion Conditioning ──→ HiFi-GAN ──→ Audio
```

### Modules

| Module | Path | Description |
|--------|------|-------------|
| **Emotion Labels** | `src/tts/emotion/emotion_labels.py` | VAD vectors, 7 emotion categories, intensity control |
| **Emotion Analyzer** | `src/tts/emotion/text_emotion_analyzer.py` | HuggingFace transformer-based auto-detection |
| **Emotion Curve** | `src/tts/emotion/emotion_curve.py` | Cubic spline interpolation for smooth transitions |
| **Text Processor** | `src/tts/text/text_processor.py` | Sentence/clause segmentation, normalization |
| **SSML Parser** | `src/tts/text/ssml_parser.py` | XML emotion markup support |
| **Acoustic Model** | `src/tts/acoustic/fastspeech2_model.py` | Emotion-aware FastSpeech2 wrapper |
| **Conditioning** | `src/tts/acoustic/emotion_conditioning.py` | Pitch/energy/rate modification from VAD |
| **Prosody** | `src/tts/acoustic/prosody_predictor.py` | Emotion-driven prosody contours |
| **Vocoder** | `src/tts/vocoder/vocoder.py` | HiFi-GAN mel-to-waveform |
| **Pipeline** | `src/tts/pipeline.py` | End-to-end orchestration |
| **Config** | `src/tts/config.py` | Central configuration |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from tts.pipeline import EmotionTTSPipeline
from tts.config import TTSConfig

pipeline = EmotionTTSPipeline(TTSConfig())

# Simple synthesis with emotion override
pipeline.synthesize("I'm so happy to see you!", emotion="happy", intensity=0.9, output_path="happy.wav")

# Auto-detect emotion from text
pipeline.synthesize("This is a very sad day.", output_path="auto.wav")
```

### Streamlit Demo

```bash
streamlit run streamlit_app.py
```

The Streamlit app uses this project as the backend. You can:

- Enter text and synthesize audio
- Choose auto-detected emotion or a manual override
- Play or download the generated WAV
- View the mel spectrogram produced from the text before vocoding

### Long-Form (Stories / Audiobooks)

```python
story = """
Once upon a time there was a cheerful little girl. She loved to play in the garden.
But one dark evening, she heard a terrible noise. She was very scared.
Then her mother came and everything was alright again. She felt safe and happy.
"""

pipeline.synthesize_long_form(story, output_path="story.wav")
```

### SSML Emotion Markup

```python
ssml = """
<speak>
  <emotion category="happy" intensity="0.9">What a wonderful day!</emotion>
  <emotion category="sad" intensity="0.6">But I miss you so much.</emotion>
  <emotion category="angry" arousal="0.95">This is unacceptable!</emotion>
</speak>
"""

pipeline.synthesize_ssml(ssml, output_path="ssml_output.wav")
```

## Emotion Model

Emotions are represented as continuous **Valence-Arousal-Dominance (VAD)** vectors:

| Emotion | Valence | Arousal | Dominance |
|---------|---------|---------|-----------|
| Happy | 0.85 | 0.70 | 0.65 |
| Sad | 0.15 | 0.25 | 0.25 |
| Angry | 0.15 | 0.85 | 0.80 |
| Fearful | 0.20 | 0.80 | 0.15 |
| Surprised | 0.60 | 0.85 | 0.40 |
| Disgusted | 0.15 | 0.55 | 0.60 |

## Testing

```bash
# Unit tests (no models required)
python -m pytest tests/ -v -m "not slow"

# Full integration tests (requires models)
python -m pytest tests/ -v -m slow
```

## Configuration

Create a `config.yml` to customize settings:

```yaml
device: cuda
sample_rate: 22050
enable_emotion_conditioning: true
emotion_confidence_threshold: 0.4
pitch_range: 3.0
energy_range: 0.5
segmentation_level: sentence
```

```python
config = TTSConfig.from_yaml("config.yml")
pipeline = EmotionTTSPipeline(config)
```

## Future Extensions

- 🔬 **Diffusion-based vocoder** for higher quality synthesis
- 🧠 **Adaptive emotional planning** using LLM-based emotion trajectories
- 🎯 **Fine-tuned emotion embeddings** injected directly into the acoustic model
- 🌐 **Multi-speaker, multi-language** emotion-aware synthesis
