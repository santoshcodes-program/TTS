"""
Emotion TTS Studio - no-key edition.

This app detects the dominant emotion from user text, displays the emotion
labels and confidence scores, and synthesizes emotional-ish speech through the
 publicly available `edge-tts` library without requiring API keys.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st

from tts.emotion.simple_emotion_detector import EmotionDetector
from tts.providers.edge_speech import EdgeSpeechSynthesizer


st.set_page_config(
    page_title="Emotion TTS Studio",
    page_icon="🔊",
    layout="wide",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    :root {
        --bg-primary: #0f0f1a;
        --bg-secondary: #161625;
        --bg-card: rgba(30, 30, 52, 0.85);
        --text-primary: #eaeaff;
        --text-secondary: #9d9dbe;
        --accent: #7c5cfc;
        --accent-glow: rgba(124, 92, 252, 0.35);
        --gradient-warm: linear-gradient(135deg, #7c5cfc 0%, #c471ed 50%, #f64f59 100%);
        --border: rgba(124, 92, 252, 0.18);
    }

    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(124,92,252,0.12), rgba(196,113,237,0.08));
        border: 1px solid var(--border);
        backdrop-filter: blur(12px);
        box-shadow: 0 0 60px rgba(124,92,252,0.08);
        margin-bottom: 1.5rem;
    }

    .hero-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        background: rgba(124,92,252,0.14);
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        margin-bottom: 0.9rem;
    }

    .hero h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
        line-height: 1.18;
        background: var(--gradient-warm);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.6rem 0;
    }

    .hero p {
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 49rem;
        margin: 0;
    }

    .panel {
        padding: 1.4rem 1.5rem;
        border-radius: 20px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }

    .metric-card {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: rgba(124,92,252,0.08);
        border: 1px solid rgba(124,92,252,0.12);
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
    }

    .metric-value {
        margin-top: 0.2rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }

    .stButton > button, .stDownloadButton > button {
        background: var(--gradient-warm) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.7rem 1.6rem !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(124, 92, 252, 0.35) !important;
    }

    h1, h2, h3, p, label, div, span {
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_detector(prefer_transformers: bool) -> EmotionDetector:
    return EmotionDetector(prefer_transformers=prefer_transformers)


@st.cache_resource(show_spinner=False)
def load_synthesizer() -> EdgeSpeechSynthesizer:
    return EdgeSpeechSynthesizer()


st.markdown(
    """
    <section class="hero">
        <div class="hero-badge">Emotion TTS Studio</div>
        <h1>Type the text, detect the emotion, and speak it back with no API key.</h1>
        <p>
            This version we have introduced emotion to the FastSpeech and added various features to the MEL spectogram 
            to dectect emotions in the input text and then moved to the hifiGan with the emotion aware MEL spectogram
            and Output the voice with indulged emotions.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Controls")
    prefer_transformers = st.toggle(
        "Use transformer detector if installed",
        value=True,
        help="If transformers is unavailable, the app automatically falls back to the built-in heuristic detector.",
    )
    


default_text = (
    "I am so happy that we finally finished this project, but I was really worried "
    "we might miss the deadline."
)
text = st.text_area(
    "Enter text",
    value=default_text,
    height=190,
    placeholder="Type the text you want to synthesize...",
)

generate = st.button("Generate emotional audio", use_container_width=False)

if generate:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Detecting emotion and generating speech..."):
            try:
                detector = load_detector(prefer_transformers)
                synthesizer = load_synthesizer()
                detection = detector.detect(text.strip())
                synthesis = synthesizer.synthesize(text.strip(), detection)

                st.session_state["tts_result"] = {
                    "text": text.strip(),
                    "detection": detection,
                    "audio_bytes": synthesis.audio_bytes,
                    "mime_type": synthesis.mime_type,
                    "voice_name": synthesis.voice_name,
                    "rate": synthesis.rate,
                    "pitch": synthesis.pitch,
                    "volume": synthesis.volume,
                }
            except Exception as exc:
                st.error(f"Synthesis failed: {exc}")
                st.caption(
                    "Install the no-key speech library with `python -m pip install edge-tts` "
                    "and make sure you have internet access."
                )


result = st.session_state.get("tts_result")

if result:
    detection = result["detection"]
    left_col, right_col = st.columns([1, 1.15], gap="large")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Audio Output")
        st.audio(result["audio_bytes"], format=result["mime_type"])
        st.download_button(
            "Download MP3",
            data=result["audio_bytes"],
            file_name="emotion-tts-output.mp3",
            mime=result["mime_type"],
        )
        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-card">
                    <div class="metric-label">Detected Emotion</div>
                    <div class="metric-value">{detection.category.value.title()}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{detection.confidence:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Detector</div>
                    <div class="metric-value">{detection.source.title()}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Voice</div>
                    <div class="metric-value">{result['voice_name']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Rate</div>
                    <div class="metric-value">{result['rate']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pitch</div>
                    <div class="metric-value">{result['pitch']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Detection evidence: {detection.evidence}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Emotion Labels")
        for label, score in sorted(detection.scores.items(), key=lambda item: item[1], reverse=True):
            st.write(f"{label.title()}: {score:.2f}")
            st.progress(float(min(1.0, max(0.0, score))))
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Input Text"):
        st.write(result["text"])

    with st.expander("Applied Prosody Settings"):
        st.json(
            {
                "voice": result["voice_name"],
                "rate": result["rate"],
                "pitch": result["pitch"],
                "volume": result["volume"],
                "dominant_emotion": detection.category.value,
                "detector_source": detection.source,
            }
        )
