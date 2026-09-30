"""
app.py - Speech-to-Text ASR Tool
================================
A simple Automatic Speech Recognition (ASR) web application built with
Python, OpenAI Whisper (local model), PyTorch, FFmpeg and Streamlit.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Safe imports: if PyTorch / Whisper are missing we show a friendly error in
# the UI instead of crashing with a long traceback.
# ---------------------------------------------------------------------------
try:
    import torch
    import whisper
    from whisper.audio import SAMPLE_RATE
    from whisper.tokenizer import LANGUAGES

    DEPENDENCY_ERROR = None
except Exception as exc:  # noqa: BLE001 - we want to catch any import problem
    torch = None
    whisper = None
    SAMPLE_RATE = 16000
    LANGUAGES = {}
    DEPENDENCY_ERROR = exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = ["wav", "mp3", "m4a", "flac"]

# Label shown in the UI -> Whisper model name
MODEL_OPTIONS = {
    "tiny  - fastest, lowest accuracy": "tiny",
    "base  - recommended (fast + good accuracy)": "base",
    "small - slower, better accuracy": "small",
}

# Label shown in the UI -> Whisper language code (None = auto-detect)
LANGUAGE_OPTIONS = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
}


# ---------------------------------------------------------------------------
# Custom exceptions (make error handling in the UI clean and readable)
# ---------------------------------------------------------------------------
class InvalidAudioError(Exception):
    """Raised when the uploaded file cannot be decoded as audio."""


class MissingDependencyError(Exception):
    """Raised when FFmpeg (or another required tool) is not available."""


class TranscriptionError(Exception):
    """Raised when the model fails to load or to transcribe."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def ffmpeg_available() -> bool:
    """Return True if the ffmpeg executable can be found on the PATH."""
    return shutil.which("ffmpeg") is not None


def format_size(num_bytes: int) -> str:
    """Convert a size in bytes to a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS."""
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Load a Whisper model once and keep it in memory.
    The first call downloads the model weights (needs internet); after that
    they are cached on disk by Whisper.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


def transcribe_audio(uploaded_file, model_name: str, language_code: str | None) -> dict:
    """
    Full ASR pipeline for one uploaded file.

    1. Save the upload to a temporary file.
    2. Decode it with FFmpeg (via whisper.load_audio) -> validates the file.
    3. Run the Whisper model.
    4. Return a dictionary with everything the UI needs.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp_path = None

    try:
        # Step 1: write upload to a temp file (delete=False is required on Windows,
        # because a file that is still open cannot be read by another process).
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Step 2: audio preprocessing -> mono, 16 kHz float32 numpy array.
        try:
            audio = whisper.load_audio(tmp_path)
        except FileNotFoundError as exc:
            # subprocess could not find the ffmpeg executable
            raise MissingDependencyError(
                "FFmpeg was not found. Install FFmpeg and restart the terminal."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise InvalidAudioError(
                "The file could not be read as audio. It may be corrupted, "
                "empty, or not a real audio file."
            ) from exc

        if audio.size == 0:
            raise InvalidAudioError("The audio file contains no audio data.")

        duration = audio.shape[0] / SAMPLE_RATE

        # Step 3: load the model (cached) and transcribe.
        try:
            model = load_model(model_name)
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(
                f"Could not load the '{model_name}' Whisper model. On first use "
                "the model must be downloaded, so check your internet connection."
            ) from exc

        started = time.perf_counter()
        try:
            result = model.transcribe(
                audio,
                language=language_code,          # None = auto-detect
                fp16=torch.cuda.is_available(),  # fp16 only works on GPU
            )
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Whisper failed during transcription: {exc}") from exc
        elapsed = time.perf_counter() - started

        detected = result.get("language", "unknown")
        return {
            "file_name": uploaded_file.name,
            "file_size": uploaded_file.size,
            "duration": duration,
            "language": LANGUAGES.get(detected, detected).title(),
            "model": model_name,
            "elapsed": elapsed,
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
        }

    finally:
        # Always remove the temporary file, even if an error occurred.
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Page title banner."""
    st.markdown(
        """
        <style>
        .banner {
            background: linear-gradient(90deg, #1f3c88, #4a6fdc);
            padding: 1.4rem 1.6rem; border-radius: 12px; color: white;
            margin-bottom: 1.2rem;
        }
        .banner h1 { margin: 0; font-size: 1.9rem; color: white; }
        .banner p  { margin: 0.3rem 0 0 0; opacity: 0.9; }
        </style>
        <div class="banner">
            <h1>🎙️ Speech-to-Text ASR Tool</h1>
            <p>Automatic Speech Recognition using OpenAI Whisper, PyTorch and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str | None, bool]:
    """Sidebar settings. Returns (model_name, language_code, show_segments)."""
    with st.sidebar:
        st.header("⚙️ Settings")
        model_label = st.selectbox("Whisper model", list(MODEL_OPTIONS), index=1)
        language_label = st.selectbox("Spoken language", list(LANGUAGE_OPTIONS), index=0)
        show_segments = st.checkbox("Show timestamps (segments)", value=True)

        st.divider()
        st.caption("Supported formats: " + ", ".join(f.upper() for f in SUPPORTED_FORMATS))
        if DEPENDENCY_ERROR is None:
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            st.caption(f"Running on: {device}")
        st.caption("Tip: 'tiny' is fastest for a quick demo.")

    return MODEL_OPTIONS[model_label], LANGUAGE_OPTIONS[language_label], show_segments


def render_result(result: dict, show_segments: bool) -> None:
    """Display processing information, transcript and download button."""
    st.success("Transcription completed successfully.")

    st.subheader("📊 Processing information")
    st.markdown(f"**Selected file:** `{result['file_name']}` ({format_size(result['file_size'])})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audio duration", format_time(result["duration"]))
    c2.metric("Detected language", result["language"])
    c3.metric("Model used", result["model"])
    c4.metric("Processing time", f"{result['elapsed']:.1f} s")

    st.subheader("📝 Transcribed text")
    if not result["text"]:
        st.warning("No speech was detected in this audio (it may be silent or only noise).")
        return

    st.text_area(
        "Transcription",
        value=result["text"],
        height=260,
        label_visibility="collapsed",
    )

    # The text file is named after the audio file, e.g. lecture_transcript.txt
    download_name = f"{Path(result['file_name']).stem}_transcript.txt"
    st.download_button(
        label="⬇️ Download transcript (.txt)",
        data=result["text"].encode("utf-8"),
        file_name=download_name,
        mime="text/plain",
    )

    if show_segments and result["segments"]:
        with st.expander("🕒 Timestamped segments"):
            lines = [
                f"[{format_time(seg['start'])} -> {format_time(seg['end'])}] {seg['text'].strip()}"
                for seg in result["segments"]
            ]
            st.text("\n".join(lines))


def run_transcription(uploaded_file, model_name: str, language_code: str | None) -> None:
    """Run the pipeline, store the result in session state, show friendly errors."""
    st.session_state.pop("result", None)
    try:
        with st.spinner(
            f"Transcribing with the '{model_name}' model... "
            "(the first run downloads the model, please wait)"
        ):
            st.session_state["result"] = transcribe_audio(uploaded_file, model_name, language_code)
    except InvalidAudioError as exc:
        st.error(f"❌ Invalid audio file. {exc}")
    except MissingDependencyError as exc:
        st.error(f"❌ Missing dependency. {exc}")
    except TranscriptionError as exc:
        st.error(f"❌ Transcription failed. {exc}")
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        st.error(f"❌ Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="ASR Speech-to-Text", page_icon="🎙️", layout="centered")
    render_header()
    model_name, language_code, show_segments = render_sidebar()

    # Stop early with a clear message if required tools are missing.
    if DEPENDENCY_ERROR is not None:
        st.error(
            "❌ Required Python packages are missing or broken "
            f"({DEPENDENCY_ERROR}). Activate your virtual environment and run: "
            "`pip install -r requirements.txt`"
        )
        st.stop()
    if not ffmpeg_available():
        st.error(
            "❌ FFmpeg is not installed or not on your PATH. Install it "
            "(Windows: `winget install Gyan.FFmpeg`), then close and reopen the "
            "terminal and restart this app."
        )
        st.stop()

    st.subheader("1️⃣ Upload an audio file")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=SUPPORTED_FORMATS,
        help="Supported formats: WAV, MP3, M4A, FLAC",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)
        st.caption(f"Selected: **{uploaded_file.name}** - {format_size(uploaded_file.size)}")

    st.subheader("2️⃣ Transcribe")
    if st.button("Transcribe Audio", type="primary"):
        if uploaded_file is None:
            st.error("❌ No file uploaded. Please upload an audio file first.")
        else:
            run_transcription(uploaded_file, model_name, language_code)

    # Results are kept in session_state so they survive Streamlit reruns
    # (for example, when the Download button is clicked).
    if "result" in st.session_state:
        st.divider()
        render_result(st.session_state["result"], show_segments)

    st.divider()
    st.caption("B.Tech CSE - ASR Tool Implementation | Python - Whisper - PyTorch - FFmpeg - Streamlit")


main()
