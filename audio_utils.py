"""
Module audio_utils.py

Provides offline voice control functionality using Vosk and sounddevice.
Defines functions to start and stop the listening thread, and events
for activation, deactivation, and exit commands.
"""

import threading
import queue
import json
from typing import Any, Dict, List

from vosk import Model, KaldiRecognizer
import sounddevice as sd
import numpy as np

from conf_utils import CONFIG

# Default parameters
_DEFAULT_DURATION = 0.025  # seconds
_DEFAULT_FS = 44100  # sampling rate

# Precompute time axis once for the default duration/fs
_TIME_AXIS = np.linspace(
    0, _DEFAULT_DURATION, int(_DEFAULT_FS * _DEFAULT_DURATION), endpoint=False
)

# Cache sine‐wave arrays by frequency (Hz)
_WAVE_CACHE: Dict[float, np.ndarray] = {}

# Queue for raw audio data
_AUDIO_QUEUE: queue.Queue[bytes] = queue.Queue()

# Events used by the application
active_event = threading.Event()  # Set when bot should process frames
stop_event = threading.Event()  # Set to request listener thread stop
exit_event = threading.Event()  # Set when exit command is received


def audio_callback(
    indata: bytes, frames: int, time_: Any, status: sd.CallbackFlags
) -> None:
    """
    Callback for the RawInputStream.

    Parameters:
        indata (bytes): Recorded audio data.
        frames (int): Number of audio frames.
        time_ (Any): Time information (ignored).
        status (sd.CallbackFlags): Stream status flags.

    Returns:
        None
    """
    if status:
        print(f"Warning: {status}")
    _AUDIO_QUEUE.put(bytes(indata))


def voice_control() -> None:
    """
    Listener thread target that processes microphone audio,
    performs speech recognition with Vosk, and adjusts events
    based on recognized commands.

    Uses configuration keys:
        - 'audio_model_path': path to the Vosk model directory
        - 'audio_activate_cmd': phrase to activate the bot
        - 'audio_deactivate_cmd': phrase to deactivate the bot
        - 'audio_exit_cmd': phrase to request exit

    Runs until stop_event is set.

    Returns:
        None
    """
    # Load commands and model path from CONFIG
    model_path: str = CONFIG.get("audio_model_path")  # type: ignore
    activate_cmd: str = CONFIG.get("audio_activate_cmd")  # type: ignore
    deactivate_cmd: str = CONFIG.get("audio_deactivate_cmd")  # type: ignore
    exit_cmd: str = CONFIG.get("audio_exit_cmd")  # type: ignore

    print(f"Loading Vosk model from '{model_path}'...")
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        print(
            f"Voice control active: say '{activate_cmd}', "
            f"'{deactivate_cmd}' or '{exit_cmd}'"
        )
        while not stop_event.is_set():
            try:
                data: bytes = _AUDIO_QUEUE.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if rec.AcceptWaveform(data):
                    res: Dict[str, Any] = json.loads(rec.Result())
                    text: str = res.get("text", "").strip().lower()
                    if not text:
                        continue

                    print(f"[VOICE] Recognized: '{text}'")
                    words: List[str] = text.split()
                    if activate_cmd in words:
                        exit_event.clear()
                        active_event.set()
                        print("⚡ Bot ACTIVATED")
                    elif deactivate_cmd in words:
                        exit_event.clear()
                        active_event.clear()
                        print("🛑 Bot DEACTIVATED")
                    elif exit_cmd in words:
                        active_event.clear()
                        exit_event.set()
                        print("👋 Bot EXIT REQUESTED")
            except Exception as e:
                print(f"❗ Error in voice_control loop: {e}")
                continue


def start_listening() -> threading.Thread:
    """
    Start the voice_control listener in its own thread.

    Returns:
        threading.Thread: Thread object running voice_control.
    """
    stop_event.clear()
    listener = threading.Thread(target=voice_control, daemon=False)
    listener.start()
    return listener


def stop_listening(thread: threading.Thread) -> None:
    """
    Signal the listener thread to stop and wait until it finishes.

    Parameters:
        thread (threading.Thread): The listener thread from start_listening().

    Returns:
        None
    """
    stop_event.set()
    if thread.is_alive():
        thread.join()


def beep(
    frequency: float = 500.0, duration: float = _DEFAULT_DURATION, fs: int = _DEFAULT_FS
) -> None:
    """
    Play a sine‐wave beep at the given frequency (Hz) for the given duration (s),
    caching the waveform so that 't' and 'wave' are computed only once per frequency.

    Args:
        frequency (float): Tone frequency in Hertz.
        duration (float): Length of the beep in seconds.
        fs (int): Sampling rate.
    """
    # Choose or recompute the time axis
    if duration == _DEFAULT_DURATION and fs == _DEFAULT_FS:
        t = _TIME_AXIS
    else:
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Fetch or build the sine‐wave for this frequency
    if duration == _DEFAULT_DURATION and fs == _DEFAULT_FS:
        wave = _WAVE_CACHE.get(frequency)
        if wave is None:
            wave = np.sin(2 * np.pi * frequency * t)
            _WAVE_CACHE[frequency] = wave
    else:
        # For non‐default params, just compute on the fly
        wave = np.sin(2 * np.pi * frequency * t)

    sd.play(wave, fs)  # non‐blocking
    # no sd.wait() so program continues immediately
