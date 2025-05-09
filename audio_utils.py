import threading
import queue
import json

from vosk import Model, KaldiRecognizer
import sounddevice as sd

from conf_utils import CONFIG

ACTIVATE_COMMAND = "jarvis vai"
DEACTIVATE_COMMAND = "jarvis fermo"
MODEL_PATH = "model/vosk-model-small-it-0.22"

audio_queue = queue.Queue()

# gli event globali
active_event = threading.Event()
stop_event = threading.Event()


def audio_callback(indata, frames, time_, status):
    if status:
        print(f"Warning: {status}")
    audio_queue.put(bytes(indata))


# Listener thread per comandi vocali
def voice_control():

    print(f"Carico modello Vosk da '{MODEL_PATH}'...")
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        print(
            f"Voice control attivo: pronuncia '{ACTIVATE_COMMAND}' o '{DEACTIVATE_COMMAND}'"
        )
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = res.get("text", "").strip().lower()
                    if not text:
                        continue

                    print(f"[VOICE] Riconosciuto: '{text}'")
                    if text == ACTIVATE_COMMAND:
                        active_event.set()
                        print("⚡ Bot AVVIATO")
                    elif text == DEACTIVATE_COMMAND:
                        active_event.clear()
                        print("⏸️ Bot IN PAUSA")
            except Exception as e:
                print(f"Errore loop voice control '{e}'")
                break


def start_listening():
    stop_event.clear()
    # Thread non-daemon per poter fare join()
    t = threading.Thread(target=voice_control, daemon=False)
    t.start()
    return t


def stop_listening(thread):
    stop_event.set()
    if thread.is_alive():
        thread.join()
