import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# In questo script non gestiamo uno stato persistente,
# ma potresti estendere il flag `activated` per usarlo nel tuo bot.
activated = False


def main():
    global activated
    q = queue.Queue()

    # Carica il modello italiano (scaricalo da https://alphacephei.com/vosk/models)
    model_path = "model/vosk-model-small-it-0.22"
    print(f"Caricando modello da '{model_path}'…")
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Warning: {status}")
        q.put(bytes(indata))

    # Apri lo stream audio in raw mode
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        print("In ascolto… pronuncia 'jarvis vai' o 'jarvis fermo'")
        try:
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip().lower()
                    if text:
                        print(f">> Riconosciuto: '{text}'")
                        if text == "jarvis vai":
                            activated = True
                            print("⚡ Bot AVVIATO")
                        elif text == "jarvis fermo":
                            activated = False
                            print("⏸️ Bot IN PAUSA")
                # altrimenti potresti usare rec.PartialResult() per wake-word spotting più rapido
        except KeyboardInterrupt:
            print("\nInterrotto dall'utente")


if __name__ == "__main__":
    main()
