from pygrabber.dshow_graph import FilterGraph
import cv2
import socket
import time
import os
import argparse
from datetime import datetime
import json

# --- Argomenti da linea di comando ---
parser = argparse.ArgumentParser(description="Webcam bot per riconoscimento immagini")
parser.add_argument("--shot", "-s", action="store_true", help="Scatta immagine dalla webcam e salva in img/")
parser.add_argument("--test", "-t", action="store_true", help="Modalità test: non invia, stampa solo")
parser.add_argument("--reset-camera", "-r", action="store_true", help="Forza la selezione della webcam")
parser.add_argument("--reset-resolution", action="store_true", help="Forza la selezione della risoluzione della webcam")
parser.add_argument("--reset-config", action="store_true", help="Resetta tutte le impostazioni salvate in config.json")
parser.add_argument("--cooldown", type=float, default=1.0, help="Secondi di pausa tra due invii dello stesso tasto")
parser.add_argument("--threshold", type=float, default=0.85, help="Soglia di matching tra 0.0 e 1.0")
parser.add_argument("--ip", "-i", type=str, default="192.168.172.89", help="Indirizzo IP del dispositivo di destinazione")
parser.add_argument("--port", "-p", type=int, default=12345, help="Porta UDP del dispositivo di destinazione")
args = parser.parse_args()

CONFIG_PATH = "config.json"

# --- Configurazione UDP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Utility config ---
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)

# --- Gestione reset configurazione ---
if args.reset_config and os.path.exists(CONFIG_PATH):
    os.remove(CONFIG_PATH)
    print("🔁 Configurazione azzerata.")

# --- Carica o aggiorna config iniziale ---
config = load_config()
config["cooldown"] = args.cooldown
config["threshold"] = args.threshold
config["ip"] = args.ip
config["port"] = args.port
save_config(config)

UDP_IP = config["ip"]
UDP_PORT = config["port"]

# --- Selezione webcam ---
def load_or_select_camera(force_select=False, force_resolution=False):
    config = load_config()
    graph = FilterGraph()
    devices = graph.get_input_devices()
    available = []

    print("🎥 Scansione webcam disponibili...")
    for idx, name in enumerate(devices):
        try:
            capabilities = graph.get_input_device_capabilities(idx)
        except:
            capabilities = []
        cap = cv2.VideoCapture(idx)
        if cap.read()[0]:
            available.append((idx, name, capabilities))
            cap.release()

    if not available:
        print("❌ Nessuna webcam funzionante trovata.")
        exit(1)

    # Cerca webcam per nome salvato
    if not force_select and "camera_name" in config:
        for idx, name, caps in available:
            if name == config["camera_name"]:
                print(f"✅ Webcam trovata per nome: '{name}' (index {idx})")
                selected_caps = caps
                break
        else:
            selected = int(input("👉 Seleziona la webcam da usare: "))
            idx, name, selected_caps = available[selected]
            config["camera_index"] = idx
            config["camera_name"] = name
            save_config(config)
            print(f"💾 Webcam selezionata salvata: '{name}' (index {idx})")
            return idx
        idx = idx
        name = name
    else:
        print("📷 Webcam disponibili:")
        for i, (idx, name, _) in enumerate(available):
            print(f"{i}: {name} (index {idx})")

        selected = int(input("👉 Seleziona la webcam da usare: "))
        idx, name, selected_caps = available[selected]
        config["camera_index"] = idx
        config["camera_name"] = name
        save_config(config)
        print(f"💾 Webcam selezionata salvata: '{name}' (index {idx})")

    if force_resolution or "camera_resolution" not in config:
        if not selected_caps:
            print("⚠️ Nessuna risoluzione disponibile via pygrabber, uso risoluzione di default.")
        else:
            print("📏 Risoluzioni disponibili:")
            for i, cap in enumerate(selected_caps):
                print(f"  {i}: {cap['width']}x{cap['height']} @ {cap['max_fps']} fps")

            selected_res = int(input("👉 Seleziona la risoluzione da usare: "))
            selected_width = selected_caps[selected_res]['width']
            selected_height = selected_caps[selected_res]['height']

            config["camera_resolution"] = [selected_width, selected_height]
            save_config(config)

    return idx

headless = not os.environ.get("DISPLAY") and os.name != "nt"
CAMERA_INDEX = load_or_select_camera(force_select=args.reset_camera, force_resolution=args.reset_resolution)
cap = cv2.VideoCapture(CAMERA_INDEX)

# --- Modalità SHOT ---
if args.shot:
    print("🎥 Premi SPAZIO per scattare, ESC per uscire")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not headless:
            cv2.imshow("Scatta immagine", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            os.makedirs("img", exist_ok=True)
            filename = f"img/screenshot-{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Immagine salvata in {filename}")
            break
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Applica la risoluzione scelta se disponibile
config = load_config()
if "camera_resolution" in config:
    width, height = config["camera_resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("📏 Risoluzione selezionata:")
    print("   Larghezza:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("   Altezza:  ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Caricamento azioni ---
actions = config.get("actions", {})
if not actions:
    print("❌ Nessuna azione definita in config.json → 'actions'")
    exit(1)

print("🔧 Azioni caricate:")
for key in sorted(actions):
    entry = actions[key]
    print(f"  '{key}' → {entry['path'] if isinstance(entry, dict) else entry}")

# --- Precaricamento template ---
templates = {}
dimensions = {}
action_data = {}

for key, info in actions.items():
    if isinstance(info, str):
        path = info
        requires = []
        requires_not = []
    else:
        path = info.get("path")
        requires = info.get("requires", [])
        requires_not = info.get("requires_not", [])

    if not os.path.exists(path):
        print(f"⚠️ Immagine mancante per '{key}': {path}")
        continue
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Impossibile caricare '{path}'")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templates[key] = gray
    dimensions[key] = gray.shape[::-1]
    action_data[key] = {
        "requires": set(requires),
        "requires_not": set(requires_not),
        "path": path
    }

# --- Ciclo principale ---
print("🔍 Bot attivo. Premi Q per uscire. Usa '+' o '-' per regolare il threshold.")
last_sent_time = {}
headless = not os.environ.get("DISPLAY") and os.name != "nt"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matched_keys = set()

    for key, template_gray in templates.items():
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= config["threshold"]:
            matched_keys.add(key)
            action_info = action_data.get(key, {})

            if not action_info["requires"].issubset(matched_keys):
                continue
            if action_info["requires_not"] & matched_keys:
                continue

            now = time.time()
            if (
                not args.test
                and key in last_sent_time
                and (now - last_sent_time[key]) < config["cooldown"]
            ):
                continue

            if args.test:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] [TEST] Match '{key}' → {max_val:.2f}")
            else:
                print(f"✅ Match '{key}' ({max_val:.2f}) → invio a {UDP_IP}:{UDP_PORT}")
                sock.sendto(key.encode(), (UDP_IP, UDP_PORT))
                last_sent_time[key] = now

            top_left = max_loc
            w, h = dimensions[key]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                frame,
                key,
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        config["threshold"] = min(1.0, config.get("threshold", 0.85) + 0.01)
        save_config(config)
        print(f"🔼 Threshold aumentato: {config['threshold']:.2f}")
    elif key == ord('-'):
        config["threshold"] = max(0.0, config.get("threshold", 0.85) - 0.01)
        save_config(config)
        print(f"🔽 Threshold diminuito: {config['threshold']:.2f}")

    if not headless:
        cv2.imshow("Webcam Bot", frame)

    time.sleep(0.2)

cap.release()
cv2.destroyAllWindows()
