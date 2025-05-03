import sys
import socket
import time
import os
import argparse
from datetime import datetime
import json

from pygrabber.dshow_graph import FilterGraph
import cv2

# --- Headless mode detection ---
HEADLESS = not os.environ.get("DISPLAY") and os.name != "nt"
CONFIG_PATH = "config.json"
CONFIG_DEFAULTS = {
    "window_size": [1280, 720],
    "sleep": 0.2,
    "cooldown": 1.0,
    "threshold": 0.85,
    "ip": "192.168.172.89",
    "port": 12345,
}
DEF_FONT=cv2.FONT_HERSHEY_SIMPLEX

# --- Argomenti da linea di comando ---
parser = argparse.ArgumentParser(description="Webcam bot per riconoscimento immagini")
parser.add_argument(
    "--shot",
    "-s",
    action="store_true",
    help="Scatta immagine dalla webcam e salva in img/",
)
parser.add_argument(
    "--test", "-t", action="store_true", help="Modalità test: non invia, stampa solo"
)
parser.add_argument(
    "--reset-camera", "-r", action="store_true", help="Forza la selezione della webcam"
)
parser.add_argument(
    "--reset-resolution",
    action="store_true",
    help="Forza la selezione della risoluzione della webcam",
)
parser.add_argument(
    "--reset-config",
    action="store_true",
    help="Resetta tutte le impostazioni salvate in config.json",
)
parser.add_argument(
    "--cooldown",
    type=float,
    default=1.0,
    help="Secondi di pausa tra due invii dello stesso tasto",
)
parser.add_argument(
    "--threshold", type=float, default=0.85, help="Soglia di matching tra 0.0 e 1.0"
)
parser.add_argument(
    "--ip",
    "-i",
    type=str,
    default="192.168.172.89",
    help="Indirizzo IP del dispositivo di destinazione",
)
parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=12345,
    help="Porta UDP del dispositivo di destinazione",
)
args = parser.parse_args()

# --- Gestione reset configurazione ---
if args.reset_config and os.path.exists(CONFIG_PATH):
    os.remove(CONFIG_PATH)
    print("🔁 Configurazione azzerata.")

# --- Carica o aggiorna config iniziale ---
config={}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        config =  json.load(config_file)

config["cooldown"] = (
    args.cooldown
    if args.cooldown != parser.get_default("cooldown")
    else config.get("cooldown", CONFIG_DEFAULTS["cooldown"])
)
config["threshold"] = (
    args.threshold
    if args.threshold != parser.get_default("threshold")
    else config.get("threshold", CONFIG_DEFAULTS["threshold"])
)
config["ip"] = (
    args.ip if args.ip != parser.get_default("ip") else config.get("ip", CONFIG_DEFAULTS["ip"])
)
config["port"] = (
    args.port
    if args.port != parser.get_default("port")
    else config.get("port", CONFIG_DEFAULTS["port"])
)
config["sleep"] = config.get("sleep", CONFIG_DEFAULTS["sleep"])
config["window_size"] = config.get("window_size", CONFIG_DEFAULTS["window_size"])

UDP_IP = config["ip"]
UDP_PORT = config["port"]

# --- Utility finestra ---
def resize_window(name, width, height):
    if not HEADLESS:
        cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, width, height)

def save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- Camera selection ---
def load_or_select_camera(force_select=False, force_resolution=False):
    filter_group = FilterGraph()
    devices = filter_group.get_input_devices()
    avalaible = []
    print("🎥 Scansione webcam disponibili...")
    for idx, name in enumerate(devices):
        try:
            cap = filter_group.get_input_device_capabilities(idx)
        except:
            cap = []
        capture = cv2.VideoCapture(idx)
        if capture.read()[0]:
            avalaible.append((idx, name, cap))
        capture.release()
    if not avalaible:
        print("❌ Nessuna webcam funzionante trovata.")
        sys.exit(1)
    if not force_select and "camera_name" in config:
        for idx, name, capabilities in avalaible:
            if name == config["camera_name"]:
                print(f"✅ Webcam trovata per nome: '{name}' (index {idx})")
                caps = capabilities
                break
        else:
            dev = int(input("👉 Seleziona la webcam da usare: "))
            idx, name, caps = avalaible[dev]
            config["camera_index"] = idx
            config["camera_name"] = name
            save_config(config)
            print(f"💾 Webcam selezionata salvata: '{name}' (index {idx})")
            return idx
    else:
        print("📷 Webcam disponibili:")
        for (idx, name, _) in enumerate(avalaible):
            print(f"{idx}: {name}")
        dev = int(input("👉 Seleziona la webcam da usare: "))
        idx, name, caps = avalaible[dev]
        config["camera_index"] = idx
        config["camera_name"] = name
        save_config(config)
        print(f"💾 Webcam selezionata salvata: '{name}' (index {idx})")
    if force_resolution or "camera_resolution" not in config:
        if not caps:
            print(
                "⚠️ Nessuna risoluzione disponibile via pygrabber, uso risoluzione di default."
            )
        else:
            print("📏 Risoluzioni disponibili:")
            for idx, capab in enumerate(caps):
                print(f"  {idx}: {capab['width']}x{capab['height']} @ {capab['max_fps']} fps")
            selected_res = int(input("👉 Seleziona la risoluzione da usare: "))
            selected_width = caps[selected_res]["width"]
            selected_height = caps[selected_res]["height"]
            config["camera_resolution"] = [selected_width, selected_height]
            save_config(config)
    return idx


# --- ROI extraction utility ---
def extract_roi(image, roi):
    if not roi:
        return image
    x, y, w, h = roi
    return image[y : y + h, x : x + w]


# --- Match ROI-aware ---
def get_roi_for_key(key, action_data):
    action_info = action_data.get(key, {})
    return action_info.get("roi", None)


# --- Template matching con ROI ---
def match_with_roi(frame, template, threshold, roi=None):
    if roi is not None:
        region = extract_roi(frame, roi)
    else:
        region = frame
    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        return True, max_val, max_loc
    return False, max_val, None


# --- Ciclo principale ---
def process_frame(frame, templates, action_data, dimensions, last_sent_time, sock):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pre_matches = {}
    confirmed_matches = {}

    # Fase 1: Rilevamento iniziale (senza dipendenze)
    for key, template in templates.items():
        roi = action_data.get(key, {}).get("roi")
        matched, max_val, max_loc = match_with_roi(gray_frame, template, config["threshold"], roi)

        if roi:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # per ora lascio commentato
            #cv2.putText(frame, key, (x, y - 5), DEF_FONT, 1, (255, 255, 0), 1)

        if matched:
            pre_matches[key] = (max_val, max_loc)

    # Fase 2: Validazione con dipendenze
    for key, (max_val, max_loc) in pre_matches.items():
        action = action_data.get(key, {})
        required = set(action.get("requires", []))
        required_not = set(action.get("requires_not", []))

        if not required.issubset(pre_matches.keys()):
            continue
        if required_not & pre_matches.keys():
            continue

        confirmed_matches[key] = (max_val, max_loc)

    # Fase 3: Visualizzazione ed invio tasti
    for key, (max_val, max_loc) in confirmed_matches.items():

        now = time.time()
        if not args.test and key in last_sent_time and (now - last_sent_time[key]) < config["cooldown"]:
            continue

        if not args.test and action_data[key].get("send", True):
            print(f"✅ Match '{key}' ({max_val:.2f}) → invio a {config['ip']}:{config['port']}")
            sock.sendto(key.encode(), (config["ip"], config["port"]))
            last_sent_time[key] = now

        roi = action_data[key].get("roi")
        w, h = dimensions[key]
        top_left = (max_loc[0] + roi[0], max_loc[1] + roi[1]) if roi else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        label = f"{key} ({max_val:.2f})"
        cv2.putText(frame, label, (top_left[0], top_left[1] - 10), DEF_FONT, 0.6, (0, 255, 0), 2)

    if not HEADLESS:
        label = f"Threshold: {config['threshold']:.2f} | Cooldown: {config['cooldown']:.1f}s | Sleep: {config['sleep']:.2f}s"
        cv2.putText(
            frame,
            label,
            (10, 30),
            DEF_FONT,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            label,
            (10, 30),
            DEF_FONT,
            1,
            (255, 255, 255),
            1,
        )
        cv2.imshow("Webcam Bot", frame)

def main():

    # --- Configurazione UDP ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(load_or_select_camera(args.reset_camera, args.reset_resolution))

    # Applica la risoluzione scelta se disponibile
    if "camera_resolution" in config:
        width, height = config["camera_resolution"]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    config["camera_resolution"] = [int(width), int(height)]
    save_config(config)
    print("📏 Risoluzione selezionata:")
    print("   Larghezza:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("   Altezza:  ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Modalità SHOT ---
    if args.shot:
        print("🎥 Premi SPAZIO per scattare, ESC per uscire")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if not HEADLESS:
                resize_window(
                    "Scatta immagine", config["window_size"][0], config["window_size"][1]
                )
                cv2.imshow("Scatta immagine", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                os.makedirs("img", exist_ok=True)
                filename = f"img/screenshot-{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"📸 Immagine salvata in {filename}")
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # --- Caricamento azioni ---
    actions = config.get("actions", {})
    if not actions:
        print("❌ Nessuna azione definita in config.json → 'actions'")
        return

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
            roi = None
        else:
            path = info.get("path")
            requires = info.get("requires", [])
            requires_not = info.get("requires_not", [])
            roi = info.get("roi")
            if roi and (not isinstance(roi, list) or len(roi) != 4 or not all(isinstance(x, int) for x in roi)):
                print(f"⚠️ ROI non valido per '{key}': {roi}")
                continue

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
            "roi": roi,
            "path": path,
            "send": info.get("send", True),  # default = True
}


    # --- Ciclo principale ---
    print("🔍 Bot attivo. Premi Q per uscire.")
    last_sent_time = {}

    if not HEADLESS:
        resize_window("Webcam Bot", config["window_size"][0], config["window_size"][1])

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        process_frame(
            frame,
            templates,
            action_data,
            dimensions,
            last_sent_time,
            sock
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        dirty_config = False
        if key == ord(","):
            config["cooldown"] = min(10.0, config.get("cooldown", 1.0) + 0.1)
            dirty_config = True
            print(f"⏫ Cooldown aumentato: {config['cooldown']:.2f}s")
        elif key == ord("."):
            config["cooldown"] = max(0.0, config.get("cooldown", 1.0) - 0.1)
            dirty_config = True
            print(f"⏬ Cooldown diminuito: {config['cooldown']:.2f}s")
        elif key == ord("*"):
            config["sleep"] = min(5.0, config.get("sleep", 0.2) + 0.05)
            dirty_config = True
            print(f"⏫ Sleep aumentato: {config['sleep']:.2f}s")
        elif key == ord("/"):
            config["sleep"] = max(0.0, config.get("sleep", 0.2) - 0.05)
            dirty_config = True
            print(f"⏬ Sleep diminuito: {config['sleep']:.2f}s")
        elif key == ord("+") or key == ord("="):
            config["threshold"] = min(1.0, config.get("threshold", 0.85) + 0.01)
            dirty_config = True
            print(f"🔼 Threshold aumentato: {config['threshold']:.2f}")
        elif key == ord("-"):
            config["threshold"] = max(0.0, config.get("threshold", 0.85) - 0.01)
            dirty_config = True
            print(f"🔽 Threshold diminuito: {config['threshold']:.2f}")

        if dirty_config:
            save_config(config)
        time.sleep(config["sleep"])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
