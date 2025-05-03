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
headless = not os.environ.get("DISPLAY") and os.name != "nt"

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

CONFIG_PATH = "config.json"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# --- Utility finestra ---
def resize_window(name, w_size, h_size):
    if not headless:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w_size, h_size)


# --- Utility config ---
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- Camera selection ---
def load_or_select_camera(force_select=False, force_resolution=False):
    fg = FilterGraph()
    devs = fg.get_input_devices()
    aval = []
    print("🎥 Scansione webcam disponibili...")
    for idx, name in enumerate(devs):
        try:
            capabilities = fg.get_input_device_capabilities(idx)
        except:
            capabilities = []
        vc = cv2.VideoCapture(idx)
        if vc.read()[0]:
            aval.append((idx, name, capabilities))
            vc.release()
    if not aval:
        print("❌ Nessuna webcam funzionante trovata.")
        sys.exit(1)
    if not force_select and "camera_name" in config:
        for idx, name, caps in aval:
            if name == config["camera_name"]:
                print(f"✅ Webcam trovata per nome: '{name}' (index {idx})")
                selected_caps = caps
                break
        else:
            selected = int(input("👉 Seleziona la webcam da usare: "))
            idx, name, selected_caps = aval[selected]
            config["camera_index"] = idx
            config["camera_name"] = name
            save_config(config)
            print(f"💾 Webcam selezionata salvata: '{name}' (index {idx})")
            return idx
    else:
        print("📷 Webcam disponibili:")
        for i, (idx, name, _) in enumerate(aval):
            print(f"{i}: {name} (index {idx})")
        selected = int(input("👉 Seleziona la webcam da usare: "))
        idx, name, selected_caps = aval[selected]
        config["camera_index"] = idx
        config["camera_name"] = name
        save_config(config)
        print(f"💾 Webcam selezionata salvata: '{name}' (index {idx})")
    if force_resolution or "camera_resolution" not in config:
        if not selected_caps:
            print(
                "⚠️ Nessuna risoluzione disponibile via pygrabber, uso risoluzione di default."
            )
        else:
            print("📏 Risoluzioni disponibili:")
            for i, capab in enumerate(selected_caps):
                print(f"  {i}: {capab['width']}x{capab['height']} @ {capab['max_fps']} fps")
            selected_res = int(input("👉 Seleziona la risoluzione da usare: "))
            selected_width = selected_caps[selected_res]["width"]
            selected_height = selected_caps[selected_res]["height"]
            config["camera_resolution"] = [selected_width, selected_height]
            save_config(config)
    return idx


# --- ROI extraction utility ---
# i: image
# r: roi info set
def extract_roi(i, r):
    if not r:
        return i
    x, y, w, h = r
    return i[y : y + h, x : x + w]


# --- Match ROI-aware ---
# k: key
# ad: action data
def get_roi_for_key(k, ad):
    action_info = ad.get(k, {})
    return action_info.get("roi", None)


# --- Template matching con ROI ---
def match_with_roi(frm, template, threshold, r=None):
    if r is not None:
        region = extract_roi(frm, r)
    else:
        region = frm
    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        return True, max_val, max_loc
    return False, max_val, None


# --- Ciclo principale ---
# frm: frame
# tmp: template
# cfg: config
# ac: action data
# ds: dimensions
# lst: last sent time
# a: args
def process_frame(
    frm, tmp, cfg, ad, ds, lst, a
):
    # ⚠️ L'ordine delle azioni nel file config.json è importante:
    #    se un'azione ha 'requires' riferiti a un'altra azione che viene dopo,
    #    potrebbe non attivarsi anche se il match è valido.
    matched_keys = set()
    gray_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    for k, template_gray in tmp.items():
        r = get_roi_for_key(k, ad)
        matched, max_val, max_loc = match_with_roi(
            gray_frame, template_gray, cfg["threshold"], r
        )

        # Draw ROI regardless of match
        if r:
            x, y, w, h = r
            cv2.rectangle(frm, (x, y), (x + w, y + h), (255, 255, 0), 1)
            cv2.putText(
                frm,
                f"{k}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        if not matched:
            continue

        matched_keys.add(k)
        action_info = ad.get(k, {})
        if not action_info.get("requires", set()).issubset(matched_keys):
            continue
        if action_info.get("requires_not", set()) & matched_keys:
            continue
        now = time.time()
        if (
            not a.test
            and k in lst
            and (now - lst[k]) < cfg["cooldown"]
        ):
            continue
        if not a.test:
            print(
                f"✅ Match '{k}' ({max_val:.2f}) → invio a {cfg['ip']}:{cfg['port']}"
            )
            sock.sendto(k.encode(), (cfg["ip"], cfg["port"]))
            lst[k] = now
        top_left = max_loc
        w, h = ds[k]
        if r:
            top_left = (top_left[0] + r[0], top_left[1] + r[1])
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frm, top_left, bottom_right, (0, 255, 0), 2)
        label = f"{k} ({max_val:.2f})"
        cv2.putText(
            frm,
            label,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


CONFIG_PATH = "config.json"

# --- Gestione reset configurazione ---
if args.reset_config and os.path.exists(CONFIG_PATH):
    os.remove(CONFIG_PATH)
    print("🔁 Configurazione azzerata.")

# --- Carica o aggiorna config iniziale ---
config = load_config()
DEFAULTS = {
    "window_size": [1280, 720],
    "sleep": 0.2,
    "cooldown": 1.0,
    "threshold": 0.85,
    "ip": "192.168.172.89",
    "port": 12345,
}
config["cooldown"] = (
    args.cooldown
    if args.cooldown != parser.get_default("cooldown")
    else config.get("cooldown", DEFAULTS["cooldown"])
)
config["threshold"] = (
    args.threshold
    if args.threshold != parser.get_default("threshold")
    else config.get("threshold", DEFAULTS["threshold"])
)
config["ip"] = (
    args.ip if args.ip != parser.get_default("ip") else config.get("ip", DEFAULTS["ip"])
)
config["port"] = (
    args.port
    if args.port != parser.get_default("port")
    else config.get("port", DEFAULTS["port"])
)
config["sleep"] = config.get("sleep", DEFAULTS["sleep"])
config["window_size"] = config.get("window_size", DEFAULTS["window_size"])
save_config(config)

UDP_IP = config["ip"]
UDP_PORT = config["port"]

# --- Configurazione UDP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

CAMERA_INDEX = load_or_select_camera(
    force_select=args.reset_camera, force_resolution=args.reset_resolution
)
cap = cv2.VideoCapture(CAMERA_INDEX)

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
        if not headless:
            resize_window(
                "Scatta immagine", config["window_size"][0], config["window_size"][1]
            )
            cv2.imshow("Scatta immagine", frame)
        key = cv2.waitKey(1) & 0xFF
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
    sys.exit()

# --- Caricamento azioni ---
actions = config.get("actions", {})
if not actions:
    print("❌ Nessuna azione definita in config.json → 'actions'")
    sys.exit(1)

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
    }

# --- Ciclo principale ---
print("🔍 Bot attivo. Premi Q per uscire.")
last_sent_time = {}

if not headless:
    resize_window("Webcam Bot", config["window_size"][0], config["window_size"][1])

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    process_frame(
        frame,
        templates,
        config,
        action_data,
        dimensions,
        last_sent_time,
        args,
    )

    if not headless:
        cv2.putText(
            frame,
            f"Threshold: {config['threshold']:.2f} | Cooldown: {config['cooldown']:.1f}s | Sleep: {config['sleep']:.2f}s",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Threshold: {config['threshold']:.2f} | Cooldown: {config['cooldown']:.1f}s | Sleep: {config['sleep']:.2f}s",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.imshow("Webcam Bot", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(","):
        config["cooldown"] = min(10.0, config.get("cooldown", 1.0) + 0.1)
        save_config(config)
        print(f"⏫ Cooldown aumentato: {config['cooldown']:.2f}s")
    elif key == ord("."):
        config["cooldown"] = max(0.0, config.get("cooldown", 1.0) - 0.1)
        save_config(config)
        print(f"⏬ Cooldown diminuito: {config['cooldown']:.2f}s")
    elif key == ord("*"):
        config["sleep"] = min(5.0, config.get("sleep", 0.2) + 0.05)
        save_config(config)
        print(f"⏫ Sleep aumentato: {config['sleep']:.2f}s")
    elif key == ord("/"):
        config["sleep"] = max(0.0, config.get("sleep", 0.2) - 0.05)
        save_config(config)
        print(f"⏬ Sleep diminuito: {config['sleep']:.2f}s")
    elif key == ord("+") or key == ord("="):
        config["threshold"] = min(1.0, config.get("threshold", 0.85) + 0.01)
        save_config(config)
        print(f"🔼 Threshold aumentato: {config['threshold']:.2f}")
    elif key == ord("-"):
        config["threshold"] = max(0.0, config.get("threshold", 0.85) - 0.01)
        save_config(config)
        print(f"🔽 Threshold diminuito: {config['threshold']:.2f}")

    time.sleep(config["sleep"])

cap.release()
cv2.destroyAllWindows()
