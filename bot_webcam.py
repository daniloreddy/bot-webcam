import socket
import time
import os
import argparse
from datetime import datetime
import json
import platform

# Solo su Windows: per rilevare la finestra attiva
try:
    import win32gui
except ImportError:
    win32gui = None

try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

import cv2

CONFIG_PATH = "config.json"
DEFAULTS = {
    "window_size": [1280, 720],
    "sleep": 0.2,
    "cooldown": 1.0,
    "threshold": 0.85,
    "ip": "192.168.172.89",
    "port": 12345,
    "headless": False,
    "game_window_title": None,
    "match_method": "TM_CCOEFF_NORMED",
    "preprocess_invert": True,
    "preprocess_blur": True,
    "preprocess_equalize": True,
    "match_sector_enabled": False,
    "match_sector_grid": [2, 2],
    "match_sector_min_success": 3,
    "match_use_mask": False,
}

DEF_FONT = cv2.FONT_HERSHEY_SIMPLEX

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
    "--reset-camera", "-rc", action="store_true", help="Forza la selezione della webcam"
)
parser.add_argument(
    "--reset-resolution",
    "-rr",
    action="store_true",
    help="Forza la selezione della risoluzione della webcam",
)
parser.add_argument(
    "--reset-config",
    "-r",
    action="store_true",
    help="Resetta tutte le impostazioni salvate in config.json",
)
parser.add_argument(
    "--cooldown",
    "-c",
    type=float,
    default=1.0,
    help="Secondi di pausa tra due invii dello stesso tasto",
)
parser.add_argument(
    "--threshold",
    "-T",
    type=float,
    default=0.85,
    help="Soglia di matching tra 0.0 e 1.0",
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
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

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
config["game_window_title"] = config.get(
    "game_window_title", DEFAULTS["game_window_title"]
)
config["match_method"] = config.get("match_method", DEFAULTS["match_method"])
config["preprocess_invert"] = config.get("preprocess_invert", DEFAULTS["preprocess_invert"])
config["preprocess_blur"] = config.get("preprocess_blur", DEFAULTS["preprocess_blur"])
config["preprocess_equalize"] = config.get("preprocess_equalize", DEFAULTS["preprocess_equalize"])


# HEADLESS: usa valore da config se presente, altrimenti autodetect
HEADLESS = config.get("headless")
if HEADLESS is None:
    if platform.system() == "Darwin":  # macOS
        HEADLESS = False
    else:
        HEADLESS = not os.environ.get("DISPLAY") and os.name != "nt"
    config["headless"] = HEADLESS
else:
    HEADLESS = bool(HEADLESS)


UDP_IP = config["ip"]
UDP_PORT = config["port"]


def is_game_window_focused(should_check_focus):
    if not should_check_focus:
        return True
    current = get_foreground_window_title()
    return current and current.lower() == config["game_window_title"].lower()


# --- Selezione finestra gioco ---
def get_foreground_window_title():
    if win32gui is None:
        return None
    hwnd = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(hwnd)


def list_open_windows():
    titles = []

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)

    win32gui.EnumWindows(enum_handler, None)
    return titles


# --- Utility finestra ---
def resize_win(name, width, height):
    if HEADLESS:
        return
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, width, height)


def save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- Camera selection ---
def load_or_select_camera(force_select=False, force_resolution=False):
    global FilterGraph
    available = []
    selected_idx = None
    selected_name = None
    selected_caps = []

    print("🎥 Scansione webcam disponibili...")

    if FilterGraph:
        try:
            filter_group = FilterGraph()
            devices = filter_group.get_input_devices()

            for idx, name in enumerate(devices):
                try:
                    caps = filter_group.get_input_device_capabilities(idx)
                except Exception:
                    caps = []
                capture = cv2.VideoCapture(idx)
                if capture.read()[0]:
                    available.append((idx, name, caps))
                capture.release()

        except Exception as e:
            print(f"⚠️ Errore durante l'uso di pygrabber: {e}")
            print("🔁 Passo al fallback con OpenCV.")
            FilterGraph = None  # forza fallback
    if not FilterGraph:
        # Fallback OpenCV
        for idx in range(10):
            capture = cv2.VideoCapture(idx)
            if capture.read()[0]:
                available.append((idx, f"Webcam {idx}", []))
                capture.release()

    if not available:
        print("❌ Nessuna webcam funzionante trovata.")
        return

    if not force_select and "camera_name" in config:
        for idx, name, caps in available:
            if name == config["camera_name"]:
                print(f"✅ Webcam trovata per nome: '{name}' (index {idx})")
                selected_idx, selected_name, selected_caps = idx, name, caps
                break

    if selected_idx is None:
        print("📷 Webcam disponibili:")
        for i, (idx, name, _) in enumerate(available):
            print(f"{i}: {name}")
        while True:
            choice = input("👉 Seleziona la webcam da usare: ").strip()
            if not choice.isdigit():
                print("❌ Inserisci solo un numero valido.")
                continue
            dev = int(choice)
            if 0 <= dev < len(available):
                selected_idx, selected_name, selected_caps = available[dev]
                config["camera_index"] = selected_idx
                config["camera_name"] = selected_name
                save_config(config)
                print(
                    f"💾 Webcam selezionata salvata: '{selected_name}' (index {selected_idx})"
                )
                break
            else:
                print(f"❌ Inserisci un numero compreso tra 0 e {len(available) - 1}.")

    # Selezione risoluzione
    if force_resolution or "camera_resolution" not in config:
        if not selected_caps:
            print(
                "⚠️ Nessuna risoluzione disponibile via pygrabber, uso risoluzione di default."
            )
        else:
            print("📏 Risoluzioni disponibili:")
            for i, capab in enumerate(selected_caps):
                print(
                    f"  {i}: {capab['width']}x{capab['height']} @ {capab['max_fps']} fps"
                )
            while True:
                res_input = input("👉 Seleziona la risoluzione da usare: ").strip()
                if not res_input.isdigit():
                    print("❌ Inserisci solo un numero valido.")
                    continue
                res = int(res_input)
                if 0 <= res < len(selected_caps):
                    selected_width = selected_caps[res]["width"]
                    selected_height = selected_caps[res]["height"]
                    config["camera_resolution"] = [selected_width, selected_height]
                    save_config(config)
                    break
                else:
                    print(f"❌ Inserisci un numero tra 0 e {len(selected_caps) - 1}.")

    return selected_idx


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


def preprocess_image(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    preprocessed = gray
    if config.get("preprocess_invert", True):
        preprocessed = cv2.bitwise_not(preprocessed)

    if config.get("preprocess_blur", True):
        preprocessed = cv2.GaussianBlur(preprocessed, (3, 3), 0)

    if config.get("preprocess_equalize", True):
        preprocessed = cv2.equalizeHist(preprocessed)

    return preprocessed

# --- Template matching con ROI ---
def match_with_roi(frame, data, threshold):
    template = data.get("template")
    mask = data.get("mask_data")
    roi = data.get("roi")

    if roi is not None:
        region = extract_roi(frame, roi)
    else:
        region = frame

    region = preprocess_image(region)
    template = preprocess_image(template)
    if mask is not None:
        mask = preprocess_image(mask)

    method_name = config.get("match_method", "TM_CCOEFF_NORMED")
    method = getattr(cv2, method_name, cv2.TM_CCOEFF_NORMED)

    use_mask = (
        config.get("match_use_mask", False)
        and mask is not None
        and method_name in ["TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"]
    )

    if config.get("match_sector_enabled", False):
        return match_template_in_sectors(region, template, mask, threshold, use_mask, method)
    else:
        return match_template(region, template, mask, threshold, use_mask, method)


def match_template(region, template, mask, threshold, use_mask, method):

    try:
        if use_mask:
            result = cv2.matchTemplate(region, template, method, mask=mask)
        else:
            result = cv2.matchTemplate(region, template, method)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            return True, max_val, max_loc
        return False, max_val, None

    except:
        return False, 0, None

# --- Matching a settori ---
def match_template_in_sectors(region, template, mask, threshold, use_mask, method):
    rows, cols = config.get("match_sector_grid", [2, 2])
    required = config.get("match_sector_min_success", 3)

    h, w = template.shape
    sector_w = w // cols
    sector_h = h // rows

    matches = 0
    best_score = 0
    best_loc = None

    for i in range(rows):
        for j in range(cols):
            x = j * sector_w
            y = i * sector_h
            sector_template = template[y:y+sector_h, x:x+sector_w]
            sector_mask = mask[y:y+sector_h, x:x+sector_w] if use_mask else None

            try:
                if use_mask:
                    result = cv2.matchTemplate(region, sector_template, method, mask=sector_mask)
                else:
                    result = cv2.matchTemplate(region, sector_template, method)

                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_loc = max_loc
                if max_val >= threshold:
                    matches += 1
            except:
                continue

    if matches >= required:
        return True, best_score, best_loc
    return False, best_score, None


def show_processed_rois(frame, gray_frame, pre_matches, action_data):
    thumb_size = 100  # dimensione anteprima
    margin = 10
    x_offset = margin
    y_offset = frame.shape[0] - thumb_size - margin

    for key in sorted(pre_matches):
        roi = action_data[key].get("roi")
        if roi:
            region = extract_roi(gray_frame, roi)
            processed = preprocess_image(region)
            thumb = cv2.resize(processed, (thumb_size, thumb_size))
            thumb_color = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

            # Calcola posizione e inserisce nel frame principale
            x_end = x_offset + thumb_size
            if x_end > frame.shape[1] - margin:
                break  # finito lo spazio orizzontale
            frame[y_offset : y_offset + thumb_size, x_offset:x_end] = thumb_color
            cv2.putText(
                frame,
                key,
                (x_offset, y_offset - 5),
                DEF_FONT,
                0.5,
                (0, 255, 255),
                1,
            )
            x_offset += thumb_size + margin


# --- Ciclo principale ---
def process_frame(
    frame, action_data, last_sent_time, sock, should_check_focus
):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pre_matches = {}
    confirmed_matches = {}

    # Fase 1: Rilevamento iniziale (senza dipendenze)
    for key, data in action_data.items():
        roi = data.get("roi")
        matched, max_val, max_loc = match_with_roi(gray_frame, data, config["threshold"])

        if roi:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # per ora lascio commentato
            # cv2.putText(frame, key, (x, y - 5), DEF_FONT, 1, (255, 255, 0), 1)

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
        action = action_data.get(key, {})
        now = time.time()
        should_send = (
            not args.test
            and action_data[key].get("send", True)
            and (
                key not in last_sent_time
                or (now - last_sent_time[key]) >= config["cooldown"]
            )
            and is_game_window_focused(should_check_focus)
        )

        if should_send:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(
                f"[{timestamp}] ✅ Match '{key}' ({max_val:.2f}) → invio a {config['ip']}:{config['port']}"
            )
            sock.sendto(key.encode(), (config["ip"], config["port"]))
            last_sent_time[key] = now

        # Mostra comunque la grafica anche se non invia
        roi = action.get("roi")
        w, h = action.get("dimensions")
        top_left = (max_loc[0] + roi[0], max_loc[1] + roi[1]) if roi else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        label = f"{key} ({max_val:.2f})"
        cv2.putText(
            frame, label, (top_left[0], top_left[1] - 10), DEF_FONT, 0.6, (0, 255, 0), 2
        )

    if HEADLESS:
        return
    
    label1 = f"Threshold: {config['threshold']:.2f} | Cooldown: {config['cooldown']:.1f}s | Sleep: {config['sleep']:.2f}s"
    label2 = f"Matching: {config.get('match_method', 'TM_CCOEFF_NORMED')}"
    label3 = f"Invert:{'Y' if config['preprocess_invert'] else 'N'} | Blur:{'Y' if config['preprocess_blur'] else 'N'} | Eq:{'Y' if config['preprocess_equalize'] else 'N'} | Sector:{'Y' if config.get('match_sector_enabled') else 'N'} | Mask:{'Y' if config.get('match_use_mask') else 'N'}"


    cv2.putText(frame, label1, (10, 30), DEF_FONT, 1, (0, 0, 0), 2)
    cv2.putText(frame, label1, (10, 30), DEF_FONT, 1, (255, 255, 255), 1)
    cv2.putText(frame, label2, (10, 65), DEF_FONT, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, label2, (10, 65), DEF_FONT, 0.8, (0, 255, 255), 1)
    cv2.putText(frame, label3, (10, 95), DEF_FONT, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, label3, (10, 95), DEF_FONT, 0.8, (0, 255, 255), 1)        

    show_processed_rois(frame, gray_frame, pre_matches, action_data)

    cv2.imshow("Webcam Bot", frame)


def take_shot(cap):
    if HEADLESS:
        print("🎥 Non è possibile acquisire uno screenshot in modalità HEADLESS")
        return
    
    print("🎥 Premi SPAZIO per scattare, ESC per uscire")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not HEADLESS:
            resize_win(
                "Scatta immagine", config["window_size"][0], config["window_size"][1]
            )
            cv2.imshow("Scatta immagine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        if key == 32: # SPAZIO
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            os.makedirs("img", exist_ok=True)
            filename = f"img/screenshot-{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Immagine salvata in {filename}")
            break
    cap.release()
    cv2.destroyAllWindows()


def choose_monitored_window():
    if win32gui is None:
        print("⚠️ Funzionalità non disponibile su questo sistema.")
        config["game_window_title"] = "unused"
        save_config(config)
        return
    titles = list_open_windows()
    if not titles:
        print("❌ Nessuna finestra attiva rilevata.")
        return

    print("🔍 Elenco delle finestre aperte:")
    titles.append("Unused")
    for i, t in enumerate(titles):
        print(f"  {i}: {t}")

    while True:
        choice = input("👉 Inserisci il numero della finestra da monitorare: ").strip()
        if choice.isdigit() and int(choice) in range(len(titles)):
            selected_title = titles[int(choice)]
            config["game_window_title"] = selected_title
            print(f"🎮 Finestra selezionata: '{selected_title}'")
            save_config(config)
            break
        else:
            print("❌ Scelta non valida. Riprova.")
    return

def handle_ocv_keys():
    dirty_config = False

    if HEADLESS:
        return False
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return True
    elif key == ord(","):
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
    elif key == ord("m"):
        available_methods = [
            "TM_CCOEFF_NORMED",
            "TM_CCORR_NORMED",
            "TM_SQDIFF_NORMED"
        ]
        current = config.get("match_method", "TM_CCOEFF_NORMED")
        idx = (available_methods.index(current) + 1) % len(available_methods)
        config["match_method"] = available_methods[idx]
        dirty_config = True
        print(f"🔁 Metodo di matching cambiato: {config['match_method']}")
    elif key == ord("i"):
        config["preprocess_invert"] = not config.get("preprocess_invert", True)
        dirty_config = True
        print(f"🌓 Inversione {'attivata' if config['preprocess_invert'] else 'disattivata'}")
    elif key == ord("b"):
        config["preprocess_blur"] = not config.get("preprocess_blur", True)
        dirty_config = True
        print(f"🌫️ Blur {'attivato' if config['preprocess_blur'] else 'disattivato'}")
    elif key == ord("e"):
        config["preprocess_equalize"] = not config.get("preprocess_equalize", True)
        dirty_config = True
        print(f"📊 Equalizzazione {'attivata' if config['preprocess_equalize'] else 'disattivata'}")
    elif key == ord("s"):
        config["match_sector_enabled"] = not config.get("match_sector_enabled", False)
        dirty_config = True
        print(f"🧩 Sector Matching {'attivato' if config['match_sector_enabled'] else 'disattivato'}")
    elif key == ord("u"):
        config["match_use_mask"] = not config.get("match_use_mask", False)
        dirty_config = True
        print(f"🎭 Uso maschere {'attivato' if config['match_use_mask'] else 'disattivato'}")

    if dirty_config:
        save_config(config)

    return False


def main():

    # --- Configurazione UDP ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(
        load_or_select_camera(args.reset_camera, args.reset_resolution)
    )

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

    if args.shot:
        take_shot(cap)
        return

    should_check_focus = True
    game_window = config.get("game_window_title")

    if game_window is None:
        choose_monitored_window()

    game_window = config.get("game_window_title")
    if game_window.lower() == "unused":
        should_check_focus = False

    if not config.get("game_window_title"):
        choose_monitored_window()

    actions = config.get("actions", {})
    if not actions:
        print("❌ Nessuna azione definita in config.json → 'actions'")
        return

    print("🔧 Azioni caricate:")
    for key in sorted(actions):
        entry = actions[key]
        print(f"  '{key}' → {entry['path'] if isinstance(entry, dict) else entry}")

    action_data = {}

    for key, info in actions.items():
        path = info.get("path") # template
        requires = info.get("requires", []) # dipendenze
        requires_not = info.get("requires_not", []) # esclusioni
        roi = info.get("roi") # roi
        if roi and (
            not isinstance(roi, list)
            or len(roi) != 4
            or not all(isinstance(x, int) for x in roi)
        ):
            print(f"⚠️ ROI non valido per '{key}': {roi}")
            info["roi"] = None

        if not os.path.exists(path):
            print(f"⚠️ Immagine mancante per '{key}': {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Impossibile caricare '{path}'")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        raw_mask = None
        mask_path = info.get("mask")
        if mask_path:
            if os.path.exists(mask_path):
                raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if raw_mask.shape != gray.shape:
                    print(f"⚠️ Maschera per '{key}' ha dimensioni diverse dal template!")
                if raw_mask is None:
                    print(f"⚠️ Impossibile caricare maschera '{mask_path}'")
            else:
                print(f"⚠️ Maschera non trovata per '{key}': {mask_path}")

        action_data[key] = {
            "requires": set(requires),
            "requires_not": set(requires_not),
            "roi": roi,
            "path": path,
            "send": info.get("send", True),
            "template": gray,
            "dimensions": gray.shape[::-1],
            "mask_data": raw_mask,
        }

    print("🔍 Bot attivo")
    if should_check_focus:
        print(f"🎯 Monitoraggio finestra: {game_window}")
    else:
        print("🎯 Nessun controllo finestra attiva (modalità 'Unused')")

    if not HEADLESS:
        print("q --> esce applicazione")
        print("+ --> aumenta threshold ricerca")
        print("- --> dimiuisce threshold ricerca")
        print(", --> aumenta cooldown")
        print(". --> diminuisce cooldown")
        print("* --> aumenta sleep frame")
        print("/ --> dimiuisce sleep frame")
        print("m --> cambia algoritmo matching (attuale:", config["match_method"], ")")
        print("i --> attiva/disattiva inversione")
        print("b --> attiva/disattiva blur")
        print("e --> attiva/disattiva equalizzazione")
        print("s --> attiva/disattiva sector matching")
        print("u --> attiva/disattiva uso maschere")

        resize_win("Webcam Bot", config["window_size"][0], config["window_size"][1])

    last_sent_time = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        process_frame(
            frame,
            action_data,
            last_sent_time,
            sock,
            should_check_focus,
        )

        if handle_ocv_keys():
            break

        time.sleep(config["sleep"])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
