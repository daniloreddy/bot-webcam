import argparse
import os
import json
import platform

args = None
CONFIG = {}
CONFIG_PATH = "config.json"
DEFAULTS = {
    "window_size": [1280, 720],
    "sleep": 0.2,
    "cooldown": 1.0,
    "threshold": 0.85,
    "ip": "192.168.178.89",
    "port": 12345,
    "headless": False,
    "game_window_title": None,
    "match_method": "TM_CCOEFF_NORMED",
    "preprocess_invert": False,
    "preprocess_blur": False,
    "preprocess_equalize": False,
    "match_sector_enabled": False,
    "match_sector_grid": [2, 2],
    "match_sector_min_success": 3,
    "match_use_mask": False,
}


def init_config():

    global args

    print("Controllo parametri riga di comando...")
    # --- Argomenti da linea di comando ---
    parser = argparse.ArgumentParser(
        description="Webcam bot per riconoscimento immagini"
    )
    parser.add_argument(
        "--shot",
        "-s",
        action="store_true",
        help="Scatta immagine dalla webcam e salva in img/",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Modalità test: non invia, stampa solo",
    )
    parser.add_argument(
        "--reset-camera",
        "-rc",
        action="store_true",
        help="Forza la selezione della webcam",
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
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            conf = json.load(f)
            CONFIG.clear()
            CONFIG.update(conf)

    print("Inizializza configurazione...")
    CONFIG["cooldown"] = (
        args.cooldown
        if args.cooldown != parser.get_default("cooldown")
        else CONFIG.get("cooldown", DEFAULTS["cooldown"])
    )
    CONFIG["threshold"] = (
        args.threshold
        if args.threshold != parser.get_default("threshold")
        else CONFIG.get("threshold", DEFAULTS["threshold"])
    )
    CONFIG["ip"] = (
        args.ip
        if args.ip != parser.get_default("ip")
        else CONFIG.get("ip", DEFAULTS["ip"])
    )
    CONFIG["port"] = (
        args.port
        if args.port != parser.get_default("port")
        else CONFIG.get("port", DEFAULTS["port"])
    )
    CONFIG["sleep"] = CONFIG.get("sleep", DEFAULTS["sleep"])
    CONFIG["window_size"] = CONFIG.get("window_size", DEFAULTS["window_size"])
    CONFIG["game_window_title"] = CONFIG.get(
        "game_window_title", DEFAULTS["game_window_title"]
    )
    CONFIG["match_method"] = CONFIG.get("match_method", DEFAULTS["match_method"])
    CONFIG["preprocess_invert"] = CONFIG.get(
        "preprocess_invert", DEFAULTS["preprocess_invert"]
    )
    CONFIG["preprocess_blur"] = CONFIG.get(
        "preprocess_blur", DEFAULTS["preprocess_blur"]
    )
    CONFIG["preprocess_equalize"] = CONFIG.get(
        "preprocess_equalize", DEFAULTS["preprocess_equalize"]
    )

    # HEADLESS: usa valore da config se presente, altrimenti autodetect
    headless = CONFIG.get("headless")
    if headless is None:
        if platform.system() == "Darwin":  # macOS
            headless = False
        else:
            headless = not os.environ.get("DISPLAY") and os.name != "nt"
        CONFIG["headless"] = headless
    else:
        headless = bool(headless)

    save_config()


def save_config():
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)
