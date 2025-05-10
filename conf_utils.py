"""
Module conf_utils.py

Handles loading, parsing, and saving the application configuration for the
webcam bot. Provides functions to initialize configuration from defaults,
command-line arguments, and an existing JSON file, and to persist changes.
"""

import argparse
import os
import json
import platform
from typing import Any, Dict, Optional

# Parsed command-line arguments
args: Optional[argparse.Namespace] = None

# Global configuration dictionary used by the application
CONFIG: Dict[str, Any] = {}

# Path to the JSON file where configuration is saved/loaded
CONFIG_PATH: str = "config.json"

# Default values for configuration keys
DEFAULTS: Dict[str, Any] = {
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
    # Audio control defaults
    "audio_model_path": "model/vosk-model-small-it-0.22",
    "audio_activate_cmd": "attiva",
    "audio_deactivate_cmd": "disattiva",
    "audio_exit_cmd": "chiudi",
}


def init_config() -> None:
    """
    Initialize the global CONFIG dictionary.

    Merges values from DEFAULTS, an existing config.json file, and
    command-line arguments. Handles reset of the configuration, sets
    platform-specific defaults (e.g. headless mode), and persists
    the result back to config.json.

    Side effects:
        - Parses sys.argv via argparse
        - Reads and writes CONFIG_PATH
        - Updates the module-level CONFIG dict
        - Updates the module-level args

    Returns:
        None
    """
    global args

    # Parse command-line arguments
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
        default=DEFAULTS["cooldown"],
        help="Secondi di pausa tra due invii dello stesso tasto",
    )
    parser.add_argument(
        "--threshold",
        "-T",
        type=float,
        default=DEFAULTS["threshold"],
        help="Soglia di matching tra 0.0 e 1.0",
    )
    parser.add_argument(
        "--ip",
        "-i",
        type=str,
        default=DEFAULTS["ip"],
        help="Indirizzo IP del dispositivo di destinazione",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULTS["port"],
        help="Porta UDP del dispositivo di destinazione",
    )

    args = parser.parse_args()

    # Reset configuration if requested
    if args.reset_config and os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)
        print("🔁 Configurazione azzerata.")

    # Load existing config.json if present
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        CONFIG.clear()
        CONFIG.update(loaded)

    # Merge defaults, saved config, and CLI overrides
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

    # Fill remaining keys from config or defaults
    for key in [
        "sleep",
        "window_size",
        "game_window_title",
        "match_method",
        "preprocess_invert",
        "preprocess_blur",
        "preprocess_equalize",
        "audio_activate_cmd",
        "audio_deactivate_cmd",
        "audio_exit_cmd",
        "audio_model_path",
    ]:
        CONFIG[key] = CONFIG.get(key, DEFAULTS[key])

    # Determine headless mode if not specified
    headless_val = CONFIG.get("headless")
    if headless_val is None:
        # On macOS use GUI, otherwise check DISPLAY env var
        if platform.system() == "Darwin":
            headless_val = False
        else:
            headless_val = not os.environ.get("DISPLAY") and os.name != "nt"
        CONFIG["headless"] = headless_val
    else:
        CONFIG["headless"] = bool(headless_val)

    # Persist configuration
    save_config()


def save_config() -> None:
    """
    Write the current CONFIG dictionary to CONFIG_PATH in JSON format.

    Overwrites any existing file. Creates human-readable indent.

    Returns:
        None
    """
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)
