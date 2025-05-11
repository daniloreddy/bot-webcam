"""
Module conf_utils.py

Handles loading, parsing, and saving the application configuration and actions for the
webcam bot. Provides functions to initialize configuration from defaults,
command-line arguments, and existing JSON files, and to persist changes to
separate config.json and actions.json files.
Adds support for specifying a custom actions file via CLI with persistence.
"""

import argparse
import os
import json
import platform
from typing import Any, Dict, Optional

# Parsed command-line arguments
args: Optional[argparse.Namespace] = None

# Global configuration and action dictionaries
CONFIG: Dict[str, Any] = {}
ACTIONS: Dict[str, Any] = {}

# Default file paths
CONF_DIR = "conf"
MODELS_DIR = "model"
IMG_DIR = "img"
CONFIG_PATH: str = os.path.normpath(os.path.join(CONF_DIR, "config.json"))
ACTIONS_PATH: str = os.path.normpath(os.path.join(CONF_DIR, "actions.json"))

# Default values for configuration keys
DEFAULTS: Dict[str, Any] = {
    "window_size": [1280, 720],
    "sleep": 0.2,
    "cooldown": 1.0,
    "threshold": 0.85,
    "ip": "192.168.178.89",
    "port": 12345,
    "headless": None,
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
    "audio_model_path": os.path.normpath(
        os.path.join(MODELS_DIR, "vosk-model-small-it-0.22")
    ),
    "audio_activate_cmd": "attiva",
    "audio_deactivate_cmd": "disattiva",
    "audio_exit_cmd": "chiudi",
    # Actions file default
    "actions_file": "actions.json",
}


def init_config() -> None:
    """
    Initialize the global CONFIG and ACTIONS dictionaries.

    Merges values from DEFAULTS, existing config.json and actions.json (or custom file) files,
    and command-line arguments. Handles reset of configuration, sets
    platform-specific defaults (e.g. headless mode), and persists the results
    back to config.json and actions.json (or custom actions file).

    Side effects:
        - Parses sys.argv via argparse
        - Reads and writes CONFIG_PATH and target actions file
        - Updates module-level CONFIG, ACTIONS, and args
    """
    global args, ACTIONS, ACTIONS_PATH

    os.makedirs("img", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("conf", exist_ok=True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Webcam bot for image recognition")
    parser.add_argument(
        "--shot",
        "-s",
        action="store_true",
        help="Capture image from webcam and save to img/",
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Test mode: do not send, only print"
    )
    parser.add_argument(
        "--reset-camera", "-rc", action="store_true", help="Force webcam selection"
    )
    parser.add_argument(
        "--reset-resolution",
        "-rr",
        action="store_true",
        help="Force webcam resolution selection",
    )
    parser.add_argument(
        "--reset-config",
        "-r",
        action="store_true",
        help="Reset all saved settings in config.json and actions file",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Force headless mode (no GUI)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Force GUI mode (not headless)",
    )
    parser.add_argument(
        "--cooldown",
        "-c",
        type=float,
        default=DEFAULTS["cooldown"],
        help="Seconds of pause between two sends of the same key",
    )
    parser.add_argument(
        "--threshold",
        "-T",
        type=float,
        default=DEFAULTS["threshold"],
        help="Matching threshold between 0.0 and 1.0",
    )
    parser.add_argument(
        "--ip",
        "-i",
        type=str,
        default=DEFAULTS["ip"],
        help="IP address of the target device",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULTS["port"],
        help="UDP port of the target device",
    )
    parser.add_argument(
        "--actions-file",
        "-a",
        type=str,
        default=DEFAULTS["actions_file"],
        help="Path to the actions JSON file",
    )
    parser.set_defaults(headless=DEFAULTS["headless"])

    args = parser.parse_args()

    # Override default actions path if provided and store in config
    ACTIONS_PATH = os.path.normpath(os.path.join(CONF_DIR, args.actions_file))

    CONFIG["actions_file"] = ACTIONS_PATH

    # Reset configuration and actions if requested
    if args.reset_config:
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
        if os.path.exists(ACTIONS_PATH):
            os.remove(ACTIONS_PATH)
        print("🔁 Configuration and actions reset.")

    CONFIG.clear()

    # Load existing config.json
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG.update(json.load(f))

    ACTIONS.clear()
    # Load actions from specified file or fallback to embedded
    if os.path.exists(ACTIONS_PATH):
        with open(ACTIONS_PATH, "r", encoding="utf-8") as f:
            ACTIONS.update(json.load(f))
    else:
        embedded = CONFIG.get("actions")
        if embedded is not None:
            ACTIONS.update(embedded)

    # Merge defaults, saved config, and CLI overrides into CONFIG
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

    # Fill remaining keys from loaded_config or defaults
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

    # Headless logic: auto-detect if not overridden
    if args.headless is None:
        headless_val = CONFIG.get("headless")
        if headless_val is None:
            if platform.system() == "Darwin":
                headless_val = False
            else:
                headless_val = not os.environ.get("DISPLAY") and os.name != "nt"
        CONFIG["headless"] = headless_val
    else:
        CONFIG["headless"] = args.headless

    # Persist configuration and actions to disk
    save_config()


def save_config() -> None:
    """
    Write the current CONFIG dictionary to CONFIG_PATH in JSON format.

    Only program configuration keys are written; embedded actions are omitted.
    """
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)
    with open(ACTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(ACTIONS, f, indent=2)
