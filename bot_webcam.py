import socket

# Solo su Windows
# per rilevare la finestra attiva
try:
    import win32gui
except ImportError:
    win32gui = None
# pre recuperare i nomi delle webcam
try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None
# ---

import cv_utils
import audio_utils
import conf_utils
from conf_utils import CONFIG


def is_game_window_focused(should_check_focus):
    if not should_check_focus:
        return True
    current = get_foreground_window_title()
    return current and current.lower() == CONFIG.get("game_window_title").lower()


# --- Selezione finestra gioco ---
def get_foreground_window_title():
    if win32gui is None:
        return None
    hwnd = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(hwnd)


def list_open_windows():
    titles = []
    if win32gui is None:
        return titles

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)

    win32gui.EnumWindows(enum_handler, None)
    return titles


# --- Camera selection ---
def load_or_select_camera(force_select=False, force_resolution=False):
    available = []
    selected_idx = None
    selected_name = None
    selected_caps = []
    use_filter_graph = True

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
                if cv_utils.try_cam(idx):
                    available.append((idx, name, caps))
        except Exception as e:
            print(f"⚠️ Errore durante l'uso di pygrabber: {e}")
            print("🔁 Passo al fallback con OpenCV.")
            use_filter_graph = False  # forza fallback

    if not use_filter_graph:
        # Fallback OpenCV
        for idx in range(10):
            if cv_utils.try_cam(idx):
                available.append((idx, f"Webcam {idx}", []))

    if not available:
        print("❌ Nessuna webcam funzionante trovata.")
        return

    if not force_select and "camera_name" in CONFIG:
        for idx, name, caps in available:
            if name == CONFIG.get("camera_name"):
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
                CONFIG["camera_index"] = selected_idx
                CONFIG["camera_name"] = selected_name
                conf_utils.save_config()
                print(
                    f"💾 Webcam selezionata salvata: '{selected_name}' (index {selected_idx})"
                )
                break
            else:
                print(f"❌ Inserisci un numero compreso tra 0 e {len(available) - 1}.")

    # Selezione risoluzione
    if force_resolution or "camera_resolution" not in CONFIG:
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
                    CONFIG["camera_resolution"] = [selected_width, selected_height]
                    conf_utils.save_config()
                    break
                else:
                    print(f"❌ Inserisci un numero tra 0 e {len(selected_caps) - 1}.")

    return selected_idx


# seleziona la finestra di gioco che deve avere il focus per abilitare l'invio
def choose_monitored_window():

    # se nella configurazione esiste restituisce il nome della finestra trovata
    game_window = CONFIG.get("game_window_title")
    if game_window is not None:
        return game_window

    # altrimenti prova a cercare le finestre (funzione disponibile solo su Windows)
    # se non disponibile win32gui disattiva
    if win32gui is None:
        print("⚠️ Funzionalità non disponibile su questo sistema.")
        CONFIG["game_window_title"] = "unused"
        conf_utils.save_config()
        return CONFIG.get("game_window_title")

    # se non ci sono finestre attive disattiva la funzione
    titles = list_open_windows()
    if not titles:
        print("❌ Nessuna finestra attiva rilevata.")
        CONFIG["game_window_title"] = "unused"
        conf_utils.save_config()
        return CONFIG.get("game_window_title")

    # chiede all'utente di scegliere la finestra da monitorare
    print("🔍 Elenco delle finestre aperte:")
    titles.append("Unused")
    for i, t in enumerate(titles):
        print(f"  {i}: {t}")

    while True:
        choice = input("👉 Inserisci il numero della finestra da monitorare: ").strip()
        if choice.isdigit() and int(choice) in range(len(titles)):
            selected_title = titles[int(choice)]
            CONFIG["game_window_title"] = selected_title
            print(f"🎮 Finestra selezionata: '{selected_title}'")
            conf_utils.save_config()
            break
        else:
            print("❌ Scelta non valida. Riprova.")
    return CONFIG.get("game_window_title")


def main():

    conf_utils.init_config()

    # Avvia il thread di riconoscimento vocale
    listener_thread = None
    if not conf_utils.args.test:
        listener_thread = audio_utils.start_listening()

    # Configurazione UDP per invio sequenze alla tastiera
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # recupera impostazioni cam
    camera_idx = load_or_select_camera(conf_utils.args.reset_camera, conf_utils.args.reset_resolution)

    try:
        # gestisce richiesta acquisizione fotogramma cam
        if conf_utils.args.shot:
            conf_utils.save_config()
            cv_utils.take_shot(camera_idx)
            return

        # gestisce selezione focus finestra
        should_check_focus = True
        game_window = choose_monitored_window()
        if game_window.lower() == "unused":
            should_check_focus = False
        # --

        # Caricamento azioni
        actions = CONFIG.get("actions", {})
        if not actions:
            print("❌ Nessuna azione definita in config.json → 'actions'")
            return

        print("🔧 Azioni caricate:")
        for key in sorted(actions):
            entry = actions[key]
            print(f"  '{key}' → {entry['path'] if isinstance(entry, dict) else entry}")

        # carica le azioni e valida e manipola le immagini da utilizzare
        actions_data = cv_utils.load_actions(actions)

        print("🔍 Bot attivo")
        if should_check_focus:
            print(f"🎯 Monitoraggio finestra: {game_window}")
        else:
            print("🎯 Nessun controllo finestra attiva (modalità 'Unused')")

        # se non sono in modalità headless stampa i comandi accettati
        if not CONFIG.get("headless"):
            print("q --> esce")
            print("+ --> aumenta threshold")
            print("- --> dimiuisce threshold")
            print(", --> aumenta cooldown")
            print(". --> diminuisce cooldown")
            print("* --> aumenta sleep frame")
            print("/ --> dimiuisce sleep frame")
            print(
                "m --> cambia algoritmo matching (attuale:", CONFIG.get("match_method"), ")"
            )
            print("i --> attiva/disattiva inversione")
            print("b --> attiva/disattiva blur")
            print("e --> attiva/disattiva equalizzazione")
            print("s --> attiva/disattiva sector matching")
            print("u --> attiva/disattiva uso maschere")

        windows_focused = is_game_window_focused(should_check_focus)

        cv_utils.frame_loop(camera_idx, actions_data, sock, windows_focused, conf_utils.args.test)
    finally:
        # ferma listener vocale
        if listener_thread:
            audio_utils.stop_listening(listener_thread)
        conf_utils.save_config()


if __name__ == "__main__":
    main()
