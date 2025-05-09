import os
from datetime import datetime
import time

import cv2
import numpy as np

from conf_utils import CONFIG
import audio_utils

# --- Voice Control Flags ---
DEF_FONT = cv2.FONT_HERSHEY_SIMPLEX


def resize_win(name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, width, height)


def try_cam(idx):
    cam_ok = False
    capture = cv2.VideoCapture(idx)
    if capture.read()[0]:
        cam_ok = True
    capture.release()
    return cam_ok


def load_camera(idx):
    cap = cv2.VideoCapture(idx)

    # prova ad impostare risoluzione custom
    if "camera_resolution" in CONFIG:
        width, height = CONFIG.get("camera_resolution")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    CONFIG["camera_resolution"] = [int(width), int(height)]

    print("📏 Risoluzione selezionata:")
    print("   Larghezza:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("   Altezza:  ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap


# Preprocessa l'immagine applicando effetti per cercare di migliorare
# il match del template
def preprocess_image(img):

    preprocessed = img

    # converte in scala di grigi (se non è gia stato fatto), questo deve essere sempre fatto perchè il template matching funziona meglio con le immagini in scala di grigi
    if len(img.shape) == 3 and img.shape[2] == 3:
        preprocessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # inverte i colori
    if CONFIG.get("preprocess_invert", True):
        preprocessed = cv2.bitwise_not(preprocessed)

    # applicata leggera sfocatura
    if CONFIG.get("preprocess_blur", True):
        preprocessed = cv2.GaussianBlur(preprocessed, (3, 3), 0)

    # applica equalizzazione
    if CONFIG.get("preprocess_equalize", True):
        preprocessed = cv2.equalizeHist(preprocessed)

    return preprocessed


# --- ROI extraction utility ---
def extract_roi(image, roi):
    if not roi:
        return image
    x, y, w, h = roi
    return image[y : y + h, x : x + w]


# individua il template nella regione di interesse del frame (o nell'interno frame se non è presente una regione di interesse)
def match_with_roi(frame, action):

    template = action.get("template")
    mask = action.get("mask_data")
    roi = action.get("roi")

    if roi is not None:
        region = extract_roi(frame, roi)
    else:
        region = frame

    region = preprocess_image(region)
    template = preprocess_image(template)
    # se presente una maschera preprocessa anceh la maschera (TODO: da valutare se necessario)
    if mask is not None:
        mask = preprocess_image(mask)

    # metodo di ricerca immagine
    method_name = CONFIG.get("match_method", "TM_CCOEFF_NORMED")
    method = getattr(cv2, method_name, cv2.TM_CCOEFF_NORMED)

    use_mask = (
        CONFIG.get("match_use_mask", False)
        and mask is not None
        and method_name in ["TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"]
    )

    threshold = CONFIG.get("threshold")
    if CONFIG.get("match_sector_enabled", False):  # ricerca per settori
        return match_template_in_sectors(
            region, template, mask, threshold, use_mask, method
        )
    else:  # ricerca immagine completa
        return match_template(region, template, mask, threshold, use_mask, method)


# ricerca il template nella regione di interesse
def match_template(
    region,  # regione di ricerca dell'immagine
    template,  # template da cercare
    mask,  # maschera da utilizzare
    threshold,  # soglia di rilevamento
    use_mask,  # flag che indica se deve essere utilizzata la maschera nella ricerca
    method,  # metodo di ricerca
):

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


# ricerca il template nella regione di interesse dividendo il template in settori
def match_template_in_sectors(
    region,  # regione di ricerca dell'immagine
    template,  # template da cercare
    mask,  # maschera da utilizzare
    threshold,  # soglia di rilevamento
    use_mask,  # flag che indica se deve essere utilizzata la maschera nella ricerca
    method,  # metodo di ricerca
):
    rows, cols = CONFIG.get("match_sector_grid", [2, 2])
    required = CONFIG.get("match_sector_min_success", 3)

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
            sector_template = template[y : y + sector_h, x : x + sector_w]
            sector_mask = mask[y : y + sector_h, x : x + sector_w] if use_mask else None

            try:
                if use_mask:
                    result = cv2.matchTemplate(
                        region, sector_template, method, mask=sector_mask
                    )
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


# mostra feedback e informazioni come overlay dell'immagine acquisita
def update_overlay(size, actions):

    print("Rigenera overlay")

    h, w = size
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # testo 1
    label1 = f"Threshold: {CONFIG['threshold']:.2f} | Cooldown: {CONFIG['cooldown']:.1f}s | Sleep: {CONFIG['sleep']:.2f}s"
    cv2.putText(overlay, label1, (10, 30), DEF_FONT, 1, (0, 0, 0), 2)
    cv2.putText(overlay, label1, (10, 30), DEF_FONT, 1, (255, 255, 255), 1)

    # testo 2
    label2 = f"Matching: {CONFIG.get('match_method')}"
    cv2.putText(overlay, label2, (10, 65), DEF_FONT, 0.8, (0, 0, 0), 2)
    cv2.putText(overlay, label2, (10, 65), DEF_FONT, 0.8, (0, 255, 255), 1)

    label3 = f"Invert:{'Y' if CONFIG['preprocess_invert'] else 'N'} | Blur:{'Y' if CONFIG['preprocess_blur'] else 'N'} | Eq:{'Y' if CONFIG['preprocess_equalize'] else 'N'} | Sector:{'Y' if CONFIG.get('match_sector_enabled') else 'N'} | Mask:{'Y' if CONFIG.get('match_use_mask') else 'N'}"
    cv2.putText(overlay, label3, (10, 95), DEF_FONT, 0.8, (0, 0, 0), 2)
    cv2.putText(overlay, label3, (10, 95), DEF_FONT, 0.8, (0, 255, 255), 1)

    # Fase 1: Rilevamento iniziale (senza dipendenze)
    for _, action_data in actions.items():
        roi = action_data.get("roi")
        # se presente roi la disegna per avere feedback visivo
        if roi:
            x, y, w, h = roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return overlay


# processa il frame acquisito dalla cam
def process_frame(
    frame,  # immagine acquisita dalla camera
    actions,  # azioni caricate
    last_sent_time,  # timestamp ultimi invii
    sock,  # socket invio key
    windows_focused,  # flag che indica se finestar gioco ha focus
    test,  # flag che indica se bot avviato in modalità test
    overlay,  # immagine da mostrare sopra al frame con le info
):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pre_matches = {}  # template trovati
    confirmed_matches = {}  # templati risconosciuti dopo validazione dipendenze

    # Fase 1: Rilevamento iniziale (senza dipendenze)
    for key, action_data in actions.items():
        roi = action_data.get("roi")
        matched, max_val, max_loc = match_with_roi(gray, action_data)

        if matched:
            pre_matches[key] = (max_val, max_loc)

    # Fase 2: Validazione con dipendenze
    for key, (max_val, max_loc) in pre_matches.items():
        action_data = actions.get(key, {})
        required = set(action_data.get("requires", []))
        required_not = set(action_data.get("requires_not", []))

        if not required.issubset(pre_matches.keys()):  # controlla dipendenze richieste
            continue
        if required_not & pre_matches.keys():  # controlla dipendenze non richieste
            continue

        confirmed_matches[key] = (max_val, max_loc)

    # Fase 3: Visualizzazione ed invio tasti
    for key, (max_val, max_loc) in confirmed_matches.items():
        action_data = actions.get(key, {})
        now = time.time()

        # valuta se inviare la sequenza alla tastiera
        should_send = (
            not test  # non devo essere in test
            and actions[key].get(
                "send", True
            )  # deve essere presente configurazione "send" a true
            and (  # valuta cooldown tra invii per evitare spam dei tasti
                key not in last_sent_time
                or (now - last_sent_time[key]) >= CONFIG.get("cooldown")
            )
            and windows_focused  # deve essere selezionata la finestra del gioco
            and audio_utils.active_event.is_set()
        )

        if should_send:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(
                f"[{timestamp}] ✅ Match '{key}' ({max_val:.2f}) → invio a {CONFIG['ip']}:{CONFIG['port']}"
            )
            sock.sendto(key.encode(), (CONFIG.get("ip"), CONFIG.get("port")))
            last_sent_time[key] = now

        if not CONFIG.get("headless"):
            # Disegna il feedback del rilevamento
            roi = action_data.get("roi")
            w, h = action_data.get("dimensions")
            top_left = (max_loc[0] + roi[0], max_loc[1] + roi[1]) if roi else max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            label = f"{key} ({max_val:.2f})"
            cv2.putText(
                frame,
                label,
                (top_left[0], top_left[1] - 10),
                DEF_FONT,
                0.6,
                (0, 255, 0),
                2,
            )

    if CONFIG.get("headless"):
        return

    draw_match_preview(frame, gray, pre_matches, actions)
    combined = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
    cv2.imshow("Webcam Bot", combined)


# mostra in sovraimpressione i template trovati
def draw_match_preview(
    frame,  # immagine acquisita dalla camera
    gray_frame,  # immagine in scala di grigi
    matches,  # elementi individuati
    action_data,  # azioni
):
    thumb_size = 100  # dimensione anteprima
    margin = 10
    x_offset = margin
    y_offset = frame.shape[0] - thumb_size - margin

    for key in sorted(matches):
        roi = action_data[key].get("roi")
        if not roi:
            continue

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


def get_cv_key():
    return cv2.waitKey(1) & 0xFF


# acquisisce uno screen di quello che vede la webcam
# da utilizzare per costruire i template e le maschere
# di riconoscimento
def take_shot(camera_idx):

    if CONFIG.get("headless"):
        print("🎥 Non è possibile acquisire uno screenshot in modalità HEADLESS")
        return

    cap = load_camera(camera_idx)

    print("🎥 Premi SPAZIO per scattare, ESC per uscire")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        win_title = "Scatta immagine"
        resize_win(
            win_title, CONFIG.get("window_size")[0], CONFIG.get("window_size")[1]
        )
        cv2.imshow(win_title, frame)

        key = get_cv_key()
        if key == 27:  # ESC
            break
        if key == 32:  # SPAZIO
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            os.makedirs("img", exist_ok=True)
            filename = f"img/screenshot-{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Immagine salvata in {filename}")
            break
    cap.release()
    cv2.destroyAllWindows()


# carica le actions definite dall'utente caricando e validando le immagini
def load_actions(actions):

    actions_data = {}

    for key, info in actions.items():

        requires = info.get("requires", [])  # dipendenze
        requires_not = info.get("requires_not", [])  # esclusioni

        # controlla template
        path = info.get("path")  # template
        if not os.path.exists(path):
            print(f"⚠️ Immagine mancante per '{key}': {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Impossibile caricare '{path}'")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # controlla roi
        roi = info.get("roi")  # roi
        if roi and (
            not isinstance(roi, list)
            or len(roi) != 4
            or not all(isinstance(x, int) for x in roi)
        ):
            print(f"⚠️ ROI non valido per '{key}': {roi}")
            info["roi"] = None

        # controlla maschera
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

        actions_data[key] = {
            "requires": set(requires),  # dipendenza: azioni richieste
            "requires_not": set(requires_not),  # dipendenza: azioni non richieste
            "roi": roi,  # region of interest
            "path": path,  # path del template
            "send": info.get("send", True),  # should send key
            "template": gray,  # immagine scala di grigi
            "dimensions": gray.shape[::-1],  # dimensioni immagine
            "mask_data": raw_mask,  # mascherda
        }

    return actions_data


# gestisce i comandi che modificano i parametri di riconoscimento e l'uscita dal programma in grafica
def handle_ocv_keys():

    dirty_config = False

    key = get_cv_key()
    if key == ord("q"):
        return True, True
    elif key == ord(","):  # aumenta cooldown invio tasti
        CONFIG["cooldown"] = min(10.0, CONFIG.get("cooldown", 1.0) + 0.1)
        dirty_config = True
        print(f"⏫ Cooldown aumentato: {CONFIG['cooldown']:.2f}s")
    elif key == ord("."):  # riduce  cooldown invio tasti
        CONFIG["cooldown"] = max(0.0, CONFIG.get("cooldown", 1.0) - 0.1)
        dirty_config = True
        print(f"⏬ Cooldown diminuito: {CONFIG['cooldown']:.2f}s")
    elif key == ord("*"):  # aumenta sleep tra un frame processato e l'altro
        CONFIG["sleep"] = min(5.0, CONFIG.get("sleep", 0.2) + 0.05)
        dirty_config = True
        print(f"⏫ Sleep aumentato: {CONFIG['sleep']:.2f}s")
    elif key == ord("/"):  # riduce sleep tra un frame processato e l'altro
        CONFIG["sleep"] = max(0.0, CONFIG.get("sleep", 0.2) - 0.05)
        dirty_config = True
        print(f"⏬ Sleep diminuito: {CONFIG['sleep']:.2f}s")
    elif key == ord("+") or key == ord("="):  # aumenta la soglia di riconoscimento
        CONFIG["threshold"] = min(1.0, CONFIG.get("threshold", 0.85) + 0.01)
        dirty_config = True
        print(f"🔼 Threshold aumentato: {CONFIG['threshold']:.2f}")
    elif key == ord("-"):  # riduce la soglia di riconoscimento
        CONFIG["threshold"] = max(0.0, CONFIG.get("threshold", 0.85) - 0.01)
        dirty_config = True
        print(f"🔽 Threshold diminuito: {CONFIG['threshold']:.2f}")
    elif key == ord("m"):  # cambia il metodo di match del template
        available_methods = ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]
        current = CONFIG.get("match_method", "TM_CCOEFF_NORMED")
        idx = (available_methods.index(current) + 1) % len(available_methods)
        CONFIG["match_method"] = available_methods[idx]
        dirty_config = True
        print(f"🔁 Metodo di matching cambiato: {CONFIG['match_method']}")
    elif key == ord("i"):  # inverte i colori dell'immagine
        CONFIG["preprocess_invert"] = not CONFIG.get("preprocess_invert", True)
        dirty_config = True
        print(
            f"🌓 Inversione {'attivata' if CONFIG['preprocess_invert'] else 'disattivata'}"
        )
    elif key == ord("b"):  # applica blur all'immagine
        CONFIG["preprocess_blur"] = not CONFIG.get("preprocess_blur", True)
        dirty_config = True
        print(f"🌫️ Blur {'attivato' if CONFIG['preprocess_blur'] else 'disattivato'}")
    elif key == ord("e"):  # equalizza immagine
        CONFIG["preprocess_equalize"] = not CONFIG.get("preprocess_equalize", True)
        dirty_config = True
        print(
            f"📊 Equalizzazione {'attivata' if CONFIG['preprocess_equalize'] else 'disattivata'}"
        )
    elif key == ord("s"):  # attiva matching in settori
        CONFIG["match_sector_enabled"] = not CONFIG.get("match_sector_enabled", False)
        dirty_config = True
        print(
            f"🧩 Sector Matching {'attivato' if CONFIG['match_sector_enabled'] else 'disattivato'}"
        )
    elif key == ord("u"):  # attiva l'uso delle maschere
        CONFIG["match_use_mask"] = not CONFIG.get("match_use_mask", False)
        dirty_config = True
        print(
            f"🎭 Uso maschere {'attivato' if CONFIG['match_use_mask'] else 'disattivato'}"
        )

    return False, dirty_config


# loop principale di acquisizione video, recupera i frame dalla fotocamera e li processa
def frame_loop(
    camera_idx,  # indice camera da utilizzare
    actions_data,  # azioni caricate
    sock,  # socket UDP a cui inviare
    windows_focused,  # finestra di gioco selezionata da focus
    test,  # programma avviato in modalità test
):

    last_sent_time = {}
    cap = load_camera(camera_idx)

    if not CONFIG.get("headless"):
        resize_win(
            "Webcam Bot", CONFIG.get("window_size")[0], CONFIG.get("window_size")[1]
        )

    error_count = 0
    max_errors = 10
    overlay = None

    # ciclo principale
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                error_count += 1
                if error_count >= max_errors:
                    print("⚠️ Troppe letture fallite...")
                    break
                continue
            error_count = 0

            if overlay is None:
                overlay = update_overlay(frame.shape[:2], actions_data)

            process_frame(
                frame,
                actions_data,
                last_sent_time,
                sock,
                windows_focused,
                test,
                overlay,
            )

            exit_loop, dirty_config = handle_ocv_keys()

            if dirty_config:
                overlay = update_overlay(frame.shape[:2], actions_data)

            if exit_loop or audio_utils.exit_event.is_set():
                break

            sleep = CONFIG.get("sleep")
            if sleep and sleep > 0:
                time.sleep(CONFIG.get("sleep"))
    finally:
        cap.release()
        cv2.destroyAllWindows()
