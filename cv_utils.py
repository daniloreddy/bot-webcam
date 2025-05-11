import os
from datetime import datetime
import time
from socket import socket as Socket
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np

from conf_utils import CONFIG, IMG_DIR
import audio_utils
import tui_utils

# --- Voice Control Flags ---
DEF_FONT = cv2.FONT_HERSHEY_SIMPLEX


def resize_win(name: str, width: int, height: int) -> None:
    """
    Resize an OpenCV window while preserving aspect ratio.

    Parameters:
        name (str): Name of the OpenCV window (created via cv2.namedWindow).
        width (int): Desired window width in pixels.
        height (int): Desired window height in pixels.
    """
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, width, height)


def try_cam(idx: int) -> bool:
    """
    Test whether a camera device can be opened and returns a frame.

    Parameters:
        idx (int): Index of the camera device to test (e.g., 0 for the first webcam).

    Returns:
        bool: True if the camera opens and a frame is successfully captured; False otherwise.
    """
    capture = cv2.VideoCapture(idx)
    ok, _ = capture.read()
    capture.release()
    return bool(ok)


def load_camera(idx: int) -> cv2.VideoCapture:
    """
    Open and configure the camera for frame capture, updating CONFIG with actual resolution.

    Parameters:
        idx (int): Index of the camera device to open.

    Returns:
        cv2.VideoCapture: The configured VideoCapture object.
    """
    cap = cv2.VideoCapture(idx)

    # Apply custom resolution if provided in CONFIG
    if "camera_resolution" in CONFIG:
        width, height = CONFIG.get("camera_resolution", [0, 0])  # type: ignore
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read back the actual resolution and store it in CONFIG
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CONFIG["camera_resolution"] = [actual_width, actual_height]

    print("📏 Risoluzione selezionata:")
    print(f"   Larghezza: {actual_width}")
    print(f"   Altezza:   {actual_height}")
    return cap


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing steps to improve template matching accuracy.

    Steps applied, in order:
    1. Convert BGR to grayscale (if needed).
    2. Invert colors (if CONFIG['preprocess_invert'] is True).
    3. Apply Gaussian blur (if CONFIG['preprocess_blur'] is True).
    4. Equalize histogram (if CONFIG['preprocess_equalize'] is True).

    Parameters:
        img (np.ndarray): Input image in BGR or grayscale format.

    Returns:
        np.ndarray: Preprocessed grayscale image.
    """
    result = img.copy()

    # Convert to grayscale if input is BGR
    if result.ndim == 3 and result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Optionally invert
    if CONFIG.get("preprocess_invert", True):
        result = cv2.bitwise_not(result)

    # Optionally blur
    if CONFIG.get("preprocess_blur", True):
        result = cv2.GaussianBlur(result, (3, 3), 0)

    # Optionally equalize histogram
    if CONFIG.get("preprocess_equalize", True):
        result = cv2.equalizeHist(result)

    return result


def extract_roi(
    image: np.ndarray, roi: Optional[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Crop the given image to a specified region of interest.

    Parameters:
        image (np.ndarray): Source image (grayscale or BGR).
        roi (Optional[Tuple[int, int, int, int]]):
            A tuple (x, y, w, h) specifying the top-left corner and size,
            or None to return the full image.

    Returns:
        np.ndarray: The cropped ROI image, or the original image if roi is None or empty.
    """
    if not roi:
        return image

    x, y, w, h = roi
    return image[y : y + h, x : x + w]


def match_with_roi(
    frame: np.ndarray, action: Dict[str, Any]
) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
    """
    Perform template matching within an optional region of interest.

    Steps:
      1. Extract ROI from frame if 'roi' in action.
      2. Preprocess region and template (and mask if present).
      3. Choose matching method from CONFIG.
      4. Dispatch to full or sector-based matching.

    Parameters:
        frame (np.ndarray): Grayscale source image.
        action (Dict[str, Any]): Dict containing:
            - 'template' (np.ndarray): Template image.
            - 'mask_data' (Optional[np.ndarray]): Optional mask.
            - 'roi' (Optional[Tuple[int,int,int,int]]): Region of interest.

    Returns:
        Tuple[bool, float, Optional[Tuple[int,int]]]:
            matched (bool): True if match score >= threshold.
            score (float): Best match score.
            location (Tuple[int,int] or None): Coordinates of best match.
    """
    # Extract template, mask, and ROI settings
    template_img: np.ndarray = action.get("template")  # type: ignore
    mask_img: Optional[np.ndarray] = action.get("mask_data")  # type: ignore
    roi: Optional[Tuple[int, int, int, int]] = action.get("roi")  # type: ignore

    # Determine search region
    search_region: np.ndarray
    if roi:
        search_region = extract_roi(frame, roi)
    else:
        search_region = frame

    # Preprocess images
    proc_region = preprocess_image(search_region)
    proc_template = preprocess_image(template_img)
    proc_mask = preprocess_image(mask_img) if mask_img is not None else None

    # Configure method and mask usage
    method_name: str = CONFIG.get("match_method", "TM_CCOEFF_NORMED")  # type: ignore
    method: int = getattr(cv2, method_name, cv2.TM_CCOEFF_NORMED)
    threshold: float = float(CONFIG.get("threshold", 0.0))  # type: ignore
    use_mask: bool = (
        CONFIG.get("match_use_mask", False)  # type: ignore
        and proc_mask is not None
        and method_name in ["TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"]
    )

    # Dispatch to sector or full matching
    if CONFIG.get("match_sector_enabled", False):  # type: ignore
        return match_template_in_sectors(
            proc_region, proc_template, proc_mask, threshold, use_mask, method
        )
    return match_template(
        proc_region, proc_template, proc_mask, threshold, use_mask, method
    )


def match_template(
    region: np.ndarray,
    template: np.ndarray,
    mask: Optional[np.ndarray],
    threshold: float,
    use_mask: bool,
    method: int,
) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
    """
    Perform standard template matching using OpenCV matchTemplate.

    Parameters:
        region (np.ndarray): Search region image.
        template (np.ndarray): Template to search for.
        mask (Optional[np.ndarray]): Optional mask for matching.
        threshold (float): Minimum score to consider a match.
        use_mask (bool): If True, apply mask in matchTemplate.
        method (int): OpenCV matching method constant.

    Returns:
        Tuple[bool, float, Optional[Tuple[int,int]]]:
            matched (bool): True if best score >= threshold.
            score (float): Best match score.
            location (Tuple[int,int] or None): Coordinates of match.
    """
    try:
        if use_mask and mask is not None:
            result = cv2.matchTemplate(region, template, method, mask=mask)
        else:
            result = cv2.matchTemplate(region, template, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            return True, max_val, max_loc
        return False, max_val, None
    except Exception:
        return False, 0.0, None


def match_template_in_sectors(
    region: np.ndarray,
    template: np.ndarray,
    mask: Optional[np.ndarray],
    threshold: float,
    use_mask: bool,
    method: int,
) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
    """
    Divide the template into a grid and match each sector independently.

    Parameters:
        region (np.ndarray): Search region image.
        template (np.ndarray): Full template image.
        mask (Optional[np.ndarray]): Mask image or None.
        threshold (float): Score threshold per sector.
        use_mask (bool): If True, apply mask per sector.
        method (int): OpenCV matching method constant.

    Returns:
        Tuple[bool, float, Optional[Tuple[int,int]]]:
            matched (bool): True if enough sectors exceed threshold.
            best_score (float): Highest sector score.
            best_loc (Tuple[int,int] or None): Location of best sector match.
    """
    # Grid configuration
    rows, cols = CONFIG.get("match_sector_grid", [2, 2])  # type: ignore
    required: int = int(CONFIG.get("match_sector_min_success", 1))  # type: ignore

    h, w = template.shape[:2]
    sector_width = w // cols
    sector_height = h // rows

    matches = 0
    best_score = 0.0
    best_loc: Optional[Tuple[int, int]] = None

    # Iterate sectors
    for i in range(rows):
        for j in range(cols):
            x0, y0 = j * sector_width, i * sector_height
            sub_tmpl = template[y0 : y0 + sector_height, x0 : x0 + sector_width]
            sub_mask = (
                mask[y0 : y0 + sector_height, x0 : x0 + sector_width]
                if use_mask and mask is not None
                else None
            )
            try:
                if use_mask and sub_mask is not None:
                    res = cv2.matchTemplate(region, sub_tmpl, method, mask=sub_mask)
                else:
                    res = cv2.matchTemplate(region, sub_tmpl, method)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_score:
                    best_score, best_loc = max_val, max_loc
                if max_val >= threshold:
                    matches += 1
            except Exception:
                continue

    if matches >= required:
        return True, best_score, best_loc
    return False, best_score, None


def update_overlay(size: Tuple[int, int], actions: Dict[str, Any]) -> np.ndarray:
    """
    Generate a visual overlay with configuration parameters and ROI outlines.

    This overlay can be blended over the video frame to provide real-time
    feedback on matching thresholds, preprocessing settings, and defined ROIs.

    Parameters:
        size (Tuple[int, int]):
            The height and width of the target frame as (height, width).
        actions (Dict[str, Any]):
            Dictionary of action definitions, each containing:
            - 'roi': Optional[Tuple[int, int, int, int]] specifying the region of interest.

    Returns:
        np.ndarray: A BGR image of the same size as 'size', containing text and ROI rectangles.
    """
    # Unpack frame dimensions
    height, width = size
    # Create a blank BGR overlay
    overlay: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

    # Configuration labels
    label1 = (
        f"Threshold: {CONFIG['threshold']:.2f} | "
        f"Cooldown: {CONFIG['cooldown']:.1f}s | "
        f"Sleep: {CONFIG['sleep']:.2f}s"
    )
    cv2.putText(overlay, label1, (10, 30), DEF_FONT, 1, (0, 0, 0), 2)
    cv2.putText(overlay, label1, (10, 30), DEF_FONT, 1, (255, 255, 255), 1)

    label2 = f"Matching: {CONFIG.get('match_method')}"
    cv2.putText(overlay, label2, (10, 65), DEF_FONT, 0.8, (0, 0, 0), 2)
    cv2.putText(overlay, label2, (10, 65), DEF_FONT, 0.8, (0, 255, 255), 1)

    label3 = (
        f"Invert:{'Y' if CONFIG.get('preprocess_invert') else 'N'} | "
        f"Blur:{'Y' if CONFIG.get('preprocess_blur') else 'N'} | "
        f"Eq:{'Y' if CONFIG.get('preprocess_equalize') else 'N'} | "
        f"Sector:{'Y' if CONFIG.get('match_sector_enabled') else 'N'} | "
        f"Mask:{'Y' if CONFIG.get('match_use_mask') else 'N'}"
    )
    cv2.putText(overlay, label3, (10, 95), DEF_FONT, 0.8, (0, 0, 0), 2)
    cv2.putText(overlay, label3, (10, 95), DEF_FONT, 0.8, (0, 255, 255), 1)

    # Draw ROI rectangles for each action
    for action in actions.values():
        roi = action.get("roi")
        if roi:
            x, y, w_box, h_box = roi  # type: ignore
            cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

    return overlay


def process_frame(
    frame: np.ndarray,
    actions: Dict[str, Any],
    last_sent_time: Dict[str, float],
    sock: Socket,
    windows_focused: bool,
    test: bool,
    overlay: np.ndarray,
) -> None:
    """
    Process a single video frame: detect templates, validate dependencies,
    draw feedback, and send commands via UDP socket if conditions met.

    Parameters:
        frame (np.ndarray): BGR frame from the camera.
        actions (Dict[str, Any]): Action definitions including 'template', 'roi', 'requires', etc.
        last_sent_time (Dict[str, float]): Mapping of action keys to last send timestamp.
        sock (socket.socket): UDP socket for sending action commands.
        windows_focused (bool): True if the game window has focus.
        test (bool): If True, do not send UDP packets but only log.
        overlay (np.ndarray): Precomputed overlay image to blend on the frame.

    Returns:
        None
    """
    gray_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pre_matches: Dict[str, Tuple[float, Tuple[int, int]]] = {}
    confirmed_matches: Dict[str, Tuple[float, Tuple[int, int]]] = {}

    # Phase 1: initial detection
    for key, action_data in actions.items():
        matched, score, loc = match_with_roi(gray_frame, action_data)
        tui_utils.tui_detect(key, score)
        if matched:
            pre_matches[key] = (score, loc)  # type: ignore

    # Phase 2: dependency validation
    for key, (score, loc) in pre_matches.items():
        action_data = actions.get(key, {})
        requires = set(action_data.get("requires", []))
        requires_not = set(action_data.get("requires_not", []))
        if not requires.issubset(pre_matches.keys()):
            continue
        if requires_not & pre_matches.keys():
            continue
        confirmed_matches[key] = (score, loc)

    # Phase 3: drawing and sending
    for key, (score, loc) in confirmed_matches.items():
        action_data = actions.get(key, {})
        now = time.time()
        should_send = (
            not test
            and action_data.get("send", True)
            and (
                key not in last_sent_time
                or (now - last_sent_time[key]) >= float(CONFIG.get("cooldown", 0.0))
            )
            and windows_focused
            and audio_utils.active_event.is_set()
        )
        if should_send:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(
                f"[{timestamp}] ✅ Match '{key}' ({score:.2f}) → {CONFIG.get('ip')}:{CONFIG.get('port')}"
            )
            sock.sendto(key.encode(), (CONFIG.get("ip"), CONFIG.get("port")))
            last_sent_time[key] = now

        if not CONFIG.get("headless", False):
            roi = action_data.get("roi")
            dims = action_data.get("dimensions", (0, 0))  # type: ignore
            w_box, h_box = dims
            top_left = (loc[0] + roi[0], loc[1] + roi[1]) if roi else loc  # type: ignore
            bottom_right = (top_left[0] + w_box, top_left[1] + h_box)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            label = f"{key} ({score:.2f})"
            cv2.putText(
                frame,
                label,
                (top_left[0], top_left[1] - 10),
                DEF_FONT,
                0.6,
                (0, 255, 0),
                2,
            )

    # Blend overlay and show preview
    if not CONFIG.get("headless", False):
        draw_match_preview(frame, gray_frame, pre_matches, actions)
        combined = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
        cv2.imshow("Webcam Bot", combined)


def draw_match_preview(
    frame: np.ndarray,
    gray_frame: np.ndarray,
    matches: Dict[str, Tuple[float, Tuple[int, int]]],
    actions: Dict[str, Any],
) -> None:
    """
    Draw thumbnail previews of matched templates at the bottom of the frame.

    Parameters:
        frame (np.ndarray): BGR frame to draw on.
        gray_frame (np.ndarray): Grayscale version of the frame for ROI extraction.
        matches (Dict[str, Tuple[float, Tuple[int,int]]]): Detected matches with scores and locations.
        actions (Dict[str, Any]): Action definitions including 'roi'.

    Returns:
        None
    """
    thumb_size: int = 100
    margin: int = 10
    x_offset: int = margin
    y_offset: int = frame.shape[0] - thumb_size - margin

    for key in sorted(matches):
        roi = actions[key].get("roi")  # type: ignore
        if not roi:
            continue
        region = extract_roi(gray_frame, roi)  # type: ignore
        processed = preprocess_image(region)
        thumb = cv2.resize(processed, (thumb_size, thumb_size))
        thumb_color = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        if x_offset + thumb_size > frame.shape[1] - margin:
            break
        frame[y_offset : y_offset + thumb_size, x_offset : x_offset + thumb_size] = (
            thumb_color
        )
        cv2.putText(
            frame, key, (x_offset, y_offset - 5), DEF_FONT, 0.5, (0, 255, 255), 1
        )
        x_offset += thumb_size + margin


def get_cv_key() -> int:
    """
    Wrapper for OpenCV waitKey to fetch a single key press.

    Returns:
        int: Lower 8 bits of the key code.
    """
    return cv2.waitKey(1) & 0xFF


def take_shot(camera_idx: int) -> Optional[str]:
    """
    Capture a single frame from the webcam and save it as an image file.

    Opens an interactive window displaying the live feed. User can press:
      - SPACE (32) to take a screenshot and save it to 'img/screenshot-YYYYMMDD-HHMMSS.png'
      - ESC (27) to exit without saving

    Parameters:
        camera_idx (int): Index of the camera device to use.

    Returns:
        Optional[str]: File path of saved screenshot if taken, None otherwise.
    """
    # Do not capture in headless mode
    if CONFIG.get("headless", False):
        print("🎥 Cannot capture screenshot in HEADLESS mode")
        return None

    cap = load_camera(camera_idx)
    window_title = "Scatta immagine"

    print("🎥 Premi SPAZIO per scattare, ESC per uscire")
    saved_path: Optional[str] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize and show window
        resize_win(
            window_title,
            CONFIG.get("window_size", [640, 480])[0],  # type: ignore
            CONFIG.get("window_size", [640, 480])[1],  # type: ignore
        )
        cv2.imshow(window_title, frame)

        key = get_cv_key()
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = os.path.normpath(
                os.path.join(IMG_DIR, f"screenshot-{timestamp}.png")
            )
            cv2.imwrite(filename, frame)
            print(f"📸 Immagine salvata in {filename}")
            saved_path = filename
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_path


def load_actions(actions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Load and validate user-defined action templates and masks.

    Reads each action's template image and optional mask, converts
    them to grayscale, checks ROI bounds, and builds a structured
    actions_data dictionary for runtime processing.

    Parameters:
        actions (Dict[str, Any]):
            Mapping of action keys to configuration dicts, each containing:
            - 'path' (str): File path to the template image.
            - 'mask' (Optional[str]): File path to a mask image (grayscale).
            - 'roi' (Optional[List[int]]): [x, y, w, h] region of interest.
            - 'requires' (Optional[List[str]]): Actions that must also match.
            - 'requires_not' (Optional[List[str]]): Actions that must not match.
            - 'send' (Optional[bool]): Whether to send the action command.

    Returns:
        Dict[str, Dict[str, Any]]:
            Processed actions data keyed by action name, each containing:
            - 'requires' (Set[str])
            - 'requires_not' (Set[str])
            - 'roi' (Optional[Tuple[int,int,int,int]])
            - 'path' (str)
            - 'send' (bool)
            - 'template' (np.ndarray): Grayscale template image.
            - 'dimensions' (Tuple[int,int]): (width, height) of template.
            - 'mask_data' (Optional[np.ndarray]): Grayscale mask image.
    """
    actions_data: Dict[str, Dict[str, Any]] = {}

    for key, info in actions.items():
        # Dependencies
        requires: Set[str] = set(info.get("requires", []))
        requires_not: Set[str] = set(info.get("requires_not", []))

        # Template path validation
        path: str = info.get("path", "")  # type: ignore
        if not os.path.exists(path):
            print(f"⚠️ Immagine mancante per '{key}': {path}")
            continue
        img: Optional[np.ndarray] = cv2.imread(path)
        if img is None:
            print(f"⚠️ Impossibile caricare '{path}'")
            continue
        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ROI validation
        raw_roi: Any = info.get("roi")
        roi: Optional[Tuple[int, int, int, int]] = None
        if (
            isinstance(raw_roi, list)
            and len(raw_roi) == 4
            and all(isinstance(x, int) for x in raw_roi)
        ):
            roi = tuple(raw_roi)  # type: ignore
        elif raw_roi is not None:
            print(f"⚠️ ROI non valido per '{key}': {raw_roi}")

        # Mask path validation
        mask_data: Optional[np.ndarray] = None
        mask_path: Optional[str] = info.get("mask")  # type: ignore
        if mask_path:
            if os.path.exists(mask_path):
                raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if raw_mask is None:
                    print(f"⚠️ Impossibile caricare maschera '{mask_path}'")
                elif raw_mask.shape != gray.shape:
                    print(f"⚠️ Maschera per '{key}' ha dimensioni diverse dal template!")
                    # Accept but warn
                    mask_data = raw_mask
                else:
                    mask_data = raw_mask
            else:
                print(f"⚠️ Maschera non trovata per '{key}': {mask_path}")

        # Build action entry
        actions_data[key] = {
            "requires": requires,
            "requires_not": requires_not,
            "roi": roi,
            "path": path,
            "send": bool(info.get("send", True)),
            "template": gray,
            "dimensions": (gray.shape[1], gray.shape[0]),  # width, height
            "mask_data": mask_data,
        }

    return actions_data


def handle_ocv_keys() -> Tuple[bool, bool]:
    """
    Process OpenCV key events to adjust configuration or request exit.

    The following keys are handled:
      - 'q': quit application
      - ',': increase cooldown (max 10.0s)
      - '.': decrease cooldown (min 0.0s)
      - '*': increase sleep interval between frames (max 5.0s)
      - '/': decrease sleep interval (min 0.0s)
      - '+', '=': increase matching threshold (max 1.0)
      - '-': decrease matching threshold (min 0.0)
      - 'm': cycle through available template matching methods
      - 'i': toggle color inversion preprocessing
      - 'b': toggle Gaussian blur preprocessing
      - 'e': toggle histogram equalization preprocessing
      - 's': toggle sector-based matching
      - 'u': toggle mask usage in matching

    Returns:
        exit_loop (bool): True if 'q' was pressed to exit.
        dirty_config (bool): True if any configuration parameter was modified.
    """
    dirty_config: bool = False
    exit_loop: bool = False

    key: int = get_cv_key()
    if key == ord("q"):
        exit_loop = True
        dirty_config = True
    elif key == ord(","):
        CONFIG["cooldown"] = min(10.0, float(CONFIG.get("cooldown", 1.0)) + 0.1)
        dirty_config = True
        print(f"⏫ Cooldown increased: {CONFIG['cooldown']:.2f}s")
    elif key == ord("."):
        CONFIG["cooldown"] = max(0.0, float(CONFIG.get("cooldown", 1.0)) - 0.1)
        dirty_config = True
        print(f"⏬ Cooldown decreased: {CONFIG['cooldown']:.2f}s")
    elif key == ord("*"):
        CONFIG["sleep"] = min(5.0, float(CONFIG.get("sleep", 0.2)) + 0.05)
        dirty_config = True
        print(f"⏫ Sleep increased: {CONFIG['sleep']:.2f}s")
    elif key == ord("/"):
        CONFIG["sleep"] = max(0.0, float(CONFIG.get("sleep", 0.2)) - 0.05)
        dirty_config = True
        print(f"⏬ Sleep decreased: {CONFIG['sleep']:.2f}s")
    elif key in (ord("+"), ord("=")):
        CONFIG["threshold"] = min(1.0, float(CONFIG.get("threshold", 0.85)) + 0.01)
        dirty_config = True
        print(f"🔼 Threshold increased: {CONFIG['threshold']:.2f}")
    elif key == ord("-"):
        CONFIG["threshold"] = max(0.0, float(CONFIG.get("threshold", 0.85)) - 0.01)
        dirty_config = True
        print(f"🔽 Threshold decreased: {CONFIG['threshold']:.2f}")
    elif key == ord("m"):
        methods = ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]
        current = CONFIG.get("match_method", methods[0])
        idx = (methods.index(current) + 1) % len(methods)
        CONFIG["match_method"] = methods[idx]
        dirty_config = True
        print(f"🔁 Matching method set to: {CONFIG['match_method']}")
    elif key == ord("i"):
        CONFIG["preprocess_invert"] = not bool(CONFIG.get("preprocess_invert", True))
        dirty_config = True
        print(
            f"🌓 Inversion {'enabled' if CONFIG['preprocess_invert'] else 'disabled'}"
        )
    elif key == ord("b"):
        CONFIG["preprocess_blur"] = not bool(CONFIG.get("preprocess_blur", True))
        dirty_config = True
        print(f"�️ Blur {'enabled' if CONFIG['preprocess_blur'] else 'disabled'}")
    elif key == ord("e"):
        CONFIG["preprocess_equalize"] = not bool(
            CONFIG.get("preprocess_equalize", True)
        )
        dirty_config = True
        print(
            f"📊 Equalization {'enabled' if CONFIG['preprocess_equalize'] else 'disabled'}"
        )
    elif key == ord("s"):
        CONFIG["match_sector_enabled"] = not bool(
            CONFIG.get("match_sector_enabled", False)
        )
        dirty_config = True
        print(
            f"🧩 Sector matching {'enabled' if CONFIG['match_sector_enabled'] else 'disabled'}"
        )
    elif key == ord("u"):
        CONFIG["match_use_mask"] = not bool(CONFIG.get("match_use_mask", False))
        dirty_config = True
        print(f"🎭 Mask usage {'enabled' if CONFIG['match_use_mask'] else 'disabled'}")

    return exit_loop, dirty_config


def frame_loop(
    camera_idx: int,
    actions_data: Dict[str, Any],
    sock: Socket,
    windows_focused: bool,
    test: bool,
) -> None:
    """
    Main loop for capturing and processing video frames from the webcam.

    Continuously reads frames from the specified camera, applies template matching
    logic via process_frame, updates overlays when configuration changes,
    handles interactive key commands via handle_ocv_keys, and sends matched
    commands over a UDP socket if all conditions are met.

    Parameters:
        camera_idx (int): Index of the camera device to open (e.g., 0 for the first camera).
        actions_data (Dict[str, Any]): Pre-loaded action definitions including templates, ROIs, and dependencies.
        sock (Socket): UDP socket used to send action command messages (keys) to the target.
        windows_focused (bool): Flag indicating whether the target application window has focus.
        test (bool): If True, runs in test mode: does not send UDP commands, only logs.

    Returns:
        None
    """
    last_sent_time: Dict[str, float] = {}
    cap = load_camera(camera_idx)

    # Prepare display window if not headless
    if not CONFIG.get("headless", False):
        width, height = CONFIG.get("window_size", [640, 480])  # type: ignore
        resize_win("Webcam Bot", width, height)

    error_count = 0
    max_errors = 10
    overlay = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                error_count += 1
                if error_count >= max_errors:
                    print("⚠️ Too many failed frame reads, exiting loop...")
                    break
                continue
            error_count = 0

            # Generate overlay on first successful frame or after config changes
            if overlay is None:
                overlay = update_overlay(frame.shape[:2], actions_data)

            # Process the current frame (matching, drawing, sending)
            process_frame(
                frame,
                actions_data,
                last_sent_time,
                sock,
                windows_focused,
                test,
                overlay,
            )

            # Handle interactive key events (exit or config adjustments)
            exit_loop, dirty_config = handle_ocv_keys()
            if dirty_config:
                overlay = update_overlay(frame.shape[:2], actions_data)

            # Break if exit requested by user or voice control
            if exit_loop or audio_utils.exit_event.is_set():
                break

            # Sleep interval between frames for CPU throttling
            sleep_time = float(CONFIG.get("sleep", 0.0))
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()
        cv2.destroyAllWindows()
