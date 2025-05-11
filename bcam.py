"""
Module bot_webcam_entry.py

Entry point for the webcam bot application. Handles:
- configuration initialization
- camera selection (including listing and resolution choose)
- game window focus selection
- loading and validating user-defined actions
- starting voice-control listener thread
- running the main frame loop
- graceful shutdown and config persistence
"""

import socket
from typing import Any, List, Optional, Tuple

# Windows only APIs
try:
    import win32gui
except ImportError:
    win32gui = None

# Optional DirectShow for camera enumeration on Windows
try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

import cv_utils
import audio_utils
import conf_utils
from conf_utils import CONFIG, ACTIONS
import tui_utils


def get_foreground_window_title() -> Optional[str]:
    """
    Retrieve the title of the currently focused window (Windows only).

    Returns:
        Optional[str]: The window title if available, None otherwise.
    """
    if win32gui is None:
        return None
    hwnd = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(hwnd)


def is_game_window_focused(should_check: bool) -> bool:
    """
    Determine whether the configured game window is in focus.

    Parameters:
        should_check (bool): If False, always return True (no focus check).

    Returns:
        bool: True if focus check disabled or if the foreground window title
              matches the configured 'game_window_title' (case-insensitive).
    """
    if not should_check:
        return True
    title = get_foreground_window_title()
    target = CONFIG.get("game_window_title") or ""
    return bool(title and title.lower() == target.lower())


def list_open_windows() -> List[str]:
    """
    List visible window titles on Windows (requires win32gui).

    Returns:
        List[str]: Titles of visible windows, empty list on non-Windows or error.
    """
    titles: List[str] = []
    if win32gui is None:
        return titles

    def _enum_handler(hwnd: int, _arg: Any) -> None:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)

    win32gui.EnumWindows(_enum_handler, None)
    return titles


def load_or_select_camera(
    force_select: bool = False, force_resolution: bool = False
) -> Optional[int]:
    """
    Discover available cameras and prompt user to select one.

    Parameters:
        force_select (bool): If True, ignore saved camera and prompt selection.
        force_resolution (bool): If True, always prompt resolution selection.

    Returns:
        Optional[int]: Selected camera index, or None if none available.
    """
    available: List[Tuple[int, str, Any]] = []
    selected_idx: Optional[int] = None
    selected_name: Optional[str] = None
    selected_caps: Any = []

    print("🎥 Scanning for available webcams...")

    # Attempt DirectShow on Windows
    use_dshow = False
    if FilterGraph:
        try:
            fg = FilterGraph()
            devices = fg.get_input_devices()
            for idx, name in enumerate(devices):
                try:
                    caps = fg.get_input_device_capabilities(idx)
                except Exception:
                    caps = []
                if cv_utils.try_cam(idx):
                    available.append((idx, name, caps))
            use_dshow = True
        except Exception as e:
            print(f"⚠️ DirectShow error: {e}, falling back to OpenCV")

    # Fallback: OpenCV enumeration
    if not use_dshow:
        for idx in range(10):
            if cv_utils.try_cam(idx):
                available.append((idx, f"Webcam {idx}", []))

    if not available:
        print("❌ No working webcams found.")
        return None

    # Try saved camera
    if not force_select and CONFIG.get("camera_name"):
        for idx, name, caps in available:
            if name == CONFIG.get("camera_name"):
                selected_idx, selected_name, selected_caps = idx, name, caps
                print(f"✅ Loaded saved webcam: '{name}' (index {idx})")
                break

    # Prompt selection if needed
    if selected_idx is None:
        print("📷 Available webcams:")
        for i, (idx, name, _) in enumerate(available):
            print(f"  {i}: {name}")
        while True:
            choice = input("👉 Select webcam index: ").strip()
            if not choice.isdigit():
                print("❌ Enter a valid number.")
                continue
            sel = int(choice)
            if 0 <= sel < len(available):
                selected_idx, selected_name, selected_caps = available[sel]
                CONFIG["camera_index"] = selected_idx
                CONFIG["camera_name"] = selected_name
                conf_utils.save_config()
                break
            print(f"❌ Please choose between 0 and {len(available)-1}.")

    # Resolution selection
    if force_resolution or "camera_resolution" not in CONFIG:
        if selected_caps:
            print("📏 Available resolutions:")
            for i, capab in enumerate(selected_caps):
                print(
                    f"  {i}: {capab['width']}x{capab['height']} @ {capab['max_fps']}fps"
                )
            while True:
                choice = input("👉 Select resolution index: ").strip()
                if not choice.isdigit():
                    print("❌ Enter a valid number.")
                    continue
                idx = int(choice)
                if 0 <= idx < len(selected_caps):
                    w = selected_caps[idx]["width"]
                    h = selected_caps[idx]["height"]
                    CONFIG["camera_resolution"] = [w, h]
                    conf_utils.save_config()
                    break
                print(f"❌ Choose between 0 and {len(selected_caps)-1}.")
        else:
            print("⚠️ No resolution info, using defaults.")

    return selected_idx


def choose_monitored_window() -> str:
    """
    Determine which window title to monitor for focus.

    Returns:
        str: The chosen window title, or 'unused' if focus check disabled.
    """
    title = CONFIG.get("game_window_title")
    if title:
        return title

    if win32gui is None:
        print("⚠️ Focus selection not available on this OS.")
        CONFIG["game_window_title"] = "unused"
        conf_utils.save_config()
        return "unused"

    windows = list_open_windows()
    if not windows:
        print("❌ No open windows detected.")
        CONFIG["game_window_title"] = "unused"
        conf_utils.save_config()
        return "unused"

    print("🔍 Open windows:")
    windows.append("unused")
    for i, w in enumerate(windows):
        print(f"  {i}: {w}")
    while True:
        choice = input("👉 Select window index: ").strip()
        if choice.isdigit() and int(choice) in range(len(windows)):
            sel = windows[int(choice)]
            CONFIG["game_window_title"] = sel
            conf_utils.save_config()
            return sel
        print("❌ Invalid selection.")


def main() -> None:
    """
    Entry point: initialize config, select camera/window, load actions,
    start voice listener, and run the processing loop.
    """
    conf_utils.init_config()

    listener: Optional[audio_utils.threading.Thread] = None
    tui_thread = None

    # Prepare UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        cam_idx = load_or_select_camera(
            conf_utils.args.reset_camera, conf_utils.args.reset_resolution
        )
        if cam_idx is None:
            return

        # Screenshot mode
        if conf_utils.args.shot:
            conf_utils.save_config()
            cv_utils.take_shot(cam_idx)
            return

        # Window focus configuration
        should_check = True
        win_title = choose_monitored_window()
        if win_title.lower() == "unused":
            should_check = False

        # Load action definitions
        actions_cfg = ACTIONS.get("actions", {})  # type: ignore
        if not actions_cfg:
            print("❌ No actions defined")
            return

        actions_data = cv_utils.load_actions(actions_cfg)  # type: ignore

        tui_thread = tui_utils.start_tui()

        # Start voice control if not in test mode
        if not conf_utils.args.test:
            listener = audio_utils.start_listening()

        print("🔍 Bot started")
        if should_check:
            print(f"🎯 Monitoring window: {win_title}")

        cv_utils.frame_loop(
            cam_idx,
            actions_data,
            sock,
            is_game_window_focused(should_check),
            conf_utils.args.test,
        )

    finally:
        tui_utils.stop_tui(tui_thread)

        if listener:
            audio_utils.stop_listening(listener)
        conf_utils.save_config()


if __name__ == "__main__":
    main()
