"""
Module tui_utils.py

Provides a cross-platform text-based user interface for headless mode using Rich.
Displays real-time configuration parameters in a live-updating table,
and a scrolling panel of recent log messages without requiring an OpenCV window.
Compatible with Windows, macOS, and Linux.
"""

import threading
import time
from collections import deque
from typing import Optional, Deque, Any
import builtins

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel

from conf_utils import CONFIG

# Event to signal the TUI thread to stop
_tui_stop_event: threading.Event = threading.Event()
# Buffer to hold recent log messages (max 10 entries)
_log_buffer: Deque[str] = deque(maxlen=10)

# Preserve the original print function
_original_print = builtins.print


def tui_log(message: str) -> None:
    """
    Append a message to the TUI log buffer.

    Parameters:
        message (str): The log message to display in the TUI.
    """
    _log_buffer.append(message)


def _build_table() -> Table:
    """
    Construct a Rich Table reflecting current CONFIG values and their shortcut commands.

    Returns:
        Table: A Rich Table with configuration parameters, commands, and current values.
    """
    table = Table(title="Webcam Bot TUI (headless mode)")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Commands", style="yellow", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Threshold", "+ / -", f"{CONFIG['threshold']:.2f}")
    table.add_row("Cooldown (s)", ", / .", f"{CONFIG['cooldown']:.2f}")
    table.add_row("Sleep (s)", "* / /", f"{CONFIG['sleep']:.2f}")
    table.add_row("Match method", "m", str(CONFIG.get("match_method")))
    table.add_row("Invert colors", "i", "Y" if CONFIG.get("preprocess_invert") else "N")
    table.add_row("Gaussian blur", "b", "Y" if CONFIG.get("preprocess_blur") else "N")
    table.add_row(
        "Histogram EQ", "e", "Y" if CONFIG.get("preprocess_equalize") else "N"
    )
    table.add_row(
        "Sector matching", "s", "Y" if CONFIG.get("match_sector_enabled") else "N"
    )
    table.add_row("Use masks", "u", "Y" if CONFIG.get("match_use_mask") else "N")

    return table


def _build_layout() -> Layout:
    """
    Create a two-pane layout: configuration table above, log panel below.
    Table pane auto-sizes to content; log pane fills remaining space.

    Returns:
        Layout: A Rich Layout object ready for rendering.
    """
    layout = Layout()
    layout.split_column(Layout(name="table"), Layout(name="logs", ratio=1))
    layout["table"].update(_build_table())

    log_panel = Panel(
        "\n".join(_log_buffer) or "<no logs>", title="Recent Logs", expand=True
    )
    layout["logs"].update(log_panel)
    return layout


def _tui_loop() -> None:
    """
    Internal loop that uses Rich Live to render and update
    the configuration table and the log panel periodically.

    Catches and logs exceptions to avoid crashing the TUI.
    """
    console = Console()
    with Live(console=console, refresh_per_second=4) as live:
        while not _tui_stop_event.is_set():
            try:
                live.update(_build_layout())
            except Exception as err:
                _original_print(f"[TUI][ERROR] {err}")
                break
            time.sleep(0.5)


def start_tui() -> threading.Thread:
    """
    Start the Rich-based TUI in a background thread.

    Overrides built-in print to redirect messages into the TUI log.

    Returns:
        threading.Thread: The thread running the TUI loop.
    """

    def _tui_print(*args: Any, sep: str = " ", end: str = "\n", **kwargs: Any) -> None:
        msg = sep.join(str(a) for a in args)
        tui_log(msg)

    builtins.print = _tui_print
    _tui_stop_event.clear()
    thread = threading.Thread(target=_tui_loop, daemon=True)
    thread.start()
    return thread


def stop_tui(thread: Optional[threading.Thread]) -> None:
    """
    Signal the TUI thread to stop and wait for it to finish.

    Restores the original print function.

    Parameters:
        thread (Optional[threading.Thread]): The TUI thread to stop.
    """
    # Restore original print
    builtins.print = _original_print
    # Signal stop and join thread
    if thread and thread.is_alive():
        _tui_stop_event.set()
        thread.join()
