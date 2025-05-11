"""
Module tui_utils.py

Provides a cross-platform text-based user interface for headless mode using Rich.
Displays real-time configuration parameters in a live-updating table,
real-time detection scores side-by-side, and a scrolling panel of recent log messages.
Compatible with Windows, macOS, and Linux.
"""

import threading
import time
from collections import deque
from typing import Optional, Deque, Any, Dict
import builtins
import math

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel

from conf_utils import CONFIG

# Event to signal the TUI thread to stop
tui_stop_event: threading.Event = threading.Event()
# Buffer to hold recent log messages (max 10 entries)
_log_buffer: Deque[str] = deque(maxlen=10)
# Buffer to hold latest detection scores per action
_detect_buffer: Dict[str, float] = {}

# Preserve the original print function
_original_print = builtins.print

UPPER_LAYOUT_SIZE: int = 14
# Maximum number of actions per column and maximum number of columns
MAX_ACTIONS_PER_COL: int = UPPER_LAYOUT_SIZE - 5
MAX_COLS: int = 3


def tui_log(message: str) -> None:
    """
    Append a log message to the TUI log buffer.

    Args:
        message (str): The log message to display in the TUI.
    """
    _log_buffer.append(message)


def tui_detect(action: str, score: float) -> None:
    """
    Store the latest detection score for an action.

    Args:
        action (str): The action key.
        score (float): The detection score.
    """
    _detect_buffer[action] = score


def _build_config_table() -> Table:
    """
    Construct a Rich Table showing current CONFIG values and their commands.

    Returns:
        Table: A Rich Table with configuration parameters, commands, and current values.
    """
    table = Table(title="Configuration")
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


def _build_detect_table() -> Table:
    """
    Construct a Rich Table displaying the latest detection scores per action,
    distributed across multiple Action/Score column pairs if exceeding MAX_ACTIONS_PER_COL.

    Returns:
        Table: A Rich Table with action keys and their current scores.
    """
    items = list(_detect_buffer.items())
    # Determine how many column pairs are needed, up to MAX_COLS
    num_chunks = min(MAX_COLS, math.ceil(len(items) / MAX_ACTIONS_PER_COL))
    # Split items into chunks of at most MAX_ACTIONS_PER_COL
    chunks = [
        items[i * MAX_ACTIONS_PER_COL : (i + 1) * MAX_ACTIONS_PER_COL]
        for i in range(num_chunks)
    ]

    table = Table(title="Detection Scores")
    # Add a pair of columns (Action, Score) for each chunk
    for _ in range(num_chunks):
        table.add_column("Action", style="green", no_wrap=True)
        table.add_column("Score", style="magenta", justify="right")

    # Use MAX_ACTIONS_PER_COL rows, filling empty cells with blanks for alignment
    for row_idx in range(MAX_ACTIONS_PER_COL):
        row_cells: list[str] = []
        for chunk in chunks:
            if row_idx < len(chunk):
                key, score = chunk[row_idx]
                row_cells.extend([key, f"{score:.2f}"])
            else:
                row_cells.extend(["", ""])
        table.add_row(*row_cells)

    return table


def _build_layout() -> Layout:
    """
    Create a layout for the TUI: top row with config and detection tables side-by-side,
    and a logs panel below filling the remaining space.

    Returns:
        Layout: A Rich Layout object ready for rendering.
    """
    up_layout = Layout(name="upper")
    logs_layout = Layout(name="logs")

    up_layout.size = UPPER_LAYOUT_SIZE

    layout = Layout()
    layout.split_column(up_layout, logs_layout)
    layout["upper"].split_row(Layout(name="config"), Layout(name="detect"))
    layout["config"].update(_build_config_table())
    layout["detect"].update(_build_detect_table())

    log_panel = Panel(
        "\n".join(_log_buffer) or "<no logs>", title="Recent Logs", expand=True
    )
    layout["logs"].update(log_panel)
    return layout


def _tui_loop() -> None:
    """
    Internal loop that uses Rich Live to render and update the layout periodically.
    Catches exceptions to avoid crashes.

    Returns:
        None
    """
    console = Console()
    with Live(console=console, refresh_per_second=2) as live:
        while not tui_stop_event.is_set():
            try:
                live.update(_build_layout())
            except Exception as err:
                _original_print(f"[TUI][ERROR] {err}")
                break
            time.sleep(0.5)


def start_tui() -> threading.Thread:
    """
    Start the Rich-based TUI in a background thread and override print to log messages.

    Returns:
        threading.Thread: The thread running the TUI loop.
    """

    def _tui_print(*args: Any, sep: str = " ", end: str = "\n", **kwargs: Any) -> None:
        msg = sep.join(str(a) for a in args)
        tui_log(msg)

    builtins.print = _tui_print
    tui_stop_event.clear()
    thread = threading.Thread(target=_tui_loop, daemon=False, name="TUIThread")
    thread.start()
    return thread


def stop_tui(thread: Optional[threading.Thread]) -> None:
    """
    Signal the TUI thread to stop, restore the original print, and wait for it to finish.

    Args:
        thread (Optional[threading.Thread]): The TUI thread to stop.
    """
    builtins.print = _original_print
    if thread and thread.is_alive():
        tui_stop_event.set()
        thread.join()
