"""Idle wakeups of the controller's own background threads (no window, no models).

usage: python bench/idle_wakeups.py [--no-ticker]
Prints its pid, then idles; measure from outside with `top -pid <pid> -stats idlew`.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from chengzhang.app import controller as C
from chengzhang.app.store import Store

ctl = C.Controller(Store(Path(tempfile.mkdtemp())))
if "--no-ticker" not in sys.argv:
    threading.Thread(target=ctl._level_ticker, daemon=True).start()
threading.Thread(target=ctl._idle_watch, daemon=True).start()
threading.Thread(target=ctl._accessibility_watch, daemon=True).start()
print(os.getpid(), flush=True)
time.sleep(40)
