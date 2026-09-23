"""Desktop companion.  pythonw -m companion.ui"""
import sys
import threading
import traceback

from PySide6.QtCore import QLockFile, Qt
from PySide6.QtWidgets import QApplication

from .. import config
from .window import Bridge, CompanionWindow, build_tray


def _log_to_file() -> None:
    """pythonw has no console, so without this every crash would vanish silently."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(config.LOG_DIR / "ui.log", "a", encoding="utf-8", buffering=1)
    if sys.stdout is None or "pythonw" in sys.executable.lower():
        sys.stdout = sys.stderr = log

    def hook(exc_type, exc, tb):
        print("UNCAUGHT:", "".join(traceback.format_exception(exc_type, exc, tb)), file=log)

    sys.excepthook = hook
    threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)
    print(f"--- {config.NAME} UI starting ---", file=log)


def main() -> int:
    _log_to_file()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName(config.NAME)

    config.DATA_HOME.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(config.DATA_HOME / "ui.lock"))
    if not lock.tryLock(100):
        return 0  # already running

    bridge = Bridge()
    win = CompanionWindow(bridge)
    holder: dict = {}

    def boot() -> None:
        try:
            from ..app import build  # heavy imports (torch, transformers) happen off the UI thread
            comp = build(confirm=win.ask_confirmation,
                         on_memory_saved=bridge.memory_saved.emit,
                         notify=bridge.notify.emit,
                         progress=bridge.progress.emit,
                         run_scheduler=True,
                         restart=bridge.restart_requested.emit)
            holder["comp"] = comp
            bridge.booted.emit(comp)
        except Exception:
            bridge.boot_failed.emit(traceback.format_exc())
            return
        # Voice loads after the brain so text chat is usable as early as possible.
        voice = ears = None
        try:
            from ..voice import Ears, Voice
            if config.VOICE_MODEL.exists():
                voice = Voice(config.VOICE_MODEL, speed=config.VOICE_SPEED)
            ears = Ears(config.WHISPER_MODEL, config.WHISPER_DIR, hint=f"{config.NAME}, {config.USER_NAME}.")
        except Exception:
            traceback.print_exc()
        bridge.voice_ready.emit(voice, ears)

    def quit_app() -> None:
        win.hide()
        comp = holder.get("comp")
        if comp:
            comp.shutdown()
        lock.unlock()
        app.quit()

    def restart_app() -> None:
        """Byte restarting itself after changing its own code: a detached helper waits for this process
        to exit (the single-instance lock), then starts a fresh one."""
        import subprocess
        cmd = f'timeout /t 5 /nobreak >nul & start "" "{sys.executable}" -m companion.ui'
        subprocess.Popen(["cmd", "/c", cmd], cwd=str(config.ROOT),
                         creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
        quit_app()

    bridge.restart_requested.connect(restart_app, Qt.QueuedConnection)  # after the reply is shown

    from ..hotkey import GlobalHotkey, HotkeyError
    stop_key = GlobalHotkey(bridge.stop_pressed.emit)  # Ctrl+Alt+Esc: stop whatever Byte is doing
    try:
        stop_key.start()
    except HotkeyError as e:
        print(f"stop hotkey unavailable: {e}")
    app.aboutToQuit.connect(stop_key.stop)

    build_tray(app, win, quit_app)
    app.aboutToQuit.connect(lambda: holder.get("comp") and holder["comp"].shutdown())
    win.show()
    threading.Thread(target=boot, daemon=True).start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
