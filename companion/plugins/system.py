"""The laptop itself: time, maths, hardware, reminders."""
import ast
import operator
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from typing import Annotated

from companion.tools import LOOK, OPEN, ToolError

ORDER = 10


def get_current_time() -> str:
    """Get the current local date, time and weekday on this computer."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculate(expression: Annotated[str, "Arithmetic expression, e.g. '4821 * 377' or '(2+3)**2 / 7'"]) -> str:
    """Evaluate an arithmetic expression exactly. Use this for any math instead of guessing."""
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise ToolError(f"Not a valid arithmetic expression: {expression!r}")
    result = _eval_node(tree.body)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return str(result)


def _eval_node(node: ast.AST) -> float:
    # Walk the syntax tree ourselves so only numbers and arithmetic can run — never eval().
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ToolError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError("Only numbers and + - * / // % ** ( ) are allowed")


def system_info() -> str:
    """Report this laptop's CPU, RAM, GPU, disk space and battery."""
    lines = [f"OS: {sys.platform}, Python {sys.version.split()[0]}", f"CPU cores: {os.cpu_count()}"]
    if sys.platform == "win32":
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = MEMSTAT()
        ms.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        lines.append(f"RAM: {ms.ullAvailPhys / 2**30:.1f} GB free of {ms.ullTotalPhys / 2**30:.1f} GB")

        class POWER(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        ps = POWER()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(ps)):
            pct = ps.BatteryLifePercent & 0xFF  # 255 means "unknown"
            if pct <= 100:
                lines.append(f"Battery: {pct}% ({'charging' if ps.ACLineStatus == 1 else 'on battery'})")
    for drive in ("C:\\", "D:\\") if sys.platform == "win32" else ("/",):
        if os.path.exists(drive):
            u = shutil.disk_usage(drive)
            lines.append(f"Disk {drive} {u.free / 2**30:.0f} GB free of {u.total / 2**30:.0f} GB")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode == 0:
            lines.append("GPU: " + out.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "\n".join(lines)


def _check_minutes(minutes: float) -> None:
    if not 0 < minutes <= 24 * 60:
        raise ToolError("Reminders must be between 0 and 1440 minutes away.")


def _dry_set_reminder(minutes: float, message: str) -> str:
    _check_minutes(minutes)
    due = datetime.now() + timedelta(minutes=minutes)
    return f"Reminder set for {due.strftime('%I:%M %p')}: {message}"


def register(registry, deps) -> None:
    reminders: list[threading.Timer] = []

    def set_reminder(minutes: Annotated[float, "Minutes from now"],
                     message: Annotated[str, "What to remind the user about"]) -> str:
        """Pop up a reminder after some minutes (only while the companion is running)."""
        _check_minutes(minutes)
        timer = threading.Timer(minutes * 60, deps.notify, args=(f"Reminder: {message}",))
        timer.daemon = True
        timer.start()
        reminders.append(timer)
        return _dry_set_reminder(minutes, message)

    registry.register(get_current_time, tier=LOOK, examples=(
        "what time is it", "what's today's date", "what day is it today"))
    registry.register(calculate, tier=LOOK, examples=(
        "what is 4821 times 377", "how much is 15% of 2400", "divide 1000 by 7"))
    registry.register(system_info, tier=LOOK, examples=(
        "how much battery do I have", "how much free disk space", "what's my GPU temperature", "how much RAM is free"))
    registry.register(set_reminder, tier=OPEN, dry=_dry_set_reminder, examples=(
        "remind me in 20 minutes to drink water", "set a timer for 10 minutes", "ping me in an hour to stretch"))
