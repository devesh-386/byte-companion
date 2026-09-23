"""Tools live in plugins: one .py file per family, found at start-up.

    plugins/files.py ─┐
    plugins/web.py   ─┼─► load_plugins() ─► each file's register(registry, deps) ─► ToolRegistry
    plugins/...      ─┘         │
                                └─ a broken plugin is skipped (and its half-registered tools removed),
                                   so one bad file never takes Byte down.

A plugin file needs:
    ORDER = 30                              # where its tools appear in the prompt (low first)
    def register(registry, deps): ...       # deps: notify(), the memory store (may be None)
"""
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..tools import ToolRegistry

BUILTIN_DIR = Path(__file__).resolve().parent
# Plugins Byte (or the user) adds live outside the code folder: ~/.companion/plugins
USER_DIR = Path(os.environ.get("COMPANION_HOME", Path.home() / ".companion")) / "plugins"
log = logging.getLogger(__name__)


@dataclass
class Deps:
    notify: Callable[[str], None] = lambda msg: None
    store: Any = None   # MemoryStore, or None when memory is off
    vision: bool = False  # the brain can see images (screen plugin)
    restart: Callable[[], None] | None = None  # relaunch Byte (the desktop app provides it)


@dataclass
class LoadReport:
    loaded: dict[str, list[str]] = field(default_factory=dict)   # plugin -> tool names
    failed: dict[str, str] = field(default_factory=dict)         # plugin -> error


def load_plugins(registry: ToolRegistry, deps: Deps, folder: Path = BUILTIN_DIR,
                 include: set[str] | None = None, min_tier: int | None = None) -> LoadReport:
    """min_tier raises every tool the folder adds to at least that tier. User/self-written plugins load
    with min_tier=CHANGE, so a plugin can't declare its own tools 'safe' and skip Allow/Deny."""
    report = _load(registry, deps, folder, include)
    if min_tier is not None:
        for names in report.loaded.values():
            for n in names:
                tool = registry.get(n)
                tool.tier = max(tool.tier, min_tier)
    return report


def _load(registry: ToolRegistry, deps: Deps, folder: Path, include: set[str] | None) -> LoadReport:
    report = LoadReport()
    modules = []
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_") or (include is not None and path.stem not in include):
            continue
        try:
            if folder == BUILTIN_DIR:  # a normal import, so tests and plugins share one copy of the module
                module = importlib.import_module(f"{__name__}.{path.stem}")
            else:
                spec = importlib.util.spec_from_file_location(f"byte_plugin_{folder.name}_{path.stem}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            if not callable(getattr(module, "register", None)):
                raise TypeError("has no register(registry, deps) function")
            modules.append((getattr(module, "ORDER", 100), path.stem, module))
        except Exception as e:  # noqa: BLE001  a plugin that won't even import is reported, not fatal
            report.failed[path.stem] = f"{type(e).__name__}: {e}"

    for _, name, module in sorted(modules, key=lambda m: (m[0], m[1])):
        before = set(registry.names())
        try:
            module.register(registry, deps)
            report.loaded[name] = [n for n in registry.names() if n not in before]
        except Exception as e:  # noqa: BLE001
            for n in set(registry.names()) - before:  # all or nothing per plugin
                registry.unregister(n)
            report.failed[name] = f"{type(e).__name__}: {e}"
    for name, err in report.failed.items():
        log.warning("plugin %s skipped: %s", name, err)
    return report
