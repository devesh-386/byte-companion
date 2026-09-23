"""Runs the task suite against the real Byte on this computer and records a scored result.

    .venv\\Scripts\\python.exe -m evals.suite                 all 10 tasks
    .venv\\Scripts\\python.exe -m evals.suite --only T04,T06  a subset
    .venv\\Scripts\\python.exe -m evals.suite --list          show the tasks

Byte uses a throwaway home folder (memory, state) so a run never touches your real memories.
Ctrl+Alt+Esc (or Ctrl+Alt+Shift+Esc if Byte's desktop app already holds that) stops the whole run.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

SCRATCH = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".companion" / "suite"
RESULTS = Path(__file__).resolve().parent / "results"


def prepare_scratch() -> Path:
    # Our own folder, recreated every run so results never depend on leftovers.
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "home").mkdir(parents=True)
    return SCRATCH


def main() -> int:
    ap = argparse.ArgumentParser(prog="evals.suite")
    ap.add_argument("--only", help="comma-separated task ids, e.g. T04,T06")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--label", help="name for this run in the results (default: model + adapter)")
    args = ap.parse_args()

    scratch = prepare_scratch()
    os.environ["COMPANION_HOME"] = str(scratch / "home")  # must be set before companion.config is imported

    from companion import config
    from companion.app import build
    from companion.cancel import STOP
    from companion.hotkey import MOD_ALT, MOD_CONTROL, MOD_SHIFT, GlobalHotkey, HotkeyError
    from companion import winutil
    from evals.tasks import TASKS, Ctx

    tasks = TASKS
    if args.only:
        wanted = {t.strip().upper() for t in args.only.split(",")}
        tasks = [t for t in TASKS if t.id in wanted]
    if args.list:
        for t in TASKS:
            print(f"{t.id} {t.slug:22} {t.ask}")
        return 0

    abort = threading.Event()

    def on_hotkey():
        abort.set()
        STOP.stop_all("suite stopped by hotkey")

    hotkey, stop_key = None, "Ctrl+Alt+Esc"
    for mods, name in ((MOD_CONTROL | MOD_ALT, "Ctrl+Alt+Esc"), (MOD_CONTROL | MOD_ALT | MOD_SHIFT, "Ctrl+Alt+Shift+Esc")):
        try:
            hotkey = GlobalHotkey(on_hotkey, modifiers=mods, hotkey_id=0xB170)
            hotkey.start()
            stop_key = name
            break
        except HotkeyError:
            hotkey = None
    print(f"Stop the run any time with {stop_key if hotkey else '(no hotkey available: use Ctrl+C)'}.\n")

    current: dict = {}

    def confirm(call, summary):
        task = current["task"]
        ok = call.name in task.allow
        current["confirms"].append({"tool": call.name, "summary": summary, "approved": ok,
                                    "expected": ok})
        return ok

    comp = build(confirm=confirm, notify=lambda m: current.setdefault("notes", []).append(m),
                 progress=lambda s: print(f"  {s}..."), context="suite")
    label = args.label or (config.active_profile().name + ("+" + config.active_lora().stem if config.active_lora() else ""))
    print(f"Byte ready ({label}). Running {len(tasks)} task(s).\n")

    results = []
    for task in tasks:
        if abort.is_set():
            break
        ctx = Ctx(scratch=scratch)
        ctx.windows_before = {w.hwnd for w in winutil.list_windows()}
        current.update(task=task, confirms=[], notes=[])
        tools: list[str] = []
        errors: list[str] = []
        row = {"id": task.id, "slug": task.slug, "prompt": None, "status": None, "detail": "", "answer": "",
               "seconds": 0.0, "tools": tools, "tool_errors": errors, "confirms": current["confirms"]}
        started = time.monotonic()
        try:
            if task.setup:
                task.setup(ctx)
            if task.precheck:
                already, why = task.check(ctx, "")
                if already:
                    row.update(status="INVALID", detail=f"already true before Byte acted: {why}")
                    results.append(row)
                    print(f"{task.id} INVALID  {why}")
                    continue
            prompt = task.prompt(ctx)
            row["prompt"] = prompt
            comp.agent.reset()
            token = STOP.new_token()
            timer = threading.Timer(task.timeout_s, token.cancel, args=("timeout",))
            timer.start()

            def on_event(e):
                if e.kind == "tool_call":
                    tools.append(e.data["name"])
                elif e.kind == "tool_error":
                    errors.append(f"{e.data['name']}: {e.data['error'][:160]}")

            try:
                answer = comp.agent.run(prompt, on_event=on_event, cancel=token)
            finally:
                timer.cancel()
                STOP.release(token)
            row["answer"] = answer
            time.sleep(2.0)  # let the desktop settle (windows close, titles update)
            if token.cancelled:
                row.update(status="TIMEOUT" if token.reason == "timeout" else "STOPPED", detail=token.reason)
            else:
                passed, why = task.check(ctx, answer)
                row.update(status="PASS" if passed else "FAIL", detail=why)
        except Exception as e:
            row.update(status="ERROR", detail=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        finally:
            if task.teardown:
                try:
                    task.teardown(ctx)
                except Exception as e:
                    row["teardown_error"] = f"{type(e).__name__}: {e}"
            row["seconds"] = round(time.monotonic() - started, 1)
            if row not in results:
                results.append(row)
        unexpected = [c["tool"] for c in row["confirms"] if not c["expected"]]
        print(f"{task.id} {row['status']:7} {row['seconds']:6.1f}s  tools={tools}"
              + (f"  UNEXPECTED={unexpected}" if unexpected else "") + f"\n      {row['detail'][:150]}")

    if hotkey:
        hotkey.stop()
    comp.shutdown(keep_server=True)

    scored = [r for r in results if r["status"] in ("PASS", "FAIL", "TIMEOUT", "ERROR")]
    passed = sum(r["status"] == "PASS" for r in scored)
    unexpected_total = sum(1 for r in results for c in r["confirms"] if not c["expected"])
    summary = {"label": label, "when": dt.datetime.now().isoformat(timespec="seconds"),
               "passed": passed, "scored": len(scored), "invalid": sum(r["status"] == "INVALID" for r in results),
               "unexpected_tier2": unexpected_total, "aborted": abort.is_set(), "results": results}
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{stamp}-{label}.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\nSCORE {passed}/{len(scored)}  (invalid {summary['invalid']}, unexpected tier-2 requests "
          f"{unexpected_total}{', ABORTED' if abort.is_set() else ''})\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
