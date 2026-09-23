"""Terminal chat.  python -m companion          (desktop UI: python -m companion.ui)"""
import argparse
import json
import os
import sys

from . import config
from .agent import Event
from .app import build
from .llm import LLMError
from .server import ServerError

DIM, CYAN, YELLOW, RED, MAGENTA, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[35m", "\033[0m"


def print_event(event: Event) -> None:
    d = event.data
    if event.kind == "text":
        print(d["text"], end="", flush=True)
    elif event.kind == "emotion" and d["label"] != "neutral":
        print(f"{MAGENTA}[{d['label']} {d['confidence']:.2f}]{RESET} ", end="", flush=True)
    elif event.kind == "memory" and d["recalled"]:
        print(f"{DIM}[recalled {len(d['recalled'])}: {d['recalled'][0]['text'][:60]}...]{RESET} ", end="", flush=True)
    elif event.kind == "tool_call":
        print(f"\n{YELLOW}  -> {d['name']}({json.dumps(d['arguments'], ensure_ascii=False)}){RESET}", flush=True)
    elif event.kind == "tool_result":
        preview = d["result"].replace("\n", " | ")
        preview = preview if len(preview) <= 160 else preview[:157] + "..."
        print(f"{DIM}  <- {preview}{RESET}", flush=True)
    elif event.kind == "tool_error":
        print(f"{RED}  !! {d['error']}{RESET}", flush=True)
    elif event.kind == "step_limit":
        print(f"\n{RED}  (hit the {d['max_steps']}-step limit){RESET}", flush=True)


def confirm_in_terminal(call, summary: str) -> bool:
    answer = input(f"\n{YELLOW}  {config.NAME} wants to run: {summary}\n  Allow? [y/N] {RESET}")
    return answer.strip().lower() in ("y", "yes")


def main() -> int:
    parser = argparse.ArgumentParser(prog="companion")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--keep-server", action="store_true",
                        help="leave llama-server running after exit (faster next start)")
    args = parser.parse_args()
    if sys.platform == "win32":
        os.system("")  # turns on ANSI colour handling in older Windows consoles

    try:
        comp = build(args.port, confirm=confirm_in_terminal,
                     on_memory_saved=lambda facts: None,
                     progress=lambda s: print(f"{DIM}{s}...{RESET}", flush=True))
    except ServerError as e:
        print(f"{RED}{e}{RESET}")
        return 1
    for note in comp.notes:
        print(f"{DIM}  {note}{RESET}")
    agent = comp.agent
    print(f"\n{CYAN}{config.NAME}{RESET} is here. Commands: /tools  /reset  /exit\n")

    try:
        while True:
            try:
                user = input(f"{CYAN}you >{RESET} ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user in ("/exit", "/quit"):
                break
            if user == "/reset":
                agent.reset()
                print(f"{DIM}(this chat cleared; long-term memory kept){RESET}\n")
                continue
            if user == "/tools":
                print(f"{DIM}{', '.join(agent.registry.names())}{RESET}\n")
                continue

            print(f"{CYAN}{config.NAME.lower()} >{RESET} ", end="", flush=True)
            try:
                agent.run(user, on_event=print_event)
            except KeyboardInterrupt:
                print(f"\n{DIM}(interrupted){RESET}")
            except LLMError as e:
                print(f"\n{RED}{e}{RESET}")
            print("\n")
    except KeyboardInterrupt:
        pass
    finally:
        comp.shutdown(keep_server=args.keep_server)
        print(f"{DIM}bye.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
