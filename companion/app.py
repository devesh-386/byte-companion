"""Assembles the whole companion (model server, LLM client, tools, emotion, memory, agent).
Shared by the terminal chat and the desktop UI."""
from dataclasses import dataclass, field
from typing import Callable

from . import config
from .actionlog import ActionLog
from .agent import Agent, ConfirmFn
from .llm import LlamaClient
from .persona import build_persona
from .plugins import USER_DIR, Deps, load_plugins
from .protocol import build_system_prompt
from .server import LlamaServer
from .toolpick import ToolPicker
from .tools import CHANGE, ToolRegistry


@dataclass
class Companion:
    server: LlamaServer
    agent: Agent
    launched_server: bool
    notes: list[str] = field(default_factory=list)
    extractor: object | None = None
    scheduler: object | None = None
    store: object | None = None

    def shutdown(self, keep_server: bool = False) -> None:
        if self.scheduler:
            self.scheduler.stop()
        if self.extractor:
            self.extractor.close()
        if self.launched_server and not keep_server:
            self.server.stop()


def make_server(port: int = config.PORT, profile=None) -> LlamaServer:
    profile = profile or config.active_profile()
    lora = config.active_lora() if profile == config.active_profile() else None
    return LlamaServer(config.LLAMA_SERVER, profile.gguf, config.HOST, port, profile.ctx,
                       config.GPU_LAYERS, config.LOG_DIR, lora=lora, profile=profile)


def build(port: int = config.PORT, confirm: ConfirmFn | None = None,
          on_memory_saved: Callable[[list[str]], None] | None = None,
          notify: Callable[[str], None] = print,
          progress: Callable[[str], None] = lambda s: None,
          run_scheduler: bool = False, context: str = "chat",
          restart: Callable[[], None] | None = None) -> Companion:
    notes: list[str] = []
    server = make_server(port)
    progress("Loading the language model onto the GPU")
    launched = server.start()
    llm = LlamaClient(server.base_url)

    registry = ToolRegistry(max_output_chars=config.TOOL_OUTPUT_MAX_CHARS)
    hooks = []
    store = embedder = None

    if (config.EMOTION_DIR / "model.pt").exists():
        progress("Loading the emotion model")
        from .hooks import EmotionHook
        from .ml.emotion import EmotionClassifier
        hooks.append(EmotionHook(EmotionClassifier(config.EMOTION_DIR)))
    else:
        notes.append("emotion model not trained yet (training/emotion/train.py)")

    extractor = None
    if config.EMBEDDER_DIR.exists():
        progress("Opening long-term memory")
        from .memory.embedder import Embedder
        from .memory.hook import FactExtractor, MemoryHook
        from .memory.store import MemoryStore
        embedder = Embedder(config.EMBEDDER_DIR)
        store = MemoryStore(config.MEMORY_DB, embedder)
        extractor = FactExtractor(llm, store, on_saved=on_memory_saved)
        hooks.append(MemoryHook(store, extractor))
        notes.append(f"memory: {store.count('fact')} facts, {store.count('episode')} conversations")
    else:
        notes.append("memory disabled (embedding model missing)")

    profile = config.active_profile()
    deps = Deps(notify=notify, store=store, vision=profile.vision, restart=restart)
    report = load_plugins(registry, deps)
    if USER_DIR.exists():
        # Plugins Byte wrote itself (or the user added): every one of their tools asks before running.
        extra = load_plugins(registry, deps, folder=USER_DIR, min_tier=CHANGE)
        report.failed.update(extra.failed)
        if extra.loaded:
            notes.append(f"your plugins: {', '.join(extra.loaded)}")
    for name, err in report.failed.items():
        notes.append(f"plugin {name} skipped: {err}")
    picker = None
    if embedder is not None and config.TOOLS_PER_TURN:
        picker = ToolPicker(embedder, registry, k=config.TOOLS_PER_TURN)
        notes.append(f"tools: {len(registry.names())} loaded, {config.TOOLS_PER_TURN} shown per message")

    if config.active_lora():
        notes.append(f"personality adapter: {config.active_lora().name}")

    persona = build_persona()
    agent = Agent(
        llm=llm,
        registry=registry,
        system_prompt=build_system_prompt(persona, registry.schemas(max_tier=CHANGE)),
        tool_picker=picker,
        prompt_for_tools=lambda schemas: build_system_prompt(persona, schemas),
        # Where mood/memories go must match how the brain's adapter was trained (see profiles.py).
        context_in_user_turn=profile.context_in_user,
        max_steps=config.MAX_AGENT_STEPS,
        history_budget_chars=int(config.HISTORY_BUDGET_TOKENS * config.CHARS_PER_TOKEN),
        hooks=hooks,
        confirm=confirm,
        max_tier=CHANGE,
        action_log=ActionLog(config.ACTION_LOG, context),
    )
    scheduler = None
    if run_scheduler and config.TASKS_FILE.exists():
        from .scheduler import Scheduler, make_task_runner
        scheduler = Scheduler(config.TASKS_FILE, config.DATA_HOME / "scheduler_state.json",
                              make_task_runner(llm, notify, config.USER_NAME, config.LOG_DIR / "scheduler.log"))
        scheduler.reload()
        scheduler.start()
        notes.append(f"scheduled tasks: {', '.join(t.name for t in scheduler.tasks if t.enabled) or 'none'}")
    return Companion(server, agent, launched, notes, extractor, scheduler, store)
