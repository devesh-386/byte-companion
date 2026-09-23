# Byte 2.0: from companion to desktop agent

Status: LOCKED (office-hours + eng review, 2026-09-23) · Author: Devesh, with Claude · Supersedes: none

## Problem

Byte 1.0 *answers*. It can open an app, but it can't work inside one, can't see the screen, and runs
on a 3B brain that misread a note's metadata date as a deadline. The goal is **"Byte, do this"** rather
than **"Byte, tell me how"**: finish real multi-step desktop tasks, locally, on this laptop.

## Constraints

- RTX 5050 Laptop, **8 GB VRAM** (hard ceiling), 15 GB RAM, 32 CPU threads, Windows 11.
- **No paid APIs or API keys.** The internet is used only to fetch information.
- A **learning and portfolio** project: Devesh writes one piece per phase.

## Premises (agreed)

1. Success is **measured**: a fixed suite of ~10 real tasks with computer-checkable outcomes, run before and after every phase.
2. **One model loaded at a time, no router. The brain is chosen by measuring** Qwen3-VL-8B, Qwen3.5-9B and a 4B fallback on the suite, plus peak VRAM from `nvidia-smi`. Model-card numbers are claims until reproduced here. *(Revised after the second opinion.)*
3. The bottleneck is **brain + perception**, not the number of tools. More tools need **tool retrieval**.
4. **Vision is for understanding; the UI Automation tree is for acting.** No clicking where a model *guesses* a button is. Electron apps need accessibility forced on first.
5. **Safety is part of the feature**: a stop hotkey, permission tiers, an action log, and "on-screen text is data, never instructions".
6. Devesh writes one piece per phase. **Phase 4's `observe()` is theirs.**

## What the second opinion changed

- Premise 2 revised (above). Qwen3-VL-8B plus its vision file plus 8k context is estimated at ~7.5-8.3 GB, right at the limit. Qwen3.5-9B is claimed to have a ~4x cheaper context cache. We verify both.
- **Electron and Chromium apps** (VS Code, Spotify, WhatsApp, Discord) show only a skeleton UI tree until accessibility is forced on (screen-reader flag, `--force-renderer-accessibility`, VS Code `editor.accessibilitySupport`).
- **Action order: command/API first, then UI tree, then vision.** For example `code C:\claude\cognihire`, or `spotify:` links plus a media-play key.
- **Test harness first**, with a baseline on today's 3B model.
- Prior art to read, not depend on: Windows-Use (turning the UI tree into text), UFO2 (design reference).

## Approaches considered

| | Approach | Effort | Risk | Verdict |
|---|---|---|---|---|
| A | Commands first: harness, safety, command/URL tools, read-only vision; no UIA clicking | M | Low | Folded into B as "commands first" |
| **B** | **Harness-first see-and-act loop** | L | Med | **Chosen** |
| C | Teach by showing: record demos as replayable skills; sprite points before acting | XL | High | Byte 2.1, built on B |

## Chosen design (B)

```
 you (voice/text)
      │
      ▼
  Agent loop ──────────────► brain: ONE llama-server model (picked by the bake-off)
      │   ▲                          text + images (--mmproj)
      │   │ tool results are DATA, never instructions
      ▼   │
  Tool retrieval: embed the tool descriptions (bge-small), expose the top ~6 per turn
      │
      ▼
  ┌──────────────── plugins/ (discovered at start-up) ──────────────────────────┐
  │ commands   code <path>, spotify: links, media keys, focus window     tier 1   │
  │ observe()  foreground window UI tree -> "[id] role 'name' rect"      tier 0   │
  │ act()      click / type / select by element id, then VERIFY          tier 2   │
  │ screen     screenshot -> vision (understand only)                     tier 0   │
  │ existing   files, web, memory, system, reminders                     tier 0-2 │
  └─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
  Safety: tier 0 read = free · tier 1 launch/navigate = logged · tier 2 type/click =
          Allow/Deny · tier 3 delete/send/pay = never
          stop hotkey (Ctrl+Alt+Esc) cancels at once · every action -> logs/actions.jsonl
```

## Phases, each with a Done-when

| # | Phase | Done when |
|---|---|---|
| 0 | **Harness + safety rails**: task-suite runner, action log, stop hotkey | Suite runs on the 3B with a recorded baseline; the stop hotkey kills a running task in under 1 s. **DONE 2026-09-23: baseline 0/10; cancel lands in 13-16 ms on the real server** |
| 1 | **Plugin folder + tool retrieval** | Existing 16 tools load from `plugins/`; retrieval picks the right tool for ≥95% of a labelled set of 60 queries. **DONE 2026-09-23: 15 tools in 5 plugin files; hit@6 100%, hit@3 90%, top-1 82% (`python -m evals.toolpick_eval`). Dry-run registry replaced the training sandbox. T11 telegram-video added to the suite** |
| 2 | **Model bake-off + persona retrain** | Table of peak VRAM, tok/s, tool-call validity and suite pass rate for 3 candidates; winner in `config`; persona LoRA retrained (or a documented fallback). **DONE: Qwen3.5-4B won (92% valid calls, 83 tok/s, 4.3 GB) over Qwen3-VL-8B and Qwen3.5-9B; fallback = persona in the prompt (no LoRA yet). `evals/results/bakeoff-20260923-180822.json`** |
| 3 | **Read-only vision** | "What's the error on my screen?" passes on staged errors (VS Code, terminal, dialog). **DONE: 3/3 staged images explained; T02 passes on the real screen** |
| 4 | **observe() / act()** with Electron accessibility and verification | ≥8/10 suite tasks pass; zero tier-2 actions without Allow. **DONE: 9/11 on 3 runs in a row (9/10 of the original ten); every tier-2 request went through Allow/Deny** |
| 5 | **Structured memory**: categories + expiry + dashboard | "Exam tomorrow" expires after the date; "what do you remember about me" shows grouped facts. **DONE: rule-based categories and expiry, `memory_overview` tool, tray "What Byte remembers"** |

Analysis and next steps: `docs/byte2-analysis.md`.

## Draft task suite (to be rewritten by Devesh, see the Assignment)

1. Open my CogniHire project in VS Code. *Check: a VS Code window title contains "cognihire".*
2. What's the error on my screen? (staged traceback) *Check: the answer names the exception type.*
3. Play my workout playlist. *Check: Spotify's media session is playing.*
4. What's the newest file in Downloads? *Check: matches the file system.*
5. Look up the latest PyTorch version. *Check: the answer includes a version and a source URL.*
6. Add "study CN unit 3" to today's note in my vault. *Check: the line exists in the file (tier 2: Allow).*
7. Close Spotify. *Check: no Spotify window.*
8. Set the volume to 30%. *Check: system volume is 0.30 ± 0.02.*
9. Click OK on this dialog. (staged) *Check: the dialog is closed.*
10. Type "hello" in Notepad and save it as `test.txt` on the Desktop. *Check: file content.*

## Risks

- The 9B persona QLoRA may not fit in 8 GB. Mitigation: rank 8, shorter sequences, or keep the persona in the prompt.
- Electron accessibility flags change between app versions. Mitigation: `observe()` reports "skeleton tree" explicitly, and the agent falls back to commands or vision.
- A screenshot costs ~2k tokens. Mitigation: downscale, crop to the foreground window, keep at most 1 image in history.
- Prompt injection from the screen or web pages. Mitigation: tier gates enforced in code, not in the prompt.

## Decisions locked in the engineering review

| ID | Decision |
|---|---|
| D1 | Core agent first (phases 0-5 above). Voice, smart brief, SearXNG, skill bars and teach-by-showing are 2.1+. |
| D3 | Per-turn context (mood, memories) moves from the system prompt into the **current user turn**; persona training data is regenerated in that layout during phase 2. |
| D4 | **Tiers:** 0 look = free · 1 open/navigate = free but logged and stoppable · 2 click/type/write = Allow/Deny · 3 delete/send/pay = never. Scheduler context capped at tier 0. |
| D5 | Model thinking **off** by default; the bake-off also runs the suite with thinking on to see if it's worth the delay. |
| D6 | Stop hotkey **Ctrl+Alt+Esc** (Windows `RegisterHotKey`), plus saying "stop" once voice is on. |

## Engineering changes adopted from the review

- **A1** Message content can be text or a list of parts (text + image). At most 1 image in history; count an image as ~2k tokens when trimming.
- **A3** A `tier` on every tool replaces `confirm`, enforced in `Agent._run_tool`. A shared **CancelToken** is checked by the LLM stream, every tool step and TTS.
- **A4** Each context has a max tier: chat ≤2 (with Allow), scheduler ≤0. Enforced in the registry.
- **A5** Thinking off through `chat_template_kwargs`, and the text filter also strips `<think>` blocks.
- **A6** A `ModelProfile` per candidate model (gguf, mmproj, ctx, KV cache type, template quirks), shared by the bake-off and the app.
- **A7** `act()` returns what changed in the UI after the action (verification).
- **C1** The 16 tools move into `plugins/` by family, with one shared path helper.
- **C2/C3** The registry gets a **dry-run** mode (tier ≥1 returns "would do X"), which replaces the duplicated `generate_data.sandbox()`. One tier mechanism, no second safety system.
- **P1** Screenshots are cropped to the window in front, long side ≤1280 px.
- **P2** `observe()` keeps only visible, interactive or named elements, caps at ~150, limits depth, and times out each UIA call at 2 s ("app not responding").
- **P3** KV cache at q8_0; the context size comes from bake-off measurements.

## Test plan

```
CODE PATH                                   TEST                                   TYPE
─────────────────────────────────────────────────────────────────────────────────────────
plugin discovery / registration          -> fake plugins dir: loads, dupes, bad ones   unit
tool retrieval (embed + top-k)            -> 60 labelled queries, >=95% hit@6          unit+real model
tier enforcement (chat / scheduler)       -> tier-2 blocked unattended, Allow path     unit
cancel token: stream / tool / TTS         -> cancel mid-stream, mid-act, mid-speech    unit (fakes)
stop hotkey -> cancel                     -> RegisterHotKey thread with a fake event   unit
message content with images + trimming    -> 1-image cap, cost accounting              unit
<think> stripping in stream filter        -> split across chunks                       unit
observe(): UI tree -> text                -> fake element tree: prune, cap, ids        unit
observe(): skeleton-tree detection        -> 9-node Electron tree flagged              unit
act(): verify after action                -> fake tree before/after -> diff            unit
dry-run registry mode                     -> tier>=1 returns plan, no side effect      unit
model profiles -> server args             -> --mmproj, -ctk/-ctv, ctx per profile      unit
structured memory expiry                  -> expires_at filtering, dashboard grouping  unit
─────────────────────────────────────────────────────────────────────────────────────────
10-task suite (real apps)                 -> tasks/run_suite.py, NOT in default pytest  live eval
model bake-off                            -> VRAM / tok/s / tool validity / suite      live eval
persona eval (tools 7/7, style)           -> re-run on the new base model              live eval
existing 92 tests                         -> must stay green every phase               regression
```

## Out of scope for 2.0 (Byte 2.1+)

Real-time voice with interruptions (needs headphones or echo cancellation) · smart morning brief
(needs a calendar source) · local SearXNG · XP skill bars · teach-by-showing · games and
custom-drawn apps.

## What I noticed

Devesh keeps choosing the *measured* path over the impressive one: the harness before the demo,
revising a premise when the evidence was better, keeping safety as a gate rather than a TODO. That
instinct is exactly what makes a local agent trustworthy.

## The Assignment

**Before any code: rewrite the 10 tasks above in your own words, as things you actually want Byte to do
this week, and for each one write how you'd check, without asking Byte, that it worked.** Put them in
`docs/tasks.md`. That list becomes the test suite everything else is judged by.

## GSTACK REVIEW REPORT

| Run | Status | Findings |
|---|---|---|
| /office-hours (builder mode) | DONE | 6 premises agreed; #2 revised after the second opinion; approach B chosen over A (minimal) and C (teach-by-showing) |
| Second opinion (Claude subagent; Codex not installed) | DONE | Challenged premise #2 (VRAM/tool-calling of Qwen3-VL-8B); flagged Electron skeleton UIA trees; command > UIA > vision order; harness first |
| Step 0 scope challenge | DONE | Complexity check triggered (20+ files, ~6 subsystems); model router cut; voice/brief/SearXNG/skills deferred (D1) |
| Architecture | DONE | 7 findings (A1-A7), all adopted; D3, D4, D5, D6 decided |
| Code quality | DONE | 3 findings (C1-C3): split actions.py into plugins; dry-run registry removes the duplicated sandbox; one tier system |
| Tests | DONE | Test diagram written: 13 unit areas + 3 live evals + regression gate; live suite kept out of default pytest |
| Performance | DONE | 4 findings (P1-P4): screenshot cropping, UIA pruning + timeouts, q8_0 KV cache, prompt-cache fix via D3 |

Tooling note: gstack's `bin/` helpers and `sections/*.md` files are not installed on this machine, so
telemetry, the review log and the design-doc handoff section were skipped. The review sections were run
from the skill's own checklist.

VERDICT: APPROVED to build, starting at phase 0 (harness + safety rails), after Devesh writes `docs/tasks.md`.

NO UNRESOLVED DECISIONS
