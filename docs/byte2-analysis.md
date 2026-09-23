# Byte 2.0: what we built, what we measured, how to make it better

Written 2026-09-23 at the end of phases 0-5. Every number here comes from a run on this laptop
(RTX 5050 8 GB); the raw results are in `evals/results/`.

## Where Byte stands

| Measure | Byte 1.0 (Qwen2.5-3B + persona LoRA) | Byte 2.0 (Qwen3.5-4B) |
|---|---|---|
| Task suite (real apps, checked by the computer) | **0/10** | **9/11** (stable across the last 3 runs; 9/10 on the original ten) |
| One-shot tool calls valid / right tool (70 requests) | 85% / 71% | 92% / 79% |
| Plain chat answered without a tool | 100% | 100% |
| Sees the screen | no | yes (3/3 staged errors explained) |
| Speed / VRAM | 78 tok/s / 2.8 GB | 83 tok/s / 4.3 GB |
| Tools | 15 | 30, in 7 plugin files, 6 shown per message |
| Safety | Allow/Deny for writes | 4 tiers in code, stop hotkey (13-16 ms), action log, overwrite guard, honesty check |

**Bigger was not better.** Qwen3-VL-8B (79% valid calls, 7.2 GB) and Qwen3.5-9B (69%, 6.8 GB) both lost to
the 4B. On 8 GB they also leave no room for anything else. Pick models by measuring, not by size.

## What moved the score (0 → 9/11)

Most of the jump came from **tools and harness**, not the model:

1. **New abilities** (phase 3-4): see the screen, read any window's buttons (UI Automation), click, type,
   keys, volume, media keys, VS Code, Spotify, find folders. Before, T02/T03/T07-T10 were impossible.
2. **Tool retrieval** (phase 1): the model sees 6 relevant tools, not 30.
3. **Tools that answer the real question**: `list_directory` had no dates, so *no* model could find the
   newest file; now it sorts by date and states `NEWEST FILE:`.
4. **Guards in code, not in the prompt**: `write_text_file` refuses to wipe a file; the honesty check
   catches "I've clicked OK" when no tool ran (T09 in run 2 did exactly that).
5. **12 steps instead of 6**: UI tasks need observe → click → observe → type → save.
6. **Fixing the harness**: checks that ran too early (VS Code takes seconds), a floating "Status" overlay that
   was screenshotted instead of the real window, a scorer that wanted the word "ValueError" when the model
   explained the error correctly.

Lesson: when an agent fails, first ask *could any model have succeeded with these tools?*

## What still fails, and why

Six suite runs on the 4B while fixing things: 4 → 8 → 9 → 7 → 9 → 9. The last three are the fair score.

- **T11 Telegram video: fails every run.** Byte opens Telegram and Saved Messages, then loops between
  `observe_window` and `focus_window` without finding a play control. Telegram *does* expose each message
  (a video shows up with a 'Duration' item), so the gap is knowing *which* element to click. The repeat guard
  catches identical calls but not A-B-A-B loops.
- **Flaky tasks (pass most runs).** T10 Notepad (once typed into the wrong place, once asked to close a window
  it shouldn't have, which was correctly denied), T05 PyTorch version (once read an older version off the
  search result), T09 (claimed "Clicking OK..." without a tool, before the honesty check covered that form).
- **Small-model habits seen in the logs:** describing an action instead of doing it; retrying a blocked
  call with the dangerous flag (`overwrite=true`); guessing window titles. Each now has a code-level guard,
  and each guard has a test built from the real transcript.
- **Barge-in:** you say "wait" and Byte keeps talking (see improvement 1).

## Improvements, ranked by value for effort

### Do next (small, high value)
1. ~~Fix barge-in~~ **DONE 2026-09-23**: `companion/aec.py` drives Windows' Voice Capture DSP (built-in AEC).
   Measured during a 12 s reply: echo RMS 0.069 → 0.020, loud blocks 65 → 5 of ~120, Whisper heard nothing of
   Byte, 0 false cut-ins in 3 trials (after dropping hallucination-prone words and using Whisper's own
   confidence). Still to confirm with a real human "wait": AEC can also dampen your voice while Byte talks.
2. **Catch A-B-A-B loops, not just repeats.** The repeat guard (added) stops `observe` twice in a row; T11
   still loops observe → focus → observe. Detect a repeating cycle in the last 4 calls and say so.
3. **Telegram video (T11).** Telegram exposes its messages as `DataItem`s; a video message has a
   'Duration' item. A small `telegram` plugin with `open_chat(name)` and `play_latest_video()` (find the
   newest message with a duration, click its thumbnail, press space) turns a 12-step puzzle into 2 calls.
   Same pattern as `play_spotify`: **a direct command beats UI automation beats vision.**
4. **Suite hygiene.** Tasks leave windows behind (a Notepad tab with unsaved "hello"). Teardowns should
   close what they opened *and* dismiss "save?" prompts they caused.

### Do soon (medium)
5. **Persona LoRA for the 4B.** Byte's voice currently comes from the prompt. Re-running
   `training/persona` on Qwen3.5-4B needs its full weights (~8 GB) and a check that PEFT supports the new
   architecture. Measure first: the prompt-only 4B may already be good enough (it stayed in character
   in every bake-off answer).
6. **Learned tool retrieval.** 30 tools: hit@6 100%, but top-1 only ~87%. Log which tool actually got
   used per request (actions.jsonl already has it) and fine-tune bge on those pairs.
7. **Thinking on for hard tasks only.** Measured off by default (D5). Try: thinking on when the first
   attempt fails, or for UI tasks with > 3 steps.
8. **Structured memory, part 2.** Categories and expiry are rule-based (fast, predictable). Next: let the
   user fix a category in the dashboard, and move reminders into memory so "remind me the day before my
   exam" works from a remembered date.

### Later (big)
9. **Teach by showing (approach C).** Record a demo ("this is how I open my Telegram video") as a replayable
   skill. The UIA tree plus the action log already contain what's needed.
10. **Always-available voice with a wake word** (openWakeWord, local) once barge-in is solid.
11. **A second, tiny model for fast things** (routing, fact extraction) so the 4B only does the hard work.

## Byte can now code itself (added 2026-09-23)

`companion/plugins/selfcode.py`: read/search its code (free); edit its code, write plugins, undo, restart
(all Allow/Deny). An edit is kept only if all tests pass (else auto-undone); every edit is backed up in
`~/.companion/self-edits`. Never editable: agent/tools/cancel/hotkey/actionlog/config, the plugin loader, the
self-edit tool, tests/ and evals/. Edits that change a `tier=` are refused, and self-written plugins' tools are
forced to Allow/Deny. Live: first attempt took 7 tries (predictable mistakes) → with a template in the tool
description, a new "countdown" plugin worked first try (it has an off-by-one; good first self-fix).

## How to judge the next change

Run both, before and after:
- `python -m evals.suite` (11 real tasks) and `python -m evals.bakeoff --only qwen3.5-4b`
- `python -m evals.toolpick_eval` plus the held-out set (`evals/tool_queries_holdout.jsonl`); only tune
  on the first.
- `python -m pytest -q` (unit tests; no GPU needed).

A change that doesn't move a number is a guess.
