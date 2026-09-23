# Byte task suite

The 11 things Byte 2.0 must be able to *do*. Each has a check that the computer runs itself, so a
pass means it really happened, not that Byte said so. Every phase is judged by how many pass.

Run: `.venv\Scripts\python.exe -m evals.suite` (takes over the screen for a few minutes; Ctrl+Alt+Esc
stops it). Results go to `evals/results/`. The code for each task is in `evals/tasks.py`, and a test
keeps this file and that code in sync, so edit both together.

Rules the runner enforces:
- **Nothing real gets touched.** Files go in a scratch folder (`%USERPROFILE%\.companion\suite\`), and
  memory uses a separate scratch database. Volume is put back afterwards, and only windows the task opened get closed.
- **A task that already passes before Byte acts is marked INVALID, not PASS** (for example, music already playing).
- **Change actions (tier 2) are only approved if the task expects them.** Anything else is denied and
  counted as an unexpected action.
- Each task has a time limit. Hitting it cancels the task through the same stop mechanism as Ctrl+Alt+Esc.

## T01 open-project
**Ask:** "Open my CogniHire project in VS Code."
**Check:** a VS Code window whose title contains "cognihire" exists (VS Code may reuse a window; if one was already open, the task is INVALID).
**Allowed tier-2:** none (opening is tier 1).

## T02 explain-screen-error
**Ask:** "There's an error on my screen. What is it?"
**Setup:** a console window runs a small Python script that crashes with `ValueError: bad config value 'port'`.
**Check:** the answer names `ValueError`, or explains its cause (the port is "eighty", not a number).
**Allowed tier-2:** none.

## T03 play-music
**Ask:** "Play some music on Spotify."
**Setup:** Spotify is running and paused (if it was playing, it gets paused).
**Check:** Spotify's window title shows a track ("Artist - Song") instead of just "Spotify".
**Allowed tier-2:** none (play/pause is tier 1).

## T04 newest-download
**Ask:** "What's the newest file in my Downloads folder?"
**Check:** the answer contains the name of the most recently modified file in Downloads.
**Allowed tier-2:** none.

## T05 latest-pytorch
**Ask:** "What's the latest version of PyTorch?"
**Check:** the answer contains the current major.minor version from PyPI (fetched live during the check).
**Allowed tier-2:** none.

## T06 add-todo
**Ask:** "Add 'study CN unit 3' to my todo list at <scratch>\todo.txt."
**Setup:** `todo.txt` exists with two other items.
**Check:** the file contains "study CN unit 3" **and still contains the two old items**, so replacing the file fails.
**Allowed tier-2:** `write_text_file`.

## T07 close-window
**Ask:** "Close the 'close-me' folder window."
**Setup:** File Explorer opens a scratch folder called `close-me`. (Not Notepad: Windows 11 Notepad opens
files as tabs in *your* existing window, so closing "its" window could close your own tabs.)
**Check:** no window titled "close-me" that the suite opened remains.
**Allowed tier-2:** `close_window`.

## T08 set-volume
**Ask:** "Set my volume to 30%."
**Setup:** the volume is set to 50% (your original level is put back afterwards).
**Check:** the master volume is between 28% and 32%.
**Allowed tier-2:** none (volume is tier 1).

## T09 dismiss-dialog
**Ask:** "A dialog popped up. Click OK on it."
**Setup:** a message box titled "Byte Suite" appears.
**Check:** the "Byte Suite" dialog is gone.
**Allowed tier-2:** `click`.

## T10 notepad-save
**Ask:** "Open Notepad, type hello, and save it as <scratch>\hello.txt."
**Check:** `hello.txt` exists and its text is exactly "hello" (spaces trimmed). Writing the file directly
without opening Notepad also passes; commands before clicking is the rule, not a trick.
**Allowed tier-2:** `write_text_file`, `type_text`, `click`, `press_keys`.

## T11 telegram-video
**Ask:** "Play the latest video in my Telegram Saved Messages."
**Needs once, by you:** a short video *with sound* in your Saved Messages (the suite never sends anything).
**Check:** Telegram has an active audio session, meaning a video is really playing, not just the chat open.
Already playing beforehand counts as INVALID.
**Allowed tier-2:** `click`, `press_keys`. Added on request after Byte proposed it; this is what Phase 4 is measured on.
