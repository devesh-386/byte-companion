import threading

import pytest

from companion.plugins import Deps, load_plugins
from companion.plugins.apps import best_app_match
from companion.plugins.web import html_to_text, parse_ddg
from companion.tools import ToolError, ToolRegistry

DDG_PAGE = """
<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org%2Fa&amp;rut=x">Python <b>3.14</b></a>
<a class="result__snippet" href="#">Released in <b>October</b>.</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></div>
<div class="result"><a class="result__a" href="https://example.com/b">No snippet here</a></div>
"""


def test_parse_ddg_unwraps_links_skips_ads_and_keeps_snippets_per_result():
    results = parse_ddg(DDG_PAGE, limit=5)
    assert results[0] == {"title": "Python 3.14", "url": "https://python.org/a", "snippet": "Released in October."}
    assert [r["url"] for r in results] == ["https://python.org/a", "https://example.com/b"]
    assert results[1]["snippet"] == ""


def test_ddg_block_page_detected():
    from companion.plugins.web import ddg_blocked
    assert ddg_blocked("<html>Unfortunately, bots use DuckDuckGo too. anomaly challenge</html>")
    assert not ddg_blocked(DDG_PAGE)


def test_html_to_text_drops_scripts_and_keeps_blocks():
    page = "<html><head><title>x</title></head><body><script>var a=1</script><h1>Hi</h1><p>Hello   there</p></body></html>"
    assert html_to_text(page) == "Hi\nHello there"


APPS = [("Spotify", "SpotifyAB.Spotify!App"), ("Visual Studio Code", "Microsoft.VisualStudioCode"),
        ("Uninstall Spotify", "x"), ("Steam Support Center", "http://support"), ("Steam", "D:/steam.exe"),
        ("Notepad++", "npp"), ("Notepad", "Microsoft.Notepad!App")]


@pytest.mark.parametrize("name,expected", [
    ("spotify", "Spotify"), ("VS Code", "Visual Studio Code"), ("steam", "Steam"),
    ("notepad", "Notepad"), ("visual", "Visual Studio Code"), ("minecraft", None),
])
def test_best_app_match(name, expected):
    match = best_app_match(name, APPS)
    assert (match[0] if match else None) == expected


def make():
    reg = ToolRegistry()
    notes = []
    load_plugins(reg, Deps(notify=notes.append))
    return reg, notes


def test_risky_tools_need_confirmation():
    reg, _ = make()
    from companion.tools import CHANGE, LOOK, OPEN
    assert reg.tier("write_text_file") == CHANGE and reg.tier("open_file") == CHANGE
    assert reg.tier("web_search") == LOOK and reg.tier("open_folder") == OPEN and reg.tier("launch_app") == OPEN


def test_write_text_file_and_append(tmp_path):
    reg, _ = make()
    target = tmp_path / "notes" / "todo.txt"
    reg.call("write_text_file", {"path": str(target), "content": "one\n"})
    reg.call("write_text_file", {"path": str(target), "content": "two\n", "append": True})
    assert target.read_text(encoding="utf-8") == "one\ntwo\n"


def test_fetch_rejects_non_http():
    reg, _ = make()
    with pytest.raises(ToolError, match="http"):
        reg.call("fetch_webpage", {"url": "file:///C:/Windows/win.ini"})


def test_open_folder_rejects_files(tmp_path):
    reg, _ = make()
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(ToolError, match="Not a folder"):
        reg.call("open_folder", {"path": str(f)})


def test_reminder_fires_notify():
    reg, notes = make()
    fired = threading.Event()
    out = reg.call("set_reminder", {"minutes": 0.001, "message": "stretch"})
    assert "stretch" in out
    for _ in range(50):
        if notes:
            break
        fired.wait(0.05)
    assert notes == ["Reminder: stretch"]
    with pytest.raises(ToolError):
        reg.call("set_reminder", {"minutes": 0, "message": "x"})



def test_write_refuses_to_wipe_an_existing_file(tmp_path):
    reg, _ = make()
    target = tmp_path / "todo.txt"
    target.write_text("- buy milk", encoding="utf-8")
    with pytest.raises(ToolError, match="append=true"):
        reg.call("write_text_file", {"path": str(target), "content": "- study"})
    reg.call("write_text_file", {"path": str(target), "content": "- study", "append": True})
    assert target.read_text(encoding="utf-8") == "- buy milk\n- study"   # new item on its own line
    reg.call("write_text_file", {"path": str(target), "content": "fresh", "overwrite": True})
    assert target.read_text(encoding="utf-8") == "fresh"


def test_list_directory_can_sort_newest_first(tmp_path):
    import os
    reg, _ = make()
    for i, name in enumerate(["old.txt", "newest.txt", "mid.txt"]):
        (tmp_path / name).write_text("x")
        os.utime(tmp_path / name, (1_700_000_000 + [0, 200, 100][i],) * 2)
    out = reg.call("list_directory", {"path": str(tmp_path), "sort": "newest"})
    names = [l.split("] ")[1].split("  (")[0] for l in out.splitlines() if l.startswith("  [")]
    assert names == ["newest.txt", "mid.txt", "old.txt"] and "newest first" in out
    assert "NEWEST FILE: newest.txt" in out
