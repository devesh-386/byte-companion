"""The web: search and read pages. The internet is only used to look things up."""
import html
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Annotated

from companion.tools import LOOK, ToolError

ORDER = 30
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def _http_get(url: str, timeout: float = 12, data: bytes | None = None) -> str:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(2_000_000).decode(charset, errors="replace")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", l).strip() for l in raw.splitlines()]
        return "\n".join(l for l in lines if len(l) > 1)


def html_to_text(page: str) -> str:
    p = _TextExtractor()
    p.feed(page)
    return p.text()


_DDG_TITLE = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S)
_DDG_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.S)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def parse_ddg(page: str, limit: int) -> list[dict]:
    titles = list(_DDG_TITLE.finditer(page))
    results = []
    for i, m in enumerate(titles):
        # Each result's snippet sits between its own title and the next result's title.
        block_end = titles[i + 1].start() if i + 1 < len(titles) else len(page)
        snip = _DDG_SNIPPET.search(page, m.end(), block_end)
        href = html.unescape(m.group("href"))
        # DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<real url>
        url = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        if "duckduckgo.com/y.js" in url:  # ads
            continue
        results.append({"title": _strip_tags(m.group("title")), "url": url,
                        "snippet": _strip_tags(snip.group("snippet")) if snip else ""})
        if len(results) >= limit:
            break
    return results


def ddg_blocked(page: str) -> bool:
    """DuckDuckGo answers too many requests with a bot-check page instead of results."""
    low = page.lower()
    return "result__a" not in low and ("anomaly" in low or "challenge" in low)


def wikipedia_search(query: str, limit: int) -> list[dict]:
    """Wikipedia's public API: free, keyless and meant for programs. Used when DuckDuckGo refuses."""
    params = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                     "format": "json", "srlimit": limit})
    data = json.loads(_http_get(f"https://en.wikipedia.org/w/api.php?{params}"))
    return [{"title": r["title"],
             "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
             "snippet": _strip_tags(r.get("snippet", ""))}
            for r in data.get("query", {}).get("search", [])]


def register(registry, deps) -> None:
    search_cache: dict[tuple[str, int], str] = {}

    def web_search(query: Annotated[str, "What to search the web for"],
                   max_results: Annotated[int, "How many results"] = 5) -> str:
        """Search the internet (DuckDuckGo) for current information, news, docs or anything you don't know."""
        limit = max(1, min(max_results, 8))
        key = (query.strip().lower(), limit)
        if key in search_cache:
            return search_cache[key]
        page = _http_get("https://html.duckduckgo.com/html/", data=urllib.parse.urlencode({"q": query}).encode())
        note = ""
        if ddg_blocked(page):
            results = wikipedia_search(query, limit)
            note = ("(The web search engine is rate-limiting us right now, so these are Wikipedia results; "
                    "they may not have the very latest news.)\n\n")
        else:
            results = parse_ddg(page, limit)
        if not results:
            return note + "No results found."
        text = note + "\n\n".join(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
                                  for i, r in enumerate(results, 1))
        search_cache[key] = text
        return text

    def fetch_webpage(url: Annotated[str, "Full http(s) URL"],
                      max_chars: Annotated[int, "Maximum characters of page text to return"] = 3000) -> str:
        """Download a web page and return its readable text. Use after web_search to read a result."""
        if not re.match(r"^https?://", url):
            raise ToolError("URL must start with http:// or https://")
        text = html_to_text(_http_get(url))
        return text[:max_chars] + (f"\n...[{len(text)} chars total]" if len(text) > max_chars else "")

    registry.register(web_search, tier=LOOK, examples=(
        "what's the latest version of pytorch", "search the web for the news today", "look up who won the match", "google something for me", "find a guide or tutorial online", "benchmarks and reviews of a product"))
    registry.register(fetch_webpage, tier=LOOK, examples=(
        "read this page https://example.com", "what does this article say", "summarise this link"))
