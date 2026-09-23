from pathlib import Path


def resolve(path: str) -> Path:
    """The one path rule every tool shares: quotes stripped, ~ expanded, relative means 'from home'."""
    p = Path(path.strip().strip('"')).expanduser()
    if not p.is_absolute():
        p = Path.home() / p
    return p.resolve()
