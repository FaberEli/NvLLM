"""Small pure rendering helpers for the `nvllm` CLI (no I/O, no dependencies)."""
from __future__ import annotations

from datetime import datetime, timezone


def human_bytes(n: int) -> str:
    """Ollama-style human size: '1.2 kB', '4.9 GB' (1e3/1e6/1e9 base)."""
    n = max(0, int(n))

    def fmt(v: float, unit: str) -> str:
        return (f"{v:.1f}" if v < 10 else f"{v:.0f}") + " " + unit

    if n >= 1e9:
        return fmt(n / 1e9, "GB")
    if n >= 1e6:
        return fmt(n / 1e6, "MB")
    if n >= 1e3:
        return fmt(n / 1e3, "kB")
    return f"{n} B"


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_past(s: str | None) -> bool:
    dt = _parse_dt(s)
    return dt is not None and dt <= datetime.now(timezone.utc)


def human_time(s: str | None) -> str:
    """Relative time, Ollama-style: 'just now', '5m ago', 'in 30s'; 'Never' if unparseable."""
    dt = _parse_dt(s)
    if dt is None:
        return "Never"
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if abs(secs) < 5:
        return "just now"
    future = secs < 0
    secs = abs(secs)
    v, unit = secs, "s"
    for name, span in (("mo", 30 * 86400), ("w", 7 * 86400), ("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= span:
            v, unit = secs // span, name
            break
    text = f"{int(v)}{unit}"
    return f"in {text}" if future else f"{text} ago"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width, borderless table (Ollama's `list`/`ps` look)."""
    widths = [
        max(len(h), *(len(r[i]) for r in rows) if rows else 0)
        for i, h in enumerate(headers)
    ]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    return "\n".join([fmt_row(headers)] + [fmt_row(r) for r in rows])
