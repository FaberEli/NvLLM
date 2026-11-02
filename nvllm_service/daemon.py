"""Local systemd user service management — the `nvllm daemon` subcommand.

``daemon`` is a *local* command (like ``serve``): it manages this machine's
systemd *user* unit directly, without talking to a running service over HTTP.
That is what makes the resident service (re)installable from the CLI at any
time — no installer re-run needed. ``install.sh`` delegates to
``nvllm daemon install`` too, so the unit template has exactly one source of
truth (this module).

Stdlib only. The systemctl calls go through one seam (``_systemctl``) the
tests monkeypatch, so the suite stays GPU-/systemd-free.
"""
from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

UNIT_NAME = "nvllm"
UNIT_FILE = "nvllm.service"
_ACTIONS = ("install", "uninstall", "status", "start", "stop", "restart")

_DAEMON_HELP = """\
nvllm daemon — manage the systemd *user* service (local command; no server needed)

  nvllm daemon install      write the unit + enable + start
                            (the default install mode since 0.8.0 — use it to
                             (re)add the service after a --no-service install)
  nvllm daemon status       installed? enabled? active?
  nvllm daemon start        start the service
  nvllm daemon stop         stop the service
  nvllm daemon restart      restart the service
  nvllm daemon uninstall    disable + remove the unit

Options:
  --port P                  install only: the unit's port (default: the port
                            recorded in install.json, else $NVLLM_PORT, else 8004)

The service runs `nvllm serve` under systemd (autostart, restart-on-crash,
logs under the install root). In containers/CI skip it (--no-service at
install time) and run `nvllm serve` in the foreground instead.
"""


def unit_path() -> Path:
    """The systemd user unit path (honors $XDG_CONFIG_HOME, like systemd)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user" / UNIT_FILE


def install_root() -> Path:
    """The install root this command manages: $NVLLM_HOME, else ~/.nvllm."""
    return Path(os.environ.get("NVLLM_HOME") or (Path.home() / ".nvllm"))


def service_port(root: Path | None = None) -> int:
    """Unit port: install.json (recorded at install) → $NVLLM_PORT → 8004."""
    root = root or install_root()
    try:
        meta = json.loads((root / "install.json").read_text(encoding="utf-8"))
        if int(meta.get("port")):
            return int(meta["port"])
    except (OSError, ValueError, TypeError):
        pass
    try:
        return int(os.environ.get("NVLLM_PORT", "8004"))
    except ValueError:
        return 8004


def render_unit(root: Path, port: int) -> str:
    """The systemd user unit text (pure — unit-tested directly)."""
    return (
        "[Unit]\n"
        "Description=nvllm — resident LLM inference service (vLLM backend)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        f"ExecStart={root}/venv/bin/nvllm serve --port {port}\n"
        f"Environment=NVLLM_LOG_DIR={root}/logs\n"
        f"Environment=NVLLM_STATE_DIR={root}/state\n"
        "Environment=NVLLM_WORKER_MAX_JOBS=2\n"
        f"WorkingDirectory={root}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    """Run `systemctl --user ARGS` — the single seam the tests monkeypatch."""
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def _systemd_available() -> bool:
    return shutil.which("systemctl") is not None


def _say(msg: str) -> None:
    print(f"nvllm: {msg}")


def _record_service(root: Path, installed: bool) -> None:
    """Keep install.json's `service` field honest (no-op when it does not exist)."""
    p = root / "install.json"
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    meta["service"] = installed
    p.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _require_unit() -> bool:
    if not unit_path().exists():
        _say(f"no systemd user service installed ({unit_path()} not found)")
        _say("install it:  nvllm daemon install")
        return False
    return True


def install(port: int | None = None, root: Path | None = None) -> int:
    """Write the unit, daemon-reload, enable + start. rc 0 on success."""
    root = root or install_root()
    port = port or service_port(root)
    unit = unit_path()
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(render_unit(root, port), encoding="utf-8")
    _say(f"wrote {unit}")
    if not _systemd_available():
        _say("systemctl not found — no systemd here (container/CI?)")
        _say("start it in the foreground instead:  nvllm serve")
        _record_service(root, False)
        return 1
    _systemctl("daemon-reload")
    r = _systemctl("enable", "--now", UNIT_NAME)
    if r.returncode != 0:
        _say(f"could not enable/start the service: {(r.stderr or r.stdout).strip() or 'unknown systemd error'}")
        _say("start it in the foreground instead:  nvllm serve")
        _record_service(root, False)
        return 1
    _say(f"systemd user service '{UNIT_NAME}' installed and started (root: {root}, port: {port})")
    _say(f"   status:  systemctl --user status {UNIT_NAME}")
    _say("   Tip: 'loginctl enable-linger $USER' keeps it running at boot without a login session.")
    _record_service(root, True)
    return 0


def uninstall(root: Path | None = None) -> int:
    """Disable + stop the service and remove its unit file."""
    root = root or install_root()
    unit = unit_path()
    if not unit.exists():
        _say(f"nothing to remove ({unit} not found)")
        _record_service(root, False)
        return 0
    _systemctl("disable", "--now", UNIT_NAME)
    unit.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    _say(f"removed the systemd user service ({unit})")
    _record_service(root, False)
    return 0


def status() -> int:
    """Report installed / enabled / active. rc 1 when not installed."""
    unit = unit_path()
    if not unit.exists():
        _say(f"not installed — no unit file at {unit}")
        _say("install it:  nvllm daemon install")
        return 1
    if not _systemd_available():
        _say(f"unit file present ({unit}) but systemctl is unavailable")
        _say("run the service in the foreground instead:  nvllm serve")
        return 1
    en = _systemctl("is-enabled", UNIT_NAME).stdout.strip()
    ac = _systemctl("is-active", UNIT_NAME).stdout.strip()
    print(f"nvllm daemon: {UNIT_NAME}")
    print(f"  unit:     {unit}")
    print(f"  enabled:  {'yes' if en == 'enabled' else f'no ({en})'}")
    print(f"  active:   {ac}")
    if ac == "active":
        print("  → the service is running:  nvllm list")
    else:
        print(f"  → not running:  nvllm daemon start   (or: systemctl --user start {UNIT_NAME})")
    return 0


def _verb(verb: str) -> int:
    """start / stop / restart the service via systemctl."""
    if not _require_unit():
        return 1
    if not _systemd_available():
        _say("systemctl not found — no systemd here (container/CI?)")
        _say("manage the foreground process instead:  nvllm serve")
        return 1
    r = _systemctl(verb, UNIT_NAME)
    if r.returncode != 0:
        _say(f"systemctl --user {verb} {UNIT_NAME} failed: {(r.stderr or r.stdout).strip() or 'unknown error'}")
        return 1
    past = {"start": "started", "stop": "stopped", "restart": "restarted"}[verb]
    _say(f"nvllm service {past}")
    return 0


def cmd_daemon(argv: list[str]) -> int:
    """``nvllm daemon <action>`` — local systemd user service management."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_DAEMON_HELP)
        return 2 if not argv else 0
    action, rest = argv[0], list(argv[1:])
    if action not in _ACTIONS:
        guess = difflib.get_close_matches(action, _ACTIONS, n=1, cutoff=0.6)
        msg = f"nvllm: unknown daemon action '{action}'"
        if guess:
            msg += f" — did you mean '{guess[0]}'?"
        print(msg + "\n  try: nvllm daemon help", file=sys.stderr)
        return 2
    port: int | None = None
    i = 0
    while i < len(rest):
        a = rest[i]
        if a in ("-h", "--help"):
            print(_DAEMON_HELP)
            return 0
        if a == "--port" and i + 1 < len(rest):
            try:
                port = int(rest[i + 1])
            except ValueError:
                print(f"nvllm: --port needs a number (got '{rest[i + 1]}')", file=sys.stderr)
                return 2
            del rest[i:i + 2]
            continue
        i += 1
    if rest:
        print(f"nvllm: unrecognized argument: {rest[0]}", file=sys.stderr)
        return 2
    if action == "install":
        return install(port=port)
    if action == "uninstall":
        return uninstall()
    if action == "status":
        return status()
    return _verb(action)
