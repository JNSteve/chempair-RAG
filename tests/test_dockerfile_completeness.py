"""Guard: every local module server.py imports must be COPY'd into the
Docker image. Added after PR #30 crashed the Railway deploy — the Dockerfile
uses an explicit COPY list, and new modules had not been added to it."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_copies_every_local_module_server_imports():
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    local_modules = sorted(
        {
            module
            for module in re.findall(r"^from (\w+) import", server, re.MULTILINE)
            if (ROOT / f"{module}.py").is_file()
        }
    )
    assert local_modules, "expected server.py to import local modules"

    missing = [
        module for module in local_modules if f"COPY {module}.py" not in dockerfile
    ]
    assert not missing, (
        f"Dockerfile is missing COPY lines for: {missing} — the deploy will "
        "crash at import. Add them to the Dockerfile."
    )


def test_dockerfile_copies_start_entrypoint_and_spec():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY start.py" in dockerfile
    assert "COPY context-bot-spec.md" in dockerfile
